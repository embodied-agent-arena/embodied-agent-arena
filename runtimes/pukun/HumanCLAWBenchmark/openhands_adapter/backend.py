"""Thin native motion/physics/metric adapter. All Habitat calls share one thread."""
import json,math
from pathlib import Path
from humanclaw_bench.evaluation.evaluator import HCFindNavInteractEvaluator
from humanclaw_bench.agent.types import PlannerResult
from humanclaw_bench.agent.skills import skill_to_text

ACTIONS={'walk_forward':0,'stop':1,'turn':2,'climb_up':3,'sit':4,'step_back':5,'side_step':6,'climb_down':7}
def native_action(agent,name,params):
    if name not in ACTIONS:raise ValueError('Unknown native action')
    def number(key,default,lo,hi):
        n=float(params.get(key,default))
        if not math.isfinite(n) or not lo<=n<=hi:raise ValueError(f'{key} must be within [{lo}, {hi}]')
        return n
    def direction():
        d=params.get('direction','left')
        if d not in ['left','right']:raise ValueError('direction must be left or right')
        return d
    if name=='walk_forward':
        speed=params.get('speed','slow')
        if speed not in ['slow','normal','fast']:raise ValueError('invalid speed')
        label=f'Walk<forward><{speed}>'
    elif name=='turn':label=f'Turn<{direction()}><{number("degrees",30,10,120)}>'
    elif name=='sit':label=f'Sit down<{number("height",.5,.15,.85)}>'
    elif name=='step_back':label=f'Step back<{number("distance",.25,.1,.6)}>'
    elif name=='side_step':label=f'Side step<{direction()}><{number("distance",.25,.1,.5)}>'
    else:label={'stop':'Stop/Stand','climb_up':'Climb upstairs<normal>','climb_down':'Walk downstairs<normal>'}[name]
    return agent._chooser_action({'action_id':ACTIONS[name],'action_name':label})

class NativeEpisode(HCFindNavInteractEvaluator):
    def run_episode_rollouts(self,episode,n_rollouts):
        assert n_rollouts==1
        self.episode=episode;self.steps=0;self.done=False;self.failed=False
        self.native_output=self.rollout_dir(episode,0);self.native_output.mkdir(parents=True,exist_ok=False)
        self.agent.reset(episode);self.observation=self._reset_env_for_rollout(episode)
        self.trajectory=self._new_trajectory_recorder(episode,0)
        self.recorder=self._new_metric_recorder(episode,0)
        return {'initialized':True}

    def context(self):
        return {'instruction':self.episode.instruction,'max_env_steps':self.episode.max_steps,'env_steps':self.steps,'terminal':self.done or self.failed,'runtime_failed':self.failed}

    def observe(self,workspace):
        from PIL import Image
        frames=Path(workspace)/'frames';frames.mkdir(exist_ok=True)
        path=frames/f'ego_{self.steps:03d}.png'
        Image.fromarray(self.observation.head_rgb).convert('RGB').save(path)
        if __import__('os').environ.get('ARENA_CASE_STUDY_DIR'):
            from embodied_harness.case_study_recording import publish_image
            publish_image(self.observation.head_rgb, 'humanclaw.head_rgb', source_path=path)
        return {**self.context(),'image_path':str(path.relative_to(workspace)),'view':'egocentric RGB; no ground-truth state'}

    def act(self,name,params,visible_state):
        if self.done or self.failed:raise RuntimeError('Episode is terminal; reset is unavailable')
        if self.steps>=self.episode.max_steps:raise RuntimeError('Native step limit reached')
        action=native_action(self.agent._planner,name,params)
        if not isinstance(visible_state,str) or len(visible_state)>2000:raise ValueError('visible_state must be concise text grounded in the last attached ego image')
        # The original scoring verifier receives the agent's explicit acknowledgement,
        # paired with semantics from exactly the pre-action image. Neither is oracle feedback.
        decision=PlannerResult(raw_plan={'visual_state_description':visible_state},action=action,
            planner_skill={'visible_state':visible_state,'action_name':action.action_name},
            verifier={'mode':'pukun_agent_action','note':'Native action-selection VLM replaced by coding policy; original scoring unchanged'})
        step=self.steps
        try:
            self.recorder.record_decision(step=step,decision=decision,find_observation=self.env.metric_find_observation())
            if action.skill=='stand':env_action={'stop':True,'skill':'stand','action':action.to_json()}
            else:
                generated=self.motion.generate(action.skill,action.cond)
                self.trajectory.record_before(step=step,action=action,action_text=skill_to_text(action),xb_world_75=generated.xb_world_75)
                env_action=generated.xb_world_75
            obs,_reward,done,info=self.env.step(env_action,reasoning=decision.raw_plan)
            if isinstance(info.get('body_state'),dict):
                self.trajectory.record_after(step=step,body_state=info['body_state'],object_states=dict(info.get('object_states') or {}))
            if action.skill!='stand':self.recorder.record_motion(step=step,action_skill=action.skill,info=info)
            self.observation=obs;self.steps+=1;self.done=bool(done or self.steps>=self.episode.max_steps)
            return {'env_steps':self.steps,'terminal':self.done,'executed_action':action.to_json()}
        except BaseException:
            self.failed=True
            raise

    def finish(self):
        self.motion.unload()
        self._finalize_rollout_artifacts(output_dir=self.native_output,trajectory=self.trajectory,
            metric_recorder=self.recorder,video_writer=None,rollout_succeeded=not self.failed)
        p=self.native_output/'metrics.json'
        return json.loads(p.read_text()) if p.exists() else None
