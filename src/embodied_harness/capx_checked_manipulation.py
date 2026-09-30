"""Bounded public-API manipulation with measured postconditions; no task oracle."""
import math


def install_checked_manipulation(session):
    import numpy as np
    import hashlib
    from pathlib import Path
    session.reset_info["checked_manipulation_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    api=session.live_api;original=api.functions
    fns=dict(original())

    def pose(value):
        if not isinstance(value,(list,tuple)) or len(value)!=2:raise ValueError('Expected (xyz metres, quaternion xyzw)')
        xyz,q=(np.asarray(v,dtype=float) for v in value)
        if xyz.shape!=(3,) or q.shape!=(4,) or not np.isfinite(xyz).all() or not np.isfinite(q).all() or np.linalg.norm(q)<1e-8:raise ValueError('Expected finite xyz[3] and nonzero xyzw[4]')
        return xyz,q/np.linalg.norm(q)

    def move_hand_checked(target_pose, arm=0, max_steps=40, position_tolerance=.02, orientation_tolerance=.35):
        """Move physically to caller world xyz + xyzw pose, then measure reach.
        At most two IK solve/motion attempts, each with max_steps (1..60) command iterations.
        Native settling/gripper steps are additional; max_steps is not a whole-primitive step count.
        Returns arrived, position_error_m and orientation_error_rad; never sets state.
        """
        xyz,q=pose(target_pose)
        if arm not in (0,1) or not 1<=max_steps<=60 or not .001<=position_tolerance<=.05 or not .01<=orientation_tolerance<=.5:raise ValueError('Invalid bounded motion arguments')
        adjusted=xyz.copy();attempts=[]
        for _ in range(2):
            joints=fns['solve_ik'](position=adjusted,quaternion_wxyz=q[[3,0,1,2]],arm=arm,offset_translation=np.zeros(3))
            joint_reached=fns['move_to_joint_positions'](target_joint_positions=np.asarray(joints,dtype=np.float32),max_steps=max_steps,settle_steps=max_steps)
            actual,actual_q=fns['get_current_eef_pose'](arm=arm)
            actual=np.asarray(actual);actual_q=np.asarray(actual_q);actual_q=actual_q/np.linalg.norm(actual_q)
            error=float(np.linalg.norm(xyz-actual));rotation=2*math.acos(float(np.clip(abs(np.dot(q,actual_q)),0,1)))
            attempts.append(dict(joint_target_reached=bool(joint_reached),actual_position=actual.tolist(),position_error_m=error,orientation_error_rad=rotation))
            if error<=position_tolerance and rotation<=orientation_tolerance:break
            adjusted+=xyz-actual
        arrived=error<=position_tolerance and rotation<=orientation_tolerance
        return dict(executed=True,arrived=arrived,requested_position=xyz.tolist(),requested_quaternion_xyzw=q.tolist(),
                    position_error_m=error,orientation_error_rad=rotation,attempts=attempts,
                    reason=None if arrived else 'measured_pose_not_reached')

    def locate_object_for_grasp(object_name):
        """Locate named object from current RGB-D and sample fresh public grasp poses.
        View changes are separate caller actions, never an implicit long search.
        No hidden object poses or task goals are read. Candidate choice remains with caller.
        """
        def locate():
            try:return fns['get_object_pose'](object_name=object_name,return_bbox_extent=True)
            except ValueError as error:
                if 'No sam3 detections' in str(error):return (None,None,None)
                raise
        found=locate()
        if not found or found[0] is None:return dict(located=False,reason='not_visible_in_current_view',next_action='Change base/camera view, then localize again')
        candidates=fns['sample_grasp_pose'](object_name=object_name)
        available=candidates is not None and candidates[0] is not None and candidates[1] is not None and len(candidates[0])>0 and len(candidates[1])>0
        return dict(located=True,object_name=object_name,pose=found,grasp_candidates=candidates,
                    grasp_available=available,source='current_public_rgbd',candidate_selection='caller')

    def grasp_object(pregrasp_pose, grasp_pose, object_name, arm=0, max_steps=40):
        """Checked grasp: planned approach, contact descent, close, planned lift.
        Pose pairs use world xyz and quaternion xyzw. Uses caller's fresh grasp candidate.
        Complete requires actual hold and measured lift. Navigation feasibility
        is attempted by the native navigation planner with collision filters off. Free-space
        plans are capped at 2*(max_steps+50) physical steps including settling.
        Failure returns stage/reason. Physical contacts and measured arrival remain active.
        """
        pre_xyz,pre_q=pose(pregrasp_pose);pose(grasp_pose)
        if isinstance(arm,bool) or arm not in (0,1) or isinstance(max_steps,bool) or not isinstance(max_steps,int) or not 1<=max_steps<=60:
            raise ValueError('Invalid bounded grasp arguments')
        if fns['check_object_in_hand'](arm=arm):return dict(executed=False,grasped=True,lifted=False,ready_to_move=False,stage='precondition',reason='arm_already_holding_object')
        control=api._arena_carry_navigation.control
        stages=[];fns['open_gripper'](arm=arm)
        for name,target in [('approach',pregrasp_pose),('descend',grasp_pose)]:
            # Approach uses the session's native planning policy. Descent
            # retains its bounded controller and measured physical arrival.
            result=(control.move_hand(pre_xyz,pre_q,arm,max_environment_steps=2*(max_steps+50))
                    if name=='approach' else move_hand_checked(target,arm=arm,max_steps=max_steps))
            stages.append(dict(stage=name,**result))
            if not result['arrived']:return dict(executed=True,grasped=False,stage=name,reason=result['reason'],stages=stages)
        fns['close_gripper'](arm=arm)
        held=bool(fns['check_object_in_hand'](arm=arm))
        if not held:return dict(executed=True,grasped=False,stage='close',reason='no_object_held',stages=stages)
        lift=control.move_hand(pre_xyz,pre_q,arm,max_environment_steps=2*(max_steps+50));stages.append(dict(stage='lift',**lift))
        held=bool(fns['check_object_in_hand'](arm=arm))
        lifted=bool(lift['arrived'] and held)
        ready=lifted
        return dict(executed=True,grasped=held,lifted=lifted,ready_to_move=ready,
                    stage='complete' if ready else 'lift',
                    reason=None if ready else lift.get('reason') or 'lift_or_hold_not_confirmed',
                    carry_start=None,initial_collision_precheck_enabled=False,
                    navigation_clearance_verified=False,stages=stages,
                    requested_object_name=object_name,object_identity_verified=False)

    def place_object_checked(target_pose, preplace_pose=None, arm=0, max_steps=40):
        """Place currently held object using caller's world end-effector pose (xyzw).
        Only open after measured arrival, then verify release and retreat. Release
        does not imply task success or correct semantic placement.
        """
        xyz,q=pose(target_pose)
        approach=pose(preplace_pose) if preplace_pose is not None else (xyz+np.array([0.,0.,.15]),q)
        if not fns['check_object_in_hand'](arm=arm):return dict(executed=False,released=False,stage='precondition',reason='no_object_held')
        stages=[]
        for name,target in [('approach',approach),('descend',(xyz,q))]:
            result=move_hand_checked(target,arm=arm,max_steps=max_steps);stages.append(dict(stage=name,**result))
            if not result['arrived']:return dict(executed=True,released=False,stage=name,reason=result['reason'],stages=stages)
            if not fns['check_object_in_hand'](arm=arm):return dict(executed=True,released=None,stage=name,reason='object_lost_before_release',stages=stages)
        fns['open_gripper'](arm=arm);released=not bool(fns['check_object_in_hand'](arm=arm))
        if released:stages.append(dict(stage='retreat',**move_hand_checked(approach,arm=arm,max_steps=max_steps)))
        retreated=bool(released and stages[-1]['stage']=='retreat' and stages[-1]['arrived'])
        return dict(executed=True,released=released,retreated=retreated,
                    stage='complete' if retreated else 'retreat' if released else 'release',stages=stages,
                    task_success_claimed=False,semantic_placement_verified=False)

    def functions():
        return {**original(),'locate_object_for_grasp':locate_object_for_grasp,'move_hand_checked':move_hand_checked,
                'grasp_object':grasp_object,'place_object_checked':place_object_checked}
    api.functions=functions
    api._arena_checked_manipulation=True
    session.reset_info['checked_manipulation']='v3: measured arrival/hold/release; native planned approach/lift; no added static-start gate'
