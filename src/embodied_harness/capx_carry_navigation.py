"""Bounded physical carry preparation and navigation for the R1Pro API.

No scene-object identity or task goal is exposed. Native grasp registration is
used only for collision attachment and retention of an already held object.
"""
import math
import time

import numpy as np


def world_goal(value):
    goal = np.asarray(value, dtype=float)
    if goal.shape != (3,) or not np.isfinite(goal).all():
        raise ValueError('goal must be finite WORLD [x_m, y_m, yaw_rad]')
    return goal


def pose_errors(actual, goal):
    return (float(np.linalg.norm(actual[:2] - goal[:2])),
            abs(math.atan2(math.sin(actual[2] - goal[2]), math.cos(actual[2] - goal[2]))))


def recoverable(status):
    return str(status).endswith(('INVALID_START_STATE_WORLD_COLLISION',
                                'INVALID_START_STATE_SELF_COLLISION'))


class CarryNavigation:
    """Two public operations sharing one bounded recovery, with measured results."""

    def __init__(self, control):
        self.control = control
        self.feedback = {}

    def _holding(self):
        return self.control.holding()

    def _preserved(self, before):
        after = self._holding()
        return all(after.get(arm) is obj for arm, obj in before.items())

    def _prepare(self, arm=None):
        if arm is not None and (isinstance(arm, bool) or arm not in (0, 1)):
            raise ValueError('arm must be 0 (left), 1 (right), or omitted')
        started = time.monotonic()
        steps_before = self.control.steps()
        held_before = self._holding()
        feedback = dict(ready_to_move=False, executed=False, stage='prepare_carry',
                        attempts=[], held_arms_before=sorted(held_before),
                        holding_preserved=True, object_identity_verified=False,
                        collision_checks_disabled=bool(getattr(self.control.generator, '_arena_planner_collisions_disabled', False)),
                        initial_collision_precheck_enabled=False,
                        navigation_clearance_verified=False)
        self.feedback = feedback
        try:
            if self.control.episode_ended():
                feedback.update(stage='episode_end', reason='native_episode_already_ended')
                return feedback
            base = self.control.base_pose()
            # Prefer an occupied arm, otherwise the hand farthest from the base.
            arms = [arm] if arm is not None else sorted(held_before) or [0, 1]
            poses = {a: self.control.eef_pose(a) for a in arms}
            selected = max(arms, key=lambda a: np.linalg.norm(poses[a][0][:2] - base[:2]))
            xyz, quat = poses[selected]
            toward_base = base[:2] - xyz[:2]
            toward_base /= max(float(np.linalg.norm(toward_base)), 1e-8)
            # Two short candidates relative to the measured initial hand pose;
            # neither resets joints nor assumes a task-specific carry posture.
            candidates = [xyz + [0., 0., .08],
                          xyz + np.r_[.12 * toward_base, .12]]
            feedback.update(stage='prepare_carry', selected_arm=selected)
            for target in candidates:
                result = self.control.move_hand(target, quat, selected)
                feedback['attempts'].append(result)
                feedback['holding_preserved'] = self._preserved(held_before)
                if not feedback['holding_preserved']:
                    feedback.update(stage='hold_check', reason='held_object_lost_or_changed')
                    return feedback
                if result.get('abort_recovery'):
                    feedback.update(reason=result['reason'])
                    return feedback
                if result['arrived']:
                    feedback.update(ready_to_move=True, stage='complete', reason=None)
                    return feedback
            feedback.update(stage='prepare_carry', reason='no_carry_posture_reached')
            return feedback
        finally:
            feedback['environment_steps'] = self.control.steps() - steps_before
            feedback['executed'] = feedback['environment_steps'] > 0
            feedback['elapsed_seconds'] = time.monotonic() - started
            feedback['held_arms_after'] = sorted(self._holding())

    def prepare_carry(self, arm=None):
        """Physically lift/retract one arm using the native planner.

        At most two ARM plans: 8 cm lift, then a candidate
        12 cm toward the base and 12 cm above the initial hand position.
        arm is 0 (left), 1 (right), or omitted to choose a held/extended arm.
        ready_to_move means a carry pose was reached while preserving any hold.
        It permits a navigation attempt; it does not certify navigation clearance
        or the identity of the held object. No extra static-start check is run.
        Print the result and end this code cell to receive fresh multi-view RGB.
        """
        return self._prepare(arm)

    def navigate_checked(self, goal, recover=True):
        """Navigate to WORLD [x metres, y metres, yaw radians], preserving hold.

        One original-target navigation, at most one prepare_carry recovery on a
        start-collision failure, then one retry ONLY if posture actually changed
        and a carry pose was reached. No alternate target, teleport, or release.
        Success requires arrival within 5 cm / 0.1 rad and retention of any held
        objects. Print the result and end this cell to receive fresh RGB.
        """
        goal = world_goal(goal)
        if not isinstance(recover, bool):
            raise ValueError('recover must be a bool')
        started = time.monotonic()
        steps_before = self.control.steps()
        before = self.control.base_pose()
        held_before = self._holding()
        feedback = dict(arrived=False, ok=False, stage='navigate', goal=goal.tolist(),
                        before=before.tolist(), navigation_attempts=[], recovery=None,
                        holding_preserved=True, held_arms_before=sorted(held_before),
                        collision_checks_disabled=bool(getattr(self.control.generator, '_arena_planner_collisions_disabled', False)), coordinate_frame='world')
        self.feedback = feedback
        try:
            if self.control.episode_ended():
                feedback.update(stage='episode_end', reason='native_episode_already_ended')
                return feedback
            for attempt in range(2):
                nav = self.control.navigate(goal)
                feedback['navigation_attempts'].append(nav)
                feedback['holding_preserved'] = self._preserved(held_before)
                if not feedback['holding_preserved']:
                    feedback.update(stage='hold_check', reason='held_object_lost_or_changed')
                    break
                if nav['arrived']:
                    feedback.update(stage='complete', reason=None)
                    break
                statuses = [p.get('status') for p in nav.get('planner_calls', [])]
                if attempt or not recover or not any(recoverable(s) for s in statuses):
                    feedback['reason'] = 'navigation_goal_not_reached'
                    break
                joints_before = self.control.joints()
                feedback['stage'] = 'prepare_carry'
                recovery = self._prepare()
                feedback['recovery'] = recovery
                self.feedback = feedback
                feedback['holding_preserved'] = self._preserved(held_before)
                if not recovery['ready_to_move'] or not feedback['holding_preserved']:
                    feedback['reason'] = recovery.get('reason') or 'held_object_lost_or_changed'
                    break
                if np.max(np.abs(self.control.joints() - joints_before)) <= 1e-4:
                    feedback['reason'] = 'recovery_did_not_change_posture'
                    break
                feedback['stage'] = 'navigate_after_recovery'
            return feedback
        except Exception as exc:
            if self.feedback is not feedback:
                feedback['recovery'] = self.feedback
            feedback['reason'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            self.feedback = feedback
            after = self.control.base_pose()
            distance, angle = pose_errors(after, goal)
            feedback.update(after=after.tolist(), distance_error_m=distance,
                            yaw_error_rad=angle, moved_m=float(np.linalg.norm(after[:2] - before[:2])),
                            arrived=bool(distance <= .05 and angle <= .1),
                            holding_preserved=self._preserved(held_before),
                            environment_steps=self.control.steps() - steps_before,
                            elapsed_seconds=time.monotonic() - started,
                            held_arms_after=sorted(self._holding()))
            feedback['executed'] = feedback['environment_steps'] > 0
            feedback['ok'] = bool(feedback['arrived'] and feedback['holding_preserved']
                                  and feedback.get('reason') is None)


class R1ProCarryControl:
    """Small adapter to existing CuRobo planning and the recorded env.step path."""

    def __init__(self, session):
        self.api = session.live_api
        self.low = session.low_level_environment
        self.controller = self.low.controller
        self.generator = self.controller._motion_generator

    def holding(self):
        return {i: self.controller.robot._ag_obj_in_hand.get(arm)
                for i, arm in enumerate(('left', 'right'))
                if self.controller.robot._ag_obj_in_hand.get(arm) is not None}

    def steps(self):
        return int(self.low._step_count)

    def episode_ended(self):
        # Only terminal status is consulted, never success or task predicates.
        return bool(getattr(self.low.env.task, '_done', False))

    def base_pose(self):
        xyz, _, yaw = self.api.get_robot_position()
        return np.array([float(xyz[0]), float(xyz[1]), float(yaw)])

    def joints(self):
        return np.asarray(self.api.get_current_joint_positions()).copy()

    def eef_pose(self, arm):
        return tuple(np.asarray(v, dtype=float).copy() for v in self.api.get_current_eef_pose(arm=arm))

    def attachments(self):
        if getattr(self.generator, '_arena_planner_collisions_disabled', False):
            return None
        robot = self.controller.robot
        result = {}
        for arm, obj in robot._ag_obj_in_hand.items():
            if obj is None:
                continue
            if not obj.root_link.collision_meshes:
                raise ValueError(f'held_object_has_no_collision_mesh: arm={arm}')
            result[robot.eef_link_names[arm]] = obj.root_link
        return result or None

    def check_start(self, arm_only=False):
        """Operator diagnostic only; public motion no longer gates on this query."""
        import torch
        from curobo.types.state import JointState
        from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection
        from .capx_navigation_guard import check_start_preserving_constraints
        gen = self.generator
        selection = CuRoboEmbodimentSelection.ARM if arm_only else CuRoboEmbodimentSelection.BASE
        planner = gen.mg[selection]
        gen.update_obstacles()
        state = JointState(position=gen.tensor_args.to_device(gen.robot.get_joint_positions().unsqueeze(0)),
                           joint_names=gen.robot_joint_names)
        gen.update_locked_joints(state, selection)
        ordered = state.get_ordered_joint_state(planner.kinematics.joint_names)
        attached = []
        try:
            attached = gen._attach_objects_to_robot(self.attachments(), None, ordered, selection)
            with torch.no_grad():
                valid, status = check_start_preserving_constraints(planner, ordered)
            return dict(valid=bool(valid), status=str(status) if status is not None else 'valid')
        finally:
            gen._detach_objects_from_robot(attached, selection)

    def move_hand(self, xyz, quat, arm, max_environment_steps=None):
        import torch
        started = time.monotonic()
        steps_before = self.steps()
        held = self.holding()
        reason = None
        completed = False
        abort_recovery = False
        planner_calls = []
        previous_arm_plans = getattr(self.generator, '_arena_checked_arm_plans', None)
        if self.generator is not None:
            self.generator._arena_checked_arm_plans = planner_calls
        self.controller.overwrite_arm(arm)
        try:
            if self.episode_ended():
                return dict(arm=arm, arrived=False, reason='native_episode_already_ended',
                            abort_recovery=True, environment_steps=0)
            motion = self.controller._move_hand(
                (torch.tensor(xyz, dtype=torch.float32), torch.tensor(quat, dtype=torch.float32)),
                lock_auxiliary_arm=True, attached_obj=self.attachments(),
                ignore_all_obstacles=False, skip_obstacle_update=False,
                ik_only=False, ik_world_collision_check=not getattr(self.generator, '_arena_planner_collisions_disabled', False))
            for action in motion:
                if max_environment_steps is not None and self.steps() - steps_before >= max_environment_steps:
                    motion.close()
                    raise RuntimeError('checked_arm_step_limit_reached')
                result = self.low.step(action)
                if any(self.holding().get(a) is not obj for a, obj in held.items()):
                    raise RuntimeError('held_object_lost_during_carry_preparation')
                if result[2] or result[3]:
                    abort_recovery = True
                    raise RuntimeError('native_episode_ended_during_carry_preparation')
            completed = True
        except TimeoutError:
            raise
        except Exception as exc:
            reason = f'{type(exc).__name__}: {exc}'
        finally:
            if self.generator is not None:
                self.generator._arena_checked_arm_plans = previous_arm_plans
        actual, actual_q = self.eef_pose(arm)
        error = float(np.linalg.norm(actual - xyz))
        rotation = 2 * math.acos(float(np.clip(abs(np.dot(quat / np.linalg.norm(quat),
                                                       actual_q / np.linalg.norm(actual_q))), 0., 1.)))
        if completed and (error > .02 or rotation > .35):
            reason = 'measured_pose_not_reached'
        return dict(arm=arm, target_position=xyz.tolist(), actual_position=actual.tolist(),
                    arrived=bool(completed and error <= .02 and rotation <= .35),
                    position_error_m=error, orientation_error_rad=rotation, reason=reason,
                    abort_recovery=abort_recovery,
                    planner_calls=planner_calls,
                    environment_steps=self.steps() - steps_before,
                    elapsed_seconds=time.monotonic() - started)

    def navigate(self, goal):
        original_step = self.low.step
        held = self.holding()
        stopped_reason = None

        def guarded_step(*args, **kwargs):
            nonlocal stopped_reason
            result = original_step(*args, **kwargs)
            if any(self.holding().get(a) is not obj for a, obj in held.items()):
                stopped_reason = 'held_object_lost_during_navigation'
            elif result[2] or result[3]:
                stopped_reason = 'native_episode_ended_during_navigation'
            if stopped_reason:
                raise RuntimeError(stopped_reason)
            return result

        self.low.step = guarded_step
        try:
            self.api.navigate_to_pose(goal)
        finally:
            self.low.step = original_step
        feedback = dict(self.api._arena_navigation_feedback)
        if stopped_reason:
            feedback.update(arrived=False, stopped_reason=stopped_reason)
        return feedback


def install_carry_navigation(session):
    import hashlib
    from pathlib import Path
    api = session.live_api
    if hasattr(api, '_arena_carry_navigation'):
        raise RuntimeError('Carry/navigation primitives already installed')
    engine = CarryNavigation(R1ProCarryControl(session))
    original = api.functions
    api.functions = lambda: {**original(), 'prepare_carry': engine.prepare_carry,
                             'navigate_checked': engine.navigate_checked}
    api._arena_carry_navigation = engine
    session.reset_info['carry_navigation'] = dict(
        version='collision-unfiltered-20260925', max_recoveries=1, max_arm_candidates=2,
        max_navigation_attempts=2, telemetry='physical steps, measured poses, held-object retention',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        teleport_enabled=False, collision_checks_disabled=bool(getattr(engine.control.generator, '_arena_planner_collisions_disabled', False)),
        initial_collision_precheck_enabled=False)
