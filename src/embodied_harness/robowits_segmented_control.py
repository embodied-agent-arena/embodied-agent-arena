"""Bounded feedback control using exactly the caller's available step budget.

This layer has no task objective, object attachment, or success-checker access.
It corrects measured waypoint error before advancing, and distinguishes a
command being executed from its requested pose being reached.
"""
import math


def _rotation_error(left, right):
    from scipy.spatial.transform import Rotation
    return float((Rotation.from_rotvec(left).inv() * Rotation.from_rotvec(right)).magnitude())


def execute(backend, single, **kw):
    from . import robowits_agent_runtime as rw
    if backend._env is None or backend.episode_status()['reached']:
        out = single(backend, **kw)
        out['target_reached'] = False
        return out
    raw, _ = rw._extract_ee_abs_action_base(backend._last_obs, action_dim=14)
    target = rw._numeric_vector(kw['target_position'], limit=3)
    if raw is None or target is None:
        return dict(executed=False, target_reached=False, error='target_or_public_ee_state_unavailable')
    orientation = rw._numeric_vector(kw.get('axis_angle'), limit=3)
    if not all(math.isfinite(float(x)) for x in target + (orientation or [])):
        return dict(executed=False, target_reached=False, error='nonfinite_pose')
    width = kw.get('gripper_width')
    if width is not None and not math.isfinite(float(width)):
        return dict(executed=False, target_reached=False, error='nonfinite_gripper_width')
    requested_limit = float(kw.get('max_translation') if kw.get('max_translation') is not None else .08)
    if not math.isfinite(requested_limit) or requested_limit <= 0:
        return dict(executed=False, target_reached=False, error='invalid_translation_bound')
    limit = rw._clamp_float(requested_limit, .005, .3)
    total = max(1, min(int(kw.get('repeat_steps') or 1), 64)) + max(0, min(int(kw.get('hold_steps') or 0), 64))
    base = rw._robowits_env_robot_base_position(backend._env)
    positions = {a: [raw[j + off] + base[j] for j in range(3)] for a, off in [('right', 0), ('left', 3)]}
    arm = str(kw.get('arm') or 'auto').lower()
    if arm not in positions:
        arm = min(positions, key=lambda a: math.dist(positions[a], target))
    po, ao, go = (0, 6, 12) if arm == 'right' else (3, 9, 13)
    orientation = orientation or list(raw[ao:ao + 3])
    steps, summaries = [], []
    used = 0
    previous_error = None
    stagnant = 0
    initial_distance = math.dist(positions[arm], target)
    pure_hold = initial_distance <= .01 and _rotation_error(raw[ao:ao + 3], orientation) <= .10
    while used < total and not backend.episode_status()['reached']:
        current, _ = rw._extract_ee_abs_action_base(backend._last_obs, action_dim=14)
        if current is None:
            break
        start = [current[j + po] + base[j] for j in range(3)]
        distance = math.dist(start, target)
        fraction = min(1., limit * .9 / max(distance, 1e-12))
        waypoint = [start[j] + fraction * (target[j] - start[j]) for j in range(3)]
        rotation_distance = _rotation_error(current[ao:ao + 3], orientation)
        rotation_target = list(current)
        rotation_target[ao:ao + 3] = orientation
        waypoint_orientation = rw._interpolate_robowits_ee_abs_action(
            current, rotation_target, arm=arm,
            alpha=min(1., .30 / max(rotation_distance, 1e-12)))[ao:ao + 3]
        allocated = min(12, total - used)
        command_steps = min(4, allocated)
        out = single(backend, **{**kw, 'arm': arm, 'target_position': waypoint,
            'axis_angle': waypoint_orientation, 'repeat_steps': command_steps,
            'hold_steps': allocated - command_steps, 'max_translation': limit})
        actual_steps = int(out.get('steps_executed') or 0)
        used += actual_steps
        steps.append(out)
        observed, _ = rw._extract_ee_abs_action_base(backend._last_obs, action_dim=14)
        if observed is None:
            break
        actual = [observed[j + po] + base[j] for j in range(3)]
        error = math.dist(actual, target)
        angle_error = _rotation_error(observed[ao:ao + 3], orientation)
        reached = error <= .01 and angle_error <= .10
        summaries.append(dict(index=len(summaries), waypoint=waypoint, actual=actual,
            position_error_m=error, orientation_error_rad=angle_error,
            steps_executed=actual_steps))
        if not out.get('executed') or actual_steps == 0:
            break
        if reached and not pure_hold:
            break
        combined_error = error + .1 * angle_error
        if previous_error is not None and previous_error - combined_error < .001:
            stagnant += 1
        else:
            stagnant = 0
        previous_error = combined_error
        if stagnant >= 4 and not pure_hold:
            break
    observed, _ = rw._extract_ee_abs_action_base(backend._last_obs, action_dim=14)
    actual = [observed[j + po] + base[j] for j in range(3)] if observed is not None else None
    error = math.dist(actual, target) if actual is not None else None
    angle_error = _rotation_error(observed[ao:ao + 3], orientation) if observed is not None else None
    reached = error is not None and error <= .01 and angle_error is not None and angle_error <= .10
    if hasattr(backend, 'record_event'):
        backend.record_event('robowits_control_diagnostics', {'segments': steps, 'step_budget': total})
    last = steps[-1] if steps else {}
    reason = None if reached else ('motion_stalled' if stagnant >= 4 else 'target_not_reached_within_step_budget')
    return dict(executed=used > 0, target_reached=reached, arrived=reached,
        segmented_control=initial_distance > limit,
        target_position_world=target, actual_position_world=actual, arm=arm,
        axis_angle=orientation, gripper_width=observed[go] if observed is not None else None,
        requested_gripper_width=width, max_translation=limit,
        requested_control_steps=total, steps_executed=used,
        action_position_robot_base_relative=last.get('action_position_robot_base_relative'),
        translation_distance=initial_distance,
        control_trajectory=dict(mode='bounded_pose_feedback',
            requested_repeat_steps=kw.get('repeat_steps'), requested_hold_steps=kw.get('hold_steps'),
            command_steps=sum(s.get('control_trajectory', {}).get('command_steps', 0) for s in steps),
            hold_steps=sum(s.get('control_trajectory', {}).get('hold_steps', 0) for s in steps)),
        termination=last.get('termination', {}),
        position_error_m=error, orientation_error_rad=angle_error,
        segments=summaries, public_state_before=steps[0].get('public_state_before', {}) if steps else {},
        public_state_after=last.get('public_state_after', {}), tracking=last.get('tracking', {}),
        episode_status=backend.episode_status(), error=reason, context=kw.get('context', {}))
