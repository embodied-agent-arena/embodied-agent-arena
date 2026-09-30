"""R1Pro navigation repair, applied only when creating a new session.

Uses the existing planner and physical executor with measured arrival. The
session's planner_collision_policy records whether planning filters are off.
"""
import math
import time
from functools import wraps


def complete_held_mesh_names(robot, object_names):
    """Bind every collision mesh of the physically held object, not only its root.

    An articulated tool's other links otherwise remain world obstacles and
    collide with its own attached geometry. No task identity is consulted.
    """
    names = list(object_names)
    requested = set(names)
    for obj in robot._ag_obj_in_hand.values():
        if obj is None:
            continue
        root_names = {mesh.prim_path for mesh in obj.root_link.collision_meshes.values()}
        if not requested.intersection(root_names):
            continue
        for link in obj.links.values():
            for mesh in link.collision_meshes.values():
                if mesh.prim_path not in names:
                    names.append(mesh.prim_path)
    return names


def install_complete_held_attachment(planner, robot):
    """Expand attach AND detach together; retain native sphere fit and margins."""
    import inspect
    original_attach = planner.attach_objects_to_robot
    original_detach = planner.detach_object_from_robot
    signature = inspect.signature(original_attach)
    attached_by_link = {}

    @wraps(original_attach)
    def attach(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        link = bound.arguments.get('link_name', 'attached_object')
        names = complete_held_mesh_names(robot, bound.arguments['object_names'])
        missing = [name for name in names if planner.world_model.get_obstacle(name) is None]
        if missing:
            raise ValueError('held_collision_mesh_missing_from_world')
        if link in attached_by_link:
            raise RuntimeError('held_collision_geometry_already_attached')
        if any(set(names).intersection(active) for active in attached_by_link.values()):
            raise RuntimeError('held_collision_geometry_already_bound_to_other_arm')
        bound.arguments['object_names'] = names
        try:
            result = original_attach(*bound.args, **bound.kwargs)
            if not result:
                raise RuntimeError('held_collision_geometry_attachment_failed')
        except Exception:
            # Native code may have disabled meshes before failing. Restore the
            # world and remove partial spheres before propagating the error.
            original_detach(object_names=names, link_name=link)
            raise
        attached_by_link[link] = names
        return result

    @wraps(original_detach)
    def detach(object_names, link_name='attached_object'):
        names = attached_by_link.get(link_name, object_names)
        result = original_detach(object_names=names, link_name=link_name)
        attached_by_link.pop(link_name, None)
        return result

    planner.attach_objects_to_robot = attach
    planner.detach_object_from_robot = detach
    planner._arena_attached_mesh_names = attached_by_link


def planner_eef_pose(world_eef_pose, world_root_pose):
    """Use the same root frame as CuRoboMotionGenerator.update_obstacles."""
    return world_root_pose.inverse().multiply(world_eef_pose)


def install_attachment_frame_fix(session):
    """Correct held-object geometry coordinates without removing any collision."""
    from curobo.types.math import Pose
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection
    generator = session.low_level_environment.controller._motion_generator
    if getattr(generator, '_arena_attachment_frame_fixed', False):
        raise RuntimeError('Attachment frame correction already installed')
    seen = set()
    for planner in generator.mg.values():
        if id(planner) in seen:
            continue
        seen.add(id(planner))
        original = planner.attach_objects_to_robot

        def make_attach(original):
            @wraps(original)
            def attach(*args, **kwargs):
                ee_pose = kwargs.get('ee_pose')
                if ee_pose is not None:
                    pos, quat_xyzw = generator.robot.root_link.get_position_orientation()
                    root = Pose(position=pos, quaternion=quat_xyzw[[3, 0, 1, 2]]).to(generator._tensor_args)
                    kwargs['ee_pose'] = planner_eef_pose(ee_pose, root)
                    generator._arena_attachment_corrections += 1
                return original(*args, **kwargs)
            return attach
        planner.attach_objects_to_robot = make_attach(original)
        install_complete_held_attachment(planner, generator.robot)

    original_attach_objects = generator._attach_objects_to_robot

    @wraps(original_attach_objects)
    def attach_objects(*args, **kwargs):
        # The native multi-arm helper does not return its cleanup list if a
        # later arm fails. Roll back successfully attached earlier arms too.
        planners = {id(p): p for p in generator.mg.values()}.values()
        before = {id(p): set(p._arena_attached_mesh_names) for p in planners}
        try:
            return original_attach_objects(*args, **kwargs)
        except Exception:
            for planner in planners:
                for link in list(planner._arena_attached_mesh_names):
                    if link not in before[id(planner)]:
                        planner.detach_object_from_robot(object_names=[], link_name=link)
            raise

    generator._attach_objects_to_robot = attach_objects

    original_trajectories = generator.compute_trajectories

    @wraps(original_trajectories)
    def trajectories(*args, **kwargs):
        # The upstream BASE caller silently skips held roots after probing a
        # nonexistent root.get_trimesh_mesh(). Attach the actual collision meshes,
        # using the same physical grasp registry as its own _get_obj_in_hand().
        if kwargs.get('emb_sel') == CuRoboEmbodimentSelection.BASE:
            attached = dict(kwargs.get('attached_obj') or {})
            arms = []
            for arm, obj in generator.robot._ag_obj_in_hand.items():
                if obj is None:
                    continue
                root = obj.root_link
                if not root.collision_meshes:
                    raise ValueError(f'held_object_has_no_collision_mesh: arm={arm}')
                attached[generator.robot.eef_link_names[arm]] = root
                arms.append(str(arm))
            kwargs['attached_obj'] = attached or None
            generator._arena_navigation_attached_arms = arms
        return original_trajectories(*args, **kwargs)

    generator.compute_trajectories = trajectories
    generator._arena_attachment_frame_fixed = True
    generator._arena_attachment_corrections = 0
    session.reset_info['attachment_frame_fix'] = 'world_eef_to_fixed_root; obstacle geometry unchanged'
    session.reset_info['held_collision_binding'] = 'physical_grasp_registry_both_arms_all_links'


def check_start_preserving_constraints(planner, start_state):
    """CuRobo's diagnostic may return with self-collision disabled: restore it."""
    costs = [planner.rollout_fn.primitive_collision_constraint,
             planner.rollout_fn.robot_self_collision_constraint]
    enabled = [cost.enabled for cost in costs]
    try:
        return planner.check_start_state(start_state)
    finally:
        for cost, was_enabled in zip(costs, enabled):
            (cost.enable_cost if was_enabled else cost.disable_cost)()


def install_navigation_guard(session, *, planner_timeout_seconds=None):
    """Replace only R1Pro navigate_to_pose; keep its bool return contract.

    A call addresses exactly the requested world x/y/yaw. The native batch
    planner decides feasibility; no extra static-start veto is applied here.
    Successful arrival is measured after execution.
    The optional planner timeout is for isolated ablations, not enabled by default.
    """
    import numpy as np
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection

    api = session.live_api
    low = session.low_level_environment
    generator = low.controller._motion_generator
    original_plan = generator.plan_batch
    state = {"active": False, "plans": []}
    if getattr(api, "_arena_navigation_guard_installed", False):
        raise RuntimeError("Navigation guard already installed in this session")
    if planner_timeout_seconds is not None and planner_timeout_seconds <= 0:
        raise ValueError("planner_timeout_seconds must be positive")

    @wraps(original_plan)
    def plan_batch(start_state, goal_pose, plan_config, link_poses=None,
                   emb_sel=CuRoboEmbodimentSelection.DEFAULT):
        arm_plans = getattr(generator, '_arena_checked_arm_plans', None)
        checked_arm = arm_plans is not None and getattr(emb_sel, 'name', None) == 'ARM'
        if not checked_arm and (not state["active"] or emb_sel != CuRoboEmbodimentSelection.BASE):
            return original_plan(start_state, goal_pose, plan_config,
                                 link_poses=link_poses, emb_sel=emb_sel)
        started = time.monotonic()
        # Delegate to the native planner with the session's collision policy;
        # no added static-start veto or synthetic successful trajectory.
        original_timeout = plan_config.timeout
        try:
            if planner_timeout_seconds is not None and not checked_arm:
                plan_config.timeout = min(original_timeout, planner_timeout_seconds)
            result, success, paths = original_plan(
                start_state, goal_pose, plan_config,
                link_poses=link_poses, emb_sel=emb_sel)
        finally:
            plan_config.timeout = original_timeout
        found = bool(success.any().item())
        (arm_plans if checked_arm else state["plans"]).append({
            "start_valid": None,
            "initial_collision_precheck_enabled": False,
            "planner_collision_checks_disabled": bool(getattr(generator, '_arena_planner_collisions_disabled', False)),
            "native_valid_query": getattr(result, "valid_query", None),
            "status": str(result.status) if result.status is not None else "success" if found else "unclassified_failure",
            "attempts": int(getattr(result, "attempts", 0) or 0),
            "elapsed_seconds": time.monotonic() - started,
            "any_trajectory_found": found,
        })
        return result, success, paths

    def current_pose():
        xyz, _quat, yaw = api.get_robot_position()
        return np.array([float(xyz[0]), float(xyz[1]), float(yaw)])

    def errors(actual, goal):
        return (float(np.linalg.norm(actual[:2] - goal[:2])),
                abs(math.atan2(math.sin(actual[2] - goal[2]),
                               math.cos(actual[2] - goal[2]))))

    def navigate_to_pose(pose_2d):
        """Navigate physically to WORLD x/y (metres) and yaw (radians).

        Read get_robot_position first; x/y are absolute world coordinates,
        not offsets or normalized proprioception. Returns True only when the
        requested pose is reached (5 cm / 0.1 rad). No implicit waypoint retries.
        On failure, inspect navigation_feedback in the action result.
        """
        goal = np.asarray(pose_2d, dtype=float)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError("pose_2d must be finite WORLD [x_m, y_m, yaw_rad]")
        before = current_pose()
        attachment_before = getattr(generator, '_arena_attachment_corrections', 0)
        generator._arena_navigation_attached_arms = []
        started = time.monotonic()
        initial_distance, initial_angle = errors(before, goal)
        state.update(active=True, plans=[])
        native_ok = initial_distance <= .05 and initial_angle <= .1
        execution_error = None
        try:
            if not native_ok:
                native_ok = bool(low._navigate_to_pose(goal))
        except Exception as exc:
            execution_error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            state["active"] = False
            after = current_pose()
            distance, angle = errors(after, goal)
            arrived = bool(native_ok and distance <= .05 and angle <= .1)
            api._arena_navigation_feedback = {
                "coordinate_frame": "world",
                "goal": goal.tolist(), "before": before.tolist(), "after": after.tolist(),
                "arrived": arrived, "native_executor_ok": bool(native_ok),
                "distance_error_m": distance, "yaw_error_rad": angle,
                "moved_m": float(np.linalg.norm(after[:2] - before[:2])),
                "progress_m": initial_distance - distance,
                "elapsed_seconds": time.monotonic() - started,
                "planner_calls": list(state["plans"]),
                "attachment_frame_corrections": getattr(generator, '_arena_attachment_corrections', 0) - attachment_before,
                "held_collision_arms": list(generator._arena_navigation_attached_arms),
                "execution_error": execution_error,
                "collision_checks_disabled": bool(getattr(generator, '_arena_planner_collisions_disabled', False)),
            }
        return arrived

    generator.plan_batch = plan_batch
    api.navigate_to_pose = navigate_to_pose
    api._arena_navigation_guard_installed = True
    session.reset_info["navigation_guard"] = {
        "version": "collision-unfiltered-revision-20260925",
        "single_requested_target": True,
        "start_validity_checked": False,
        "initial_collision_precheck_enabled": False,
        "measured_arrival": True,
        "planner_timeout_seconds": planner_timeout_seconds,
        "collision_checks_disabled": bool(getattr(generator, '_arena_planner_collisions_disabled', False)),
        "teleport_enabled": False,
    }
