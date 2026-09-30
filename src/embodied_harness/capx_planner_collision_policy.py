"""User-requested collision-unfiltered planning for simulated R1Pro only.

CuRobo geometry is a planning approximation, separate from PhysX contact.
Disable both world/self collision costs and constraints across every solver.
Pose, joint-limit, timing and measured execution checks remain in force.
"""
import inspect
from functools import wraps


COLLISION_TERMS = (
    'primitive_collision_cost', 'primitive_collision_constraint',
    'robot_self_collision_cost', 'robot_self_collision_constraint',
)


def collision_terms(generator):
    seen = set()
    for selection, planner in generator.mg.items():
        for rollout in [planner.rollout_fn, *planner.get_all_rollout_instances()]:
            for name in COLLISION_TERMS:
                cost = getattr(rollout, name, None)
                if cost is not None and id(cost) not in seen:
                    seen.add(id(cost))
                    yield str(selection), name, cost


def planner_collision_status(generator):
    terms = list(collision_terms(generator))
    return dict(
        policy='collision_unfiltered_r1pro_20260925',
        initial_collision_precheck_enabled=False,
        world_collision_filter_enabled=False,
        self_collision_filter_enabled=False,
        collision_terms=len(terms),
        all_disabled=bool(terms) and all(
            not c.enabled and c.weight.count_nonzero().item() == 0
            and c._weight.count_nonzero().item() == 0 for _, _, c in terms),
        embodiments=sorted({selection for selection, _, _ in terms}),
        physics_contacts_modified=False,
        joint_limits_modified=False,
        teleport_enabled=False,
    )


def install_planner_collision_policy(session):
    generator = session.low_level_environment.controller._motion_generator
    if getattr(generator, '_arena_planner_collisions_disabled', False):
        raise RuntimeError('Planner collision policy already installed')
    # In-place tensor writes also affect graphs captured during native warmup.
    # Clear the saved weight too: native IK/start diagnostics can call
    # enable_cost(), which must not silently restore collision filtering.
    for _, _, cost in collision_terms(generator):
        cost._weight.zero_()
        cost.disable_cost()
    status = planner_collision_status(generator)
    if not status['all_disabled']:
        raise RuntimeError('R1Pro planner collision policy was not fully applied')

    original = generator.compute_trajectories
    signature = inspect.signature(original)

    @wraps(original)
    def trajectories(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.arguments['ik_world_collision_check'] = False
        # Collision attachments serve only the disabled planner geometry.
        # Physical grasp attachment and object retention are untouched.
        bound.arguments['attached_obj'] = None
        return original(*bound.args, **bound.kwargs)

    generator.compute_trajectories = trajectories
    generator._arena_planner_collisions_disabled = True
    session.reset_info['planner_collision_policy'] = status
