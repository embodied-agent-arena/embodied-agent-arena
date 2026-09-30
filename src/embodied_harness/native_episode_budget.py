"""Opt-in W4 native episode limits; applied once after reset, before any action.

No controller/action-repeat/physics-timestep changes; task failure and success
conditions remain native. Unlimited horizons stay unlimited.
"""
import math
import os


def apply_native_episode_budget(backend, task):
    factor = int(os.environ.get('ARENA_NATIVE_EPISODE_BUDGET_MULTIPLIER', '1'))
    if factor == 1:
        return
    if factor != 2:
        raise ValueError('W4 episode budget supports only the frozen 2x multiplier')
    if 'native_episode_budget' in task.metadata:
        return
    bench = os.environ.get('ARENA_CASE_STUDY_BENCHMARK', '')
    env = getattr(backend, '_env', None)
    changes = []
    touched = set()

    def scale(obj, attr, path, unit, *, active=True, note=None):
        if obj is None or not hasattr(obj, attr):
            return
        key = (id(obj), attr)
        if key in touched:
            return
        touched.add(key)
        before = getattr(obj, attr)
        if before is None:
            return
        if not isinstance(before, (int, float)) or isinstance(before, bool):
            raise TypeError('Unexpected native episode limit: ' + path)
        finite = math.isfinite(before) and before > 0
        after = before * factor if finite else before
        if finite:
            setattr(obj, attr, after)
            if getattr(obj, attr) != after:
                raise RuntimeError('Native episode limit did not take effect: ' + path)
        changes.append(dict(path=path, unit=unit, before=before if math.isfinite(before) else 'unlimited',
                            after=after if math.isfinite(after) else 'unlimited', active=active,
                            changed=finite, note=note))

    def wrappers(obj):
        seen = set()
        while obj is not None and id(obj) not in seen:
            seen.add(id(obj))
            yield obj
            obj = vars(obj).get('env')

    if bench == 'maniskill':
        for i, obj in enumerate(wrappers(env)):
            if '_max_episode_steps' in vars(obj):
                scale(obj, '_max_episode_steps', f'env.wrapper[{i}]._max_episode_steps', 'control_steps')
                # Gym's cached spec is informational; keep it aligned locally.
                spec = getattr(obj, 'spec', None)
                if spec is not None:
                    spec.max_episode_steps = obj._max_episode_steps
        if not changes:
            raise RuntimeError('ManiSkill native time-limit wrapper was not found')
        task.metadata.setdefault('native_evaluation_protocol', {})['max_episode_steps'] = min(c['after'] for c in changes)
    elif bench == 'cliport':
        scale(backend._task, 'max_steps', 'task.max_steps', 'pick_place_actions')
    elif bench in ('robocasa', 'robocasa365'):
        obj = getattr(env, 'unwrapped', env)
        scale(obj, 'horizon', 'env.horizon', 'control_steps', active=not obj.ignore_done,
              note='Existing ignore_done=True leaves episode time termination disabled; flag unchanged.')
    elif bench == 'robotwin2':
        scale(env, 'step_lim', 'env.step_lim', 'native_take_action_calls',
              note='Native take_action limit; planned motion primitives can use scene.step directly.')
    elif bench == 'robowits':
        children = getattr(env, 'envs', None)
        objects = children if children is not None else [env]
        for i, child in enumerate(objects):
            for j, obj in enumerate(wrappers(child)):
                if '_max_episode_steps' in vars(obj):
                    scale(obj, '_max_episode_steps', f'env[{i}].wrapper[{j}]._max_episode_steps', 'control_steps')
        if not changes:
            raise RuntimeError('RoboWits native step limit was not found')
    elif bench == 'vlabench':
        scale(backend.runtime.env, '_time_limit', 'env._time_limit', 'simulation_seconds')
    elif bench == 'capx':
        low = backend._live_session.low_level_environment
        if backend._native_api_name == 'R1ProControlApi':
            conditions = low.env.task._termination_conditions
            for name, condition in conditions.items():
                if type(condition).__name__ == 'Timeout':
                    scale(condition, '_max_steps', 'env.task._termination_conditions.' + name + '._max_steps', 'control_steps')
            if not changes:
                raise RuntimeError('CaPX BEHAVIOR native timeout condition was not found')
        else:
            scale(low, 'max_steps', 'low_level.max_steps', 'wrapper_step_calls', active=False,
                  note='Public robot API normally bypasses the wrapper step counter.')
            obj = getattr(low, 'robosuite_env', None)
            if obj is None:
                obj = low.handle.env
            found = False
            for i, inner in enumerate(wrappers(obj)):
                if 'horizon' in vars(inner):
                    found = True
                    scale(inner, 'horizon', f'simulator.wrapper[{i}].horizon', 'control_steps', active=not getattr(inner, 'ignore_done', False))
            if not found:
                raise RuntimeError('CaPX tabletop simulator horizon was not found')
    elif bench not in ('calvin', 'rlbench', 'vimabench'):
        raise ValueError('Unknown native episode budget benchmark: ' + bench)
    result = dict(multiplier=factor, benchmark=bench, limits=changes,
                  no_finite_active_limit=not any(c['active'] and c['changed'] for c in changes),
                  policy='Double finite native episode limits after reset; no reset/replay, controller or success/failure criterion changes.')
    task.metadata['native_episode_budget'] = result
    backend.record_event('native_episode_budget', result)
