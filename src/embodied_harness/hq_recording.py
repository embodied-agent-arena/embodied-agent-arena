"""Evaluator-only renders after every control step, on the simulator owner thread.

Never modifies observations, advances physics, or exposes footage to the agent.
"""
import os


def install(backend):
    if not os.environ.get('ARENA_CASE_STUDY_DIR'):
        return
    from .case_study_recording import publish_image, _error
    if (os.environ.get('ARENA_CASE_STUDY_BENCHMARK') == 'capx'
            and getattr(backend, '_native_api_name', None) == 'R1ProControlApi'):
        _install_capx_behavior_external(backend)
        return
    if (os.environ.get('ARENA_CASE_STUDY_BENCHMARK') == 'capx'
            and getattr(backend, '_native_api_name', None) in
                {'FrankaLiberoApi', 'FrankaControlApi', 'FrankaControlSpillWipeApi'}):
        _install_capx_tabletop(backend)
        return
    env = getattr(backend, '_env', None)
    runtime = getattr(backend, 'runtime', None)
    is_vla = runtime is not None and getattr(runtime, 'env', None) is not None
    if is_vla:
        env = runtime.env
    if env is None or getattr(env, '_arena_hq_installed', False):
        return
    bench = os.environ.get('ARENA_CASE_STUDY_BENCHMARK', '')
    if bench in ('cliport', 'vimabench', 'rlbench', 'robotwin2'):
        _install_inner_steps(backend, env, bench)
        return
    if bench not in ('robocasa', 'robocasa365', 'vlabench'):
        return  # RoboWits pixels and CaPX multi-camera observations use native taps.
    original = env.step
    counter = [0]
    errors = [0]

    def capture():
        try:
            if is_vla:
                physics = env.physics
                image = physics.render(height=512, width=512, camera_id=0)
                publish_image(image, 'hq.vlabench.camera0', source_metadata={
                    'control_step': counter[0], 'simulation_time': float(physics.data.time),
                    'recording_only': True, 'resolution': 'native_render_512'})
            else:
                obj = getattr(env, 'unwrapped', env)
                sim = obj.sim
                names = list(getattr(obj, 'camera_names', []))
                names = [n for n in names if 'agentview_left' in n or 'eye_in_hand' in n] or names[:1]
                for name in names:
                    image = sim.render(width=512, height=512, camera_name=name, depth=False)
                    publish_image(image[::-1], 'hq.' + bench + '.' + name, source_metadata={
                        'control_step': counter[0], 'simulation_time': float(sim.data.time),
                        'recording_only': True, 'vertical_flip': True,
                        'resolution': 'native_render_512'})
        except Exception as exc:
            errors[0] += 1
            if errors[0] <= 5:
                _error('hq_capture: ' + type(exc).__name__ + ': ' + str(exc))

    def step(*args, **kwargs):
        value = original(*args, **kwargs)
        counter[0] += 1
        capture()
        return value

    step._case_study_tap = True  # original already contains the observation tap
    env.step = step
    env._arena_hq_installed = True
    capture()


def _install_inner_steps(backend, env, bench):
    """Record inside long native actions, without adding physics steps."""
    from .case_study_recording import publish_image, _error
    from .w4_rgb import collect_views
    clock = [0.0]
    last = [-1e9]
    errors = [0]
    if bench in ('cliport', 'vimabench'):
        import pybullet as p
        client = getattr(env, 'client_id', 0)
        dt = float(p.getPhysicsEngineParameters(physicsClientId=client)['fixedTimeStep'])
        owner, method = env, 'step_simulation'
    elif bench == 'rlbench':
        owner = backend._task._scene
        dt = float(owner.pyrep.get_simulation_timestep())
        method = 'step'
    else:
        owner, method = env.scene, 'step'
        dt = float(owner.get_timestep())
    original = getattr(owner, method)
    if getattr(original, '_arena_inner_recording', False):
        return

    def capture():
        if clock[0] - last[0] < .05 - 1e-9:
            return
        last[0] = clock[0]
        try:
            for name, pixels, kind in collect_views(backend, bench):
                if kind == 'scene':
                    publish_image(pixels, 'hq.' + bench + '.' + name, source_metadata={
                        'simulation_time': clock[0], 'clock_origin': 'recording_hook_install',
                        'recording_only': True, 'native_step_seconds': dt})
        except Exception as exc:
            errors[0] += 1
            if errors[0] <= 5:
                _error('inner_step_capture: ' + type(exc).__name__ + ': ' + str(exc))

    def step(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] += dt
        capture()
        return result
    step._arena_inner_recording = True
    try:
        setattr(owner, method, step)
    except (AttributeError, TypeError) as exc:
        _error('inner_step_install: ' + str(exc))
        return
    env._arena_hq_installed = True
    capture()


def _install_capx_tabletop(backend):
    """Sample real low-level tabletop control steps, using the public cameras."""
    from .case_study_recording import publish_image, _error
    from .w4_rgb import collect_views, simulation_time
    low = backend._live_session.low_level_environment
    if getattr(low, '_arena_hq_installed', False):
        return
    # Public API methods bypass low.step; intercept the actual simulator step.
    owner = getattr(low, 'robosuite_env', None)
    if owner is None:
        owner = low.handle.env
    original = owner.step
    last = [-1e9]
    errors = [0]
    def capture():
        try:
            now = simulation_time(backend)
            if now is not None and now - last[0] < .05 - 1e-9:
                return
            for name, pixels, kind in collect_views(backend, 'capx'):
                publish_image(pixels, 'hq.capx.' + name, source_metadata={
                    'simulation_time': now, 'recording_only': True})
            if now is not None:
                last[0] = now
        except Exception as exc:
            errors[0] += 1
            if errors[0] <= 5:
                _error('capx_tabletop_recording: ' + type(exc).__name__ + ': ' + str(exc))
    def step(*args, **kwargs):
        result = original(*args, **kwargs)
        capture()
        return result
    step._case_study_tap = True
    owner.step = step
    low._arena_hq_installed = True
    capture()


def _install_capx_behavior_external(backend):
    """Stream the existing external camera for review, without exposing it to the model."""
    from .case_study_recording import publish_image, _error
    from .w4_rgb import simulation_time
    low = backend._live_session.low_level_environment
    if getattr(low, '_arena_hq_installed', False):
        return
    sensor = low.env.external_sensors.get('external_camera')
    if sensor is None:
        _error('capx_external_recording: external_camera missing')
        return
    original = low.step
    last = [-1e9]
    errors = [0]

    def capture():
        now = simulation_time(backend)
        if now is not None and now - last[0] < 1 / 15 - 1e-9:
            return
        try:
            rgb = sensor.get_obs()[0]['rgb'][:, :, :3]
            publish_image(rgb, 'hq.capx.external', source_metadata={
                'simulation_time': now, 'recording_only': True})
            if now is not None:
                last[0] = now
        except Exception as exc:
            errors[0] += 1
            if errors[0] <= 5:
                _error('capx_external_recording: ' + type(exc).__name__ + ': ' + str(exc))

    def step(*args, **kwargs):
        result = original(*args, **kwargs)
        capture()
        return result

    step._case_study_tap = True
    low.step = step
    low._arena_hq_installed = True
