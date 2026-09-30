"""Small compatibility fix for the pinned CaPX LIBERO reset sequence."""
from functools import wraps


def install_reset_fix():
    from capx.envs.simulators.libero import FrankaLiberoEnv
    original = FrankaLiberoEnv.reset
    if getattr(original, '_arena_init_state_fix', False):
        return

    @wraps(original)
    def reset(self, *, seed=None, options=None):
        handle = self.handle
        states = handle.init_states
        if states is None or len(states) == 0:
            raise RuntimeError('LIBERO requires the official initial-state asset')
        index = 0 if seed is None else int(seed) - 1
        if not 0 <= index < len(states):
            raise ValueError('LIBERO trial seed must identify an existing 1-based initial state')
        original_handle_reset = handle.reset

        def fixed_handle_reset(seed=None):
            handle.env.seed(seed)
            handle.env.reset()
            observation = handle.env.set_init_state(states[index])
            # The pinned wrapper would immediately reset again and discard
            # the selected state. Skip only that duplicate branch while the
            # original robot initialization / ten settling steps still run.
            handle.init_states = None
            return observation, {'initial_state_index': index}

        handle.reset = fixed_handle_reset
        try:
            observation, info = original(self, seed=seed, options=options)
            info['harness_initial_state_index'] = index
            return observation, info
        finally:
            handle.reset = original_handle_reset
            handle.init_states = states

    reset._arena_init_state_fix = True
    FrankaLiberoEnv.reset = reset
