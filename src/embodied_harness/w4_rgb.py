"""W4-only observation adapter. No actions, verifier data, VDM, or reset calls."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
import uuid
from pathlib import Path

SUPPORTED = frozenset({'calvin', 'cliport', 'rlbench', 'robocasa', 'robocasa365',
                       'robotwin2', 'robowits', 'vimabench', 'vlabench', 'maniskill', 'capx'})
INSTRUCTIONS = ('Current RGB observations are attached as separate images in the listed camera order. '
                'All scene views belong to the current observation. Prompt-reference images are task instructions. '
                'Use these images with the public tools and execution feedback. Yield after a bounded action block '
                'to receive refreshed images. No image differencing model is used.')


class VisualInputError(ValueError):
    pass


def array_rgb(value):
    import numpy as np
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    a = np.asarray(value)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
        a = a.transpose(1, 2, 0)
    if a.ndim != 3 or a.shape[-1] not in (3, 4) or min(a.shape[:2]) < 8:
        raise VisualInputError(f'Invalid RGB shape {a.shape}')
    if a.dtype.kind == 'f':
        if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
            raise VisualInputError('RGB float values must be finite and in [0,1]')
        a = (a * 255).round()
    elif a.dtype.kind not in 'ui' or a.min() < 0 or a.max() > 255:
        raise VisualInputError('Invalid RGB values')
    return a[..., :3].astype('uint8').copy()


def _leaves(value, prefix=''):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, f'{prefix}/{key}')
    else:
        yield prefix, value


def _require(mapping, names):
    missing = [n for n in names if n not in mapping or mapping[n] is None]
    if missing:
        raise VisualInputError(f'Missing required cameras: {missing}; available={list(mapping)}')
    return [(n, mapping[n], 'scene') for n in names]


def collect_views(backend, benchmark):
    """Called on the simulator owner thread, using explicitly selected camera APIs."""
    env = getattr(backend, '_env', None)
    if benchmark in {'robocasa', 'robocasa365'}:
        names = ['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand']
        obj = getattr(env, 'unwrapped', env)
        return [(n, obj.sim.render(width=512, height=512, camera_name=n, depth=False)[::-1], 'scene') for n in names]
    if benchmark == 'calvin':
        return _require(env.get_obs()['rgb_obs'], ['rgb_static', 'rgb_gripper'])
    if benchmark == 'cliport':
        obs = env._get_obs()
        colors = obs['color']
        if len(colors) != 3:
            raise VisualInputError('CLIPort requires exactly the three native RGB views')
        return [(n, a, 'scene') for n, a in zip(['front', 'left', 'right'], colors)]
    if benchmark == 'rlbench':
        obs = backend._task.get_observation()
        names = ['front', 'wrist', 'overhead']
        return _require({n: getattr(obs, n + '_rgb', None) for n in names}, names)
    if benchmark == 'robotwin2':
        obs = env.get_obs()['observation']
        names = ['head_camera', 'left_camera', 'right_camera']
        return _require({n: v.get('rgb') for n, v in obs.items()}, names)
    if benchmark == 'robowits':
        children = getattr(env, 'envs', None)
        if children is not None:
            if len(children) != 1:
                raise VisualInputError('Expected one RoboWits episode')
            env = children[0]
        env = getattr(env, 'unwrapped', env)
        obs = env.get_observations()
        return _require(obs['pixels'], ['ego', 'wrist_left', 'wrist_right'])
    if benchmark == 'vimabench':
        obs = env._get_obs()
        views = _require(obs['rgb'], ['front', 'top'])
        # Reference assets are part of the task. Preserve placeholder/view identity.
        for path, value in _leaves(env.prompt_assets):
            if '/rgb/' in path or path.endswith('/rgb'):
                views.append(('prompt' + path, value, 'prompt_reference'))
        if not any(v[2] == 'prompt_reference' for v in views) and env.prompt_assets:
            raise VisualInputError('VIMA prompt assets exist but no RGB reference was found')
        return views
    if benchmark == 'vlabench':
        physics = backend.runtime.env.physics
        names = [physics.model.id2name(i, 'camera') for i in range(physics.model.ncam)]
        if not names or any(not n for n in names):
            raise VisualInputError('VLABench camera indices must resolve to names')
        return [(n, physics.render(height=480, width=480, camera_id=i), 'scene') for i, n in enumerate(names)]
    if benchmark == 'maniskill':
        obj = getattr(env, 'unwrapped', env)
        obs = obj.get_obs()
        sensors = obs.get('sensor_data', {})
        names = sorted(n for n, v in sensors.items() if 'rgb' in v)
        if not names:
            raise VisualInputError('ManiSkill observation sensors contain no RGB')
        return _require({n: sensors[n]['rgb'] for n in names}, names)
    if benchmark == 'capx':
        session = backend._live_session
        low = session.low_level_environment
        if getattr(backend, '_native_api_name', None) == 'R1ProControlApi':
            # Refresh the renderer after reset/action without ticking physics.
            # Fixed count across all models; capture_rgb checks simulation time.
            import omnigibson as og
            for _ in range(8):
                og.sim.render()
            raw = backend._public_live_observation()
            matches = {}
            for path, value in _leaves(raw):
                if not path.endswith('/rgb'):
                    continue
                for name, marker in [('ego', ':zed_link:'), ('left_wrist', ':left_realsense_link:'), ('right_wrist', ':right_realsense_link:')]:
                    if marker in path:
                        if name in matches:
                            raise VisualInputError(f'Ambiguous CaPX camera {name}')
                        matches[name] = value
            return _require(matches, ['ego', 'left_wrist', 'right_wrist'])
        # Robosuite's public primary render and wrist render, not R1Pro sensors.
        with session.activate_runtime_workdir():
            return [('agentview', low.render(), 'scene'), ('wrist', low.render_wrist(), 'scene')]
    raise VisualInputError(f'Unsupported W4 RGB benchmark: {benchmark}')


def simulation_time(backend):
    import sys
    env = getattr(backend, '_env', None)
    runtime = getattr(backend, 'runtime', None)
    session = getattr(backend, '_live_session', None)
    if getattr(backend, '_native_api_name', None) == 'R1ProControlApi':
        og = sys.modules.get('omnigibson')
        value = getattr(getattr(og, 'sim', None), 'current_time', None)
        if value is not None:
            return float(value)
    candidates = [env, getattr(runtime, 'env', None),
                  getattr(getattr(session, 'low_level_environment', None), 'robosuite_env', None),
                  getattr(getattr(getattr(session, 'low_level_environment', None), 'handle', None), 'env', None)]
    for obj in candidates:
        for attr in ('sim', 'physics'):
            data = getattr(getattr(obj, attr, None), 'data', None)
            if data is not None and hasattr(data, 'time'):
                return float(data.time)
    return None


def capture_rgb(backend, benchmark, directory, observation_id, episode_id):
    from PIL import Image
    before = simulation_time(backend)
    views = collect_views(backend, benchmark)
    after = simulation_time(backend)
    if before is not None and before != after:
        raise VisualInputError('RGB capture advanced simulation time')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    images = []
    for name, pixels, kind in views:
        a = array_rgb(pixels)
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, format='PNG', compress_level=6)
        data = buf.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        path = directory / (digest + '.png')
        if not path.exists():
            path.write_bytes(data)
        images.append(dict(path=str(path.resolve()), sha256=digest, camera_id=name, kind=kind,
                           width=a.shape[1], height=a.shape[0], observation_id=observation_id,
                           episode_id=episode_id, simulation_time=after))
    if not images:
        raise VisualInputError('RGB observation is empty')
    return images


class W4RGBFeedback:
    def __init__(self, backend, benchmark, directory):
        self.backend, self.benchmark = backend, benchmark
        self.directory = Path(directory).resolve()
        self.episode_id = uuid.uuid4().hex
        self.images = []
        self.signature = None

    def refresh(self, turn):
        from .subprocess_backend_bridge import SubprocessBackendBridge
        args = dict(benchmark=self.benchmark, directory=str(self.directory / 'images'),
                    observation_id=f'{self.episode_id}:{turn}', episode_id=self.episode_id)
        if isinstance(self.backend, SubprocessBackendBridge):
            self.images = self.backend.capture_rgb(**args)
        else:
            self.images = capture_rgb(self.backend, **args)
        signature = [(x['camera_id'], x['kind'], x['width'], x['height']) for x in self.images]
        preset_file = os.environ.get('ARENA_W4_CAMERA_PRESET_FILE')
        if preset_file:
            task_id = os.environ.get('ARENA_CASE_STUDY_TASK_ID')
            presets = json.loads(Path(preset_file).read_text())
            expected = presets.get(task_id)
            if expected is None or signature != [tuple(row) for row in expected]:
                raise VisualInputError('Camera set/order/resolution differs from the accepted task preset')
        if self.signature is not None and signature != self.signature:
            raise VisualInputError('Camera set/order/resolution changed within the episode')
        self.signature = signature
        with (self.directory / 'observations.jsonl').open('a') as f:
            f.write(json.dumps(dict(turn=turn, images=self.images, wall_time=time.time())) + '\n')

    @property
    def paths(self):
        return [Path(x['path']) for x in self.images]

    def describe(self):
        return '\nCurrent RGB images in order: ' + json.dumps([
            {k: row[k] for k in ('camera_id', 'kind', 'width', 'height')} for row in self.images])


def archive_request(directory, payload, image_paths, observation=None):
    """Archive final wire body with exact data URLs replaced by reversible byte refs."""
    doc = json.loads(payload)
    refs = iter(image_paths or [])
    count = 0
    for msg in doc['messages']:
        if isinstance(msg['content'], list):
            for part in msg['content']:
                if part.get('type') == 'image_url':
                    path = Path(next(refs))
                    data = path.read_bytes()
                    url = part['image_url']['url']
                    if base64.b64decode(url.split(',', 1)[1]) != data:
                        raise VisualInputError('Archived image differs from serialized request')
                    part['image_url'] = dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(),
                                             data_url_prefix=url.split(',', 1)[0] + ',')
                    count += 1
    if count != len(image_paths or []):
        raise VisualInputError('Serialized image count differs from selected observation')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    request_id = uuid.uuid4().hex
    path = root / (request_id + '.json')
    path.write_text(json.dumps(dict(request_id=request_id, wall_time=time.time(), image_count=count,
                                    wire_sha256=hashlib.sha256(payload).hexdigest(), payload=doc,
                                    observation=observation), ensure_ascii=False) + '\n')
    return request_id


def archive_codex_request(directory, command, prompt, image_paths, observation=None):
    """Local CLI inputs only; does not claim access to the CLI's remote wire body."""
    refs = []
    targets = [command[i + 1] for i, value in enumerate(command[:-1]) if value == '--image']
    if len(targets) != len(image_paths or []):
        raise VisualInputError('Codex attachment count mismatch')
    for source, target in zip(image_paths or [], targets):
        data = Path(source).read_bytes()
        if data != Path(target).read_bytes():
            raise VisualInputError('Codex attachment copy differs from archived RGB')
        refs.append(dict(path=str(source), cli_path=target, sha256=hashlib.sha256(data).hexdigest()))
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    request_id = uuid.uuid4().hex
    (root / (request_id + '.json')).write_text(json.dumps(dict(
        request_id=request_id, transport='codex-exec', wall_time=time.time(),
        command=command, stdin=prompt, image_count=len(refs), images=refs,
        observation=observation), ensure_ascii=False) + '\n')
    return request_id
