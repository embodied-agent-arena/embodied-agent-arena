import base64
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
from embodied_harness.w4_rgb import W4RGBFeedback, VisualInputError, array_rgb, capture_rgb, collect_views
from embodied_harness.native_agent_loop import OpenAICompatibleModelClient



class RGBTests(unittest.TestCase):
    def test_current_frame_refresh_and_exact_wire_archive(self):
        pixels = np.zeros((16, 24, 3), dtype=np.uint8)
        env = NS(get_obs=lambda: {'rgb_obs': {'rgb_static': pixels, 'rgb_gripper': pixels}})
        with tempfile.TemporaryDirectory() as directory:
            feedback = W4RGBFeedback(NS(_env=env), 'calvin', directory)
            feedback.refresh(1)
            before = feedback.images[0]['sha256']
            pixels[:] = 150
            feedback.refresh(2)
            self.assertNotEqual(before, feedback.images[0]['sha256'])
            self.assertEqual(feedback.images[0]['sha256'], feedback.images[1]['sha256'])
            self.assertEqual(len(list((Path(directory) / 'images').glob('*.png'))), 2)
            config = NS(model='fixture', base_url='https://invalid.example', api_key='NOT-SAVED', temperature=0,
                        request_timeout_seconds=1, input_cost_per_million=None, output_cost_per_million=None)
            client = OpenAICompatibleModelClient(config)
            client.request_archive_dir = Path(directory) / 'requests'
            client.request_observation = {'turn': 2, 'images': feedback.images}
            body = {'choices': [{'message': {'content': 'print(1)'}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 5, 'completion_tokens': 3, 'total_tokens': 8}}
            class Response:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def read(self): return json.dumps(body).encode()
            seen = []
            def send(request, **kwargs):
                seen.append(json.loads(request.data))
                return Response()
            with patch('embodied_harness.native_agent_loop.urlopen', side_effect=send):
                client.complete([{'role': 'user', 'content': 'current cameras'}], max_tokens=32, image_paths=feedback.paths)
            wire_images = [p for p in seen[0]['messages'][-1]['content'] if p['type'] == 'image_url']
            self.assertEqual(len(wire_images), 2)
            for part, path in zip(wire_images, feedback.paths):
                self.assertEqual(base64.b64decode(part['image_url']['url'].split(',')[1]), path.read_bytes())
            saved = (client.request_archive_dir / (client.last_request_id + '.json')).read_text()
            self.assertNotIn('NOT-SAVED', saved)
            self.assertEqual(json.loads(saved)['image_count'], 2)
            self.assertEqual(json.loads(saved)['observation']['turn'], 2)

    def test_codex_archives_actual_copied_attachments(self):
        from embodied_harness.w4_rgb import archive_codex_request
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, target = root / 'source.png', root / 'cli.png'
            source.write_bytes(b'image-bytes')
            target.write_bytes(source.read_bytes())
            request_id = archive_codex_request(root / 'requests',
                ['codex', '--image', str(target), '-'], 'prompt', [source], {'turn': 1})
            record = json.loads((root / 'requests' / (request_id + '.json')).read_text())
            self.assertEqual(record['image_count'], 1)
            self.assertEqual(record['stdin'], 'prompt')
            target.write_bytes(b'changed')
            with self.assertRaises(VisualInputError):
                archive_codex_request(root, ['codex', '--image', str(target)], 'prompt', [source])

    def test_camera_missing_and_resolution_change_fail_closed(self):
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        cameras = {'rgb_static': rgb, 'rgb_gripper': rgb}
        with tempfile.TemporaryDirectory() as d:
            f = W4RGBFeedback(NS(_env=NS(get_obs=lambda: {'rgb_obs': cameras})), 'calvin', d)
            f.refresh(1)
            cameras['rgb_gripper'] = np.zeros((20, 20, 3), dtype=np.uint8)
            with self.assertRaises(VisualInputError): f.refresh(2)
            del cameras['rgb_gripper']
            with self.assertRaises(VisualInputError): f.refresh(3)

    def test_vima_prompt_references_and_channel_order(self):
        rgb = np.zeros((3, 16, 20), dtype=np.uint8)
        rgb[0] = 255
        env = NS(_get_obs=lambda: {'rgb': {'front': rgb, 'top': rgb}},
                 prompt_assets={'target': {'rgb': {'front': rgb}}})
        views = collect_views(NS(_env=env), 'vimabench')
        self.assertEqual([x[2] for x in views], ['scene', 'scene', 'prompt_reference'])
        self.assertEqual(array_rgb(rgb)[0, 0].tolist(), [255, 0, 0])

    def test_capture_cannot_advance_physics(self):
        data = NS(time=0.)
        def render(**kwargs):
            data.time += 0.1
            return np.zeros((16, 16, 3), dtype=np.uint8)
        backend = NS(_env=NS(sim=NS(data=data, render=render)))
        with tempfile.TemporaryDirectory() as d, self.assertRaises(VisualInputError):
            capture_rgb(backend, 'robocasa', d, 'obs', 'episode')




if __name__ == '__main__':
    unittest.main()
