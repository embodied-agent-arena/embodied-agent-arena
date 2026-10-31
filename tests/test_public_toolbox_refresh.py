"""Updated media must remain tied to its own model, task, and run outcome."""
import hashlib
import json
import unittest
from html.parser import HTMLParser
from pathlib import Path

PUBLIC = Path(__file__).resolve().parents[1] / 'public'


class Media(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.videos = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        if tag == 'video':
            self.videos.append(dict(attrs))


class ToolboxRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comparison = json.loads((PUBLIC / 'data/case-comparisons.json').read_text())
        cls.refresh = json.loads((PUBLIC / 'data/toolbox-recording-refresh.json').read_text())

    def test_ten_updated_recordings_have_intact_media_and_frame_indexes(self):
        records = self.refresh['records']
        self.assertEqual(len(records), 10)
        self.assertEqual(len({r['modelKey'] for r in records}), 10)
        for r in records:
            with self.subTest(model=r['modelKey']):
                proof = json.loads((PUBLIC / r['provenance']).read_text())
                self.assertEqual(r['taskId'], self.refresh['taskId'])
                self.assertEqual(hashlib.sha256((PUBLIC / r['video']).read_bytes()).hexdigest(), r['videoSha256'])
                self.assertTrue((PUBLIC / r['poster']).is_file())
                frames = [json.loads(line) for line in (PUBLIC / proof['frameIndex']).read_text().splitlines()]
                self.assertEqual(len(frames), r['frameCount'])
                self.assertEqual([f['frame'] for f in frames], list(range(r['frameCount'])))
                self.assertEqual({f['source'] for f in frames}, {'hq.capx.head'})
                self.assertFalse(r['countInEvaluation'])
                self.assertIn(r['outcome'], {'task_failure', 'budget_exhausted'})
                for key in r:
                    self.assertEqual(r[key], proof[key])
        self.assertEqual(next(r for r in records if r['modelKey'] == 'sol')['modelId'], 'gpt-6-sol')

    def test_overview_and_original_records_remain_distinguishable(self):
        overview = json.loads((PUBLIC / 'data/report/overview-provenance.json').read_text())
        tiles = [t for t in overview['tiles'] if t.get('updatedRecording')]
        self.assertEqual(len(tiles), 10)
        self.assertEqual(overview['updatedRecordingCount'], 10)
        for tile in tiles:
            u = tile['updatedRecording']
            self.assertEqual(tile['modelKey'], u['modelKey'])
            self.assertEqual(tile['taskId'], u['taskId'])
            self.assertEqual(tile['previewVideo'], u['video'])
            self.assertEqual(tile['poster'], u['poster'])
            original = json.loads((PUBLIC / tile['detailUrl']).read_text())
            self.assertEqual(original['reportSha256'], tile['reportSha256'])
            self.assertEqual(original['updatedRecording'], u)
            self.assertNotEqual(tile['sourceVideo'], tile['previewVideo'])

    def test_wall_uses_multiple_lightweight_sources_with_lazy_playback(self):
        media = Media((PUBLIC / 'index.html').read_text()).videos
        wall = [v for v in media if 'hero-wall-' in v.get('data-src', '')]
        self.assertEqual(len(wall), 12)
        sources = {v['data-src'] for v in wall}
        self.assertEqual(len(sources), 8)
        self.assertLess(sum((PUBLIC / s).stat().st_size for s in sources), 1024 * 1024)
        for v in wall:
            self.assertIn('muted', v)
            self.assertIn('playsinline', v)
            self.assertEqual(v['preload'], 'none')
            self.assertNotIn('autoplay', v)
            self.assertNotIn('src', v)
            self.assertTrue((PUBLIC / v['poster']).is_file())

    def test_noisy_other_tasks_retain_records_without_automatic_footage(self):
        groups = [g for g in self.comparison['groups'] if g.get('previewPolicy') == 'records-only']
        self.assertEqual(len(groups), 2)
        for g in groups:
            self.assertEqual(len(g['runIds']), 10)
            for rid in g['runIds']:
                r = self.comparison['runs'][rid]
                self.assertEqual(r['previewPolicy'], 'records-only')
                self.assertEqual(r['taskId'], g['taskId'])
                self.assertTrue((PUBLIC / r['detailUrl']).is_file())
                self.assertNotIn('updatedRecording', r)


if __name__ == '__main__':
    unittest.main()
