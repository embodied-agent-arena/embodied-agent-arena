"""Protect public comparisons against cross-task and cross-execution joins."""
import json
import unittest
from pathlib import Path

PUBLIC = Path(__file__).resolve().parents[1] / 'public'


def read(name):
    return json.loads((PUBLIC / name).read_text())


class ExecutionComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = read('data/execution/index.json')
        cls.media = read('data/case-media.json')
        cls.audit = read('data/execution/identity-audit.json')

    def test_each_gallery_group_is_one_task(self):
        seen = set()
        for group in self.media['groups']:
            with self.subTest(group=group['id']):
                self.assertEqual(group['galleryMode'], 'same-task')
                self.assertEqual({self.media['runs'][r]['taskId'] for r in group['runIds']}, {group['taskId']})
                self.assertFalse(seen.intersection(group['runIds']))
                seen.update(group['runIds'])
        self.assertEqual(seen, {r for r, d in self.media['runs'].items() if not d.get('demoOnly')})

    def test_execution_groups_share_native_task_configuration(self):
        for story in self.index['stories']:
            with self.subTest(task=story['id']):
                records = self.audit['comparisons'][story['id']]
                self.assertEqual(set(records), {r['id'] for r in story['runs']})
                self.assertEqual(len({r['taskId'] for r in records.values()}), 1)
                self.assertEqual(len({r['identitySha256'] for r in records.values()}), 1)
                for run in story['runs']:
                    self.assertEqual(records[run['id']]['taskId'], run['taskId'])
                    self.assertEqual(records[run['id']]['reportSha256'], run['reportSha256'])

    def test_video_cannot_illustrate_a_different_execution(self):
        mismatches = 0
        for story in self.index['stories']:
            for run in story['runs']:
                with self.subTest(run=run['id']):
                    detail = read(run['detailUrl'])
                    source = self.media['runs'][run['id']]
                    matches = source['sourceReportSha256'] == detail['reportSha256']
                    self.assertEqual(matches, detail['mediaMatchesReport'])
                    if not matches:
                        mismatches += 1
                        self.assertIsNone(detail['video'])
                        self.assertIsNone(detail['poster'])
                    else:
                        self.assertEqual(detail['video'], source['video'])
                    self.assertEqual(run['video'], detail['video'])
                    for field in ('video', 'poster'):
                        if detail.get(field):
                            self.assertTrue((PUBLIC / detail[field]).is_file())
        # This known regression is why same task alone is insufficient.
        self.assertGreater(mismatches, 0)

    def test_progress_uses_native_outcomes_not_tool_call_count(self):
        for story in self.index['stories']:
            for run in story['runs']:
                with self.subTest(run=run['id']):
                    detail = read(run['detailUrl'])
                    p = detail['progress']
                    self.assertNotIn('call', p['label'].lower())
                    self.assertLessEqual(p['value'], p['total'])
                    self.assertGreaterEqual(p['value'], 0)
                    if p['label'] == 'Task completion':
                        self.assertEqual(p['value'], int(detail['success']))
                    for moment in detail['milestones']:
                        self.assertLess(moment['index'], len(detail['steps']))
                        self.assertEqual(moment['at'], detail['steps'][moment['index']]['at'])


if __name__ == '__main__':
    unittest.main()
