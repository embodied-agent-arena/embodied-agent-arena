"""Replacement results, traces and replays must describe the same native run."""
import collections
import hashlib
import importlib.util
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / 'public'
spec = importlib.util.spec_from_file_location('execution_health', ROOT/'scripts/audit_case_execution_health.py')
health = importlib.util.module_from_spec(spec)
spec.loader.exec_module(health)


def read(name):
    return json.loads((PUB/name).read_text())


class HumanClawRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = read('data/report/humanclaw-qwen-refresh.json')
        cls.records = cls.audit['records']

    def test_complete_fixed_population_and_original_ids(self):
        self.assertEqual(len(self.records), 39)
        self.assertEqual(collections.Counter(r['modelKey'] for r in self.records),
                         {'qwen': 13, 'qwen35-397b': 13, 'qwen35-27b': 13})
        by_task = collections.defaultdict(list)
        for r in self.records:
            by_task[r['taskId']].append(r)
            self.assertEqual(r['seed'], 42)
            self.assertRegex(r['recordId'], r'^[a-f0-9]{20}$')
            self.assertNotEqual(r['recordId'], r['replacementRecordId'])
        self.assertEqual(len(by_task), 13)
        for runs in by_task.values():
            self.assertEqual(len(runs), 3)
            self.assertEqual(len({r['identitySha256'] for r in runs}), 1)

    def test_replays_preserve_exact_report_and_frame_order(self):
        with_video = [r for r in self.records if r['video']]
        self.assertEqual(len(with_video), 38)
        for r in self.records:
            d = read(r['detailUrl'])
            p = read(r['mediaProvenance'])
            for key in ['recordId', 'taskId', 'modelKey', 'reportSha256', 'video', 'poster', 'success', 'auditStatus']:
                self.assertEqual(d[key], r[key])
            self.assertEqual(p['sourceReportSha256'], r['reportSha256'])
            self.assertEqual(p['nativeIdentity'], r['nativeIdentity'])
            if r['video']:
                self.assertEqual(r['mediaKind'], 'Saved observations')
                self.assertEqual(r['mediaReportSha256'], r['reportSha256'])
                self.assertEqual(hashlib.sha256((PUB/r['video']).read_bytes()).hexdigest(), p['videoSha256'])
                self.assertEqual(hashlib.sha256((PUB/r['poster']).read_bytes()).hexdigest(), p['posterSha256'])
                self.assertEqual(p['sourceObservationFrames'], r['observationFrameCount'])
                self.assertEqual(p['videoDurationSeconds'], r['observationFrameCount']/2)
                self.assertEqual([f['video_frame_order'] for f in p['observations']], list(range(r['observationFrameCount'])))
            else:
                self.assertEqual(r['replacementRecordId'], 'qwen-07')
                self.assertEqual(r['outcome'], 'budget_exhausted')
                self.assertEqual(r['stats']['environmentSteps'], 0)
                self.assertIsNone(r['mediaReportSha256'])
                self.assertIsNone(r['poster'])

    def test_provider_interruptions_are_not_hidden_by_recordings(self):
        interrupted = []
        stopped = []
        for r in self.records:
            d = read(r['detailUrl'])
            issue = health.api_interruption(d)
            if issue:
                interrupted.append(r['replacementRecordId'])
                self.assertEqual(issue['phase'], 'after_actions')
                self.assertEqual(r['auditStatus'], 'provider_interruption')
                self.assertTrue(r['video'])
                self.assertFalse(r['success'])
            if r['auditStatus'] == 'task_failure_after_agent_stop':
                stopped.append(r)
                self.assertEqual(r['outcome'], 'agent_failure')
                self.assertTrue(d['verifier']['success']['active_stop'])
                self.assertIsNone(issue)
        self.assertEqual(set(interrupted), {'qwen-02', 'qwen-08'})
        self.assertEqual(len(stopped), 3)

    def test_native_actions_and_public_privacy(self):
        for r in self.records:
            d = read(r['detailUrl'])
            self.assertEqual(len(d['steps']), r['stats']['rounds'])
            calls = [c for s in d['steps'] for c in s['calls']]
            self.assertEqual(len(calls), r['stats']['actionSubmissions'])
            for c in calls:
                self.assertIn('arguments', c)
                self.assertIn('output', c)
                self.assertIn('ok', c)
            for path in [r['detailUrl'], r['mediaProvenance']]:
                text = (PUB/path).read_text()
                for forbidden in ['/home/', '/Users/', 'api.gpt.ge', '"base_url"', '"request_id"', '"authorization"']:
                    self.assertNotIn(forbidden, text)
                self.assertIsNone(re.search(r'\bsk-[A-Za-z0-9_-]{16,}', text))
                self.assertIsNone(re.search(r'[\u4e00-\u9fff]', text))

    def test_all_current_atlas_and_overview_links_match(self):
        cases = read('data/case-comparisons.json')
        selected = [cases['runs'][rid] for rid in self.audit['updatedAtlasRecords']]
        self.assertEqual(len(selected), 18)
        for r in selected:
            d = read(r['detailUrl'])
            self.assertEqual(r['recordId'], d['recordId'])
            self.assertEqual(r['sourceReportSha256'], d['reportSha256'])
            self.assertEqual(r['video'], d['video'])
        overview = read('data/report/overview-provenance.json')
        chair = [r for r in overview['tiles'] if r['benchmark'] == 'HumanCLAW' and r['modelKey'].startswith('qwen')]
        self.assertEqual([r['status'] for r in chair], ['Completed', 'Completed', 'Not completed'])
        for r in chair:
            self.assertTrue(r['sourceVideo'])
            self.assertEqual(r['mediaKind'], 'Saved observations')
            self.assertIsNone(r['sceneContext'])


if __name__ == '__main__':
    unittest.main()
