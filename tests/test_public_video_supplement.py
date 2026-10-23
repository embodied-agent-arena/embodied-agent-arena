"""Supplemental clips must belong to the displayed evaluation, with truthful media types."""
import hashlib
import json
import unittest
from pathlib import Path

PUB=Path(__file__).resolve().parents[1]/'public'


def read(path):
    return json.loads((PUB/path).read_text())


class VideoSupplementTests(unittest.TestCase):
    def test_every_clip_matches_its_task_model_and_report(self):
        audit=read('data/video-supplement-audit.json')
        runs=read('data/case-comparisons.json')['runs']
        self.assertEqual(audit['matchedAssignments'],70)
        self.assertEqual(len(audit['entries']),65)
        self.assertEqual(len(audit['gaps']),5)
        for entry in audit['entries']:
            with self.subTest(run=entry['runId']):
                run=runs[entry['runId']]
                self.assertEqual((run['taskId'],run['modelKey'],run['sourceReportSha256']),
                                 (entry['taskId'],entry['modelKey'],entry['reportSha256']))
                self.assertEqual(run['sourceReportSha256'],run['mediaReportSha256'])
                detail=read(run['detailUrl'])
                for field in ['video','poster','mediaKind','mediaNote']:
                    self.assertEqual(run[field],detail[field])
                self.assertEqual(hashlib.sha256((PUB/run['video']).read_bytes()).hexdigest(),entry['videoSha256'])
                self.assertEqual(hashlib.sha256((PUB/run['poster']).read_bytes()).hexdigest(),entry['posterSha256'])
                provenance=read(entry['provenance'])
                for private_prefix in ['/Users/','/home/','/mnt/','/data/']:
                    self.assertNotIn(private_prefix,json.dumps(provenance))
                self.assertEqual(provenance['sourceReportSha256'],entry['reportSha256'])
                if entry['mediaKind']=='Verified action replay':
                    replay=provenance['replayVerification']
                    self.assertTrue(replay['all_states_equal'])
                    self.assertEqual(len(replay['checks']),replay['actions'])
                    self.assertTrue(all(c['before_equal'] and c['after_equal'] and c['response_equal'] for c in replay['checks']))
                if entry['mediaKind']=='Saved-state render':
                    self.assertEqual(provenance['physics_steps'],0)
                    self.assertEqual(provenance['vlm_calls'],0)
        overview=read('data/report/overview-provenance.json')
        self.assertEqual(sum(bool(t['sourceVideo'] or t['sourceAnimation']) for t in overview['tiles'] if t['column']<=7),68)
        self.assertEqual(overview['recordingCount'],107)
        self.assertEqual(overview['demoCount'],3)

    def test_missing_video_never_borrows_another_run(self):
        runs=read('data/case-comparisons.json')['runs']
        audit=read('data/video-supplement-audit.json')
        for gap in audit['gaps']:
            run=runs[gap['runId']]
            if gap.get('resolvedBy'):
                replacement=gap['resolvedBy']
                self.assertEqual(run['sourceReportSha256'],replacement['reportSha256'])
                self.assertEqual(run['video'],replacement['video'])
                self.assertEqual(run['mediaProvenance'],replacement['provenance'])
                self.assertNotEqual(run['sourceReportSha256'],gap['reportSha256'])
                continue
            self.assertIsNone(run['video'])
            self.assertIsNone(run['animation'])
            self.assertEqual(run['sourceReportSha256'],gap['reportSha256'])
        self.assertEqual({(g['modelKey'],g['benchmark']) for g in audit['gaps']},
                         {('astra','CALVIN'),('astra','RLBench'),('qwen','HumanCLAW'),('qwen35-397b','HumanCLAW'),('qwen35-27b','HumanCLAW')})


if __name__=='__main__':
    unittest.main()
