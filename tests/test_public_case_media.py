"""Media recovery and shared-scene fallback must never change task or model evidence."""
import hashlib
import json
import unittest
from pathlib import Path

PUB=Path(__file__).resolve().parents[1]/'public'
def read(path): return json.loads((PUB/path).read_text())

class CaseMediaTests(unittest.TestCase):
    def test_every_task_has_verified_context(self):
        data=read('data/case-comparisons.json')
        self.assertEqual(len(data['groups']),109)
        for group in data['groups']:
            context=group['sceneContext']; source=data['runs'][context['sourceRunId']]
            self.assertIn(source['id'],group['runIds'])
            self.assertEqual(context['taskId'],group['taskId'])
            self.assertEqual(context['taskId'],source['taskId'])
            self.assertEqual(context['image'],source['poster'])
            self.assertTrue((PUB/context['image']).is_file())
            self.assertEqual(source['sourceReportSha256'],source['mediaReportSha256'])
            if context['kind']=='scene':
                self.assertNotEqual(source['mediaKind'],'Animated execution log')
            else:
                self.assertEqual(context['kind'],'action_log')
                self.assertEqual(source['mediaKind'],'Animated execution log')

    def test_recovered_video_matches_own_evaluation(self):
        data=read('data/case-comparisons.json'); audit=read('data/case-recovery-audit.json')
        self.assertEqual(len(audit['entries']),30)
        for entry in audit['entries']:
            run=data['runs'][entry['runId']];detail=read(run['detailUrl'])
            self.assertEqual((run['taskId'],run['modelKey'],run['sourceReportSha256']),
                             (entry['taskId'],entry['modelKey'],entry['reportSha256']))
            self.assertEqual(hashlib.sha256((PUB/entry['video']).read_bytes()).hexdigest(),entry['videoSha256'])
            self.assertEqual(run['video'],detail['video'])
            self.assertEqual(run['poster'],detail['poster'])
            self.assertTrue(run['mediaMatchesReport'])
            self.assertTrue(detail['mediaMatchesReport'])
            self.assertEqual(run['success'],detail['success'])

    def test_context_is_not_promoted_to_a_model_recording(self):
        overview=read('data/report/overview-provenance.json')
        contexts=[c for c in overview['tiles'] if c.get('sceneContext')]
        self.assertEqual(len(contexts),3)
        for c in contexts:
            self.assertIsNone(c['sourceVideo']);self.assertIsNone(c['sourceAnimation'])
            self.assertIsNone(c['poster']);self.assertIsNone(c['mediaReportSha256'])
            self.assertEqual(c['sceneContext']['taskId'],c['taskId'])

    def test_execution_explorer_context_uses_same_task(self):
        for s in read('data/execution/complete-index.json')['stories']:
            self.assertEqual({r['taskId'] for r in s['runs']},{s['sceneContext']['taskId']})

if __name__=='__main__':unittest.main()
