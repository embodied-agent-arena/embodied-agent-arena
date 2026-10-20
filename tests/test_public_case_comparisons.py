"""Every gallery model must refer to the exact task and execution shown."""
import json
import unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'public'
def read(p):return json.loads((P/p).read_text())

class CompleteCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=read('data/case-comparisons.json')
        cls.audit=read('data/case-comparison-audit.json')

    def test_ten_models_for_every_task(self):
        d=self.data
        self.assertEqual(len(d['models']),10)
        self.assertEqual(len(d['groups']),109)
        for g in d['groups']:
            with self.subTest(task=g['id']):
                rr=[d['runs'][i] for i in g['runIds']]
                self.assertEqual([r['model'] for r in rr],d['models'])
                self.assertEqual({r['taskId'] for r in rr},{g['taskId']})
                self.assertEqual({r['identitySha256'] for r in rr if r['executionStarted']},{g['identitySha256']})
                for r in rr:
                    if not r['executionStarted']:
                        self.assertFalse(r['success'])
                        self.assertEqual(r['modelRounds'],0)
                        self.assertFalse(r['video'] or r['animation'])

    def test_media_and_details_belong_to_identical_execution(self):
        for r in self.data['runs'].values():
            with self.subTest(record=r['id']):
                detail=read(r['detailUrl'])
                self.assertEqual(detail['taskId'],r['taskId'])
                self.assertEqual(detail['modelKey'],r['modelKey'])
                self.assertEqual(detail['reportSha256'],r['sourceReportSha256'])
                self.assertEqual(detail['recordId'],r['recordId'])
                self.assertEqual(detail['success'],r['success'])
                if r['video'] or r['animation']:
                    self.assertEqual(r['mediaReportSha256'],r['sourceReportSha256'])
                    for key in ['video','animation','poster']:
                        if r.get(key):self.assertTrue((P/r[key]).is_file())
                else:self.assertIsNone(r['mediaReportSha256'])

    def test_legacy_aliases_point_to_exact_task(self):
        groups={g['id']:g for g in self.data['groups']}
        for c in read('data/case-studies.json'):
            if not c['predictions']:
                self.assertEqual(groups[self.data['aliases'][c['id']]]['taskId'],c['sample'])

    def test_explorer_all_nine_selectable(self):
        originals={s['id']:s for s in read('data/execution/index.json')['stories']}
        for s in read('data/execution/complete-index.json')['stories']:
            with self.subTest(story=s['id']):
                self.assertEqual([r['model'] for r in s['runs']],self.data['models'][:9])
                self.assertEqual(len({r['taskId'] for r in s['runs']}),1)
                for old in originals[s['id']]['runs']:
                    self.assertEqual(old,next(r for r in s['runs'] if r['modelKey']==old['modelKey']))
                for r in s['runs']:
                    detail=read(r['detailUrl'])
                    self.assertEqual(detail['taskId'],r['taskId'])
                    self.assertEqual(detail['reportSha256'],r['reportSha256'])
                    self.assertEqual(detail['video'],r['video'])

if __name__=='__main__':unittest.main()
