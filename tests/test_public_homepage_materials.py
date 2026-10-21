"""Footage provenance must survive propagation into the gallery and execution view."""
import unittest,json,hashlib
from pathlib import Path
PUB=Path(__file__).resolve().parents[1]/'public'
def read(s):return json.loads((PUB/s).read_text())
class HomepageMaterialsTests(unittest.TestCase):
 def test_supplement_and_all_derived_views(self):
  a=read('data/homepage-supplement-audit.json');runs=read('data/case-comparisons.json')['runs'];self.assertEqual(len(a['entries']),17)
  for e in a['entries']:
   r=runs[e['runId']];d=read(r['detailUrl']);proof=read(e['provenance'])
   self.assertEqual((r['modelKey'],r['taskId'],r['sourceReportSha256']),(e['modelKey'],e['taskId'],e['reportSha256']))
   self.assertEqual(r['outcome'],proof['original_outcome'])
   if e['availability']=='demo_video':
    self.assertFalse(r['demoMedia']['countInEvaluation']);self.assertIsNone(r['video']);self.assertNotEqual(r['demoMedia']['reportSha256'],r['sourceReportSha256']);self.assertEqual(r['demoMedia'],d['demoMedia'])
   else:
    self.assertEqual(r['sourceReportSha256'],r['mediaReportSha256'])
    for k in ['video','animation','poster','mediaKind']:self.assertEqual(r[k],d[k])
   if e['availability']=='observations_only':
    self.assertIsNone(r['video']);self.assertEqual([f['round'] for f in r['frames']],list(range(1,25)))
    for f in r['frames']:self.assertEqual(hashlib.sha256((PUB/f['image']).read_bytes()).hexdigest(),f['sha256'])
  index=read('data/execution/complete-index.json')
  for story in index['stories']:
   for r in story['runs']:
    d=read(r['detailUrl'])
    if r.get('demoMedia'):self.assertEqual(r['demoMedia'],d['demoMedia'])
    if 'homepage-supplement' in (r.get('video') or r.get('animation') or ''):
     self.assertEqual(r['video'],d['video']);self.assertEqual(r.get('animation'),d.get('animation'))
