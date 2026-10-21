"""Outcome comparisons must share exact tasks and retain previous-model results."""
import collections,json,re,unittest
from pathlib import Path
U=Path(__file__).resolve().parents[1]/'public'
class PlanningOutcomes(unittest.TestCase):
 def test_matched_population_and_preserved_results(self):
  base=json.loads((U/'data/report/metrics.json').read_text())
  data=json.loads((U/'data/report/planning-outcomes.json').read_text())
  self.assertEqual(data['models'],base['models'])
  original={t['taskId']:t for t in base['planning']['tasks']}
  self.assertEqual(set(original),{t['taskId'] for t in data['tasks']})
  self.assertEqual(len(data['tasks']),157)
  for t in data['tasks']:
   self.assertEqual(set(t['outcomes']),set(data['models']))
   self.assertEqual(set(t['successModels']),{m for m,o in t['outcomes'].items() if o=='success'})
   for m in base['models']:self.assertEqual(t['outcomes'][m],original[t['taskId']]['outcomes'][m])
   self.assertRegex(t['argonReportSha256'],r'^[a-f0-9]{64}$')
  self.assertEqual(data['allFail'],sum(not t['successModels'] for t in data['tasks']))
  self.assertEqual(data['atLeastOneSucceeds'],sum(bool(t['successModels']) for t in data['tasks']))
  self.assertEqual(data['allSucceed'],sum(len(t['successModels'])==10 for t in data['tasks']))
  self.assertEqual((data['allFail'],data['atLeastOneSucceeds'],data['allSucceed']),(15,142,8))
  study=json.loads((U/'data/study.json').read_text())
  names={p['model']:p['name'] for p in base['profiles']}
  for row in data['sourceResults']:
   model=next(m for m in study['models'] if m['name']==names[row['model']])
   self.assertEqual(round(100*row['completed']/157,1),float(model['scores'][6]))
   self.assertEqual(row['completed'],sum(s['completed'] for s in row['sourceResults']))
  # All ten models now have hash-matched adopted logs.
  self.assertEqual(base['execution']['modelCount'],10)
 def test_case_links_preserve_exact_task_identity(self):
  gallery=json.loads((U/'data/case-comparisons.json').read_text())
  cases={g['id']:g['taskId'] for g in gallery['groups']}
  data=json.loads((U/'data/report/planning-outcomes.json').read_text())
  for t in data['tasks']:
   if t['caseId']:self.assertEqual(cases[t['caseId']],t['taskId'])
 def test_every_aggregate_profile_has_a_display_name(self):
  data=json.loads((U/'data/report/metrics.json').read_text())
  for p in data['profiles']:self.assertIn(p['model'],data['shortNames'])
if __name__=='__main__':unittest.main()
