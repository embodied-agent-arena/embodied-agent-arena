"""Adopted Argon logs must join exact runs and preserve measured progress."""
import json,statistics,unittest
from pathlib import Path
U=Path(__file__).resolve().parents[1]/'public'
def read(p):return json.loads((U/p).read_text())
class ArgonLogs(unittest.TestCase):
 def test_every_gallery_report_matches_adopted_log(self):
  a=read('data/argon-logs-audit.json');c=read('data/case-comparisons.json')
  self.assertEqual(len(a['caseEvidence']),109)
  self.assertTrue(a['historicalAttemptsExcluded'])
  for e in a['caseEvidence']:
   r=c['runs'][e['runId']];d=read(r['detailUrl'])
   self.assertEqual((e['taskId'],e['reportSha256']),(r['taskId'],r['sourceReportSha256']))
   self.assertEqual(e['reportSha256'],d['reportSha256'])
   self.assertTrue(any(s.get('code') or s.get('calls') or s.get('feedback') for s in d['steps']))
   self.assertTrue(e['adoptedAttemptId'])
 def test_planning_resources_and_outcomes_share_population(self):
  episodes=[r for r in read('data/report/planning-episodes.json') if r['model']=='gemini4argon']
  tasks={t['taskId']:t for t in read('data/report/planning-outcomes.json')['tasks']}
  m=next(m for m in read('data/report/metrics.json')['planning']['models'] if m['model']=='gemini4argon')
  self.assertEqual(len(episodes),157);self.assertEqual(len({r['taskId'] for r in episodes}),157)
  self.assertEqual(sum(r['outcome']=='success' for r in episodes),110)
  self.assertEqual(sum(bool(r['executionErrorRounds']) for r in episodes),28)
  self.assertEqual(sum(r['recoveredAfterError'] for r in episodes),16)
  for r in episodes:
   self.assertEqual(r['reportSha256'],tasks[r['taskId']]['argonReportSha256'])
   self.assertEqual(r['outcome'],tasks[r['taskId']]['outcomes']['gemini4argon'])
   self.assertIsNotNone(r['tokens'])
   if r['recoveredAfterError']:self.assertEqual(r['outcome'],'success');self.assertGreater(r['rounds'],r['firstErrorRound'])
  self.assertEqual(m['time']['median'],statistics.median(r['elapsedSeconds'] for r in episodes))
  self.assertAlmostEqual(m['unsuccessfulTokenShare'],sum(r['tokens'] for r in episodes if r['outcome']!='success')/sum(r['tokens'] for r in episodes))
 def test_actions_are_distinct_from_native_steps_and_progress(self):
  stories={s['id']:s for s in read('data/execution/complete-index.json')['stories']}
  def detail(story):return read(next(r['detailUrl'] for r in stories[story]['runs'] if r['modelKey']=='gemini4argon'))
  c=detail('sequence');self.assertEqual(c['callCounts']['submit_calvin_action'],475);self.assertEqual(c['callCounts']['observe_calvin_state'],408)
  self.assertEqual(c['progress']['value'],5);self.assertEqual(c['loopBudgets'],{'max_turns':40,'timeout_seconds':7200})
  p=detail('precision');self.assertEqual(p['stats']['nativeSteps'],23);self.assertEqual(p['callCounts']['apply_maniskill_action'],38);self.assertFalse(p['success'])
  self.assertIn('max_total_tokens',p['stopLimits']);self.assertAlmostEqual(p['stats']['finalGoalDistanceCm'],15.94)
  r=detail('recovery');self.assertEqual([s['score'] for s in r['steps']],[0,0,.3,.3,.5,.5,.9,1])
if __name__=='__main__':unittest.main()
