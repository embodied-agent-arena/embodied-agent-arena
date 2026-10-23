"""API failures, budget exhaustion, and generated-code failures are distinct."""
import csv
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('health', ROOT/'scripts/audit_case_execution_health.py')
health = importlib.util.module_from_spec(spec)
spec.loader.exec_module(health)


class ExecutionHealthTests(unittest.TestCase):
    def test_request_failure_before_execution(self):
        step = {'error': 'ModelRequestError: Model API transport failure: URLError',
                'feedback': {'stage': 'model_completion'}}
        self.assertEqual(health.api_interruption({'steps': [step]})['phase'], 'before_actions')

    def test_request_failure_after_summarized_execution(self):
        steps = [{'feedback': {'stage': 'execution', 'execution_ok': True}},
                 {'error': 'Model API HTTP 429', 'feedback': {'stage': 'model_completion'}}]
        self.assertEqual(health.api_interruption({'steps': steps})['phase'], 'after_actions')

    def test_budget_runtime_and_recovered_api_errors_are_not_pending(self):
        for detail in [
            {'steps': [{'error': 'loop budget exhausted: max_visible_tokens_per_response',
                        'feedback': {'stage': 'model_completion'}, 'budgetStop': True}]},
            {'steps': [{'error': 'NameError: target is not defined', 'code': 'move(target)',
                        'feedback': {'stage': 'execution'}}]},
            {'steps': [{'error': 'Model API HTTP 500', 'feedback': {'stage': 'model_completion'}},
                       {'success': True, 'feedback': {'stage': 'execution'}}]},
        ]:
            self.assertIsNone(health.api_interruption(detail))

    def test_verified_provider_error_can_surface_as_empty_model_choices(self):
        detail = {'auditStatus': 'provider_interruption', 'providerErrorCode': 'data_inspection_failed',
                  'steps': [{'feedback': {'stage': 'execution'}},
                            {'error': 'ModelResponseError: Expected exactly one model choice',
                             'feedback': {'stage': 'model_completion', 'providerErrorCode': 'data_inspection_failed'}}]}
        self.assertEqual(health.api_interruption(detail)['phase'], 'after_actions')
        detail.pop('providerErrorCode')
        self.assertIsNone(health.api_interruption(detail))

    def test_audit_covers_every_public_record_and_preserves_source_outcomes(self):
        public = ROOT/'public'
        data = json.loads((public/'data/case-comparisons.json').read_text())
        report = json.loads((public/'data/report/case-execution-health.json').read_text())
        found = {}
        for rid, run in data['runs'].items():
            detail = json.loads((public/run['detailUrl']).read_text())
            issue = health.api_interruption(detail)
            if issue:
                found[rid] = issue
                audited = report['runs'][rid]
                self.assertEqual(audited['recordedOutcome'], run['outcome'])
                self.assertEqual(audited['reportSha256'], detail['reportSha256'])
                self.assertEqual(audited['taskId'], detail['taskId'])
        self.assertEqual(set(found), set(report['runs']))
        self.assertEqual(report['checkedRecords'], len(data['runs']))
        self.assertEqual(report['errorCount'], 21)
        self.assertEqual(report['totalErrorCount'], 23)
        self.assertEqual(report['additionalErrorCount'], 2)
        self.assertEqual(report['rerunCount'], 0)
        self.assertEqual(report['pendingRerunCount'], 0)
        for issue in [*report['runs'].values(), *report['additionalRuns'].values()]:
            self.assertEqual(issue['status'], 'counted_failure')
            self.assertTrue(issue['countedAsFailure'])
            self.assertTrue(issue['final'])
            self.assertFalse(issue['rerunRequired'])
        self.assertEqual(report['byPhase'], {'before_actions': 9, 'after_actions': 12})


    def test_error_inventory_has_no_pending_reruns(self):
        public=ROOT/'public'
        rows=list(csv.DictReader((public/'data/report/case-api-errors.csv').read_text().splitlines()))
        self.assertEqual(len(rows),23)
        self.assertEqual(len({(r['model_key'],r['task_id']) for r in rows}),23)
        for row in rows:
            self.assertEqual(row['status'],'API error')
            self.assertEqual(row['counted_as_failure'],'True')
            self.assertEqual(row['rerun_required'],'False')
            detail=json.loads((public/row['detail_url']).read_text())
            self.assertFalse(detail['success'])
            self.assertEqual(row['recorded_outcome'],detail['outcome'])
        self.assertEqual((public/'data/report/case-api-errors.csv').read_bytes(),
                         (public/'data/report/case-reruns.csv').read_bytes())
        for filename in ['case-studies.js','execution-analysis.js','humanclaw-refresh.html']:
            text=(public/filename).read_text()
            self.assertNotIn('Rerun required',text)
            self.assertNotIn('replacement evaluation needed',text)


if __name__ == '__main__':
    unittest.main()
