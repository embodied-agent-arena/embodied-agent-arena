#!/usr/bin/env python3
"""Separate API-interrupted attempts from environment/runtime outcomes.

This adds presentation metadata and a final error inventory. It never changes an original
outcome, score, report, or the published aggregate evaluation population.
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / 'public'
API_ERROR = re.compile(r'Model (?:API (?:transport failure|HTTP \d{3})|request failed)', re.I)
API_MODELS = {
    'astra': 'gpt-6-astra',
    'gemini': 'gemini-3.8-flash', 'qwen': 'qwen3.8-max',
    'qwen35-397b': 'qwen3.5-397b-a17b', 'qwen35-27b': 'qwen3.5-27b',
}


def api_interruption(detail):
    """A terminal model-request failure is a final counted failure; recovered errors are not.

    Budget stops and failures in generated code are distinct, even when the
    archive's coarse outcome field calls both kinds a runtime_failure.
    """
    steps = detail.get('steps') or []
    if not steps:
        return None
    last = steps[-1]
    error = str(last.get('error') or '')
    # Some providers return an error envelope that the runner reports as an
    # empty-choice parsing error. The replacement import verifies that envelope
    # and publishes only its error code, never its request identifier or body.
    verified_provider_error = (detail.get('auditStatus') == 'provider_interruption'
                               and detail.get('providerErrorCode')
                               and (last.get('feedback') or {}).get('providerErrorCode') == detail['providerErrorCode'])
    if (not API_ERROR.search(error) and not verified_provider_error) or last.get('budgetStop'):
        return None
    if last.get('code') or last.get('calls'):
        return None
    stage = (last.get('feedback') or {}).get('stage')
    if stage not in (None, 'model_completion'):
        return None
    acted = any(s.get('code') or s.get('calls') or
                (s.get('feedback') or {}).get('stage') == 'execution' for s in steps[:-1])
    if verified_provider_error:
        error = 'Model request failed: provider error ' + detail['providerErrorCode'] + '.'
    return {'status': 'counted_failure', 'reason': 'api_failure',
            'displayStatus': 'API error', 'final': True, 'countedAsFailure': True, 'rerunRequired': False,
            'phase': 'after_actions' if acted else 'before_actions',
            'error': error, 'failedRound': last.get('round')}


POLICY = ('Terminal API interruptions are final errors and count as unsuccessful attempts in the existing evaluation population. '
          'No reruns are pending. Original outcomes, native success flags, report identities, recordings and aggregate scores are preserved. '
          'API errors remain distinct from generated-code or environment runtime errors and budget stops.')


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':'))+'\n')


def build():
    data = json.loads((PUB/'data/case-comparisons.json').read_text())
    refresh = json.loads((PUB/'data/report/humanclaw-qwen-refresh.json').read_text())
    issues, additional, rows, seen, by_report = {}, {}, [], set(), {}

    def inspect(run, case_id, identity, scope):
        key = run['modelKey'], run['taskId']
        if key in seen:
            return
        seen.add(key)
        detail = json.loads((PUB/run['detailUrl']).read_text())
        digest = run.get('sourceReportSha256') or run['reportSha256']
        assert detail['taskId'] == run['taskId'] and detail['modelKey'] == run['modelKey']
        assert detail['reportSha256'] == digest
        issue = api_interruption(detail)
        if not issue:
            return
        assert not run['success'] and not detail['success'], 'An API interruption must already be counted as unsuccessful.'
        enriched = {**issue, 'caseId': case_id, 'taskId': run['taskId'], 'modelKey': run['modelKey'],
                    'recordId': run['recordId'], 'reportSha256': digest, 'recordedOutcome': run['outcome'], 'scope': scope}
        (issues if scope == 'case_atlas' else additional)[run['id']] = enriched
        by_report[digest] = enriched
        rows.append({'scope': scope, 'status': 'API error', 'counted_as_failure': True, 'rerun_required': False,
                     'model': run['model'], 'model_key': run['modelKey'],
                     'api_model_id': API_MODELS.get(run['modelKey'], ''),
                     'case_id': case_id, 'task_id': run['taskId'],
                     'native_identity': json.dumps(identity, sort_keys=True),
                     'record_id': run['recordId'], 'phase': issue['phase'], 'error': issue['error'],
                     'recorded_outcome': run['outcome'], 'report_sha256': digest,
                     'recorded_budgets': json.dumps(run.get('budgets', {}), sort_keys=True),
                     'detail_url': run['detailUrl']})

    for group in data['groups']:
        for rid in group['runIds']:
            run = data['runs'][rid]
            assert run['taskId'] == group['taskId']
            inspect(run, group['id'], group['nativeIdentity'], 'case_atlas')
    for run in refresh['records']:
        inspect(run, run['id'], run['nativeIdentity'], 'humanclaw_refresh')
    rows.sort(key=lambda r:(r['scope'], r['task_id'], r['model_key']))
    report = {'checkedRecords': len(data['runs']), 'checkedTasks': len(data['groups']),
              'checkedAdditionalRecords': len(seen)-len(data['runs']), 'totalCheckedRecords': len(seen),
              'errorCount': len(issues), 'additionalErrorCount': len(additional), 'totalErrorCount': len(rows),
              'rerunCount': 0, 'pendingRerunCount': 0,
              'byPhase': dict(Counter(i['phase'] for i in issues.values())),
              'allByPhase': dict(Counter(i['phase'] for i in by_report.values())),
              'runs': issues, 'additionalRuns': additional,
              'policy': POLICY, 'policyDate': '2026-10-23',
              'inventory': 'data/report/case-api-errors.csv',
              'compatibilityCsv': 'case-reruns.csv is a legacy download URL for the same final error inventory, not a rerun queue.'}
    (PUB/'data/report/case-execution-health.json').write_text(json.dumps(report, indent=2)+'\n')
    for name in ['case-api-errors.csv', 'case-reruns.csv']:
        with (PUB/'data/report'/name).open('w', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)

    # Attach the same final classification to all current aliases and summary
    # records. Report hashes prevent historical attempts from inheriting a new result.
    def attach(value):
        changed = False
        if isinstance(value, dict):
            if 'modelKey' in value and 'outcome' in value:
                digest = value.get('sourceReportSha256') or value.get('reportSha256')
                issue = by_report.get(digest)
                if issue:
                    assert issue['modelKey'] == value['modelKey'] and issue['taskId'] == value['taskId']
                    if value.get('executionReview') != issue:
                        value['executionReview'] = issue; changed = True
            for key, child in value.items():
                if key != 'executionReview':
                    changed = attach(child) or changed
        elif isinstance(value, list):
            for child in value: changed = attach(child) or changed
        return changed
    paths = [PUB/'data/case-comparisons.json', PUB/'data/case-media.json', PUB/'data/report/humanclaw-qwen-refresh.json']
    paths += list((PUB/'data/execution').glob('*.json')) + list((PUB/'data/case-records').glob('*.json'))
    for path in paths:
        value = json.loads(path.read_text())
        if attach(value): write(path, value)
    print(f"Audited {len(seen)} distinct records: {len(issues)} atlas + {len(additional)} additional API errors; all final counted failures, 0 pending reruns.")


if __name__ == '__main__':
    build()
