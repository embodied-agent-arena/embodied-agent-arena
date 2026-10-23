#!/usr/bin/env python3
"""Separate API-interrupted attempts from environment/runtime outcomes.

This adds presentation metadata and a rerun queue. It never changes an original
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
    """A terminal model-request failure needs a rerun; recovered errors do not.

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
    return {'status': 'rerun_required', 'reason': 'api_failure',
            'phase': 'after_actions' if acted else 'before_actions',
            'error': error, 'failedRound': last.get('round')}


def build():
    data = json.loads((PUB/'data/case-comparisons.json').read_text())
    issues, rows = {}, []
    for group in data['groups']:
        for rid in group['runIds']:
            run = data['runs'][rid]
            detail = json.loads((PUB/run['detailUrl']).read_text())
            assert detail['taskId'] == run['taskId'] == group['taskId']
            assert detail['modelKey'] == run['modelKey']
            assert detail['reportSha256'] == run['sourceReportSha256']
            issue = api_interruption(detail)
            if not issue:
                continue
            issues[rid] = {**issue, 'caseId': group['id'], 'taskId': run['taskId'],
                           'modelKey': run['modelKey'], 'recordId': run['recordId'],
                           'reportSha256': run['sourceReportSha256'],
                           'recordedOutcome': run['outcome']}
            rows.append({'priority': 'homepage' if group['id']=='showcase-humanclaw' else 'case_atlas',
                         'model': run['model'], 'model_key': run['modelKey'],
                         'api_model_id': API_MODELS.get(run['modelKey'], ''),
                         'case_id': group['id'], 'task_id': run['taskId'],
                         'native_identity': json.dumps(group['nativeIdentity'], sort_keys=True),
                         'record_id': run['recordId'], 'phase': issue['phase'],
                         'error': issue['error'], 'recorded_outcome': run['outcome'],
                         'report_sha256': run['sourceReportSha256'],
                         'recorded_budgets': json.dumps(run.get('budgets', {}), sort_keys=True),
                         'detail_url': run['detailUrl']})
    rows.sort(key=lambda r:(r['priority']!='homepage', r['task_id'], r['model_key']))
    report = {'checkedRecords': len(data['runs']), 'checkedTasks': len(data['groups']),
              'rerunCount': len(issues), 'byPhase': dict(Counter(i['phase'] for i in issues.values())),
              'runs': issues,
              'policy': 'API-interrupted attempts require reruns. Original outcomes and aggregate scores are preserved until replacement runs are supplied.'}
    (PUB/'data/report/case-execution-health.json').write_text(json.dumps(report, indent=2)+'\n')
    with (PUB/'data/report/case-reruns.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=['priority', 'model', 'model_key', 'api_model_id',
                                                'case_id', 'task_id', 'native_identity', 'record_id',
                                                'phase', 'error', 'recorded_outcome', 'report_sha256',
                                                'recorded_budgets', 'detail_url'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Audited {len(data['runs'])} records: {len(issues)} API-interrupted attempts; {report['byPhase']}")


if __name__ == '__main__':
    build()
