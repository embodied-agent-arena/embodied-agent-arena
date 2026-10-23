#!/usr/bin/env python3
"""Verify adopted Planning records, including an explicit HumanCLAW replacement overlay.

With --logs and --argon-logs, all adopted reports are freshly verified. When the
original archives are unavailable, --baseline-public-data preserves their prior
verification attribution and proves retained rows are unchanged; only overlay
reports are described as freshly verified in that mode.
"""
import argparse, collections, csv, gzip, hashlib, io, json, zipfile
from pathlib import Path


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def canonical_blob(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def overlay_records(root):
    """Read latest supplied rows; retain raw outcomes and verify normalization."""
    records = {}
    rows = list(csv.DictReader((root / 'case_results.csv').open(encoding='utf-8-sig')))
    for row in rows:
        directory = root / row['record_path']
        blob = (directory / 'runner_report.json').read_bytes()
        report = json.loads(blob)
        manifest = json.loads((directory / 'execution_manifest.json').read_text())
        errors = json.loads((directory / 'provider_errors.json').read_text())
        native = report['pukun']['run_summary']['tasks'][0]
        stages = report['pukun']['run_summary']['native_metrics']['success']
        task = manifest['tasks'][0]
        assert sha(blob) == row['source_report_sha256']
        assert report['outcome'] == row['raw_outcome']
        assert report['model']['model'] == row['api_model']
        assert task['task_id'] == row['case_id']
        assert native['task_id'] == task['variation'] == report['expected_task_id']
        assert report['actual_task_ids'] == [native['task_id']]
        assert task['seed'] == 42
        assert native['success'] == (row['success'] == 'True')
        provider = 'data_inspection_failed' in json.dumps(errors).lower()
        if provider:
            assert not native['success']
            outcome = 'provider_interruption'
        elif row['audit_status'] == 'task_failure_after_agent_stop':
            assert report['outcome'] == 'agent_failure' and stages['active_stop'] and not native['success']
            outcome = 'task_failure'
        else:
            outcome = report['outcome']
        assert (outcome == 'success') == native['success']
        key = (row['model_id'], row['case_id'])
        assert key not in records
        records[key] = dict(blob=blob, rawOutcome=report['outcome'], outcome=outcome,
                            providerInterruption=provider, replacementRecordId=row['record_id'],
                            nativeTaskId=native['task_id'], seed=task['seed'],
                            reportSha256=sha(blob), reportPath=row['record_path'] + '/runner_report.json')
    assert len(records) == 39
    assert collections.Counter(m for m, _ in records) == {'qwen': 13, 'qwen35-397b': 13, 'qwen35-27b': 13}
    return records


def counts(tasks):
    c = collections.Counter(('astra' in t['successModels'], 'opus55' in t['successModels']) for t in tasks)
    return dict(tasks=len(tasks), both=c[True, True], astraCompletedOpusIncomplete=c[True, False],
                opusCompletedAstraIncomplete=c[False, True], neither=c[False, False],
                astraAloneAmongTen=sum(t['successModels'] == ['astra'] for t in tasks),
                noneOfTen=sum(not t['successModels'] for t in tasks),
                allTen=sum(len(t['successModels']) == 10 for t in tasks))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--logs', type=Path)
    ap.add_argument('--argon-logs', type=Path)
    ap.add_argument('--humanclaw-results', type=Path, help='Extracted latest HumanCLAW results package')
    ap.add_argument('--baseline-public-data', type=Path, help='Pre-refresh public data ZIP with its historic audit')
    ap.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'public/data/report/planning-comparison-audit.json')
    a = ap.parse_args()
    full = a.logs is not None and a.argon_logs is not None
    if (a.logs is None) != (a.argon_logs is None):
        ap.error('Supply both raw archives, or neither with an audited baseline and overlay.')
    if not full and not (a.baseline_public_data and a.humanclaw_results):
        ap.error('Without raw archives, --baseline-public-data and --humanclaw-results are required.')
    u = Path(__file__).resolve().parents[1] / 'public'
    data = json.loads((u / 'data/report/planning-outcomes.json').read_text())
    eps = json.loads((u / 'data/report/planning-episodes.json').read_text())
    tasks = {t['taskId']: t for t in data['tasks']}
    idx = {(e['model'], e['taskId']): e for e in eps}
    assert len(tasks) == 157 and len(idx) == len(eps) == 1570
    overlay = overlay_records(a.humanclaw_results) if a.humanclaw_results else {}
    verified = set()
    ledger = []
    historical = None

    def check(model, task, outcome, record, blob, replacement=None):
        key = (model, task)
        t, e = tasks[task], idx[key]
        report = json.loads(blob)
        report = report.get('runner_report', report)
        assert e['recordId'] == t['recordIds'][model] == record
        assert sha(blob) == e['reportSha256']
        assert outcome == e['outcome'] == t['outcomes'][model]
        assert (outcome == 'success') == (model in t['successModels'])
        if replacement:
            assert report['outcome'] == replacement['rawOutcome'] == e['rawOutcome']
            assert e['replacementRecordId'] == replacement['replacementRecordId']
            assert e['nativeTaskId'] == replacement['nativeTaskId'] and e['seed'] == replacement['seed']
            assert e['providerInterruption'] == replacement['providerInterruption']
            assert t['rawOutcomes'][model] == replacement['rawOutcome']
            assert t['replacementReportSha256'][model] == sha(blob)
            ledger.append(dict(model=model, taskId=task, recordId=record,
                               **{k: v for k, v in replacement.items() if k != 'blob'},
                               supersededReportSha256=e['supersededReportSha256']))
        else:
            assert report['outcome'] == outcome
        assert key not in verified
        verified.add(key)

    if full:
        with zipfile.ZipFile(a.logs) as z:
            prefix = next(n[:-len('trials.csv')] for n in z.namelist() if n.endswith('/trials.csv') and not n.startswith('__MACOSX/'))
            rows = csv.DictReader(io.StringIO(z.read(prefix + 'trials.csv').decode('utf-8-sig')))
            for r in rows:
                if r['dataset'] != 'main' or r['wave'] != 'W3':
                    continue
                blob = gzip.decompress(z.read(prefix + r['report_object']))
                assert sha(blob) == r['source_observed_sha256']
                key = (r['model'], r['task_id'])
                replacement = overlay.get(key)
                if replacement:
                    assert idx[key]['supersededReportSha256'] == sha(blob)
                    assert json.loads(blob).get('runner_report', json.loads(blob))['outcome'] == r['outcome']
                    check(*key, replacement['outcome'], r['record_id'], replacement['blob'], replacement)
                else:
                    check(*key, r['outcome'], r['record_id'], blob)
        with zipfile.ZipFile(a.argon_logs) as z:
            prefix = next(n[:-9] for n in z.namelist() if n.endswith('/slots.csv') and not n.startswith('__MACOSX/'))
            rows = csv.DictReader(io.StringIO(z.read(prefix + 'slots.csv').decode('utf-8-sig')))
            for r in rows:
                if r['scope'] != 'w3':
                    continue
                assert r['adopted'] == 'True'
                t = tasks[r['task_id']]
                blob = gzip.decompress(z.read(prefix + r['report_object']))
                assert sha(blob) == t['argonReportSha256']
                check('gemini4argon', r['task_id'], r['outcome'], t['recordIds']['gemini4argon'], blob)
        archives = [a.logs.name, a.argon_logs.name]
        fresh, retained = 1570, 0
        verification = 'Every currently adopted raw report was freshly verified against its SHA-256, task, record ID and final interpreted outcome; replacement raw labels are retained in the ledger.'
    else:
        with zipfile.ZipFile(a.baseline_public_data) as z:
            prefix = 'public/data/report/'
            old_bytes = z.read(prefix + 'planning-episodes.json')
            old_eps = json.loads(old_bytes)
            old_tasks = {t['taskId']: t for t in json.loads(z.read(prefix + 'planning-outcomes.json'))['tasks']}
            historic_bytes = z.read(prefix + 'planning-comparison-audit.json')
            historic = json.loads(historic_bytes)
        baseline = {(e['model'], e['taskId']): e for e in old_eps}
        assert len(baseline) == len(old_eps) == historic['verifiedEpisodes'] == 1570
        assert set(baseline) == set(idx) and set(old_tasks) == set(tasks)
        retained_keys = sorted(set(idx) - set(overlay))
        for key in retained_keys:
            assert idx[key] == baseline[key], 'A retained episode changed without raw-source verification: ' + str(key)
            m, task = key
            assert tasks[task]['outcomes'][m] == old_tasks[task]['outcomes'][m] == idx[key]['outcome']
            assert tasks[task]['recordIds'][m] == old_tasks[task]['recordIds'][m] == idx[key]['recordId']
            assert (idx[key]['outcome'] == 'success') == (m in tasks[task]['successModels'])
            verified.add(key)
        for key, replacement in overlay.items():
            old, e = baseline[key], idx[key]
            assert old['benchmark'] == e['benchmark'] == 'HumanCLAW'
            assert old['reportSha256'] == e['supersededReportSha256']
            assert old['recordId'] == e['recordId']
            check(*key, replacement['outcome'], e['recordId'], replacement['blob'], replacement)
        fresh, retained = len(overlay), len(retained_keys)
        assert (fresh, retained) == (39, 1531)
        archives = historic['sourceArchives']
        historical = dict(baselineArchive=a.baseline_public_data.name,
                          baselineArchiveSha256=sha(a.baseline_public_data.read_bytes()),
                          baselineEpisodesSha256=sha(old_bytes), priorAuditSha256=sha(historic_bytes),
                          priorAudit=historic, retainedEpisodes=retained,
                          retainedRowsCanonicalSha256=sha(canonical_blob([baseline[k] for k in retained_keys])),
                          currentRetainedRowsCanonicalSha256=sha(canonical_blob([idx[k] for k in retained_keys])),
                          verification='All 1,531 retained episode objects and their task outcomes/record IDs match the pre-refresh backup. Their original raw-report verification is inherited from the embedded prior audit, not rerun.')
        verification = 'Fresh raw-report verification for 39 HumanCLAW replacements; 1,531 unchanged retained rows inherit the explicitly preserved historical two-archive verification. The original raw archives were not re-read in this audit.'
    assert verified == set(idx)
    assert len(ledger) == len(overlay)
    result = dict(scope='All 157 Planning tasks, one adopted episode per model; not all 1,000 benchmark cases.',
                  models=data['models'], verifiedEpisodes=len(verified), freshVerifiedEpisodes=fresh,
                  historicalVerifiedRetainedEpisodes=retained,
                  sourceArchives=archives, verification=verification,
                  pair=['astra', 'opus55'], counts=counts(data['tasks']),
                  bySource={s: counts([t for t in tasks.values() if t['benchmark'] == s]) for s in sorted({t['benchmark'] for t in tasks.values()})})
    if historical:
        result['historicalVerification'] = historical
    if overlay:
        result['humanclawRefresh'] = dict(package=a.humanclaw_results.name, records=len(ledger),
                                        selection='Latest per-case attempts, not best outcomes.',
                                        providerInterruptions=sum(r['providerInterruption'] for r in ledger),
                                        rawOutcomesRetained=True, reports=sorted(ledger, key=lambda r: (r['model'], r['taskId'])))
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(counts=result['counts'], freshVerifiedEpisodes=fresh, historicalRetainedEpisodes=retained)))


if __name__ == '__main__':
    main()
