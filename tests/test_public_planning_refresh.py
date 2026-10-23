"""The HumanCLAW replacement must preserve identity and honest audit attribution."""
import collections
import json
from pathlib import Path

PUBLIC = Path(__file__).resolve().parents[1] / 'public'


def read(name):
    return json.loads((PUBLIC / 'data' / name).read_text())


def test_replacement_hash_ledger_and_stop_semantics():
    audit = read('report/planning-comparison-audit.json')
    episodes = {(r['model'], r['taskId']): r for r in read('report/planning-episodes.json')}
    tasks = {t['taskId']: t for t in read('report/planning-outcomes.json')['tasks']}
    rows = audit['humanclawRefresh']['reports']
    assert len(rows) == len({(r['model'], r['taskId']) for r in rows}) == 39
    assert collections.Counter(r['model'] for r in rows) == {'qwen': 13, 'qwen35-397b': 13, 'qwen35-27b': 13}
    assert sum(r['providerInterruption'] for r in rows) == 2
    assert sum(r['rawOutcome'] == 'agent_failure' and r['outcome'] == 'task_failure' for r in rows) == 3
    for r in rows:
        e, t = episodes[r['model'], r['taskId']], tasks[r['taskId']]
        assert r['recordId'] == e['recordId'] == t['recordIds'][r['model']]
        assert r['reportSha256'] == e['reportSha256'] == t['replacementReportSha256'][r['model']]
        assert r['supersededReportSha256'] == e['supersededReportSha256']
        assert r['rawOutcome'] == e['rawOutcome'] == t['rawOutcomes'][r['model']]
        assert r['outcome'] == e['outcome'] == t['outcomes'][r['model']]
        assert r['seed'] == e['seed'] == 42
        assert r['nativeTaskId'] == e['nativeTaskId']
        assert len(r['reportSha256']) == 64 and r['reportSha256'] != r['supersededReportSha256']
        assert (r['outcome'] == 'provider_interruption') == r['providerInterruption']


def test_retained_history_is_not_claimed_as_fresh_verification():
    audit = read('report/planning-comparison-audit.json')
    assert audit['verifiedEpisodes'] == 1570
    assert audit['freshVerifiedEpisodes'] == 39
    assert audit['historicalVerifiedRetainedEpisodes'] == 1531
    h = audit['historicalVerification']
    assert h['retainedEpisodes'] == 1531
    assert h['retainedRowsCanonicalSha256'] == h['currentRetainedRowsCanonicalSha256']
    assert h['priorAudit']['verifiedEpisodes'] == 1570
    assert h['priorAudit']['sourceArchives'] == audit['sourceArchives']
    assert 'not re-read' in audit['verification']


def test_planning_counts_stay_consistent_across_public_surfaces():
    episodes = read('report/planning-episodes.json')
    outcome = read('report/planning-outcomes.json')
    metrics = read('report/metrics.json')
    charts = read('report/domain-charts.json')['planning']
    study = read('study.json')
    assert len(episodes) == 1570
    assert len({(r['model'], r['taskId']) for r in episodes}) == 1570
    aliases = {'Astra':'astra', 'Sol':'sol', 'Fable':'fable', 'Qwen-Max':'qwen',
               'Gemini 3.8':'gemini', 'Qwen-397B':'qwen35-397b', 'Qwen-27B':'qwen35-27b',
               'Opus 5.5':'opus55', 'DeepSeek-1017':'deepseek', 'Gemini 4 Argon':'gemini4argon'}
    for m in metrics['planning']['models']:
        rows = [r for r in episodes if r['model'] == m['model']]
        count = sum(r['outcome'] == 'success' for r in rows)
        assert len(rows) == m['n'] == 157
        assert count == m['completed']
        assert count == sum(m['model'] in t['successModels'] for t in outcome['tasks'])
        chart = next(c for c in charts['models'] if c['model'] == m['model'])
        assert count == chart['count']
        assert abs(chart['value'] - count / 157 * 100) < 1e-10
        profile = next(s for s in study['models'] if aliases[s['alias']] == m['model'])
        assert float(profile['scores'][6]) == round(count / 157 * 100, 1)
    histogram = collections.Counter(len(t['successModels']) for t in outcome['tasks'])
    assert charts['histogram'] == [histogram[i] for i in range(11)]
    assert histogram[0] == outcome['allFail'] == 15
    assert histogram[10] == outcome['allSucceed'] == 8
