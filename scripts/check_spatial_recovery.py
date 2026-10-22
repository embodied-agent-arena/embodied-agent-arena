#!/usr/bin/env python3
"""Recompute the published Spatial and CALVIN diagnostics from released records."""
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'public/data/spatial-recovery'


def aggregate(rows):
    ids = {r['task_id'] for r in rows}
    runs = sorted({r['run'] for r in rows}, key=int)
    assert len(rows) == len(ids) * len(runs)
    assert len({(r['task_id'], r['run']) for r in rows}) == len(rows)
    counts = [sum(r['success'] == 'True' for r in rows if r['run'] == run) for run in runs]
    rates = [100 * n / len(ids) for n in counts]
    return dict(cases=len(ids), runs=len(runs), counts=counts, mean=statistics.mean(rates),
                sd=statistics.stdev(rates), run_means=rates)


def main():
    spatial = json.loads((DATA / 'spatial-summary.json').read_text())
    with (DATA / 'spatial-trials.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 10560 and all(r['eligible'] == 'True' for r in rows)
    for panel in spatial['panels'].values():
        for group in panel:
            selected = [r for r in rows if r['source'] == group['source'] and r['category'] == group['category']]
            for model in spatial['models']:
                assert aggregate([r for r in selected if r['model'] == model]) == group['results'][model]
    with (ROOT / 'public/data/model-comparison.csv').open() as f:
        main_rows = list(csv.DictReader(f))
    for source, result in spatial['sources'].items():
        for model in spatial['models']:
            assert aggregate([r for r in rows if r['model'] == model and r['source'] == source]) == result[model]
    main_spatial = [r for r in main_rows if r['domain'] == 'Spatial Reasoning' and r['metric'] == 'Pass@1']
    assert len(main_spatial) == 10
    for model, expected in zip(spatial['models'], main_spatial):
        actual = aggregate([r for r in rows if r['model'] == model])
        assert round(actual['mean'], 1) == float(expected['value'])
        assert round(actual['sd'], 1) == float(expected['run_sd'])
    # The formerly ambiguous label belongs to the original Gemini 3.8 records.
    left = spatial['panels']['a'][0]['results']
    assert math.isclose(left['gemini']['mean'], 96.66666666666667)
    assert left['gemini4argon']['mean'] == 100
    calvin = json.loads((DATA / 'calvin-summary.json').read_text())
    episodes = json.loads((DATA / 'calvin-episodes.json').read_text())
    assert len(episodes) == len({(r['model'], r['task_id']) for r in episodes}) == 100
    task_sets = [{r['task_id'] for r in episodes if r['model'] == m} for m in spatial['models']]
    assert len(task_sets[0]) == 10 and all(ids == task_sets[0] for ids in task_sets)
    for e in episodes:
        errors = [t['round'] for t in e['events'] if t['execution_error']]
        assert errors == e['error_rounds'] and len(e['sha256']) == 64
        first = min(errors, default=math.inf)
        assert e['resumed'] == any(t['round'] > first and t['execution_ok'] is True for t in e['events'])
        assert e['recovered'] == bool(e['success'] and any(t['round'] > first and t['native_success'] for t in e['events']))
        assert e['credited_prefix'] == (e['prefix'] or 0)
        assert e['success'] == (e['prefix'] == 5)
    for expected in calvin['models']:
        rr = [e for e in episodes if e['model'] == expected['model']]
        actual = dict(model=expected['model'], n=len(rr), mean_prefix=statistics.mean(e['credited_prefix'] for e in rr),
                      prefix_observed=sum(e['prefix'] is not None for e in rr),
                      reached=[sum(e['credited_prefix'] >= k for e in rr) for k in range(1, 6)],
                      error_episodes=sum(bool(e['error_rounds']) for e in rr), resumed=sum(e['resumed'] for e in rr),
                      completed_after_error=sum(e['recovered'] for e in rr), completed=sum(e['success'] for e in rr))
        assert actual == expected
    assert calvin['total'] == dict(n=100, error_episodes=65, resumed=62, completed_after_error=3, completed=9)
    assert sum(e['prefix'] is None for e in episodes) == 6
    print('Verified 110 spatial subgroup cells, 60 source cells, and 100 CALVIN episodes: 65 → 62 → 3.')


if __name__ == '__main__':
    main()
