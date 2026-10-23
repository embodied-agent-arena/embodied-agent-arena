#!/usr/bin/env python3
"""Refresh the three technical-report tables from already adopted, audited data.

This renderer does not import analysis data or rewrite the report's narrative,
project homepage, or claim ledger. --analysis must describe the current canonical
10-model evaluation; stale or partial analyses are rejected before any write.
The existing report is authoritative for prose. A checked-in template is used
only when public/technical-report.html does not yet exist.
"""
from pathlib import Path
import argparse
from collections import Counter
import html
import json
import math
import re


MODEL_COUNT = 10
TASK_COUNT = 157
TABLE_CAPTIONS = {
    'sources': 'Completed Planning tasks, by source',
    'resources': 'Planning episode time, token allocation and recovery',
    'profiles': 'Region representation and downstream task completion',
}
TABLE_PATTERN = re.compile(
    r'(<div class="report-table-scroll">\s*<table\b[^>]*>\s*'
    r'<caption>(?P<caption>.*?)</caption>.*?<tbody>)(?P<body>.*?)'
    r'(</tbody>\s*</table>\s*</div>)', re.S,
)


class AuditError(ValueError):
    """An input differs from the current, reviewed evaluation."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def read_json(path):
    return json.loads(path.read_text())


def keyed(rows, fields, label):
    result = {tuple(row[field] for field in fields): row for row in rows}
    require(len(result) == len(rows), f'Duplicate {label} identities')
    return result


def checked_inputs(repo, analysis):
    """Validate all inputs before rendering or touching a destination file."""
    public = repo / 'public'
    current = read_json(public / 'data/report/metrics.json')
    episodes = read_json(public / 'data/report/planning-episodes.json')
    coverage = read_json(public / 'data/report/planning-outcomes.json')
    study = read_json(public / 'data/study.json')
    supplied = read_json(analysis / 'report-metrics.json')
    supplied_episodes = read_json(analysis / 'planning-episodes.json')

    models = current['models']
    require(len(models) == len(set(models)) == MODEL_COUNT,
            'Canonical metrics must contain all ten models')
    require(supplied.get('models') == models,
            'Analysis model population differs from current canonical data')
    current_rows = keyed(episodes, ('model', 'taskId'), 'canonical episode')
    supplied_rows = keyed(supplied_episodes, ('model', 'taskId'), 'analysis episode')
    require(len(current_rows) == MODEL_COUNT * TASK_COUNT,
            'Canonical Planning data must contain 1,570 assignments')
    require(Counter(r['model'] for r in episodes) == {m: TASK_COUNT for m in models},
            'Canonical Planning data must contain 157 assignments per model')
    require(current_rows.keys() == supplied_rows.keys(),
            'Analysis episode population differs from current canonical data')
    for key, current_row in current_rows.items():
        require(current_row == supplied_rows[key],
                f'Stale analysis episode {key}: use the adopted report, identity, '
                'outcome, usage and duration, including current replacement hashes')
    # Do not silently replace old profile values with study.json. That previously
    # concealed stale analyses while restoring old Planning aggregates and prose.
    require(supplied.get('profiles') == current['profiles'],
            'Analysis comparison profiles differ from current canonical data')
    require(supplied.get('planning') == current['planning'],
            'Analysis Planning aggregates differ from current canonical data')

    plan = current['planning']
    require(plan['nTasks'] == TASK_COUNT and plan['nEpisodes'] == len(episodes),
            'Canonical Planning denominators are inconsistent')
    by_model = keyed(plan['models'], ('model',), 'Planning model')
    profiles = keyed(current['profiles'], ('model',), 'comparison profile')
    require(set(by_model) == set(profiles) == {(m,) for m in models},
            'Canonical aggregates and profiles must include all ten models')
    by_name = keyed(study['models'], ('name',), 'study model')
    require(len(by_name) == MODEL_COUNT, 'Study data must include all ten models')
    for profile in current['profiles']:
        require((profile['name'],) in by_name, 'Unknown model in comparison profiles')
        displayed = by_name[profile['name'],]
        require(profile['scores'] == displayed['scores'] and
                profile['runs'] == displayed['metric_runs'],
                f'Canonical profile disagrees with study.json: {profile["name"]}')
        means = [float(value.split(' ± ')[0]) for value in profile['scores']]
        require(profile['means'] == means,
                f'Canonical means disagree with displayed scores: {profile["name"]}')

    tasks = keyed(coverage['tasks'], ('taskId',), 'outcome task')
    source_rows = keyed(coverage['sourceResults'], ('model',), 'source summary')
    require(len(tasks) == TASK_COUNT and set(source_rows) == set(by_model),
            'Outcome matrix must include 157 tasks and ten source summaries')
    task_ids = {key[0] for key in tasks}
    for model in models:
        rows = [r for r in episodes if r['model'] == model]
        require({r['taskId'] for r in rows} == task_ids,
                f'Outcome matrix and episode task identities differ for {model}')
        aggregate = by_model[model,]
        completed = sum(r['outcome'] == 'success' for r in rows)
        require(aggregate['n'] == TASK_COUNT and aggregate['completed'] == completed,
                f'Canonical completion count does not reproduce episodes: {model}')
        require(float(profiles[model,]['scores'][6]) == round(100 * completed / TASK_COUNT, 1),
                f'Main-table Planning score does not reproduce episodes: {model}')
        known = [r for r in rows if r['tokens'] is not None]
        failed_tokens = sum(r['tokens'] for r in known if r['outcome'] != 'success')
        total_tokens = sum(r['tokens'] for r in known)
        require(aggregate['tokens']['n'] == len(known) and math.isclose(
            aggregate['unsuccessfulTokenShare'], failed_tokens / total_tokens,
            rel_tol=1e-12), f'Canonical token coverage or failure share is inconsistent: {model}')
        require(source_rows[model,]['completed'] == completed,
                f'Outcome source total is inconsistent: {model}')
        for source in source_rows[model,]['sourceResults']:
            subset = [r for r in rows if r['benchmark'] == source['benchmark']]
            require(source['n'] == len(subset) and source['completed'] == sum(
                r['outcome'] == 'success' for r in subset),
                f'Outcome source summary is inconsistent: {model}/{source["benchmark"]}')
        for row in rows:
            task = tasks[row['taskId'],]
            require((model in task['successModels']) == (row['outcome'] == 'success'),
                    f'Outcome matrix disagrees with episode success: {model}/{row["taskId"]}')
            require(task['recordIds'][model] == row['recordId'],
                    f'Outcome matrix record identity is stale: {model}/{row["taskId"]}')
    return current, coverage


def table_rows(metrics, coverage):
    """Only measured rows are generated; approved captions and prose stay put."""
    escape = html.escape
    profiles = {p['model']: p for p in metrics['profiles']}
    source_results = {r['model']: r for r in coverage['sourceResults']}
    sources, resources, comparison = [], [], []
    for model in metrics['models']:
        summary = source_results[model]
        cells = ''.join(f'<td>{s["completed"]}/{s["n"]}</td>' for s in summary['sourceResults'])
        sources.append(f'<tr><th scope="row">{escape(profiles[model]["name"])}</th>'
                       f'{cells}<td><strong>{summary["completed"]}/{TASK_COUNT}</strong></td></tr>')
    for model in metrics['planning']['models']:
        resources.append(
            f'<tr><th scope="row">{escape(model["name"])}</th>'
            f'<td>{model["completed"]}/{TASK_COUNT}</td>'
            f'<td>{model["time"]["median"]:.1f}</td>'
            f'<td>{model["time"]["q25"]:.1f}–{model["time"]["q75"]:.1f}</td>'
            f'<td>{model["unsuccessfulTokenShare"] * 100:.1f}%</td>'
            f'<td>{model["tokens"]["n"]}/{TASK_COUNT}</td>'
            f'<td>{model["recoveredAfterError"]}/{model["episodesWithExecutionErrors"]}</td></tr>')
    for profile in metrics['profiles']:
        value = profile['means']
        comparison.append(
            f'<tr><th scope="row">{escape(profile["name"])}</th>'
            f'<td>{value[4]:.3f}</td><td>{value[5]:.3f}</td>'
            f'<td>{value[5]-value[4]:+.3f}</td><td>{value[6]:.1f}</td>'
            f'<td>{value[7]:.1f}</td></tr>')
    return {'sources': ''.join(sources), 'resources': ''.join(resources),
            'profiles': ''.join(comparison)}


def refresh_tables(page, rows, coverage):
    found = Counter()

    def replace(match):
        caption = html.unescape(re.sub('<[^>]+>', '', match['caption']))
        keys = [key for key, prefix in TABLE_CAPTIONS.items() if caption.startswith(prefix)]
        require(len(keys) == 1, f'Unexpected report table: {caption}')
        key = keys[0]
        found[key] += 1
        if key == 'sources':
            # A changed source order must never put valid counts under wrong headers.
            header = re.search(r'<thead>(.*?)</thead>', match.group(1), re.S)
            require(header is not None, 'Source table is missing its column headings')
            headings = [html.unescape(re.sub('<[^>]+>', '', text)).strip()
                        for text in re.findall(r'<th[^>]*>(.*?)</th>', header[1], re.S)]
            expected = ['Model'] + [s['benchmark'] for s in coverage['sourceResults'][0]['sourceResults']] + ['All']
            require(headings == expected, 'Source table headings differ from canonical source order')
            require(all([s['benchmark'] for s in model['sourceResults']] == expected[1:-1]
                        for model in coverage['sourceResults']),
                    'Source-summary order differs across models')
        return match.group(1) + rows[key] + match.group(4)

    result = TABLE_PATTERN.sub(replace, page)
    require(found == {key: 1 for key in TABLE_CAPTIONS},
            'Report must have exactly one source, resource and profile table')
    return result


def build(repo, analysis):
    metrics, coverage = checked_inputs(repo, analysis)
    target = repo / 'public/technical-report.html'
    template = repo / 'scripts/templates/technical-report.html'
    original = target.read_text() if target.exists() else template.read_text()
    rendered = refresh_tables(original, table_rows(metrics, coverage), coverage)
    # All validation and rendering are complete before the sole permitted write.
    if not target.exists() or rendered != original:
        target.write_text(rendered)
    return {'models': MODEL_COUNT, 'planningReports': MODEL_COUNT * TASK_COUNT,
            'tablesRefreshed': 3, 'reportBytes': len(rendered.encode()),
            'narrativePreserved': True, 'canonicalDataModified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True,
                        help='Audited analysis matching the currently adopted canonical records')
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = build(args.repo.resolve(), args.analysis.resolve())
    except (AuditError, FileNotFoundError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.exit(2, f'Report not changed: {error}\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
