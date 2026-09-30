"""Conservative, read-only reuse of completed case results across campaigns."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


def identity(manifest, task):
    profile = dict(manifest['model_profile'])
    # Installation location is not a model setting. Runtime/budget remain bound.
    profile.pop('codex_executable', None)
    profile.pop('cursor_executable', None)
    case = {k: v for k, v in task.items() if k not in {'task_id', 'resources', 'enabled'}}
    context = {'profile': profile, 'runtime': manifest.get('runtime', {}),
               'execution': manifest.get('execution', {}), 'case': case}
    return hashlib.sha256(json.dumps(context, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def discover(root, output, extra=()):
    paths = list((root / 'reports/campaigns').glob('*/checkpoint.json'))
    historical = root / 'artifacts/campaign-history-20260914/history.json'
    if historical.is_file():
        paths.append(historical)
    for path in extra:
        path = Path(path).resolve()
        if path.is_dir():
            path /= 'checkpoint.json'
        if not path.is_file():
            raise ValueError('Reuse source does not exist: ' + str(path))
        paths.append(path)
    return sorted({p.resolve() for p in paths if p.parent.resolve() != output.resolve()})


def reusable_results(manifest, paths):
    # Import lazily to avoid the campaign/status parser dependency cycle.
    from .campaign import status_result
    wanted = {identity(manifest, t): t['task_id'] for t in manifest['tasks']}
    candidates = {}
    diagnostics = []
    for path in paths:
        try:
            data = json.loads(path.read_text())
            if data.get('schema') == 'campaign-history/v1':
                records = data['records']
            else:
                source = json.loads((path.parent / 'effective-manifest.json').read_text())
                tasks = {t['task_id']: t for t in source['tasks']}
                latest = dict(data.get('latest', {}))
                batch = data.get('active_batch')
                if batch:
                    for status in (path.parent / batch['directory'] / 'statuses').glob('*.json'):
                        r = status_result(status)
                        if r:
                            latest[r['task_id']] = r
                records = [{'identity': identity(source, tasks[k]), 'status_file': r['status_file']}
                           for k, r in latest.items() if k in tasks]
            for record in records:
                key = wanted.get(record['identity'])
                if key is None:
                    continue
                status_path = Path(record['status_file'])
                raw = status_path.read_bytes()
                if record.get('status_sha256') and hashlib.sha256(raw).hexdigest() != record['status_sha256']:
                    diagnostics.append({'source': str(status_path), 'reason': 'status_changed'})
                    continue
                status = json.loads(raw)
                result = status_result(status_path)
                if not result or not result['report_available']:
                    continue
                if record.get('report_sha256') and hashlib.sha256(Path(result['runner_report']).read_bytes()).hexdigest() != record['report_sha256']:
                    diagnostics.append({'source': str(status_path), 'reason': 'report_changed'})
                    continue
                # Latest attempt, never best outcome. Do not silently use a new
                # success over a later error from another recovery campaign.
                order = (status.get('finished_at') or '', status_path.stat().st_mtime_ns, str(status_path))
                if key not in candidates or order > candidates[key][0]:
                    result = copy.deepcopy(result)
                    result.update(task_id=key, reused=True, source_task_id=result['task_id'], source_index=str(path))
                    candidates[key] = (order, result)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            diagnostics.append({'source': str(path), 'reason': type(exc).__name__})
    return {k: v[1] for k, v in candidates.items()}, diagnostics
