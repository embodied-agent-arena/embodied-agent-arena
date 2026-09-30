"""Portable entry point around the existing benchmark runners and scorers."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from .paths import resolve_project_root


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def confined(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f'Data path escapes the dataset: {relative}')
    return path


def dataset_root(value: str | None) -> Path:
    if value or os.environ.get('EMBODIED_ARENA_DATA_ROOT'):
        path = Path(value or os.environ['EMBODIED_ARENA_DATA_ROOT']).expanduser().resolve()
    else:
        root = resolve_project_root()
        path = root / 'data' if (root / 'data/cases.jsonl').exists() else root.parent / 'data'
    if not (path / 'cases.jsonl').is_file():
        raise ValueError(f'Dataset not found at {path}; pass --data-root or set EMBODIED_ARENA_DATA_ROOT')
    return path


def read_cases(root: Path) -> list[dict]:
    rows = [json.loads(line) for line in (root / 'cases.jsonl').read_text().splitlines() if line.strip()]
    ids = [row['task_id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate task IDs in dataset')
    return rows


def select_cases(cases: list[dict], args) -> list[dict]:
    filters = {'wave': args.wave, 'benchmark_id': args.benchmark, 'task_id': args.case}
    selected = cases
    for key, values in filters.items():
        if not values:
            continue
        wanted = {item for value in values for item in value.split(',')}
        unknown = wanted - {row[key] for row in cases}
        if unknown:
            raise ValueError(f'Unknown {key}: {sorted(unknown)}')
        selected = [row for row in selected if row[key] in wanted]
    if not selected:
        raise ValueError('No cases match the selection')
    return selected


def option(args: list[str], flag: str) -> str | None:
    result = None
    for index, arg in enumerate(args):
        if arg == flag and index + 1 < len(args):
            result = args[index + 1]
        elif arg.startswith(flag + '='):
            result = arg.split('=', 1)[1]
    return result


def override_w4_request(task: dict, *, max_tokens=None, timeout=None):
    """Apply an explicit request override without changing other phase budgets."""
    if task.get('wave') != 'W4':
        return
    for flag, value in [('--max-tokens', max_tokens), ('--request-timeout-seconds', timeout)]:
        if value is None:
            continue
        args = iter(task['runner_args'])
        kept = []
        for arg in args:
            if arg == flag:
                next(args)
            elif not arg.startswith(flag + '='):
                kept.append(arg)
        task['runner_args'] = kept + [flag, str(value)]


def expand_paths(value: Any, bindings: dict[str, str]) -> Any:
    if isinstance(value, str):
        for key, path in bindings.items():
            value = value.replace('${' + key + '}', path)
        if '${' in value:
            raise ValueError(f'Unresolved path variable: {value}')
        return value
    if isinstance(value, list):
        return [expand_paths(item, bindings) for item in value]
    if isinstance(value, dict):
        return {key: expand_paths(item, bindings) for key, item in value.items()}
    return value


def path_bindings(root: Path, data: Path) -> dict[str, str]:
    return {'PROJECT_ROOT': str(root), 'DATA_ROOT': str(data),
            'EXTERNAL_ROOT': str(Path(os.environ.get('EMBODIED_ARENA_EXTERNAL_ROOT', root / 'external')).resolve()),
            'CACHE_ROOT': str(root / '.cache')}


def check_dataset(data: Path, *, hashes: bool = False) -> dict:
    manifest = json.loads((data / 'manifest.json').read_text())
    cases = read_cases(data)
    errors = []
    if len(cases) != manifest['case_count']:
        errors.append('Case count differs from manifest')
    if dict(Counter(t['wave'] for t in cases)) != manifest['waves']:
        errors.append('Wave counts differ from manifest')
    if dict(Counter(t['benchmark_id'] for t in cases)) != manifest['benchmarks']:
        errors.append('Benchmark counts differ from manifest')
    if file_sha256(data / 'cases.jsonl') != manifest['cases_sha256']:
        errors.append('cases.jsonl checksum mismatch')
    catalog_ids = {}
    checked_files = 0
    for group in manifest['asset_groups']:
        base = confined(data, group['path'])
        receipt_path = base / 'download_receipt.json'
        if not receipt_path.is_file():
            errors.append(f'Missing receipt: {group["id"]}'); continue
        if file_sha256(receipt_path) != group['receipt_sha256']:
            errors.append(f'Receipt checksum mismatch: {group["id"]}')
        receipt = json.loads(receipt_path.read_text())
        for filename, digest in receipt['files'].items():
            path = confined(base, filename)
            checked_files += 1
            if not path.is_file():
                errors.append(f'Missing asset: {group["path"]}/{filename}')
            elif hashes and file_sha256(path) != digest:
                errors.append(f'Asset checksum mismatch: {group["path"]}/{filename}')
        if (base / 'catalog.jsonl').is_file():
            rows = [json.loads(line) for line in (base/'catalog.jsonl').read_text().splitlines() if line.strip()]
            catalog_ids[str(base)] = {r['sample_id'] for r in rows}
    for name, digest in manifest['episode_files'].items():
        path = confined(data, name)
        checked_files += 1
        if not path.is_file() or file_sha256(path) != digest:
            errors.append(f'Episode file missing or changed: {name}')
    groups = {str(confined(data,g['path'])):g for g in manifest['asset_groups']}
    for task in cases:
        args = expand_paths(task['runner_args'], path_bindings(resolve_project_root(), data))
        base = option(args, '--data-root')
        if base:
            group = groups.get(base)
            if group is None:
                errors.append(f'Unknown data root: {task["task_id"]}')
            elif option(args, '--data-receipt-sha256') != group['receipt_sha256']:
                errors.append(f'Task receipt differs: {task["task_id"]}')
            if base in catalog_ids and option(args, '--sample-id') not in catalog_ids[base]:
                errors.append(f'Missing sample: {task["task_id"]}')
        if task['wave'] == 'W2' and option(args, '--perception') != 'none':
            errors.append(f'Default perception must be none: {task["task_id"]}')
    return {'ok': not errors, 'cases': len(cases), 'waves': manifest['waves'],
            'benchmark_adapters': len(manifest['benchmarks']), 'checked_files': checked_files,
            'asset_hashes_checked': hashes, 'errors': errors}


def environment(root: Path, data: Path) -> dict[str, str]:
    """Only public runtime configuration belongs in the generated manifest."""
    pukun = root / 'runtimes/pukun'
    return {'EMBODIED_ARENA_ROOT': str(root),
            'EMBODIED_ARENA_DATA_ROOT': str(data),
            'EMBODIED_ARENA_EXTERNAL_ROOT': path_bindings(root, data)['EXTERNAL_ROOT'],
            'PUKUN_ROOT': str(pukun),
            'AGENTIC_EMBODIED_ARENA_PUKUN_ROOT': str(pukun),
            'EMBODIED_ARENA_TRACESPATIAL_ROOT': str(root / 'runtimes/tracespatial'),
            'PYTHONPATH': str(root/'src') + os.pathsep + str(pukun/'scripts'),
            'ARENA_CASE_STUDY_HZ': '0'}


def run_cases(root: Path, data: Path, cases: list[dict], forwarded: list[str]) -> int:
    from .campaign import main as campaign_main
    if forwarded[:1] == ['--']:
        forwarded = forwarded[1:]
    forbidden = {'--manifest', '--suite'}
    if any(arg.split('=', 1)[0] in forbidden for arg in forwarded):
        raise ValueError('Use --wave/--benchmark/--case to select the packaged tasks')
    api = option(forwarded, '--provider') == 'openai-compatible'
    if api:
        cases = copy.deepcopy(cases)
        for task in cases:
            override_w4_request(task, max_tokens=option(forwarded, '--max-output-tokens'),
                                timeout=option(forwarded, '--request-timeout'))
    if api and any(t['wave'] == 'W4' for t in cases):
        for flag, value in [('--max-output-tokens', '3600'), ('--request-timeout', '720')]:
            if not any(a.split('=', 1)[0] == flag for a in forwarded):
                forwarded.extend([flag, value])
    manifest = make_manifest(root, data, cases)
    # Source data remains immutable. Each invocation has a separate temporary manifest.
    with tempfile.TemporaryDirectory(prefix='arena-manifest-') as tmp:
        path = Path(tmp) / 'tasks.json'
        path.write_text(json.dumps(manifest))
        return campaign_main(root, ['--manifest',str(path),'--no-reuse',*forwarded])


def make_manifest(root: Path, data: Path, cases: list[dict]) -> dict:
    fields = {'benchmark_id','case_id','task_id','split','variation','seed','runner_args','env','budgets','resources','enabled'}
    selected = [{k:v for k,v in case.items() if k in fields} for case in cases]
    manifest = {'schema_version':'agentic-embodied-arena/full-evaluation/v1',
                'tasks':expand_paths(selected, path_bindings(root, data)),
                'runtime':{'environment':environment(root,data)},
                'harness':{'max_workers':1,'gpus':[],'gpu_memory_gb':24,'gpu_memory_reserve_gb':2}}
    return manifest


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite_json(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [finite_json(v) for v in value]
    return value


def score_predictions(data: Path, cases: list[dict], predictions: Path) -> list[dict]:
    predicted = {}
    known = {case['task_id'] for case in cases}
    for line in predictions.read_text().splitlines():
        if not line.strip(): continue
        row = json.loads(line); task_id = row['task_id']
        if task_id not in known:
            raise ValueError(f'Prediction outside selected tasks: {task_id}')
        if task_id in predicted:
            raise ValueError(f'Duplicate prediction: {task_id}')
        predicted[task_id] = row['answer']
    from . import w1_scoring, w5_scoring
    catalog_cache = {}
    adapters = {}
    records = []
    try:
        for task in cases:
            if task['wave'] not in {'W1','W2','W5'}:
                raise ValueError('W3/W4 must be scored by their live native environment')
            args = expand_paths(task['runner_args'], path_bindings(resolve_project_root(), data))
            base = Path(option(args, '--data-root'))
            sample_id = option(args, '--sample-id'); benchmark = option(args,'--benchmark')
            if task['task_id'] not in predicted:
                metrics = {'submission_valid':False,'passed':False,'reason':'missing_prediction'}
            else:
                answer = predicted[task['task_id']]
                if task['wave'] == 'W2':
                    from w2_harness.offline_arena.adapters import create_adapter
                    if benchmark not in adapters:
                        adapters[benchmark] = create_adapter(benchmark, data_root=base)
                    adapter = adapters[benchmark]
                    sample = adapter.load_sample(sample_id)
                    metrics = adapter.evaluate_private(sample, adapter.parse_submission(sample, answer))
                else:
                    if base not in catalog_cache:
                        catalog_cache[base] = {row['sample_id']: row for row in
                            (json.loads(line) for line in (base/'catalog.jsonl').read_text().splitlines() if line.strip())}
                    case = catalog_cache[base][sample_id]
                    if benchmark.startswith('tracespatial_'):
                        from .w1_tracespatial import score_trajectory
                        metrics = score_trajectory(answer, case, base)
                    elif task['wave'] == 'W1':
                        metrics = w1_scoring.score_case(w1_scoring.parse_submission(answer),case)
                    else:
                        metrics = w5_scoring.score_case(w5_scoring.parse_submission(answer),case,base)
            records.append({'task_id':task['task_id'],'benchmark_id':task['benchmark_id'],
                            'wave':task['wave'],'metrics':finite_json(metrics)})
    finally:
        for adapter in adapters.values(): adapter.close()
    return records


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['batch']:
        from .batch import main as batch_main
        return batch_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command',required=True)
    subs.add_parser('batch', help='Configurable multi-model supervision; see arena batch --help')
    for name in ['list','validate','run','score']:
        p = subs.add_parser(name)
        p.add_argument('--data-root')
        if name != 'validate':
            p.add_argument('--wave',action='append',default=[])
            p.add_argument('--benchmark',action='append',default=[])
            p.add_argument('--case',action='append',default=[])
        if name == 'validate': p.add_argument('--hashes',action='store_true')
        if name == 'score':
            p.add_argument('--predictions',type=Path,required=True)
            p.add_argument('--output',type=Path,required=True)
    args, forwarded = parser.parse_known_args(argv)
    if forwarded and args.command != 'run': parser.error('unrecognized arguments: '+' '.join(forwarded))
    try:
        root = resolve_project_root(); data = dataset_root(args.data_root)
        if args.command == 'validate':
            result=check_dataset(data,hashes=args.hashes)
            print(json.dumps(result,indent=2));return 0 if result['ok'] else 1
        cases=select_cases(read_cases(data),args)
        if args.command == 'list':
            print(json.dumps({'case_count':len(cases),'waves':dict(Counter(t['wave'] for t in cases)),
                              'benchmarks':dict(sorted(Counter(t['benchmark_id'] for t in cases).items()))},indent=2))
            return 0
        if args.command == 'run': return run_cases(root,data,cases,forwarded)
        if not (args.wave or args.benchmark or args.case):
            cases=[t for t in cases if t['wave'] in {'W1','W2','W5'}]
        rows=score_predictions(data,cases,args.predictions)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows))
        print(json.dumps({'scored_cases':len(rows),'output':str(args.output)}));return 0
    except (ValueError,OSError,KeyError) as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr);return 2


if __name__ == '__main__':
    raise SystemExit(main())
