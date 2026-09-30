"""Admission checks for suites that preserve a baseline and add probed cases."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .full_evaluation import expand_manifest


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_digest(task: dict) -> str:
    return hashlib.sha256(json.dumps(task, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate_probe(receipt: dict, task: dict) -> None:
    key = task['task_id']
    if receipt.get('task_id') != key or receipt.get('benchmark') != task['benchmark_id']:
        raise ValueError('probe identity mismatch: ' + key)
    if receipt.get('passed') is not True or not receipt.get('finished_at') or receipt.get('stage') != 'complete':
        raise ValueError('native probe not complete: ' + key)
    if receipt.get('model_calls') != 0:
        raise ValueError('expected a no-model probe: ' + key)
    if receipt.get('schema') == 'candidate-native-probe/v1':
        required = ['content_seal', 'binding', 'reset', 'observe', 'action', 'observe_after', 'verify_called']
        if not all(receipt.get('stages', {}).get(k) is True for k in required):
            raise ValueError('incomplete native probe: ' + key)
        if not receipt.get('action', {}).get('ok') or not isinstance(receipt.get('verification'), dict):
            raise ValueError('missing action or verifier result: ' + key)
        if receipt['verification'].get('metadata', {}).get('verifier_available') is False:
            raise ValueError('verifier unavailable: ' + key)
    elif receipt.get('schema') == 'candidate-pukun-probe/v1':
        required = ['process_exit_zero', 'task_binding', 'completed_one', 'action_step', 'no_code_exception', 'no_timeout']
        if not all(receipt.get('checks', {}).get(k) is True for k in required):
            raise ValueError('incomplete Pukun probe: ' + key)
        args = task['runner_args']
        expected = args[args.index('--expected-task-id') + 1]
        rows = receipt.get('report', {}).get('run_summary', {}).get('tasks', [])
        if len(rows) != 1 or rows[0].get('task_id') != expected:
            raise ValueError('Pukun executed a different task: ' + key)
    else:
        raise ValueError('unknown probe schema: ' + key)


def validate_verified_suite(document: dict) -> None:
    readiness = document.get('readiness', {})
    if readiness.get('schema') != 'native-probe-admission/v1':
        raise ValueError('verified suite requires native-probe admission metadata')
    baseline = readiness['baseline']
    path = Path(baseline['path'])
    if file_digest(path) != baseline['sha256']:
        raise ValueError('verified suite baseline changed')
    old = {t.task_id: t.to_dict() for t in expand_manifest(json.loads(path.read_text()))}
    tasks = [t.to_dict() for t in expand_manifest(document)]
    current = {t['task_id']: t for t in tasks}
    if len(current) != len(tasks):
        raise ValueError('duplicate task in verified suite')
    if any(current.get(key) != task for key, task in old.items()):
        raise ValueError('verified suite changed or removed a baseline case')
    references = readiness['supplement_receipts']
    if set(references) != set(current) - set(old):
        raise ValueError('every supplement must have exactly one probe receipt')
    for key, reference in references.items():
        path = Path(reference['path'])
        if file_digest(path) != reference['sha256']:
            raise ValueError('native probe receipt changed: ' + key)
        if task_digest(current[key]) != reference['task_sha256']:
            raise ValueError('probed task configuration changed: ' + key)
        validate_probe(json.loads(path.read_text()), current[key])
