"""Configurable multi-model supervision around the existing campaign runner.

Each model/task/round is an independent trial. Attempts are retained; a valid
wrong answer is terminal. Launch intents, exclusive job locks, and Linux process
identities let a restarted supervisor adopt work without launching it twice.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

from . import release
from .campaign import ending_http_codes, try_lock
from .full_evaluation import TERMINAL_STATUSES
from .paths import resolve_project_root


ACTIVE = {'starting', 'running'}
RETRY_KINDS = {'network', 'timeout', 'invalid', 'runtime'}
QUEUE_RESERVE_GB = 0.001
MODEL_FIELDS = {'name', 'model', 'provider', 'base_url', 'env_file', 'workers', 'rounds',
                'max_output_tokens', 'request_timeout', 'temperature', 'reasoning_effort',
                'codex_executable', 'cursor_executable'}
CONFIG_FIELDS = {'data_root', 'selection', 'models', 'defaults', 'max_workers',
                 'workers_per_model', 'rounds', 'batch_size', 'max_attempts', 'retry_on',
                 'backoff_seconds', 'max_backoff_seconds', 'poll_seconds', 'resources',
                 'budgets', 'benchmark_limits', 'port_base'}


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def positive(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise ValueError(f'{name} must be a {"nonnegative" if zero else "positive"} integer')
    return value


def duration(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f'{name} must be finite and {"nonnegative" if zero else "positive"}')
    return value


def only_fields(value, fields, name):
    if not isinstance(value, dict) or set(value) - fields:
        raise ValueError(f'Invalid {name} fields; supported: {", ".join(sorted(fields))}')


def resolve_file(value, base):
    p = Path(value).expanduser()
    return str((base / p).resolve() if not p.is_absolute() else p.resolve())


def code_identity(root):
    """Bind executable code and task configuration, excluding caches and outputs."""
    files = {}
    for folder in ('src', 'scripts', 'runtimes/pukun', 'runtimes/tracespatial/upstream', 'configs', 'benchmarks'):
        for p in (root / folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py', '.json', '.yaml', '.yml', '.md'}:
                files[str(p.relative_to(root))] = release.file_sha256(p)
    return digest(files)


def make_plan(config_path, root, *, data_override=None):
    path = Path(config_path).resolve()
    config = read(path)
    only_fields(config, CONFIG_FIELDS, 'batch configuration')
    defaults = config.get('defaults', {})
    only_fields(defaults, MODEL_FIELDS - {'name', 'model', 'workers', 'rounds'}, 'model defaults')
    data_value = str(Path(data_override).resolve()) if data_override else config.get('data_root')
    data = release.dataset_root(resolve_file(data_value, path.parent) if data_value else None)
    check = release.check_dataset(data, hashes=True)
    if not check['ok']:
        raise ValueError('Dataset validation failed: ' + '; '.join(check['errors']))
    selection = config.get('selection', {})
    only_fields(selection, {'waves', 'benchmarks', 'cases'}, 'selection')
    for k, values in selection.items():
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ValueError(f'selection.{k} must be a list of strings')
    tasks = release.select_cases(release.read_cases(data), SimpleNamespace(
        wave=selection.get('waves', []), benchmark=selection.get('benchmarks', []), case=selection.get('cases', [])))
    budgets = config.get('budgets', {})
    only_fields(budgets, {'max_agent_iterations', 'max_total_tokens', 'timeout_seconds', 'llm_num_retries'}, 'budget overrides')
    for name, value in budgets.items():
        positive(value, 'budgets.' + name, zero=name == 'llm_num_retries')
    tasks = copy.deepcopy(tasks)
    for task in tasks:
        task['budgets'].update(budgets)
        if task['wave'] == 'W3' and task['budgets'].get('llm_num_retries', 0):
            raise ValueError('W3 preserves its no-request-retry loop; use whole-case network retries')
        if task['wave'] == 'W4' and 'max_agent_iterations' in budgets:
            args = task['runner_args']
            if '--max-verifier-calls' in args:
                index = args.index('--max-verifier-calls') + 1
                args[index] = str(max(int(args[index]), budgets['max_agent_iterations'] + 1))
    total = positive(config.get('max_workers', 8), 'max_workers')
    per_model = positive(config.get('workers_per_model', total), 'workers_per_model')
    rounds = positive(config.get('rounds', 1), 'rounds')
    resources = config.get('resources', {})
    only_fields(resources, {'gpus', 'gpu_memory_gb', 'gpu_reserve_gb'}, 'resources')
    devices = resources.get('gpus', [])
    if not isinstance(devices, list) or any(isinstance(d, bool) or not isinstance(d, (str, int)) or not str(d) or ',' in str(d) for d in devices):
        raise ValueError('resources.gpus must be a list of device IDs')
    devices = [str(d) for d in devices]
    if len(set(devices)) != len(devices):
        raise ValueError('Duplicate GPU IDs')
    memory = duration(resources.get('gpu_memory_gb', 24), 'gpu_memory_gb')
    reserve = duration(resources.get('gpu_reserve_gb', 2), 'gpu_reserve_gb', zero=True)
    if reserve >= memory:
        raise ValueError('GPU reserve must be smaller than GPU capacity')
    for task in tasks:
        requested = task.get('resources', {}).get('gpu_memory_gb', 0)
        if requested and (not devices or requested > memory - max(reserve, QUEUE_RESERVE_GB)):
            raise ValueError(f'No configured GPU can admit {task["task_id"]}: requires {requested} GiB')
    rows = config.get('models')
    if not isinstance(rows, list) or not rows:
        raise ValueError('models must contain at least one model')
    models = {}
    for row in rows:
        only_fields(row, MODEL_FIELDS, 'model')
        model = {**defaults, **row}
        name = model.get('name', '')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name) or name in models:
            raise ValueError('Model names must be unique filesystem-safe aliases')
        if not isinstance(model.get('model'), str) or not model['model'].strip():
            raise ValueError('Each model needs its exact model ID')
        model['provider'] = model.get('provider', 'openai-compatible')
        if model['provider'] not in {'openai-compatible', 'codex-exec', 'cursor-exec'}:
            raise ValueError('Unsupported model provider')
        if model['provider'] != 'openai-compatible':
            if any(model.get(k) is not None for k in ('base_url', 'env_file', 'max_output_tokens', 'request_timeout')) or model.get('temperature', 0) != 0:
                raise ValueError('CLI profiles use their saved login and native phase settings; omit API settings')
        if model['provider'] != 'codex-exec' and any(model.get(k) for k in ('codex_executable', 'reasoning_effort')):
            raise ValueError('Codex executable and reasoning_effort require codex-exec')
        if model['provider'] != 'cursor-exec' and model.get('cursor_executable'):
            raise ValueError('Cursor executable requires cursor-exec')
        model['workers'] = positive(model.get('workers', per_model), name + '.workers')
        model['rounds'] = positive(model.get('rounds', rounds), name + '.rounds')
        if model.get('env_file'):
            model['env_file'] = resolve_file(model['env_file'], path.parent)
        if model.get('base_url'):
            url = urlsplit(model['base_url'])
            if url.scheme not in {'http', 'https'} or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError('base_url must be a credential-free HTTP(S) endpoint')
        for key in ('max_output_tokens', 'request_timeout'):
            if key in model:
                duration(model[key], key) if key == 'request_timeout' else positive(model[key], key)
        if 'temperature' in model:
            duration(model['temperature'], 'temperature', zero=True)
            if model['temperature'] > 2:
                raise ValueError('temperature must be between 0 and 2')
        models[name] = model
    retry_on = config.get('retry_on', ['network'])
    if not isinstance(retry_on, list) or any(x not in RETRY_KINDS for x in retry_on):
        raise ValueError('retry_on supports network, timeout, invalid, runtime')
    limits = config.get('benchmark_limits', {'vlabench': 1})
    if not isinstance(limits, dict):
        raise ValueError('benchmark_limits must be a mapping')
    known = {t['benchmark_id'] for t in tasks} | {group(t) for t in tasks}
    for name, value in limits.items():
        positive(value, 'benchmark_limits.' + name)
        if name not in known and name != 'vlabench':
            raise ValueError('Unknown benchmark limit: ' + name)
    plan = dict(schema_version=1, project_root=str(root), data_root=str(data),
                dataset_manifest_sha256=release.file_sha256(data/'manifest.json'),
                code_sha256=code_identity(root), models=models, tasks=tasks,
                max_workers=total, batch_size=positive(config.get('batch_size', 1), 'batch_size'),
                max_attempts=positive(config.get('max_attempts', 3), 'max_attempts'),
                retry_on=retry_on, backoff_seconds=duration(config.get('backoff_seconds', 30), 'backoff_seconds', zero=True),
                max_backoff_seconds=duration(config.get('max_backoff_seconds', 900), 'max_backoff_seconds', zero=True),
                poll_seconds=duration(config.get('poll_seconds', 5), 'poll_seconds'),
                resources=dict(gpus=devices, gpu_memory_gb=memory, gpu_reserve_gb=reserve),
                benchmark_limits=limits, port_base=positive(config.get('port_base', 20000), 'port_base'))
    if plan['port_base'] < 1024 or plan['port_base'] + total * 3 > 65535:
        raise ValueError('port_base must leave three usable service ports per worker')
    plan['plan_sha256'] = digest(plan)
    return plan


def group(task):
    return task.get('env', {}).get('ARENA_REPORTING_BENCHMARK', task['benchmark_id'])


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (OSError, IndexError):
        return None


def alive(record):
    return bool(record and record.get('start_ticks') and process_identity(record['pid']) == record['start_ticks'])


def live_job_processes(folder):
    needle = str(Path(folder).resolve()).encode()
    found = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        pid = int(path.parent.name)
        if pid == os.getpid():
            continue
        try:
            if any(needle in arg for arg in path.read_bytes().split(b'\0')):
                stamp = process_identity(pid)
                if stamp:
                    found.append(dict(pid=pid, start_ticks=stamp))
        except OSError:
            continue
    return found


def classify(status, report):
    native = report.get('native_result') or {}
    if native.get('submission_valid') is True:
        return 'completed'
    codes = ending_http_codes(status, report)
    if set(codes) & {401, 402, 403}:
        return 'credentials'
    if set(codes) & {400, 404, 422}:
        return 'configuration'
    if any(c in {408, 409, 425, 429} or c >= 500 for c in codes):
        return 'network'
    # Inspect terminal errors only, not old recovered errors or model reasoning.
    exception = report.get('exception') or status.get('exception') or {}
    episode = report.get('episode_loop') or (report.get('event_log_paths') or {}).get('episode_loop') or {}
    turns = episode.get('turns') or []
    last = turns[-1] if turns else {}
    message = str(exception.get('message', '')) + ' ' + str(last.get('error') or last.get('author_error') or report.get('error') or '')
    if 'Model API transport failure' in message or (exception.get('category') == 'model_api' and exception.get('retryable')):
        return 'network'
    outcome = status.get('outcome') or report.get('outcome')
    if outcome == 'timeout':
        return 'timeout'
    if outcome in {'runtime_failure', 'verifier_failure'} or not report:
        return 'runtime'
    if native.get('submission_valid') is False:
        return 'invalid'
    return 'completed' if outcome in TERMINAL_STATUSES else 'runtime'


class Supervisor:
    def __init__(self, output, plan, *, resume=False, retry_paused=False, command_factory=None):
        self.out, self.plan = Path(output), plan
        self.root = Path(plan['project_root'])
        self.tasks = {t['task_id']: t for t in plan['tasks']}
        self.command_factory = command_factory
        self.children = {}
        self.gpu_lease = None
        self.stop = False
        path = self.out/'state.json'
        if path.exists():
            if not resume:
                raise ValueError('Existing state requires --resume')
            self.state = read(path)
            if self.state['plan_sha256'] != plan['plan_sha256']:
                raise ValueError('Saved batch configuration differs; use a new output directory')
        else:
            if resume:
                raise ValueError('No saved state to resume')
            slots = {}
            for alias, model in plan['models'].items():
                for rnd in range(1, model['rounds'] + 1):
                    for task in plan['tasks']:
                        key = f'{alias}|{rnd}|{task["task_id"]}'
                        slots[key] = dict(key=key, alias=alias, round=rnd, task_id=task['task_id'],
                                          attempts=0, state='pending', ready_at=0, history=[])
            self.state = dict(plan_sha256=plan['plan_sha256'], slots=slots, jobs=[], paused_models={}, phase='running')
        if retry_paused:
            self.state['paused_models'] = {}
            for slot in self.state['slots'].values():
                if slot['state'] == 'paused' and slot['attempts'] < plan['max_attempts']:
                    slot.update(state='pending', ready_at=0)

    def save(self):
        self.state['updated_at'] = time.time()
        atomic(self.out/'state.json', self.state)
        active = [j for j in self.state['jobs'] if j['state'] in ACTIVE]
        allocated = sum(len(j['keys']) for j in active)
        assert allocated <= self.plan['max_workers']
        summary = {}
        for alias, model in self.plan['models'].items():
            slots = [s for s in self.state['slots'].values() if s['alias'] == alias]
            counts = Counter(s['state'] for s in slots)
            used = sum(len(j['keys']) for j in active if j['alias'] == alias)
            assert used <= model['workers']
            summary[alias] = dict(target=len(slots), states=dict(counts), allocated_workers=used,
                                  attempts=sum(s['attempts'] for s in slots), paused_reason=self.state['paused_models'].get(alias))
        atomic(self.out/'status.json', dict(phase=self.state['phase'], updated_at=self.state['updated_at'],
               allocated_workers=allocated, max_workers=self.plan['max_workers'], models=summary))

    def obtain_gpu_lease(self):
        if self.gpu_lease is not None:
            return True
        path = self.root/'.cache/gptge-batch.lock'
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open('a')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return False
        self.gpu_lease = handle
        return True

    def allocation(self, task):
        active = [j for j in self.state['jobs'] if j['state'] in ACTIVE]
        limits = self.plan['benchmark_limits']
        for name in {group(task), task['benchmark_id']} & set(limits):
            count = sum(sum(name in {group(self.tasks[self.state['slots'][k]['task_id']]), self.tasks[self.state['slots'][k]['task_id']]['benchmark_id']}
                            for k in j['keys']) for j in active)
            if count >= limits[name]:
                return False
        memory = float(task.get('resources', {}).get('gpu_memory_gb', 0))
        if not memory:
            return dict(gpu=None, memory=0, exclusive=False)
        exclusive = task.get('resources', {}).get('exclusive_gpu', False)
        capacity = self.plan['resources']['gpu_memory_gb'] - max(self.plan['resources']['gpu_reserve_gb'], QUEUE_RESERVE_GB)
        for gpu in self.plan['resources']['gpus']:
            occupants = [j for j in active if j['allocation']['gpu'] == gpu]
            if occupants and (exclusive or any(j['allocation'].get('exclusive') for j in occupants)):
                continue
            used = sum(j['allocation']['memory'] for j in occupants)
            if used + memory <= capacity and self.obtain_gpu_lease():
                return dict(gpu=gpu, memory=memory, exclusive=exclusive)
        return False

    def service_ports(self):
        active = {j.get('port_base') for j in self.state['jobs'] if j['state'] in ACTIVE}
        for base in range(self.plan['port_base'], self.plan['port_base'] + 3*self.plan['max_workers'], 3):
            if base in active:
                continue
            sockets = []
            try:
                for port in range(base, base+3):
                    sock = socket.socket(); sockets.append(sock); sock.bind(('127.0.0.1', port))
                return base
            except OSError:
                continue
            finally:
                for sock in sockets:
                    sock.close()
        return None

    def command(self, job):
        if self.command_factory is not None:
            return self.command_factory(job)
        model = self.plan['models'][job['alias']]
        folder = self.out/'jobs'/job['id']
        command = [sys.executable, str(self.root/'scripts/run_benchmarks.py'), '--manifest', str(folder/'manifest.json'),
                   '--provider', model['provider'], '--model', model['model'], '--workers', str(len(job['keys'])),
                   '--batch-size', str(len(job['keys'])), '--no-reuse', '--output-dir', str(folder/'campaign')]
        fields = {'env_file':'--env-file', 'base_url':'--base-url', 'max_output_tokens':'--max-output-tokens',
                  'request_timeout':'--request-timeout', 'temperature':'--temperature', 'reasoning_effort':'--reasoning-effort',
                  'codex_executable':'--codex-executable', 'cursor_executable':'--cursor-executable'}
        settings = dict(model)
        is_w4 = self.tasks[self.state['slots'][job['keys'][0]]['task_id']]['wave'] == 'W4'
        if model['provider'] == 'openai-compatible':
            settings.setdefault('max_output_tokens', 3600 if is_w4 else 1800)
            settings.setdefault('request_timeout', 720 if is_w4 else 360)
        for key, flag in fields.items():
            if settings.get(key) is not None:
                command.extend([flag, str(settings[key])])
        allocation = job['allocation']
        command.extend(['--gpus', allocation['gpu'] or ''])
        if allocation['gpu'] is not None:
            resources = self.plan['resources']
            # The child requires an explicit reservation for a separate queue;
            # the parent holds the shared queue lease and accounts for all jobs.
            command.extend(['--gpu-memory-gb', str(resources['gpu_memory_gb']),
                            '--gpu-reserve-gb', str(max(0, resources['gpu_reserve_gb'] - QUEUE_RESERVE_GB)),
                            '--gpu-queue-lock', str(folder/'gpu.lock'), '--external-gpu-reserve-gb', str(QUEUE_RESERVE_GB)])
        return command

    def spawn(self, job):
        if job['allocation']['gpu'] is not None and not self.obtain_gpu_lease():
            return  # A recovered launch intent waits for other GPU jobs to drain.
        folder = self.out/'jobs'/job['id']
        atomic(folder/'command.json', self.command(job))
        env = dict(os.environ)
        env['PYTHONPATH'] = str(self.root/'src')
        inherited = (self.gpu_lease.fileno(),) if job['allocation']['gpu'] is not None and self.gpu_lease is not None else ()
        if inherited:
            env['ARENA_BATCH_GPU_LEASE_FD'] = str(inherited[0])
        else:
            env.pop('ARENA_BATCH_GPU_LEASE_FD', None)
        with (folder/'runner.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-m', 'embodied_harness.batch', '--job-wrapper', str(folder)],
                        cwd=self.root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True, pass_fds=inherited)
        self.children[job['id']] = child
        job.update(state='running', started=time.time())
        self.save()

    def finish_job(self, job):
        folder = self.out/'jobs'/job['id']
        reports = {}
        for path in sorted((folder/'campaign').glob('batches/*/statuses/*.json')):
            try:
                status = read(path)
                if status.get('outcome') not in TERMINAL_STATUSES:
                    continue
                report_path = Path(status.get('artifacts', {}).get('runner_report', ''))
                reports[status['task_id']] = (status, read(report_path) if report_path.is_file() else {}, path, report_path)
            except (OSError, ValueError, KeyError):
                continue
        job.update(state='finished', ended=time.time())
        for key in job['keys']:
            slot = self.state['slots'][key]
            status, report, path, report_path = reports.get(slot['task_id'], ({}, {}, None, None))
            kind = classify(status, report)
            record = dict(job=job['id'], kind=kind, outcome=status.get('outcome') or report.get('outcome'),
                          status_file=str(path) if path else None, runner_report=str(report_path) if report_path else None)
            slot['history'].append(record)
            slot['result'] = record
            if kind == 'completed':
                slot['state'] = 'completed'
            elif kind in {'credentials', 'configuration'}:
                self.state['paused_models'][slot['alias']] = kind
                slot['state'] = 'paused' if slot['attempts'] < self.plan['max_attempts'] else 'failed'
            elif kind in self.plan['retry_on'] and slot['attempts'] < self.plan['max_attempts']:
                slot['state'] = 'pending'
                slot['ready_at'] = time.time() + min(self.plan['max_backoff_seconds'], self.plan['backoff_seconds'] * 2**min(slot['attempts']-1, 20))
            else:
                slot['state'] = 'failed'
        self.save()

    def refresh(self):
        for job in self.state['jobs']:
            if job['state'] not in ACTIVE:
                continue
            folder = self.out/'jobs'/job['id']
            child = self.children.get(job['id'])
            if child is not None:
                child.poll()
            records = [read(folder/n) for n in ('launch.json', 'campaign-process.json') if (folder/n).exists()]
            if any(alive(r) for r in records) or (child is not None and child.poll() is None):
                continue
            if live_job_processes(folder):
                continue
            if records or (folder/'exit.json').exists():
                self.finish_job(job)
            else:
                self.spawn(job)  # Durable intent, same job lock: no second paid attempt.

    def schedule(self):
        # Least occupied runnable model first; unused global capacity is available
        # to every model up to its configured ceiling, rather than a fixed share.
        while not self.stop:
            active = [j for j in self.state['jobs'] if j['state'] in ACTIVE]
            used = sum(len(j['keys']) for j in active)
            if used >= self.plan['max_workers']:
                return
            by_model = Counter()
            for j in active:
                by_model[j['alias']] += len(j['keys'])
            scheduled = False
            for alias in sorted(self.plan['models'], key=lambda a: by_model[a]):
                model = self.plan['models'][alias]
                if alias in self.state['paused_models'] or by_model[alias] >= model['workers']:
                    continue
                ready = [s for s in self.state['slots'].values() if s['alias'] == alias and s['state'] == 'pending' and s['ready_at'] <= time.time()]
                ready.sort(key=lambda s: (s['attempts'], s['round'], s['task_id']))
                for slot in ready:
                    task = self.tasks[slot['task_id']]
                    allocation = self.allocation(task)
                    if allocation is False:
                        continue
                    ports = self.service_ports() if task['benchmark_id'] == 'capx' else None
                    if task['benchmark_id'] == 'capx' and ports is None:
                        continue
                    keys = [slot['key']]
                    # CPU-only homogeneous chunks can use the existing in-process
                    # campaign scheduler. GPU jobs retain per-case reservations.
                    ceiling = min(self.plan['batch_size'], self.plan['max_workers']-used, model['workers']-by_model[alias])
                    if not allocation['memory'] and not ({group(task), task['benchmark_id']} & set(self.plan['benchmark_limits'])):
                        seen = {slot['task_id']}
                        for peer in ready:
                            t = self.tasks[peer['task_id']]
                            if len(keys) >= ceiling:
                                break
                            if peer['task_id'] not in seen and t['wave'] == task['wave'] and group(t) == group(task) and not t.get('resources', {}).get('gpu_memory_gb'):
                                keys.append(peer['key']); seen.add(peer['task_id'])
                    job = dict(id=f'{len(self.state["jobs"]):07d}-{alias}', alias=alias, keys=keys,
                               allocation=allocation, port_base=ports, state='starting')
                    selected = [copy.deepcopy(self.tasks[self.state['slots'][k]['task_id']]) for k in keys]
                    for t in selected:
                        release.override_w4_request(t, max_tokens=model.get('max_output_tokens'),
                                                    timeout=model.get('request_timeout'))
                        if ports is not None:
                            for i, service in enumerate(('SAM3', 'CONTACT_GRASPNET', 'PYROKI')):
                                t.setdefault('env', {})['AGENTIC_EMBODIED_ARENA_CAPX_' + service + '_PORT'] = str(ports+i)
                    manifest = release.make_manifest(self.root, Path(self.plan['data_root']), selected)
                    atomic(self.out/'jobs'/job['id']/'manifest.json', manifest)
                    for key in keys:
                        self.state['slots'][key]['state'] = 'active'
                        self.state['slots'][key]['attempts'] += 1
                    self.state['jobs'].append(job)
                    self.save()  # Persist intent before launching a process.
                    self.spawn(job)
                    scheduled = True
                    break
                if scheduled:
                    break
            if not scheduled:
                return

    def run(self):
        self.state['phase'] = 'running'
        self.save()
        try:
            while not self.stop:
                self.refresh()
                self.schedule()
                active = any(j['state'] in ACTIVE for j in self.state['jobs'])
                pending = any(s['state'] == 'pending' and s['alias'] not in self.state['paused_models'] for s in self.state['slots'].values())
                if not active and not pending:
                    complete = all(s['state'] == 'completed' for s in self.state['slots'].values())
                    self.state['phase'] = 'complete' if complete else 'needs_attention'
                    self.save()
                    return 0 if complete else 2
                time.sleep(self.plan['poll_seconds'])
            for job in self.state['jobs']:
                if job['state'] in ACTIVE:
                    p = self.out/'jobs'/job['id']/'launch.json'
                    if p.exists() and alive(read(p)):
                        os.kill(read(p)['pid'], signal.SIGTERM)
            self.state['phase'] = 'interrupted'
            self.save()
            return 130
        finally:
            if self.gpu_lease is not None:
                self.gpu_lease.close()


def job_wrapper(folder):
    folder = Path(folder)
    with (folder/'job.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        if (folder/'exit.json').exists():
            return 0
        # Handles a wrapper killed between spawning and recording the child PID.
        while live_job_processes(folder):
            time.sleep(1)
            if (folder/'exit.json').exists():
                return 0
        atomic(folder/'launch.json', dict(pid=os.getpid(), start_ticks=process_identity(os.getpid())))
        lease_fd = os.environ.get('ARENA_BATCH_GPU_LEASE_FD')
        # Preserve the cooperative reservation even if the supervisor and its
        # wrapper die while the actual campaign is still running.
        child = subprocess.Popen(read(folder/'command.json'),
                                 pass_fds=(int(lease_fd),) if lease_fd else ())
        atomic(folder/'campaign-process.json', dict(pid=child.pid, start_ticks=process_identity(child.pid)))
        def stop(signum, frame):
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        rc = child.wait()
        atomic(folder/'exit.json', dict(exit_code=rc, ended=time.time()))
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--data-root')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--execute', action='store_true', help='Start evaluations; default is plan-only')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-paused', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--job-wrapper', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.job_wrapper:
        return job_wrapper(args.job_wrapper)
    if not args.output_dir:
        parser.error('--output-dir is required')
    out = args.output_dir.resolve()
    try:
        if args.status:
            print(json.dumps(read(out/'status.json'), indent=2)); return 0
        if args.retry_paused and not args.resume:
            raise ValueError('--retry-paused requires --resume')
        root = resolve_project_root()
        if args.resume:
            plan = read(out/'plan.json')
            stored_digest = plan['plan_sha256']
            if digest({k:v for k,v in plan.items() if k != 'plan_sha256'}) != stored_digest:
                raise ValueError('Saved plan was modified')
            if code_identity(root) != plan['code_sha256'] or release.file_sha256(Path(plan['data_root'])/'manifest.json') != plan['dataset_manifest_sha256']:
                raise ValueError('Code or dataset changed since this batch started; use a new output directory')
            if not release.check_dataset(Path(plan['data_root']), hashes=True)['ok']:
                raise ValueError('Dataset content changed since this batch started')
            if args.config and make_plan(args.config, root, data_override=args.data_root) != plan:
                raise ValueError('Resume configuration differs from the saved plan')
            if args.data_root and str(Path(args.data_root).resolve()) != plan['data_root']:
                raise ValueError('Resume dataset path differs')
        else:
            if not args.config:
                raise ValueError('--config is required for a new batch')
            plan = make_plan(args.config, root, data_override=args.data_root)
        targets = {a:len(plan['tasks'])*m['rounds'] for a,m in plan['models'].items()}
        if not args.execute:
            print(json.dumps(dict(execute=False, tasks=len(plan['tasks']), targets=targets,
                  trials=sum(targets.values()), max_workers=plan['max_workers'],
                  model_worker_limits={a:m['workers'] for a,m in plan['models'].items()}, output=str(out)), indent=2))
            return 0
        out.mkdir(parents=True, exist_ok=True)
        with try_lock(out/'supervisor.lock'):
            if (out/'plan.json').exists() and read(out/'plan.json') != plan:
                raise ValueError('Output contains a different plan')
            supervisor = Supervisor(out, plan, resume=args.resume, retry_paused=args.retry_paused)
            atomic(out/'plan.json', plan)
            def stop(signum, frame):
                supervisor.stop = True
            old = {sig:signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
            try:
                return supervisor.run()
            finally:
                for sig, handler in old.items():
                    signal.signal(sig, handler)
    except (OSError, ValueError, KeyError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
