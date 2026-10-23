#!/usr/bin/env python3
"""Import audited HumanCLAW replacement results and observation-frame replays.

Run after legacy case builders. Inputs remain private; only selected trace fields,
native outcomes, public media and digest-based provenance are exported. Existing
public record IDs and deep links stay stable. No aggregate benchmark scores change.
"""
import argparse
import collections
import copy
import csv
import hashlib
import html
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / 'public'
DATE = '2026-10-23'
MODELS = {'qwen': 'Qwen 3.8 Max', 'qwen35-397b': 'Qwen 3.5 397B-A17B',
          'qwen35-27b': 'Qwen 3.5 27B'}
NOTE = ('Chronological head-camera observations from this evaluation, displayed at two '
        'observations per second. This is not continuous action footage; playback time '
        'does not represent episode elapsed time. No interpolation or rerendering.')


def read(p):
    return json.loads(p.read_text())


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, separators=(',', ':')) + '\n')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p):
    return list(csv.DictReader(p.open(encoding='utf-8-sig')))


def lines(p):
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def clean(value):
    """Keep recorded execution evidence, excluding host paths and provider payloads."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()
                if not re.search(r'path|file|credential|api.?key|authorization|request.?id|base.?url', k, re.I)}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r'/(?:home|Users|opt|mnt|tmp|data)/[^\s\"\'<>),]+', '[local file]', value)
        value = re.sub(r'https?://[^\s\"\'<>),]+', '[service URL]', value)
        value = re.sub(r'\bsk-[A-Za-z0-9_-]{16,}', '[redacted]', value)
        value = re.sub(r'(?i)Bearer\s+\S+', 'Bearer [redacted]', value)
        return value
    return value


def status(r):
    if r.get('auditStatus') == 'provider_interruption':
        return 'Rerun required'
    if r.get('auditStatus') == 'task_failure_after_agent_stop':
        return 'Not completed'
    return {'success': 'Completed', 'budget_exhausted': 'Budget reached',
            'task_failure': 'Not completed', 'runtime_failure': 'Runtime error',
            'timeout': 'Timed out', 'agent_failure': 'Agent error'}.get(r['outcome'], 'Not completed')


def verify_pack(pack):
    checks = read(pack / 'checksums.json')
    for name, digest in checks.items():
        assert sha(pack / name) == digest, f'Checksum mismatch: {pack.name}/{name}'


def make_record(row, result_pack, video_pack, validation, ffmpeg):
    source_id = row['record_id']
    folder = result_pack / row['record_path']
    report_path = folder / 'runner_report.json'
    assert sha(report_path) == row['source_report_sha256']
    report = read(report_path)
    manifest = read(folder / 'execution_manifest.json')
    task = manifest['tasks'][0]
    native_id = f"{row['scene_id']}:{row['episode_id']}:{row['target']}"
    assert task['task_id'] == row['case_id']
    assert task['seed'] == 42 and task['variation'] == native_id
    assert report['expected_task_id'] == native_id
    assert report['actual_task_ids'] == [native_id]
    assert manifest['model_profile']['model'] == row['api_model'] == report['model']['model']
    assert report['outcome'] == row['raw_outcome']
    native = report['pukun']['run_summary']['native_metrics']
    assert str(native['episode']['scene_id']) == row['scene_id']
    assert str(native['episode']['episode_id']) == row['episode_id']
    assert native['episode']['object_category'] == row['target']
    verified_success = row['success'] == 'True'
    assert verified_success == (report['outcome'] == 'success')
    original = manifest['execution_revision']
    record_id = original['old_record_id']
    rid = 'humanclaw-refresh-' + source_id
    rel = f'assets/humanclaw-qwen-20261023/{source_id}'
    dest = PUB / rel
    dest.mkdir(parents=True, exist_ok=True)
    frame_index = read(folder / 'frame_index.json')
    assert len(frame_index) == int(row['head_observation_frames'])
    video = poster = None
    if row['video_path']:
        media = validation[source_id]
        src = video_pack / row['video_path']
        assert media['path'] == row['video_path'] and media['decode_ok']
        assert media['sha256'] == sha(src)
        assert media['source_observation_frames'] == len(frame_index)
        assert (media['width'], media['height']) == (448, 448)
        assert media['duration_seconds'] == len(frame_index) / 2
        exported_frames = read(video_pack / 'frame_indices' / (source_id + '.json'))
        assert exported_frames == frame_index
        video = rel + '/clip.mp4'
        poster = rel + '/poster.jpg'
        if not (PUB / video).exists() or sha(PUB / video) != media['sha256']:
            shutil.copyfile(src, PUB / video)
        if not (PUB / poster).exists():
            subprocess.run([str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-y',
                            '-i', str(src), '-frames:v', '1', '-q:v', '2', str(PUB / poster)], check=True)
    else:
        assert source_id == 'qwen-07' and not frame_index
        assert int(row['env_steps']) == 0
    identity = {'task': native_id}
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    usage = report['llm_usage']
    loop = report['episode_loop']
    traces = {r['turn']: r for r in lines(folder / 'execution_trace.jsonl')}
    actions = collections.defaultdict(list)
    event_path = folder / 'native_events.jsonl'
    if not event_path.exists():
        assert source_id == 'qwen-07' and int(row['env_steps']) == 0
    for event in lines(event_path) if event_path.exists() else []:
        if event['event'] == 'native_action':
            actions[event['turn']].append(clean({k: v for k, v in event.items() if k not in ['event', 'turn']}))
    provider_codes = sorted({e['code'] for e in read(folder / 'provider_errors.json') if e.get('code')})
    interrupted = row['audit_status'] == 'provider_interruption'
    if interrupted:
        assert source_id in ['qwen-02', 'qwen-08'] and provider_codes == ['data_inspection_failed']
    steps, cumulative = [], 0
    previous = 0
    for turn in loop['turns']:
        number = turn['turn']
        trace = traces.get(number, {})
        calls = actions[number]
        cumulative += len(calls)
        error = clean(turn.get('error'))
        budget_stop = turn.get('stop_reason') == 'budget_exhausted'
        at = turn.get('elapsed_seconds')
        final = number == loop['turns'][-1]['turn']
        accepted = final and verified_success
        feedback = clean({k: v for k, v in turn.items() if k not in ['turn', 'elapsed_seconds', 'error']})
        if final:
            feedback['nativeVerification'] = clean(native['success'])
            feedback['terminalVerifierSuccess'] = verified_success
            if interrupted:
                feedback['providerErrorCode'] = provider_codes[0]
        call_rows = [dict(name=c['primitive'], arguments=c.get('arguments', {}), output=c.get('result', {}),
                          ok=not bool(c.get('result', {}).get('error')),
                          agentVisibleState=c.get('visible_state')) for c in calls]
        source_code = trace.get('cell_file_content') or ''
        display_code = clean(source_code)
        # One delivered comment mixes a Chinese adjective into an English
        # sentence. Translate only that comment; preserve executable statements.
        comment_translated = 'its近' in display_code
        if comment_translated:
            display_code = display_code.replace('its近', 'its near')
        title = ('Task complete' if accepted else 'Provider interruption' if final and interrupted else
                 'Budget reached' if budget_stop else 'Execution error' if error else
                 ' · '.join(dict.fromkeys(c['primitive'].replace('_', ' ') for c in calls)) or 'Observe and decide')
        steps.append(dict(round=number, at=at, duration=at-previous if at is not None else None,
                          title=title, error=error, budgetStop=budget_stop, success=accepted,
                          score=None, goalsMet=None, goalsTotal=None,
                          environmentSteps=trace.get('env_steps'), currentSubgoal=None,
                          toolCalls=len(calls), cumulativeToolCalls=cumulative,
                          code=display_code,
                          sourceCodeSha256=hashlib.sha256(source_code.encode()).hexdigest() if source_code else None,
                          commentTranslatedToEnglish=comment_translated,
                          stdout=clean(trace.get('cell_stdout_content', '')),
                          stderr=clean(trace.get('cell_stderr_content', '')),
                          feedback=feedback, calls=call_rows))
        if at is not None:
            previous = at
    limits = report['model']['runtime_limits']
    budgets = {**clean(task['budgets']), **limits, 'max_rounds': loop['max_turns'],
               'timeout_seconds': loop['timeout_seconds']}
    interaction = native['success']['is_interact_episode']
    target = row['target'].replace('_', ' ')
    instruction = f'Find the {target}, approach until the body touches it, and ' + (f'sit on the {target}.' if interaction else 'stop.')
    note = ('The provider interrupted this attempt after actions had been executed; a replacement evaluation is still required.' if interrupted else
            'The agent stopped, but the native success check was not satisfied.' if row['audit_status'] == 'task_failure_after_agent_stop' else '')
    provenance = dict(date=DATE, sourceRecordId=source_id, recordId=record_id,
                      modelKey=row['model_id'], apiModel=row['api_model'], taskId=row['case_id'],
                      nativeIdentity=identity, seed=42, sourceReportSha256=row['source_report_sha256'],
                      previousReportSha256=original['old_source_sha256'],
                      sourcePackages=[result_pack.name, video_pack.name],
                      video=video, videoSha256=sha(PUB / video) if video else None,
                      poster=poster, posterSha256=sha(PUB / poster) if poster else None,
                      mediaKind='Saved observations' if video else 'Scene only',
                      sourceObservationFrames=len(frame_index),
                      observations=[{k: f[k] for k in ['sha256', 'tick', 'video_frame_order', 'video_timestamp_seconds', 'width', 'height'] if k in f}
                                    for f in frame_index],
                      videoDurationSeconds=validation[source_id]['duration_seconds'] if video else None,
                      explanation=NOTE if video else 'No observation frames were saved before the response budget stopped this attempt.')
    write(dest / 'provenance.json', provenance)
    statistics = dict(elapsedSeconds=float(row['elapsed_seconds']), modelCalls=int(row['model_responses']),
                      rounds=len(steps), tokens=int(row['total_tokens']), promptTokens=int(row['prompt_tokens']),
                      completionTokens=int(row['completion_tokens']), codeErrors=sum(bool(s['error']) and not s['budgetStop'] and s['feedback'].get('stage') == 'execution' for s in steps),
                      toolCalls=cumulative, environmentSteps=int(row['env_steps']), apiSeconds=None,
                      apiMeanSeconds=None, completedSubgoals=None, requiredSubgoals=None, nativeSteps=None,
                      finalScore=None, finalGoalDistanceCm=float(row['final_distance_m']) * 100,
                      finalGoalAngleDegrees=None, actionSubmissions=cumulative)
    moments = sorted({0, len(steps)-1} | {i for i, s in enumerate(steps) if s['error'] or s['success']})
    detail = dict(id=rid, model=MODELS[row['model_id']], modelKey=row['model_id'], benchmark='HumanCLAW',
                  taskId=row['case_id'], instruction=instruction, outcome=report['outcome'], rawOutcome=report['outcome'],
                  auditStatus=row['audit_status'], success=verified_success, protocol='API recovery · native interface',
                  stats=statistics, budgets=budgets, stopLimits=report.get('budget_exhausted', []),
                  verifier=clean(native), callCounts=dict(collections.Counter(c['name'] for s in steps for c in s['calls'])),
                  video=video, animation=None, poster=poster, frames=[],
                  mediaKind='Saved observations' if video else 'Execution log',
                  mediaNote=NOTE if video else provenance['explanation'], videoNote=NOTE if video else provenance['explanation'],
                  evidenceLevel='action_trace', timingNote='Trace timestamps use execution time; reported elapsed time also includes worker setup and cleanup. Observation playback has a separate clock.',
                  callCoverage='Recorded native action events. Observation and helper calls are not enumerated; visible-state descriptions are supplied by the agent.',
                  recordId=record_id, sourceRecordId=source_id, replacementRecordId=source_id,
                  reportSha256=row['source_report_sha256'], sourceReportSha256=row['source_report_sha256'],
                  previousReportSha256=original['old_source_sha256'],
                  mediaReportSha256=row['source_report_sha256'] if video else None,
                  mediaProvenance=rel+'/provenance.json', mediaMatchesReport=bool(video),
                  observationFrameCount=len(frame_index), nativeIdentity=identity, identitySha256=identity_sha,
                  executionStarted=True, seed=42, steps=steps, unassignedCells=[],
                  detailUrl=f'data/execution/{rid}.json', note=note,
                  progress={'value': int(verified_success), 'total': 1, 'label': 'Task completion'},
                  milestones=[dict(index=i, at=steps[i]['at'], round=steps[i]['round'], label=steps[i]['title'],
                                   kind='success' if steps[i]['success'] else 'error' if steps[i]['error'] else 'action', detail='') for i in moments])
    if interrupted:
        detail['providerErrorCode'] = provider_codes[0]
    write(PUB / detail['detailUrl'], detail)
    return detail, provenance


def summary(detail):
    return {k: v for k, v in detail.items() if k not in ['steps', 'unassignedCells', 'verifier', 'callCounts', 'milestones']}


def replace_detail(old, new):
    replacement = copy.deepcopy(new)
    replacement.update(id=old['id'], detailUrl=old.get('detailUrl', new['detailUrl']))
    assert old['recordId'] == new['recordId'], (old['id'], old['recordId'], new['recordId'])
    return replacement


def media_fields(d):
    fields = {k: v for k, v in summary(d).items() if k not in ['stats', 'reportSha256', 'progress', 'stopLimits']}
    fields.update(elapsedSeconds=d['stats']['elapsedSeconds'], modelRounds=d['stats']['rounds'],
                  tokens=d['stats']['tokens'], progress=d['progress'], detailedTraceAvailable=True,
                  demoOnly=False, reportedOutcome=d['outcome'], recorded=True)
    return fields


def build(result_pack, video_pack, ffmpeg):
    verify_pack(result_pack)
    verify_pack(video_pack)
    cases = rows(result_pack / 'case_results.csv')
    videos = {r['record_id']: r for r in rows(video_pack / 'video_index.csv')}
    validation = {r['record_id']: r for r in read(video_pack / 'video_validation.json')}
    assert len(cases) == len(videos) == 39 and len(validation) == 38
    updates, proofs = {}, []
    for row in cases:
        v = videos[row['record_id']]
        for field in ['model_id', 'api_model', 'case_id', 'source_report_sha256', 'video_path', 'head_observation_frames']:
            assert row[field] == v[field], (row['record_id'], field)
        detail, proof = make_record(row, result_pack, video_pack, validation, ffmpeg)
        key = (detail['taskId'], detail['modelKey'])
        assert key not in updates
        updates[key] = detail
        proofs.append(proof)
    # Preserve existing case and execution URLs; replace their evidence atomically.
    for directory in ['data/case-records', 'data/execution']:
        for path in (PUB / directory).glob('*.json'):
            old = read(path)
            if not isinstance(old, dict) or 'steps' not in old:
                continue
            d = updates.get((old.get('taskId'), old.get('modelKey')))
            if d:
                write(path, replace_detail(old, d))
    media_path = PUB / 'data/case-media.json'
    media = read(media_path)
    present = set()
    for rid, run in media['runs'].items():
        key = run.get('taskId'), run.get('modelKey')
        if key in updates:
            d = updates[key]
            old_detail = run.get('detailUrl')
            run.update(media_fields(d), id=rid)
            if old_detail:
                run['detailUrl'] = old_detail
            present.add(key)
    groups = {g['taskId']: g for g in media['groups']}
    for key, d in updates.items():
        if key not in present:
            media['runs'][d['id']] = media_fields(d)
        if key[0] not in groups:
            g = dict(id='humanclaw-refresh-task-'+hashlib.sha256(key[0].encode()).hexdigest()[:10],
                     taskId=key[0], sample=key[0], domain='Task Planning', domainKey='planning',
                     benchmark='HumanCLAW', title=d['instruction'], shortTask=d['instruction'],
                     question=d['instruction'], predictions=[], images=[], galleryMode='same-task', runIds=[])
            media['groups'].append(g)
            groups[key[0]] = g
        for rid, run in media['runs'].items():
            if (run['taskId'], run['modelKey']) == key and rid not in groups[key[0]]['runIds']:
                groups[key[0]]['runIds'].append(rid)
    media['sourceDate'] = DATE
    write(media_path, media)
    comparisons = read(PUB / 'data/case-comparisons.json')
    changed_runs = []
    for rid, r in comparisons['runs'].items():
        d = updates.get((r.get('taskId'), r.get('modelKey')))
        if d:
            keep = {k: r[k] for k in ['id', 'detailUrl', 'taskLabel', 'comparisonInstruction', 'comparisonTaskLabel'] if k in r}
            assert r['recordId'] == d['recordId']
            r.update(media_fields(d), **keep)
            r.pop('executionReview', None)
            r.pop('demoMedia', None)
            changed_runs.append(rid)
    for g in comparisons['groups']:
        g['videoCount'] = sum(bool(comparisons['runs'][rid].get('video') or comparisons['runs'][rid].get('animation')) for rid in g['runIds'])
        if g.get('sceneContext'):
            source = comparisons['runs'][g['sceneContext']['sourceRunId']]
            g['sceneContext'].update(image=source['poster'], model=source['model'])
            g['poster'] = source['poster']
    comparisons['updated'] = DATE
    write(PUB / 'data/case-comparisons.json', comparisons)
    for name in ['index.json', 'complete-index.json']:
        path = PUB / 'data/execution' / name
        index = read(path)
        for story in index['stories']:
            for i, old in enumerate(story['runs']):
                d = updates.get((old.get('taskId'), old.get('modelKey')))
                if d:
                    detail = replace_detail(old, d)
                    write(PUB / detail['detailUrl'], detail)
                    story['runs'][i] = summary(detail)
            group = next((g for g in comparisons['groups'] if g['taskId'] == story['runs'][0]['taskId']), None)
            if group:
                story['sceneContext'] = group['sceneContext']
        index['updated'] = DATE
        index['roundCount'] = sum(r['stats']['rounds'] for s in index['stories'] for r in s['runs'])
        index['toolCallCount'] = sum(r['stats'].get('toolCalls') or 0 for s in index['stories'] for r in s['runs'])
        write(path, index)
    # Refresh evidence-ledger joins. Earlier package gaps remain a historical record.
    path = PUB / 'data/case-comparison-audit.json'
    audit = read(path)
    for g in comparisons['groups']:
        for rid in g['runIds']:
            r = comparisons['runs'][rid]
            if rid in changed_runs:
                audit['comparisons'][g['id']][r['modelKey']].update(
                    recordId=r['recordId'], reportSha256=r['sourceReportSha256'],
                    mediaReportSha256=r['mediaReportSha256'], executionStarted=True,
                    replacementRecordId=r['replacementRecordId'])
    audit['updated'] = DATE
    audit['videoCount'] = sum(bool(r.get('video') or r.get('animation')) for r in comparisons['runs'].values())
    audit['sourceArchives'] = sorted(set(audit.get('sourceArchives', [])) | {result_pack.name, video_pack.name})
    write(path, audit)
    path = PUB / 'data/execution/identity-audit.json'
    audit = read(path)
    index = read(PUB / 'data/execution/index.json')
    for story in index['stories']:
        for r in story['runs']:
            if (r['taskId'], r['modelKey']) in updates:
                audit['comparisons'][story['id']][r['id']].update(
                    reportSha256=r['reportSha256'], replacementRecordId=r['replacementRecordId'],
                    sourceVideoSha256=sha(PUB/r['video']) if r['video'] else None,
                    video=r['video'], poster=r['poster'])
    audit['date'] = DATE
    write(path, audit)
    path = PUB / 'data/video-supplement-audit.json'
    audit = read(path)
    for gap in audit['gaps']:
        d = updates.get((gap['taskId'], gap['modelKey']))
        if d:
            gap['resolvedBy'] = dict(date=DATE, replacementRecordId=d['replacementRecordId'],
                                     reportSha256=d['reportSha256'], video=d['video'],
                                     provenance=d['mediaProvenance'])
    write(path, audit)
    exported = [summary(updates[(r['case_id'], r['model_id'])]) for r in cases]
    audit = dict(date=DATE, recordCount=39, videoCount=38, seed=42,
                 sourcePackages=[result_pack.name, video_pack.name],
                 matching=['model', 'canonical task ID', 'native scene/episode/target', 'seed', 'report SHA-256', 'video SHA-256', 'ordered observation-frame hashes'],
                 updatedAtlasRecords=changed_runs, records=exported,
                 policy='Latest delivered attempts replace earlier records; they are not selected by score. Provider interruptions remain Rerun required. Observation replays are not continuous action recordings.')
    write(PUB / 'data/report/humanclaw-qwen-refresh.json', audit)
    build_index(exported)
    print(f'Imported {len(updates)} records, 38 observation replays, {len(changed_runs)} atlas rows; no aggregate scores changed.')


def build_index(records):
    h = html.escape
    by_task = collections.defaultdict(dict)
    for r in records:
        by_task[r['taskId']][r['modelKey']] = r
    body = []
    for task, rr in by_task.items():
        first = next(iter(rr.values()))
        cells = []
        for model in MODELS:
            r = rr[model]
            if r['video']:
                media = f'<video controls muted playsinline preload="none" poster="{h(r["poster"])}" src="{h(r["video"])}" aria-label="{h(r["model"])} observed frames"></video><span class="kind">Observed frames</span>'
            else:
                context = next(v for v in rr.values() if v['poster'])
                media = f'<img src="{h(context["poster"])}" alt="Same-task scene from {h(context["model"])}; no frames for this run"><span class="kind">Scene only</span>'
            cells.append(f'<td><div class="media">{media}</div><strong>{status(r)}</strong><br><a href="{h(r["detailUrl"])}">Execution trace ↗</a></td>')
        body.append(f'<tr><th scope="row">{h(first["instruction"])}<small>{h(first["nativeIdentity"]["task"])}</small></th>{"".join(cells)}</tr>')
    page = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HumanCLAW Qwen Runs · Embodied Agent Arena</title><style>body{font:16px/1.5 system-ui,sans-serif;color:#203541;margin:32px auto;padding:0 20px;max-width:1250px}h1{font-size:clamp(26px,4vw,40px);margin-bottom:12px}a{color:#426e5b}p{max-width:90ch}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;min-width:760px}th,td{padding:14px;vertical-align:top;border-bottom:1px solid #dce3e0}th{text-align:left;width:25%}thead th{white-space:nowrap}small{display:block;font-size:12px;color:#637279;overflow-wrap:anywhere;margin-top:8px}video,img{width:100%;aspect-ratio:1;object-fit:contain;background:#14262b;display:block;border-radius:8px}.media{position:relative}.kind{position:absolute;top:8px;left:8px;background:#fffdf3;padding:2px 6px;border-radius:4px;font-size:12px;pointer-events:none}td strong{display:inline-block;margin-top:8px}</style></head><body><a href="index.html#cases">← Project page</a><h1>HumanCLAW: Refreshed Qwen Runs</h1><p>Three models on the same 13 tasks, seed 42. The latest delivered attempts replace the earlier records. Two provider-interrupted attempts still require reruns; one budget-limited attempt has no saved frames.</p><p><strong>Observed frames</strong> replays saved head-camera observations at two images per second. They are not continuous action footage, and playback duration is separate from task elapsed time. <strong>Scene only</strong> is a same-task reference image from another model, not a recording of the missing run.</p><p><a href="data/report/humanclaw-qwen-refresh.json">Results and provenance ↗</a></p><div class="scroll"><table><thead><tr><th>Task · Native case ID</th>' + ''.join(f'<th>{h(n)}</th>' for n in MODELS.values()) + '</tr></thead><tbody>' + ''.join(body) + '</tbody></table></div></body></html>'
    (PUB / 'humanclaw-refresh.html').write_text(page)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--videos', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    args = parser.parse_args()
    build(args.results, args.videos, args.ffmpeg)
