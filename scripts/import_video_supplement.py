#!/usr/bin/env python3
"""Attach supplemental media only to its exact model, task and source report."""
import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT/'public'
DEST = 'assets/video-supplement-20261021'
KINDS = {
    'native_video': ('Video excerpt', 'Excerpt from the original recording; playback time is separate from episode time.'),
    'saved_observations': ('Saved observations', 'Original saved observations shown in order at uniform display intervals.'),
    'verified_action_replay': ('Verified action replay', 'Original actions and seed replayed; recorded states and action responses match the verification log.'),
    'saved_state_render': ('Saved-state render', 'Rendered from saved body and object poses, without advancing physics or calling the model.'),
}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pretty = path.name in {'case-media.json', 'case-studies.json', 'release.json'}
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2 if pretty else None, separators=None if pretty else (',', ':'))+'\n')


def key(run):
    return run.get('modelKey'), run.get('taskId'), run.get('sourceReportSha256', run.get('reportSha256'))


def build(archive):
    comparisons = read(PUB/'data/case-comparisons.json')
    by_assignment = {(r['modelKey'], r['taskId']): r for r in comparisons['runs'].values()}
    assert len(by_assignment) == len(comparisons['runs'])
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    audit_path = PUB/'data/video-supplement-audit.json'
    previous = read(audit_path) if audit_path.exists() else {}
    old_entries = {r['runId']:r for r in previous.get('entries', [])} if previous.get('archiveSha256') == digest else {}
    assets, updates, entries, gaps = {}, {}, [], []
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.endswith('/catalog.json') and not n.startswith('__MACOSX/')]
        assert len(names) == 1
        prefix = names[0][:-len('catalog.json')]
        catalog = json.loads(z.read(names[0]))
        assert len(catalog) == 70 and len({(c['model'],c['task_id']) for c in catalog}) == 70
        def payload(name):
            path = PurePosixPath(name)
            assert not path.is_absolute() and '..' not in path.parts
            return z.read(prefix+name)
        for c in catalog:
            run = by_assignment[(c['model'], c['task_id'])]
            assert run['sourceReportSha256'] == c['source_sha256']
            assert run['outcome'] == c['outcome']
            if c.get('record_id') is not None:
                assert c['record_id'] == run['recordId']
            assert c['task_id'].split('--')[0] == c['benchmark']
            if not c.get('video'):
                assert not (run.get('video') or run.get('animation'))
                gaps.append(dict(runId=run['id'], model=run['model'], modelKey=run['modelKey'], benchmark=run['benchmark'],
                                 taskId=run['taskId'], recordId=run['recordId'], reportSha256=run['sourceReportSha256'], reason=c['issue']))
                continue
            assert not c.get('new_demo')
            kind, note = KINDS[c['media_kind']]
            video, poster = payload(c['video']), payload(c['poster'])
            assert hashlib.sha256(video).hexdigest() == c['video_sha256']
            meta = c['media']
            evidence = dict(sourceReportSha256=c['source_sha256'], mediaKind=kind, durationSeconds=c['duration_s'],
                            videoSha256=c['video_sha256'], posterSha256=hashlib.sha256(poster).hexdigest(),
                            note=note, modelKey=c['model'], taskId=c['task_id'])
            for field in ['source_sha256','source_trajectory_sha256','source_duration_s','clip_start_s','clip_end_s',
                          'output_fps','source_fps','frames','physics_steps','vlm_calls','display_ms_per_observation']:
                if field in meta: evidence[field] = meta[field]
            if c['media_kind'] == 'verified_action_replay':
                verification = json.loads(payload(c['verification']))
                assert verification['all_states_equal']
                assert len(verification['checks']) == verification['actions']
                assert all(v['before_equal'] and v['after_equal'] and v['response_equal'] for v in verification['checks'])
                evidence['replayVerification'] = {k:v for k,v in verification.items() if k != 'trace'}
            elif c['media_kind'] == 'saved_state_render':
                assert meta['physics_steps'] == 0 and meta['vlm_calls'] == 0
                assert meta['source_trajectory_sha256']
            elif c['media_kind'] == 'saved_observations':
                index_name = meta.get('deduplicated_source_index')
                if index_name:
                    obs = json.loads(payload(index_name))
                    evidence['observations'] = [{k:v for k,v in item.items() if k not in ['path','original_path','index','feedback']} for item in obs]
            directory = f"{DEST}/{c['benchmark']}/{c['model']}"
            poster_name = 'poster'+PurePosixPath(c['poster']).suffix
            assets[directory+'/clip.mp4'] = video
            assets[directory+'/'+poster_name] = poster
            assets[directory+'/provenance.json'] = (json.dumps(evidence,ensure_ascii=False,separators=(',',':'))+'\n').encode()
            patch = dict(video=directory+'/clip.mp4', animation=None, poster=directory+'/'+poster_name,
                         mediaKind=kind, mediaNote=note, mediaProvenance=directory+'/provenance.json',
                         mediaDurationSeconds=c['duration_s'], mediaReportSha256=c['source_sha256'])
            updates[key(run)] = patch
            old = old_entries.get(run['id'])
            entry = dict(runId=run['id'], model=run['model'], modelKey=run['modelKey'], benchmark=run['benchmark'],
                         taskId=run['taskId'], recordId=run['recordId'], reportSha256=run['sourceReportSha256'],
                         mediaKind=kind, video=patch['video'], poster=patch['poster'], provenance=patch['mediaProvenance'],
                         videoSha256=c['video_sha256'], posterSha256=hashlib.sha256(poster).hexdigest(),
                         filledEmptyCell=old['filledEmptyCell'] if old else not bool(run.get('video') or run.get('animation')),
                         addedScenePreview=old['addedScenePreview'] if old else not bool(run.get('video') or run.get('animation')) or run.get('mediaKind')=='Animated execution log')
            entries.append(entry)
    assert len(entries)==65 and len(gaps)==5
    # Validate the whole archive before mutating any public file.
    for name, data in assets.items():
        path=PUB/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    for run in comparisons['runs'].values():
        if key(run) in updates:
            run.update(updates[key(run)])
    for group in comparisons['groups']:
        runs=[comparisons['runs'][rid] for rid in group['runIds']]
        group['videoCount']=sum(bool(r.get('video') or r.get('animation')) for r in runs)
        preview=next((r for r in runs if r.get('poster') and (r.get('video') or r.get('animation'))),None)
        if preview: group['poster']=preview['poster']
    write(PUB/'data/case-comparisons.json', comparisons)
    # Keep legacy media maps and detailed trace views synchronized by source digest.
    media=read(PUB/'data/case-media.json')
    for run in media['runs'].values():
        if key(run) in updates: run.update(updates[key(run)])
    write(PUB/'data/case-media.json', media)
    legacy_path=PUB/'data/case-studies.json'
    legacy=read(legacy_path)
    names={r['model']:r['modelKey'] for r in comparisons['runs'].values()}
    for case in legacy:
        k=(names.get(case.get('model')),case.get('taskId'),case.get('sourceReportSha256'))
        if k in updates and not case.get('predictions'):
            case.update(updates[k])
    write(legacy_path,legacy)
    detail_paths={PUB/r['detailUrl'] for r in comparisons['runs'].values() if key(r) in updates}
    for name in ['index.json','complete-index.json']:
        path=PUB/'data/execution'/name; index=read(path)
        for story in index['stories']:
            for run in story['runs']:
                if key(run) in updates:
                    patch=updates[key(run)];run.update(patch);run['videoNote']=patch['mediaNote'];run['mediaMatchesReport']=True
                    detail_paths.add(PUB/run['detailUrl'])
        write(path,index)
    for path in detail_paths:
        detail=read(path)
        if key(detail) in updates:
            patch=updates[key(detail)];detail.update(patch);detail['videoNote']=patch['mediaNote'];detail['mediaMatchesReport']=True
            write(path,detail)
    audit=dict(archive=archive.name,archiveSha256=digest,matchedAssignments=70,availableMedia=65,missingMedia=5,
               newlyFilledCells=sum(e['filledEmptyCell'] for e in entries),addedScenePreviews=sum(e['addedScenePreview'] for e in entries),
               mediaKinds=dict(Counter(e['mediaKind'] for e in entries)),entries=entries,gaps=gaps)
    write(audit_path,audit)
    release=read(PUB/'data/release.json')
    release['playable_previews']=sum(bool(r.get('video') or r.get('animation')) for r in comparisons['runs'].values())
    release['recordings']=sum(bool(r.get('video') or r.get('animation')) and r.get('mediaKind')!='Animated execution log' for r in comparisons['runs'].values())
    release['recording_definition']='Scene previews: original recordings, saved observations, verified action replays and saved-state renders; excludes animated execution-log plots.'
    release['video_supplement']=dict(archive=archive.name,matchedAssignments=70,availableMedia=65,missingMedia=5)
    write(PUB/'data/release.json',release)
    print(json.dumps({k:v for k,v in audit.items() if k not in ['entries','gaps']},indent=2))
    print('All-gallery scene previews:',release['recordings'],'; all playable previews:',release['playable_previews'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    build(parser.parse_args().archive.resolve())
