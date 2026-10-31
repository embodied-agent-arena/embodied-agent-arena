#!/usr/bin/env python3
"""Import the ten supplied toolbox recordings without changing scored episodes."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'public'
ASSETS = 'assets/toolbox-recordings-20261024'
TASK = 'capx--capx_current_interface_live--test--env_configs-r1pro-b1k_outfit_a_basic_toolbox.yam--0--1cd2784f91bf'
ARCHIVED_TASKS = {
    'capx--capx_current_interface_live--test--env_configs-r1pro-b1k_putting_shoes_on_rack.yaml--0--7c76679a3a9c',
    'capx--capx_current_interface_live--test--env_configs-r1pro-b1k_can_meat.yaml--0--4dbe460b28c8',
}


def main(source, ffmpeg):
    rows = list(csv.DictReader((source / 'manifest.csv').open(encoding='utf-8-sig')))
    assert len(rows) == 10 and len({r['alias'] for r in rows}) == 10
    updates = {}
    for r in rows:
        assert r['case_id'] == TASK
        video = source / r['video']
        assert hashlib.sha256(video.read_bytes()).hexdigest() == r['sha256']
        summary = json.loads((source / 'frame-index' / (r['alias'] + '.summary.json')).read_text())
        assert summary['complete'] and not summary['error'] and summary['dropped'] == 0
        assert summary['written'] == int(r['frames']) and summary['source'] == 'hq.capx.head'
        dest = PUBLIC / ASSETS / r['alias']
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(video, dest / 'clip.mp4')
        subprocess.run([str(ffmpeg), '-v', 'error', '-i', str(video), '-f', 'null', '-'], check=True)
        subprocess.run([str(ffmpeg), '-v', 'error', '-y', '-ss', '1', '-i', str(video),
                        '-frames:v', '1', '-q:v', '2', str(dest / 'poster.jpg')], check=True)
        shutil.copyfile(source / 'frame-index' / (r['alias'] + '.jsonl'), dest / 'frames.jsonl')
        updates[r['alias']] = dict(
            modelKey=r['alias'], taskId=TASK, recordingDate='2026-10-24',
            modelId='gpt-6-sol' if r['alias'] == 'sol' else r['model_id'],
            video=f"{ASSETS}/{r['alias']}/clip.mp4", poster=f"{ASSETS}/{r['alias']}/poster.jpg",
            videoSha256=r['sha256'], frameCount=int(r['frames']), playbackSeconds=float(r['duration_seconds']),
            elapsedSeconds=float(r['run_elapsed_seconds']), outcome=r['outcome'],
            attempt=int(r['attempt']), cameraSource=summary['source'], cameraView='head',
            mediaKind='Updated recording', countInEvaluation=False,
            provenance=f"{ASSETS}/{r['alias']}/provenance.json",
            mediaNote='Head-camera recording from the 24 October release. This separate run does not change the paper’s aggregate results.')
        proof = dict(updates[r['alias']], archive=source.name + '.zip',
                     frameIndex=f"{ASSETS}/{r['alias']}/frames.jsonl", encoderSummary=summary,
                     modelIdCorrection='Sol model ID corrected to gpt-6-sol by the author on 2026-10-31.' if r['alias']=='sol' else None)
        (dest / 'provenance.json').write_text(json.dumps(proof, indent=2)+'\n')

    def annotate(obj):
        if isinstance(obj, dict):
            key = obj.get('modelKey')
            if obj.get('taskId', obj.get('sample')) in ARCHIVED_TASKS:
                obj['previewPolicy'] = 'records-only'
            if key in updates and obj.get('taskId', obj.get('sample')) == TASK:
                obj['updatedRecording'] = updates[key]
            if obj.get('id') == 'showcase-capx_behavior1k':
                obj['cameraNote'] = 'Updated recordings · 24 October · Robot head camera'
                obj['poster'] = updates['astra']['poster']
            for key, value in list(obj.items()):
                if key != 'updatedRecording': annotate(value)
        elif isinstance(obj, list):
            for value in obj: annotate(value)

    paths = [PUBLIC/'data'/f for f in ['case-comparisons.json','case-media.json','case-studies.json']]
    paths += list((PUBLIC/'data/case-records').glob('*.json')) + list((PUBLIC/'data/execution').rglob('*.json'))
    changed = []
    for path in paths:
        obj = json.loads(path.read_text()); before = json.dumps(obj)
        annotate(obj)
        if json.dumps(obj) != before:
            pretty = path.name == 'case-studies.json'
            path.write_text(json.dumps(obj,ensure_ascii=False,indent=2 if pretty else None,separators=None if pretty else (',',':'))+'\n')
            changed.append(str(path.relative_to(ROOT)))
    audit = dict(date='2026-10-31',sourceArchive=source.name+'.zip',taskId=TASK,
                 scorePolicy='Preserve original scored episodes; the supplied recordings are separate runs.',
                 records=list(updates.values()))
    (PUBLIC/'data/toolbox-recording-refresh.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(f'Imported and decoded {len(updates)} recordings; annotated {len(changed)} data files.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--ffmpeg',type=Path,required=True)
    args = parser.parse_args()
    main(args.source,args.ffmpeg)
