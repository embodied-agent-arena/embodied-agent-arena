#!/usr/bin/env python3
"""Recover overlooked clips only when task, model and evaluated report all match."""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / 'public'
def read(p): return json.loads(p.read_text())
def write(p, obj): p.write_text(json.dumps(obj, ensure_ascii=False, separators=(',', ':'))+'\n')
def key(r): return r.get('taskId'), r.get('modelKey'), r.get('sourceReportSha256', r.get('reportSha256'))
def sha(b): return hashlib.sha256(b).hexdigest()

def build(archives):
    data = read(PUB/'data/case-comparisons.json')
    lookup = {key(r): r for r in data['runs'].values()}
    updates, entries = {}, []
    for archive in sorted(archives.glob('*case-study*.zip')):
        with zipfile.ZipFile(archive) as z:
            names = set(z.namelist())
            for member in sorted(names):
                if not member.endswith('case.json') or not any(w in member for w in ['/W3/', '/W4/']): continue
                case = json.loads(z.read(member))
                k = case.get('task_id'), case.get('model'), case.get('source_sha256')
                run = lookup.get(k)
                if not run or run.get('video') or run.get('animation') or k in updates: continue
                folder = member.rsplit('/', 1)[0]
                if (case.get('media') or {}).get('kind') != 'native_video': continue
                if folder+'/clip.mp4' not in names or folder+'/animation.gif' not in names: continue
                assert case['outcome'] == run['outcome']
                dest = PUB/'assets/recovered-cases-20261022'/run['recordId']
                dest.mkdir(parents=True, exist_ok=True)
                clip = z.read(folder+'/clip.mp4')
                (dest/'clip.mp4').write_bytes(clip)
                with Image.open(io.BytesIO(z.read(folder+'/animation.gif'))) as im:
                    im.convert('RGB').save(dest/'poster.webp', quality=90)
                rel = dest.relative_to(PUB).as_posix()
                patch = dict(video=rel+'/clip.mp4', animation=None, poster=rel+'/poster.webp',
                             mediaKind='Video excerpt', mediaReportSha256=k[2],
                             mediaProvenance=rel+'/provenance.json',
                             mediaNote='Excerpt from this evaluation recording; playback time is separate from episode time.')
                proof = dict(archive=archive.name, member=member, modelKey=k[1], taskId=k[0],
                             reportSha256=k[2], video=patch['video'], videoSha256=sha(clip),
                             sourceCaseSha256=sha(z.read(member)), recording=case['media'])
                write(dest/'provenance.json', proof)
                updates[k] = patch
                entries.append(dict(runId=run['id'], **{k:proof[k] for k in ['archive','member','modelKey','taskId','reportSha256','video','videoSha256']}, provenance=rel+'/provenance.json'))
    def sync(obj):
        if isinstance(obj, dict):
            patch = updates.get(key(obj))
            if patch: obj.update(patch)
            for v in list(obj.values()):
                if isinstance(v, (dict,list)): sync(v)
        elif isinstance(obj, list):
            for v in obj: sync(v)
    paths = [PUB/'data'/n for n in ['case-comparisons.json','case-media.json','case-studies.json']]
    paths += list((PUB/'data/case-records').glob('*.json')) + list((PUB/'data/execution').rglob('*.json'))
    for p in set(paths):
        obj=read(p); before=json.dumps(obj); sync(obj)
        if json.dumps(obj)!=before: write(p,obj)
    data=read(PUB/'data/case-comparisons.json')
    for group in data['groups']:
        rows=[data['runs'][r] for r in group['runIds']]
        group['videoCount']=sum(bool(r.get('video') or r.get('animation')) for r in rows)
        previews=[r for r in rows if r.get('poster') and (r.get('video') or r.get('animation'))]
        preview=next((r for r in previews if r['mediaKind']!='Animated execution log'), next(iter(previews),None))
        assert preview, group['id']
        group['poster']=preview['poster']
        # A context image is never assigned to another model's recording fields.
        group['sceneContext']=dict(image=preview['poster'], sourceRunId=preview['id'],
                                   taskId=group['taskId'], model=preview['model'],
                                   kind='action_log' if preview['mediaKind']=='Animated execution log' else 'scene')
    write(PUB/'data/case-comparisons.json',data)
    by_task={g['taskId']:g for g in data['groups']}
    for filename in ['index.json','complete-index.json']:
        path=PUB/'data/execution'/filename; index=read(path)
        for story in index['stories']:
            task_ids={r['taskId'] for r in story['runs']}
            assert len(task_ids)==1
            group=by_task.get(next(iter(task_ids)))
            if group: story['sceneContext']=group['sceneContext']
        write(path,index)
    audit=PUB/'data/case-recovery-audit.json'
    previous=read(audit)['entries'] if audit.exists() else []
    write(audit,dict(date='2026-10-22',matching=['taskId','modelKey','reportSha256'],entries=previous+entries))
    release=read(PUB/'data/release.json')
    release['playable_previews']=sum(bool(r.get('video') or r.get('animation')) for r in data['runs'].values())
    release['recordings']=sum(bool(r.get('video') or r.get('animation')) and r.get('mediaKind')!='Animated execution log' for r in data['runs'].values())
    write(PUB/'data/release.json',release)
    print(f'Recovered {len(entries)} exact-report clips; {len(data["groups"])} task groups have visual or log evidence.')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--archives',type=Path,required=True)
    build(ap.parse_args().archives)
