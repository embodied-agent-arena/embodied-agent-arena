#!/usr/bin/env python3
"""Import exact-report footage; keep independent demos separate from evaluated runs."""
import argparse, hashlib, json
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]; PUB=ROOT/'public'
def read(p): return json.loads(p.read_text())
def write(p,d): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,separators=(',',':'))+'\n')
def digest(b): return hashlib.sha256(b).hexdigest()
def key(r): return r.get('modelKey'),r.get('taskId'),r.get('sourceReportSha256',r.get('reportSha256'))
def build(materials):
    hashes=read(materials/'files-sha256.json');assert digest((materials/'website-media-map.json').read_bytes())==hashes['website-media-map.json']
    manifest=read(materials/'website-media-map.json');data=read(PUB/'data/case-comparisons.json');updates={};assets={};entries=[]
    for c in manifest:
        r=data['runs']['record-'+c['record_id']]
        assert (r['modelKey'],r['taskId'],r['sourceReportSha256'],r['outcome'])==(c['model'],c['task_id'],c['original_report_sha256'],c['original_outcome'])
        case=read(materials/c['media_case_json'])
        assert (case['model'],case['task_id'],case['source_sha256'])==(c['model'],c['task_id'],c['media_report_sha256'])
        if c['demo_only']:
            assert (case['reference_record_id'],case['reference_report_sha256'],case['count_in_evaluation'])==(c['record_id'],c['original_report_sha256'],False)
        else:
            assert case['record_id']==c['record_id'] and case['outcome']==c['original_outcome']
        folder=f"assets/homepage-supplement-20261022/{c['record_id']}"
        def add(source,name):
            p=(materials/source).resolve();assert p.is_relative_to(materials.resolve());b=p.read_bytes();assert digest(b)==hashes[source];assets[folder+'/'+name]=b;return folder+'/'+name
        poster=add(c['poster'],'poster'+Path(c['poster']).suffix)
        proof={k:c[k] for k in ['record_id','model','task_id','original_report_sha256','original_outcome','media_origin','availability','demo_only','count_in_evaluation','demo_id','demo_outcome','media_report_sha256']}
        patch={}
        if c['demo_only']:
            assert c['count_in_evaluation'] is False and c['media_report_sha256']!=r['sourceReportSha256']
            demo=dict(video=add(c['preview_video'],'demo.mp4'),poster=poster,mediaKind='Same-task demo',mediaNote='Separate recording of this task; excluded from evaluation. The result and timeline below belong to the original evaluation.',modelKey=r['modelKey'],taskId=r['taskId'],reportSha256=c['media_report_sha256'],originalReportSha256=r['sourceReportSha256'],outcome=c['demo_outcome'],countInEvaluation=False)
            patch={'demoMedia':demo};proof['video']=demo['video'];proof['videoSha256']=digest(assets[demo['video']])
        else:
            assert c['media_report_sha256']==r['sourceReportSha256']
            patch=dict(poster=poster,mediaReportSha256=r['sourceReportSha256'],mediaProvenance=folder+'/provenance.json')
            if c['preview_video']:
                patch.update(video=add(c['preview_video'],'clip.mp4'),animation=None,mediaKind='Video excerpt',mediaNote='Excerpt from the original evaluation recording; playback time is separate from episode time.')
                proof['video']=patch['video'];proof['videoSha256']=digest(assets[patch['video']])
            else:
                assert c['availability']=='observations_only'
                observations=[json.loads(l) for l in (materials/c['observations_index']).read_text().splitlines() if l.strip()]
                frames=[];images=[]
                for obs in observations:
                    im=next(i for i in obs['images'] if i['camera_id']=='robot0_agentview_left')
                    assert digest((materials/im['path']).read_bytes())==im['sha256']
                    path=add(im['path'],f"observation-{obs['turn']:03}.png")
                    frames.append(dict(image=path,label=f"Round {obs['turn']} · input observation",round=obs['turn'],sha256=im['sha256']))
                    images.append(Image.open(materials/im['path']).convert('RGB'))
                assert len(frames)==24 and [f['round'] for f in frames]==list(range(1,25))
                import io
                out=io.BytesIO();images[0].save(out,format='WEBP',save_all=True,append_images=images[1:],duration=900,loop=0,quality=85)
                assets[folder+'/observations.webp']=out.getvalue()
                patch.update(video=None,animation=folder+'/observations.webp',mediaKind='Saved observations',mediaNote='24 original input RGB observations, one per decision round. This slideshow does not show the intervening actions.',frames=frames)
                proof['observations']=frames;proof['animation']=patch['animation'];proof['animationSha256']=digest(out.getvalue())
        assets[folder+'/provenance.json']=(json.dumps(proof,separators=(',',':'))+'\n').encode();updates[key(r)]=patch
        entries.append(dict(runId=r['id'],modelKey=r['modelKey'],taskId=r['taskId'],reportSha256=r['sourceReportSha256'],availability=c['availability'],provenance=folder+'/provenance.json'))
    assert len(entries)==17
    for name,b in assets.items():p=PUB/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    def sync(obj):
        if isinstance(obj,dict):
            patch=updates.get(key(obj))
            if patch:
                obj.update(patch)
                if 'mediaNote' in patch:obj['videoNote']=patch['mediaNote'];obj['mediaMatchesReport']=True
            for v in list(obj.values()):
                if isinstance(v,(dict,list)):sync(v)
        elif isinstance(obj,list):
            for v in obj:sync(v)
    # Only known public indexes and exact-match record files, never metrics.
    paths=[PUB/'data/case-comparisons.json',PUB/'data/case-media.json',PUB/'data/case-studies.json',PUB/'data/execution/index.json',PUB/'data/execution/complete-index.json']
    paths+=list((PUB/'data/case-records').glob('*.json'))+list((PUB/'data/execution').glob('**/*.json'))
    for p in set(paths):
        obj=read(p);before=json.dumps(obj);sync(obj)
        if json.dumps(obj)!=before:write(p,obj)
    comp=read(PUB/'data/case-comparisons.json')
    for g in comp['groups']:g['videoCount']=sum(bool(comp['runs'][rid].get('video') or comp['runs'][rid].get('animation')) for rid in g['runIds'])
    write(PUB/'data/case-comparisons.json',comp)
    audit=dict(date='2026-10-22',archive='homepage-missing-materials-complete-20261021.zip',matchedAssignments=17,originalVideos=13,independentDemos=3,observationSequences=1,entries=entries)
    write(PUB/'data/homepage-supplement-audit.json',audit)
    release=read(PUB/'data/release.json');release['homepage_supplement']={k:v for k,v in audit.items() if k!='entries'}
    release['playable_previews']=sum(bool(r.get('video') or r.get('animation')) for r in comp['runs'].values())
    release['recordings']=sum(bool(r.get('video') or r.get('animation')) and r.get('mediaKind')!='Animated execution log' for r in comp['runs'].values())
    release['independent_demo_previews']=3;write(PUB/'data/release.json',release)
    print(json.dumps({k:v for k,v in audit.items() if k!='entries'},indent=2));print('Scene previews',release['recordings'])
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--materials',type=Path,required=True);build(ap.parse_args().materials)
