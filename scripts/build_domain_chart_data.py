"""Export native web-chart inputs from the same retained evidence as Figures 4–8."""
import argparse, hashlib, json, re, xml.etree.ElementTree as ET
from pathlib import Path
R = Path(__file__).resolve().parents[1]
P = R.parents[2]
U = R / 'public'
inputs = {}
def read(path):
    raw = path.read_bytes()
    inputs[str(path.relative_to(P))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)
spatial = read(U/'data/spatial-recovery/spatial-summary.json')
calvin = read(U/'data/spatial-recovery/calvin-summary.json')
planning = read(U/'data/report/planning-outcomes.json')
metrics = read(U/'data/report/metrics.json')
geo = read(P/'output/geometry-deep-dive/geometry-data.json')
motion = read(P/'work/figure-audit-20261020/motion-data.json')
argon = read(P/'work/gemini4argon-20261021/derived.json')
models = spatial['models']
# Use the exact discrete ramp embedded in the current editable paper figure.
f = U/'assets/analysis/figure5-spatial.drawio'
inputs[str(f.relative_to(P))] = hashlib.sha256(f.read_bytes()).hexdigest()
ramp=[]
for cell in ET.parse(f).findall('.//mxCell'):
    g=cell.find('mxGeometry')
    style=dict(s.split('=',1) for s in cell.get('style','').split(';') if '=' in s)
    if g is not None and abs(float(g.get('y','-1'))-419)<.01 and float(g.get('width',0))<4:
        ramp.append((float(g.get('x')),style['fillColor']))
ramp=[c for _,c in sorted(ramp)]; assert len(ramp)==100, len(ramp)
# Definitions follow the source-native category taxonomy, not guessed difficulty levels.
definitions = [
('Left/right','Image-relative left/right','Identify whether one object lies to the left or right of another in the supplied view.'),
('Relations','Object relations','Infer the spatial relation between objects across the supplied MindCube views.'),
('Shift','Camera shift','Infer how the camera moves between views of the same scene.'),
('Actor','Actor-relative left/right','Judge left and right from the depicted person’s own orientation, rather than the image frame.'),
('Facing','Object-facing side','Identify which side of an object—front, back, left, or right—faces another specified object.'),
('Hard','Four-way video directions','From a video, imagine standing beside one object and facing a second. Locate a third object in the front-left, front-right, back-left, or back-right quadrant.'),
('Near','Closer depth','Choose the object that is closer to the camera in BOP-ASK.'),
('Far','Farther depth','Choose the object that is farther from the camera in BOP-ASK.'),
('Camera','Camera-relative depth','Compare objects by their distance to the camera in 3DSRBench.'),
('Object','Inter-object proximity','Identify which object is closer to a specified reference object in 3DSRBench.'),
('Relative','Relative distance','Choose the nearest object to a reference object in a video scene, measuring between their closest points.')]
taxonomy = read(P/'sources/materials-20260920/W2/w2-error-taxonomy-20260918/snapshot.json')
for row,(short,title,definition) in zip(spatial['panels']['a']+spatial['panels']['b'],definitions):
    assert row['label']==short
    row.update(title=title,definition=definition)
    samples=[t for t in taxonomy['tasks'] if t['analysis_type']==row['category']]
    if samples: row['exampleQuestion']=samples[0]['question']
source_names={'w2_3dsrbench':'3DSRBench','w2_bop_ask':'BOP-ASK','w2_mindcube':'MindCube','w2_mmsi_bench':'MMSI-Bench','w2_robospatial':'RoboSpatial-Home','w2_vsi_bench':'VSI-Bench'}
source_defs=['Spatial relations and imagined viewpoints in static images.','Object-pose, position, and depth questions.','Spatial inference across multiple views of a scene.','Multi-image spatial reasoning.','Spatial reasoning in home scenes.','Video-based spatial reasoning.']
spatial['sourceInfo']=[{'key':k,'label':v,'definition':d}for (k,v),d in zip(source_names.items(),source_defs)]
geo_defs={
'focal':'Estimate focal length from a GeoProbe image; error is mean absolute percentage error.',
'depth':'Estimate depth in GeoProbe; error is mean absolute relative error, expressed as a percentage.',
'size':'Estimate physical object size; error is mean absolute percentage error.',
'height':'Estimate target height using a reference object of known size; error is mean absolute percentage error.',
'ratio':'Estimate the target-to-reference height ratio; error is mean absolute percentage error.',
'influx':'Estimate focal length on InFlux photographs; error is mean absolute percentage error.'}
reductions=[]
for key in ['focal','depth','size','height','ratio','influx']:
    t=next(t for t in geo['tasks'] if t['id']==key);a=t['results']['astra'];b=t['results'][t['peer']]
    reductions.append(dict(label=t['label'],definition=geo_defs[key],value=(1-a['mean']/b['mean'])*100,peer=t['peer'],astra=a['mean'],peerMean=b['mean'],cases=a['case_count']))
# The scatter in the current paper uses the complementary table's display precision.
f=P/'work/deepseek-update-20261019/source/sections/table_profile_full.tex'
s=f.read_text(); inputs[str(f.relative_to(P))]=hashlib.sha256(f.read_bytes()).hexdigest()
def table_row(prefix):
    line=next(l for l in s.splitlines() if l.startswith(prefix))
    return [float(v.strip().rstrip('\\')) for v in line.split(' & ')[2:]]
errors=table_row('3D trace RMSE'); successes=table_row('3D constraint success')
trace=[dict(model=m,rmse=x,success=y,cases=56)for m,x,y in zip(models,errors,successes)]
motion_names=['Fixed camera','Small shift','Large shift','Photographs']
motion_defs=['Object displacement with a fixed camera in Blender.','Object displacement with a small camera-view change in Blender.','Object displacement with a large camera-view change in Blender.','Object displacement between real photographs.']
motion_rows=[dict(label=l,definition=d,results={('opus55' if m=='opus' else m):dict(z[i],runs=5) for m,z in motion.items()}) for i,(l,d) in enumerate(zip(motion_names,motion_defs))]
# Box values retain unrounded mean and SD; polygon values match the released table.
stats=read(P/'output/run-statistics/run-statistics.json')
regions=[]
for m in models:
    profile=next(x for x in metrics['profiles'] if x['model']==m)
    box=stats['main']['opus' if m=='opus55' else m][4]
    regions.append(dict(model=m,box=box['mean']*100,sd=box['sd']*100,runs=box['runs'],polygon=profile['means'][5]*100,cases=70))
sources=[dict(label=l,value=argon['polygon'][k]*100,definition=d,model='gemini4argon')for k,l,d in [
('w5_umd','UMD','Functional regions on tools: localize the part that supports the requested action.'),
('w5_ragnet','RAGNet','Action-conditioned regions in interaction scenes.'),
('w5_reasonaff','ReasonAff','Reason about action-conditioned target regions in interaction scenes.')]]
counts=[sum(m in t['successModels'] for t in planning['tasks']) for m in models]
hist=[sum(len(t['successModels'])==i for t in planning['tasks']) for i in range(11)]
execution = read(U/'data/execution/complete-index.json')
story = next(s for s in execution['stories'] if s['id']=='recovery')
progress=[]
for m in ['opus55','gemini4argon','deepseek']:
    run=next(r for r in story['runs'] if r['modelKey']==m)
    detail=read(U/run['detailUrl'])
    progress.append(dict(model=m,rounds=[dict(round=s['round'],score=s['score']) for s in detail['steps']]))
out=dict(version='20261022',models=models,heatRamp=ramp,progress=progress,geometry=dict(reductions=reductions,motion=motion_rows,trace=trace),spatial=spatial,affordance=dict(regions=regions,sources=sources),planning=dict(total=len(planning['tasks']),models=[dict(model=m,count=n,value=n/len(planning['tasks'])*100)for m,n in zip(models,counts)],histogram=hist),manipulation=calvin,provenance=inputs)
dest=U/'data/report/domain-charts.json'; data=json.dumps(out,ensure_ascii=False,indent=2)+'\n'
parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
if args.check: assert dest.read_text()==data,'Chart export differs from source evidence'
else: dest.write_text(data)
print(f'Verified five domains, {len(models)} models, {len(ramp)} original heatmap colours; {len(inputs)} source hashes.')
