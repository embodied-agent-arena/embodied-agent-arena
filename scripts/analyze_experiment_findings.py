"""Recompute descriptive experiment findings from the public adopted-result manifests.

Run from any directory. Prints JSON; does not modify inputs or the manuscript.
"""
from pathlib import Path
import collections,hashlib,itertools,json
r=Path(__file__).resolve().parents[1];u=r/'public';w=r/'docs'
files=['report/planning-outcomes.json','spatial-recovery/manipulation-sources.json','spatial-recovery/calvin-summary.json','spatial-recovery/spatial-summary.json']
data={f:json.loads((u/'data'/f).read_text()) for f in files}
d=data[files[0]];ts=d['tasks'];ms=d['models'];assert len(ts)==157 and len({t['taskId'] for t in ts})==157
sets={m:{t['taskId'] for t in ts if m in t['successModels']} for m in ms}
reduced=[t for t in ts if t['benchmark'] not in ('ScienceWorld','VirtualHome')];assert len(reduced)==63
out={'input_sha256':{f:hashlib.sha256((u/'data'/f).read_bytes()).hexdigest() for f in files},'planning':{'histogram':dict(sorted(collections.Counter(len(t['successModels']) for t in ts).items())),'full_successes':{m:len(v) for m,v in sets.items()},'source_exclusion':{m:{'successes':sum(m in t['successModels'] for t in reduced),'n':len(reduced)}for m in ms},'all_union':len(set.union(*sets.values())),'pairs':sorted([{'models':c,'union':len(sets[c[0]]|sets[c[1]])}for c in itertools.combinations(ms,2)],key=lambda x:-x['union'])}}
assert out['planning']['pairs'][0]=={'models':('astra','gemini4argon'),'union':137}
assert out['planning']['all_union']==142
for m in ms:assert out['planning']['source_exclusion'][m]['successes']/63 < len(sets[m])/157
man=data[files[1]];out['manipulation_reweighting']={}
for m in ('astra','opus55','gemini4argon'):
 rows=man['sources'];assert sum(r['n'] for r in rows)==182 and len(rows)==13 and all(r['successes'][m] is not None for r in rows)
 out['manipulation_reweighting'][m]={'successes':sum(r['successes'][m] for r in rows),'assignments':182,'micro_percent':100*sum(r['successes'][m] for r in rows)/182,'equal_source_percent':100*sum(r['successes'][m]/r['n'] for r in rows)/13}
cs=data[files[2]];out['calvin']={'reached':[sum(m['reached'][i] for m in cs['models']) for i in range(5)],'unverified_prefixes':sum(m['n']-m['prefix_observed'] for m in cs['models']),'totals':cs['total']};assert out['calvin']['reached']==[60,40,25,18,9]
print(json.dumps(out,indent=2))
