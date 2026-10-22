#!/usr/bin/env python3
"""Verify toolbox assignments against raw reports and identify recording cameras.

Requires the original local material bundles. Only the verified Qwen head-camera
recording is replaced; external-camera clips remain explicitly identified.
"""
import argparse
import gzip
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / 'public'
GROUP = 'showcase-capx_behavior1k'


def read(path):
    return json.loads(path.read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    pretty = path.name in {'case-media.json', 'case-studies.json'}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2 if pretty else None,
                               separators=None if pretty else (',', ':'))+'\n')


def build(work, downloads, ffmpeg):
    data = read(PUB/'data/case-comparisons.json')
    group = next(g for g in data['groups'] if g['id'] == GROUP)
    cached = read(work/'case-comparison-20261020/reports.json')
    patches, entries = {}, []
    destination = 'assets/toolbox-cameras-20261022'
    dest = PUB/destination
    dest.mkdir(parents=True, exist_ok=True)
    supplement = zipfile.ZipFile(downloads/'homepage-missing-materials-complete-20261021.zip')
    logs = zipfile.ZipFile(downloads/'gemini4argon-logs-20261021.zip')
    for rid in group['runIds']:
        run = data['runs'][rid]
        model = run['modelKey']
        if model == 'gemini4argon':
            folder = work/'gemini4argon-20261021/inputs/gemini4argon-case-study-20261021/W4/capx_behavior1k/W4-capx_behavior1k-gemini4argon-01'
            digest = run['sourceReportSha256']
            raw_bytes = gzip.decompress(logs.read(f'gemini4argon-logs-20261021/objects/{digest[:2]}/{digest}.json.gz'))
            assert sha(raw_bytes) == digest
            raw = json.loads(raw_bytes)
        elif model == 'qwen':
            folder = None
            item = cached[run['recordId']]
            assert item['row']['source_observed_sha256'] == run['sourceReportSha256']
            raw = item['raw']
        else:
            folder = (work/'deepseek-update-20261019/inputs/deepseek-case-study-20261019' if model == 'deepseek'
                      else work/'case-gallery-20261015/case-study-all-models-20261014')/'W4/capx_behavior1k'/f'W4-capx_behavior1k-{model}-01'
            raw_bytes = (work/f'technical-report-20261020/reports/W4-capx_behavior1k-{model}-01.json').read_bytes()
            assert sha(raw_bytes) == run['sourceReportSha256']
            raw = json.loads(raw_bytes)
        binding = {k:v for k,v in raw['pool_binding'].items() if k != 'config_file'}
        identity = dict(binding=binding, entry=raw['pool_coordinate']['entry_id'], revision=raw['pool_coordinate']['source_revision'])
        assert raw['task_id'] == group['taskId'] == run['taskId']
        assert identity == group['nativeIdentity'] == run['nativeIdentity']
        if folder:
            case_bytes = (folder/'case.json').read_bytes()
            case = json.loads(case_bytes)
            camera = case['media']['camera']
            assert read(folder/'recording-summary.json')['source'] == camera
            assert sha((folder/'clip.mp4').read_bytes()) == sha((PUB/run['video']).read_bytes())
        else:
            prefix = 'homepage-missing-materials-complete-20261021/cases/'+run['recordId']+'/'
            case_bytes = supplement.read(prefix+'case.json')
            case = json.loads(case_bytes)
            head = 'original-recording/video/68a1c88497cad2bc-1790238351519653248-3221126'
            summary = json.loads(supplement.read(prefix+head+'.summary.json'))
            camera = summary['source']
            assert ':zed_link:Camera:0.rgb' in camera and summary['complete']
            video = supplement.read(prefix+head+'.mp4')
            listed = next(f for f in case['package_files'] if f['path'].endswith(head+'.mp4'))
            assert sha(video) == listed['sha256']
            (dest/'qwen-head.mp4').write_bytes(video)
            subprocess.run([str(ffmpeg), '-y', '-hide_banner', '-loglevel', 'error', '-i', str(dest/'qwen-head.mp4'),
                            '-frames:v', '1', str(dest/'qwen-head.jpg')], check=True)
        assert case['task_id'] == run['taskId'] and case['model'] == model
        assert case['source_sha256'] == run['sourceReportSha256'] == run['mediaReportSha256']
        external = camera == 'hq.capx.external'
        patch = dict(cameraView='external' if external else 'head', cameraLabel='External camera' if external else 'Robot head camera',
                     cameraSource=camera)
        if model == 'qwen':
            patch.update(video=destination+'/qwen-head.mp4', poster=destination+'/qwen-head.jpg', animation=None,
                         mediaProvenance=destination+'/qwen-provenance.json',
                         mediaNote='Original robot head-camera recording from this evaluation; playback time is separate from episode time.')
            write(dest/'qwen-provenance.json', dict(taskId=run['taskId'], modelKey=model, reportSha256=run['sourceReportSha256'],
                  archive=Path(supplement.filename).name, member=prefix+head+'.mp4', cameraSource=camera,
                  sourceVideoSha256=sha(video), videoSha256=sha(video), video=patch['video'],
                  sourceCaseSha256=sha(case_bytes), recordingSummary=summary))
        elif external:
            patch['mediaNote'] = 'External recording camera for this same task and evaluation. This camera is not the robot observation; playback time is separate from episode time.'
        if model == 'gemini4argon':
            manifest = json.loads(logs.read('gemini4argon-logs-20261021/cases/31593afb9cac5340cc2a.json'))
            assert manifest['task']['report_object'].endswith(run['sourceReportSha256']+'.json.gz')
            candidates = []
            for request in manifest['requests']:
                meta = json.loads(gzip.decompress(logs.read('gemini4argon-logs-20261021/'+request['metadata'])))
                obs = meta['observation']
                for img in obs.get('images', []):
                    if img['camera_id'] == 'ego':
                        archived = next(i for i in request['images'] if i['sha256'] == img['sha256'])
                        candidates.append((obs['turn'], img, archived))
            turn, img, archived = min(candidates, key=lambda item:item[0])
            image = logs.read('gemini4argon-logs-20261021/'+archived['bundle_path'])
            assert sha(image) == img['sha256']
            (dest/'gemini4-head-observation.png').write_bytes(image)
            patch['observationPreview'] = dict(image=destination+'/gemini4-head-observation.png', camera='ego', round=turn,
                sha256=sha(image), reportSha256=run['sourceReportSha256'], observationId=img['observation_id'])
        patches[(model, run['taskId'], run['sourceReportSha256'])] = patch
        entries.append(dict(runId=rid, modelKey=model, taskId=run['taskId'], reportSha256=run['sourceReportSha256'],
                            identitySha256=run['identitySha256'], sourceCaseSha256=sha(case_bytes),
                            originalCamera=case['media']['camera'], **{**patch, 'video':patch.get('video', run['video'])},
                            videoSha256=sha((PUB/patch.get('video', run['video'])).read_bytes())))
    supplement.close()
    logs.close()
    assert len(entries) == 10 and sum(e['cameraView']=='external' for e in entries) == 2
    def sync(obj):
        if isinstance(obj, dict):
            key = obj.get('modelKey'), obj.get('taskId'), obj.get('sourceReportSha256', obj.get('reportSha256'))
            if key in patches:
                obj.update(patches[key])
                if 'videoNote' in obj: obj['videoNote'] = obj.get('mediaNote', obj['videoNote'])
            if obj.get('id') == GROUP:
                obj['cameraNote'] = 'Same task and seed. DeepSeek and Gemini 4 use an external recording camera.'
            for value in list(obj.values()): sync(value)
        elif isinstance(obj, list):
            for value in obj: sync(value)
    paths = [PUB/'data'/f for f in ['case-comparisons.json', 'case-media.json', 'case-studies.json']]
    paths += list((PUB/'data/case-records').glob('*.json')) + list((PUB/'data/execution').rglob('*.json'))
    for path in paths:
        obj = read(path)
        before = json.dumps(obj)
        sync(obj)
        if json.dumps(obj) != before: write(path, obj)
    write(PUB/'data/toolbox-camera-audit.json', dict(date='2026-10-22', groupId=GROUP, taskId=group['taskId'],
          nativeIdentity=group['nativeIdentity'], matching=['taskId', 'config_sha256', 'seed', 'source_revision', 'reportSha256'],
          conclusion='All ten evaluations use the same native task. Recording camera selection differs.', entries=entries))
    print('Verified ten report identities and media sources; restored Qwen head camera; labeled two external recordings.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--downloads', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    args = parser.parse_args()
    build(args.work, args.downloads, args.ffmpeg)
