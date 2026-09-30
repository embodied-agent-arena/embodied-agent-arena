"""Evaluator-only sampling of the latest published RGB frames.

Two sampling modes share one producer/sampler split:

``primitive_event`` (default for real cases) samples when a native frame is
published, so one recorded frame corresponds to one environment interaction.
``wall_clock`` keeps the original fixed 1 Hz tick for backward compatibility.

Wall-clock sampling was throttling the taps to one publish per second, which
both stored long runs of byte-identical frames and dropped the rapid bursts of
primitive calls that matter most (e.g. RoboCasa refining a button contact).
Event mode removes the tap throttle and skips unchanged streams instead.

Simulator APIs run only on their existing owner thread. The sampler never calls
observe(), render(), step(), a model, or a verifier. Repeated/stale frames and
missing sources are explicit. No recording paths are added to model feedback.
"""
from __future__ import annotations
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import time
import uuid

_ENV='ARENA_CASE_STUDY_DIR'
_MODES=('primitive_event','wall_clock')
_last_publish={}
# Set by producers, waited on by the sampler thread, so sampling stays off the
# simulator's owner thread while still tracking every published frame.
_publish_signal=threading.Event()
# Serializes publishes.jsonl so a burst cannot lose intermediate frames.
_publish_lock=threading.Lock()
# Minimum seconds between tap publishes; event mode lifts it to capture bursts.
_tap_min_interval=[0.0]

def _atomic(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_bytes(data);os.replace(tmp,path)

def _error(message):
    root=os.environ.get(_ENV)
    if root:
        try:
            with (Path(root)/'publisher-errors.jsonl').open('a') as f:
                f.write(json.dumps({'unix_time':time.time(),'error':str(message)})+'\n')
        except OSError:pass

def publish_image(image, source, *, source_path=None, source_metadata=None):
    """Called by native frame producers, never by the sampler thread."""
    root=os.environ.get(_ENV)
    if not root:return
    try:
        from PIL import Image
        if isinstance(image,(str,Path)):
            with Image.open(image) as im:im.load();image=im.convert('RGB')
        elif isinstance(image,bytes):
            with Image.open(io.BytesIO(image)) as im:im.load();image=im.convert('RGB')
        elif not isinstance(image,Image.Image):
            import numpy as np
            if hasattr(image,'detach'):image=image.detach().cpu().numpy()
            a=np.asarray(image)
            if a.ndim==4 and a.shape[0]==1:a=a[0]
            if a.ndim==3 and a.shape[0] in (3,4) and a.shape[-1] not in (3,4):a=a.transpose(1,2,0)
            if a.ndim!=3 or a.shape[-1] not in (3,4) or min(a.shape[:2])<8:return
            if a.dtype.kind=='f':
                if not np.isfinite(a).all() or a.min()<0 or a.max()>1:return
                a=(a*255).round().astype('uint8')
            elif a.dtype.kind in 'ui' and a.min()>=0 and a.max()<=255:a=a.astype('uint8')
            else:return
            image=Image.fromarray(a[...,:3])
        image=image.convert('RGB')
        if os.environ.get('ARENA_CASE_STUDY_FORMAT') == 'mp4':
            from .stream_video import publish
            publish(image, source, source_metadata)
            return
        buf=io.BytesIO();image.save(buf,format='PNG',compress_level=1);data=buf.getvalue()
        digest=hashlib.sha256(data).hexdigest();root=Path(root)
        blob=root/'blobs'/f'{digest}.png'
        if not blob.exists():_atomic(blob,data)
        stream=hashlib.sha256(source.encode()).hexdigest()[:16]
        now=time.time()
        previous=_last_publish.get((str(root),source))
        changed=previous is None or previous[0]!=digest
        content_updated=now if changed else previous[1]
        _last_publish[(str(root),source)]=(digest,content_updated)
        meta=dict(stream=stream,source=source,source_path=str(source_path) if source_path else None,sha256=digest,blob=str(blob.relative_to(root)),published_unix=now,content_updated_unix=content_updated,width=image.width,height=image.height,source_metadata=source_metadata or {})
        _atomic(root/'incoming'/f'{stream}.json',json.dumps(meta).encode())
        # incoming/ is latest-only per stream. The append-only log is what
        # event sampling reads, so a same-stream burst cannot collapse.
        with _publish_lock:
            with (root/'publishes.jsonl').open('a') as f:
                f.write(json.dumps(meta)+'\n')
        _publish_signal.set()
    except Exception as exc:_error(f'{source}: {type(exc).__name__}: {exc}')

def publish_observation(value, source='observation', *, _depth=0, source_metadata=None):
    if not os.environ.get(_ENV) or _depth>9:return
    try:
        name=source.lower()
        leaf=name.rsplit('.',1)[-1]
        if any(k in leaf for k in ('depth','segment','mask','normal','pointcloud')):return
        if isinstance(value,dict):
            for key,item in value.items():publish_observation(item,source+'.'+str(key),_depth=_depth+1,source_metadata=source_metadata)
        elif hasattr(value,'shape'):
            if any(k in name for k in ('rgb','color','image','frame','pixels')):
                if len(value.shape)==4 and value.shape[0]>1:
                    for i in range(min(16,value.shape[0])):publish_image(value[i],f'{source}.{i}',source_metadata=source_metadata)
                else:publish_image(value,source,source_metadata=source_metadata)
        elif isinstance(value,(list,tuple)):
            # Public CLIPort raw pixels can be nested Python lists.
            if value and isinstance(value[0],list) and any(k in name for k in ('rgb','color','image')):
                import numpy as np
                a=np.asarray(value)
                if a.ndim==3 and a.shape[-1] in (3,4):publish_image(a,source,source_metadata=source_metadata);return
            for i,item in enumerate(value[:16]):publish_observation(item,f'{source}.{i}',_depth=_depth+1,source_metadata=source_metadata)
        elif isinstance(value,str):
            if value.startswith('data:image/'):
                publish_image(base64.b64decode(value.split(',',1)[1]),source,source_metadata=source_metadata)
            elif leaf in ('image_path','rgb_path','frame_path') and Path(value).is_file():
                publish_image(value,source,source_path=value,source_metadata=source_metadata)
        elif hasattr(value,'__dict__'):
            publish_observation(vars(value),source,_depth=_depth+1,source_metadata=source_metadata)
    except Exception as exc:_error(f'{source}: {type(exc).__name__}: {exc}')

def capture_backend(backend, result=None):
    if not os.environ.get(_ENV):return
    for name in ('_last_obs','_last_observation','_current_runtime_observation'):
        value=vars(backend).get(name)
        if value is not None:publish_observation(value,'native'+name)
    if result is not None:
        value=getattr(result,'output',None) or getattr(result,'data',None)
        if value:publish_observation(value,'native_result')
    install_native_taps(backend)
    if os.environ.get("ARENA_HQ_RECORDING") == "1":
        from .hq_recording import install
        install(backend)

def install_native_taps(backend):
    """Observe values already returned by native APIs; do not make extra calls."""
    if not os.environ.get(_ENV):return
    objects=[('backend',backend)]
    for key in ('_env','_task_env'):
        obj=vars(backend).get(key)
        if obj is not None:objects.append((key,obj))
    runtime=vars(backend).get('runtime')
    if runtime is not None and getattr(runtime,'env',None) is not None:
        objects.append(('vlabench_environment',runtime.env))
    session=vars(backend).get('_live_session')
    if session is not None:
        obj=getattr(session,'low_level_environment',None)
        if obj is not None:objects.append(('capx_environment',obj))
    for label,obj in objects:
        for name in ('get_obs','get_observation','render','step','_public_live_observation'):
            original=getattr(obj,name,None)
            if not callable(original) or getattr(original,'_case_study_tap',False):continue
            def make_tap(fn,source):
                stamp=[-1e9]
                def tap(*args,**kwargs):
                    result=fn(*args,**kwargs)
                    now=time.monotonic()
                    if now-stamp[0]>=_tap_min_interval[0]:
                        metadata = None
                        if os.environ.get('ARENA_CASE_STUDY_FORMAT') == 'mp4':
                            from .w4_rgb import simulation_time
                            metadata = {'simulation_time': simulation_time(backend)}
                        stamp[0]=now;publish_observation(result,source+('.rgb' if source.endswith('.render') else ''),source_metadata=metadata)
                    return result
                tap._case_study_tap=True
                return tap
            try:setattr(obj,name,make_tap(original,f'{label}.{name}'))
            except (AttributeError,TypeError):pass

class WallClockRecorder:
    def __init__(self,directory, *, metadata=None, interval=1.0, mode='wall_clock'):
        if mode not in _MODES:raise ValueError(f'Unknown case-study mode: {mode!r}')
        if mode=='wall_clock' and interval!=1.0:raise ValueError('Case-study recorder is fixed at 1 Hz')
        self.mode=mode
        self.root=Path(directory);self.root.mkdir(parents=True,exist_ok=True)
        self.metadata=metadata or {};self.start=time.monotonic();self.started_unix=time.time()
        self.stop_event=threading.Event();self.thread=None;self.previous={};self.count=0;self.errors=[]
        self.session=uuid.uuid4().hex[:12]
        self.index=self.root/f'frames-{self.session}.jsonl'
        self.consumed=0
    def _link(self, m, now):
        source=self.root/m['blob']
        if self.root.resolve() not in source.resolve().parents:raise ValueError('Frame outside recording root')
        prior=self.previous.get(m['stream'])
        if self.mode=='primitive_event' and prior==m['sha256']:return None
        frame=self.root/'frames'/self.session/m['stream']/f'{self.count:08d}.png';frame.parent.mkdir(parents=True,exist_ok=True)
        os.link(source,frame)
        self.previous[m['stream']]=m['sha256']
        return dict(m,path=str(frame.relative_to(self.root)),repeated=prior==m['sha256'],source_age_seconds=max(0,now-m['published_unix']),content_age_seconds=max(0,now-m['content_updated_unix']))
    def _emit(self, kind, frames, now, elapsed):
        if self.mode=='primitive_event' and not frames and kind!='final':return None
        row=dict(tick=self.count,kind=kind,elapsed_seconds=elapsed,unix_time=now,frames=frames,missing_frame=not frames)
        with self.index.open('a') as f:f.write(json.dumps(row)+'\n')
        self.count+=1
        return row
    def _sample_event(self, kind, now, elapsed):
        log=self.root/'publishes.jsonl'
        with _publish_lock:
            lines=log.read_text().splitlines() if log.is_file() else []
            new=lines[self.consumed:]
            self.consumed=len(lines)
        last=None
        for line in new:
            try:
                frame=self._link(json.loads(line),now)
                if frame is None:continue
                last=self._emit('final' if kind=='final' else 'event',[frame],now,elapsed)
            except (OSError,ValueError,KeyError,json.JSONDecodeError) as exc:self.errors.append(str(exc))
        return last
    def sample(self, kind="periodic"):
        now=time.time();elapsed=time.monotonic()-self.start
        if self.mode=='primitive_event':
            return self._sample_event(kind,now,elapsed)
        frames=[]
        for p in sorted((self.root/'incoming').glob('*.json')):
            try:
                frame=self._link(json.loads(p.read_text()),now)
                if frame:frames.append(frame)
            except (OSError,ValueError,KeyError) as exc:self.errors.append(str(exc))
        return self._emit(kind,frames,now,elapsed)
    def _run(self):
        if self.mode=='primitive_event':
            while not self.stop_event.is_set():
                # Short poll so a publish that lands during a sample is not
                # missed, and so stop() is honoured promptly.
                if not _publish_signal.wait(0.2):continue
                _publish_signal.clear()
                try:self.sample(kind='event')
                except Exception as exc:self.errors.append(str(exc))
            return
        deadline=self.start
        while not self.stop_event.is_set():
            try:self.sample()
            except Exception as exc:self.errors.append(str(exc))
            deadline+=1.0
            # Do not synthesize past samples after a delayed disk write.
            if deadline<time.monotonic():deadline=time.monotonic()+1.0
            self.stop_event.wait(max(0,deadline-time.monotonic()))
    def start_recording(self):
        name='case-study-event' if self.mode=='primitive_event' else 'case-study-1hz'
        self.thread=threading.Thread(target=self._run,name=name,daemon=True);self.thread.start();return self
    def close(self):
        self.stop_event.set()
        if self.thread:self.thread.join(timeout=15)
        # Preserve a short episode's only frame, or a newly published final view.
        # This endpoint sample is explicitly labeled, never counted as a 1 Hz tick.
        try:
            if self.mode=='primitive_event':
                # Drain any publishes the sampler thread has not indexed yet.
                self.sample(kind='event')
            else:
                incoming=[json.loads(p.read_text()) for p in (self.root/'incoming').glob('*.json')]
                if any(self.previous.get(m['stream']) != m['sha256'] for m in incoming):
                    self.sample(kind='final')
        except Exception as exc:self.errors.append(str(exc))
        event=self.mode=='primitive_event'
        summary=dict(schema='arena-case-study/v1',mode=self.mode,hz=None if event else 1,time_basis='primitive_event' if event else 'wall_clock',frame_policy='changed_published_rgb_per_interaction' if event else 'latest_published_rgb',session=self.session,started_unix=self.started_unix,finished_unix=time.time(),ticks=self.count,streams=len(self.previous),index=self.index.name,errors=self.errors,writer_stopped=not self.thread or not self.thread.is_alive(),metadata=self.metadata)
        _atomic(self.root/f'summary-{self.session}.json',json.dumps(summary,indent=2).encode())
        return summary

def start_for_case(arguments):
    if os.environ.get('ARENA_CASE_STUDY_HZ')!='1':return None
    bench=os.environ.get('ARENA_CASE_STUDY_BENCHMARK','')
    # Text/symbolic environments render nothing; W1/W2/W5 are open-loop and
    # their showcase material is an offline prediction-vs-GT overlay, not a
    # frame strip. deployment/recording-scope.json is the primary gate; this is
    # a backstop for hand-run cases that bypass the derived manifest.
    if bench in ('scienceworld_text','virtualhome_symbolic'):return None
    if bench.startswith(('w1_','w2_','w5_')):return None
    mode=os.environ.get('ARENA_CASE_STUDY_MODE') or 'primitive_event'
    if mode not in _MODES:raise ValueError(f'Unknown ARENA_CASE_STUDY_MODE: {mode!r}')
    if mode=='primitive_event':_tap_min_interval[0]=0.0
    def arg(flag):
        for i,a in enumerate(arguments):
            if a==flag and i+1<len(arguments):return arguments[i+1]
            if a.startswith(flag+'='):return a.split('=',1)[1]
    output=arg('--output')
    if not output:raise ValueError('Recording requires a case output path')
    # Optional campaign-wide once-only claim, used by the four CaPX demos.
    # A failed recording remains a recorded attempt; retries never silently
    # create additional videos. Model-input archives are independent.
    once_root=os.environ.get('ARENA_CASE_STUDY_ONCE_ROOT')
    if once_root:
        identity=json.dumps([os.environ.get('ARENA_CASE_STUDY_TASK_ID'),
            os.environ.get('ARENA_CASE_STUDY_MODEL') or arg('--model')])
        root=Path(once_root);root.mkdir(parents=True,exist_ok=True)
        claim=root/(hashlib.sha256(identity.encode()).hexdigest()+'.json')
        try:
            with claim.open('x') as f:json.dump(dict(identity=json.loads(identity),output=output,claimed_at=time.time()),f)
        except FileExistsError:
            os.environ.pop(_ENV,None)
            return None
    directory=Path(output).resolve().parent/'case_study'
    directory.mkdir(parents=True,exist_ok=True)
    if os.environ.get('ARENA_CASE_STUDY_FORMAT') == 'mp4':
        from .stream_video import VideoRecorder
        os.environ[_ENV] = str(directory)
        return VideoRecorder(directory, dict(benchmark=bench,
            task_id=os.environ.get('ARENA_CASE_STUDY_TASK_ID'),
            model=os.environ.get('ARENA_CASE_STUDY_MODEL') or arg('--model')))
    # A retry gets a fresh source set; past ticks/blobs remain intact.
    for old in (directory/'incoming').glob('*.json'):old.unlink()
    log=directory/'publishes.jsonl'
    if log.exists():log.unlink()
    _publish_signal.clear()
    os.environ[_ENV]=str(directory)
    return WallClockRecorder(directory,mode=mode,metadata=dict(benchmark=bench,task_id=os.environ.get('ARENA_CASE_STUDY_TASK_ID'),model=os.environ.get('ARENA_CASE_STUDY_MODEL') or arg('--model'),runner_output=output)).start_recording()
