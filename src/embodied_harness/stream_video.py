"""Bounded asynchronous video sink for existing native RGB taps; no PNG frame spool."""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

_sinks = {}
_lock = threading.Lock()


def selected_source(source):
    bench = os.environ.get('ARENA_CASE_STUDY_BENCHMARK', '')
    if bench in {'robocasa', 'robocasa365', 'vlabench'}:
        return source.startswith('hq.' + bench + '.')
    if bench in {'cliport', 'vimabench', 'rlbench', 'robotwin2'}:
        return source.startswith('hq.' + bench + '.')
    if bench == 'capx':
        if source.startswith('hq.capx.'):
            return True
        return source.startswith('capx_environment.step.') and any(
            marker in source for marker in (':zed_link:', ':left_realsense_link:', ':right_realsense_link:'))
    # Prefer native step streams over duplicate backend/result taps.
    return '.step.' in source and not source.startswith('native')


class VideoSink:
    def __init__(self, root, source, width, height, fps=20):
        self.root = Path(root) / 'video'
        self.root.mkdir(parents=True, exist_ok=True)
        self.source, self.width, self.height, self.fps = source, width, height, fps
        self.stream = hashlib.sha256(source.encode()).hexdigest()[:16]
        self.session = f'{time.time_ns()}-{os.getpid()}'
        self.stem = self.root / f'{self.stream}-{self.session}'
        self.queue = queue.Queue(maxsize=16)
        self.last_sim_time = None
        self.error = None
        self.closed = False
        self.accepted = self.dropped = self.written = 0
        self.thread = threading.Thread(target=self._encode, name='rgb-video', daemon=True)
        self.thread.start()

    def push(self, image, metadata):
        if self.closed:
            raise RuntimeError('Video sink is closed')
        if image.size != (self.width, self.height):
            raise ValueError('Video resolution changed midstream')
        sim_time = metadata.get('simulation_time')
        if sim_time is not None and self.last_sim_time is not None:
            if sim_time < self.last_sim_time:
                raise ValueError('Simulation time moved backwards')
            if sim_time - self.last_sim_time < 1 / self.fps - 1e-9:
                return
        # Known simulation time gates sampling; native event streams without it
        # preserve events and declare their timebase instead of claiming 20Hz capture.
        packet = (image.tobytes(), dict(metadata, wall_time=time.time(),
                    timebase='simulation' if sim_time is not None else 'native_event'))
        try:
            self.queue.put_nowait(packet)
            self.accepted += 1
            self.last_sim_time = sim_time
        except queue.Full:
            self.dropped += 1

    def _encode(self):
        proc = None
        try:
            with self.stem.with_suffix('.stderr').open('wb') as log, self.stem.with_suffix('.jsonl').open('w') as index:
                proc = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                    '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{self.width}x{self.height}',
                    '-r', str(self.fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'veryfast',
                    '-crf', '18', '-pix_fmt', 'yuv420p', '-threads', '1',
                    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                    '-movflags', '+frag_keyframe+empty_moov+default_base_moof', str(self.stem.with_suffix('.mp4'))],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    pixels, metadata = item
                    proc.stdin.write(pixels)
                    index.write(json.dumps(dict(frame=self.written, source=self.source, **metadata)) + '\n')
                    self.written += 1
                proc.stdin.close()
                if proc.wait(timeout=30) != 0:
                    raise RuntimeError('ffmpeg encoding failed; see stderr')
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()

    def close(self):
        if self.closed:
            return
        self.closed = True
        while self.thread.is_alive():
            try:
                self.queue.put(None, timeout=0.2)
                break
            except queue.Full:
                continue
        self.thread.join(timeout=40)
        self.stem.with_suffix('.summary.json').write_text(json.dumps(dict(source=self.source,
            accepted=self.accepted, written=self.written, dropped=self.dropped, error=self.error,
            encoder_alive=self.thread.is_alive(), playback_fps=self.fps,
            complete=self.error is None and self.dropped == 0 and not self.thread.is_alive()
                     and self.written == self.accepted,
            note='Playback fps is not proof of native sampling frequency; inspect timebase and timestamps.')) + '\n')


def publish(image, source, metadata):
    if not selected_source(source):
        return
    root = os.environ['ARENA_CASE_STUDY_DIR']
    with _lock:
        key = (root, source)
        if key not in _sinks:
            # CaPX R1Pro's recorded step stream advances at 15 Hz. Keep the
            # playback timebase aligned with simulation time; 20 fps made it
            # appear 4/3 faster and exaggerated renderer flicker.
            fps = 15 if source.startswith('capx_environment.step.') or source == 'hq.capx.external' else 20
            _sinks[key] = VideoSink(root, source, image.width, image.height, fps=fps)
        _sinks[key].push(image, metadata or {})


def close_all():
    with _lock:
        for sink in _sinks.values():
            sink.close()
        _sinks.clear()


class VideoRecorder:
    """Case-level lifecycle; actual encoders belong to the simulator worker."""
    def __init__(self, root, metadata):
        self.root, self.metadata = Path(root), metadata
        self.root.mkdir(parents=True, exist_ok=True)
        self.started = time.time()

    def close(self):
        close_all()
        summaries = [json.loads(p.read_text()) for p in (self.root / 'video').glob('*.summary.json')]
        movies = list((self.root / 'video').glob('*.mp4'))
        publisher_errors = self.root / 'publisher-errors.jsonl'
        (self.root / 'video-summary.json').write_text(json.dumps(dict(
            schema='arena-stream-video/v1', started=self.started, finished=time.time(),
            metadata=self.metadata, streams=len(summaries),
            complete=bool(summaries) and len(summaries) == len(movies)
                     and all(x['complete'] for x in summaries)
                     and not (publisher_errors.exists() and publisher_errors.stat().st_size),
            summaries=summaries), indent=2) + '\n')


atexit.register(close_all)
