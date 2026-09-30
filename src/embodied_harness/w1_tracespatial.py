"""RGB-only paired TraceSpatial task adapter; GT and geometry remain outside the worker."""
import json,os,subprocess
from pathlib import Path
from .w1_runtime import W1Session
from .paths import resolve_project_root

BENCHMARKS={'tracespatial_2d':'TraceSpatial-Bench 2D','tracespatial_3d':'TraceSpatial-Bench 3D'}

def score_trajectory(answer, case, data_root, *, stderr_path=None):
    root = Path(os.environ.get('EMBODIED_ARENA_TRACESPATIAL_ROOT',
                              resolve_project_root() / 'runtimes/tracespatial'))
    python = Path(os.environ.get('EMBODIED_ARENA_TRACESPATIAL_PYTHON',
                                root / 'scorer-env/bin/python'))
    if not python.is_file():
        raise RuntimeError('TraceSpatial scoring requires its CPU environment; install runtimes/tracespatial/requirements.lock.txt or set EMBODIED_ARENA_TRACESPATIAL_PYTHON')
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    proc=subprocess.run([str(python),str(root/'score_worker.py')],
        input=json.dumps({'answer':answer,'case':case,'data_root':str(data_root)}),
        text=True,capture_output=True,timeout=300,env=env)
    if stderr_path is not None: Path(stderr_path).write_text(proc.stderr)
    if proc.returncode: raise RuntimeError('TraceSpatial scorer failed: ' + proc.stderr[-1500:])
    return json.loads(proc.stdout)

class TraceSpatialSession(W1Session):
    def observe(self):
        result=super().observe()
        result['image_note']='Receive RGB after observing and yielding. Submit x,y in 0-1000; in 3D append depth in meters. Geometry and reference annotations are private.'
        return result

    def submit(self,kwargs):
        if self.executor!='probe' and (self.turn<2 or self.pending_media or not self.image_bundle):
            raise ValueError('First observe and yield; submit only after receiving the RGB image')
        if self.submitted is not None:raise ValueError('Answer already submitted')
        self.submitted=kwargs.get('answer',kwargs)
        self.steps+=1;self.terminal=True
        # Keep metric tooling inaccessible to the model process; invoke it only after commit.
        self.evaluation=score_trajectory(self.submitted,self.case,self.data_root,
                                       stderr_path=self.outputs/'private-scoring.stderr.log')
        self.event(dict(event='submitted',prediction=self.submitted,evaluation=self.evaluation))
        return self.observe()

    def finish(self):
        if self.evaluation is None:return None
        m=self.evaluation;valid=m['submission_valid'];passed=m.get('passed')
        return dict(submission_valid=valid,prediction=self.submitted,passed=passed,metrics=m,official_score=None,official_score_eligible=False,score_kind='pinned_official_metric_adapter',task_outcome='invalid_prediction' if not valid else 'scored' if passed is None else 'correct' if passed else 'incorrect')
