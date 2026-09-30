"""Host-memory admission before loading the VLABench simulator; no episode retry."""
import fcntl
import os
from pathlib import Path
from .paths import get_project_paths

_LEASE = None

def available_gib():
    rows=Path('/proc/meminfo').read_text().splitlines()
    return next(int(x.split()[1]) for x in rows if x.startswith('MemAvailable:')) / 1024**2

def admit_vlabench():
    global _LEASE
    if _LEASE is not None:return
    lock=Path(os.environ.get('ARENA_VLABENCH_ADMISSION_LOCK', str(get_project_paths().artifact_root / 'locks/vlabench-native.lock')))
    lock.parent.mkdir(parents=True,exist_ok=True)
    lease=lock.open('a+')
    try:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        free=available_gib()
        if free < 24:
            raise RuntimeError(f'vlabench_resource_admission_deferred: host MemAvailable={free:.2f} GiB, require 24 GiB')
    except BlockingIOError:
        lease.close()
        raise RuntimeError('vlabench_resource_admission_deferred: another VLABench native episode holds the lease') from None
    except Exception:
        lease.close();raise
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[name]='1'
    _LEASE=lease
