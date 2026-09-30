"""Read the selected release task pool without exposing private fields to agents."""
import json
import os
from pathlib import Path


def load_selected_pool():
    filename = os.environ.get('EMBODIED_ARENA_W3_TASK_POOL')
    if not filename:
        return None
    def expand(value):
        if isinstance(value, str):
            for variable in ('DATA_ROOT', 'EXTERNAL_ROOT'):
                token = '${' + variable + '}'
                if token in value:
                    setting = os.environ.get('EMBODIED_ARENA_' + variable)
                    if not setting:
                        raise ValueError('Missing EMBODIED_ARENA_' + variable)
                    value = value.replace(token, setting)
            if '${' in value:
                raise ValueError('Unresolved task-pool path variable')
            return value
        if isinstance(value, list): return [expand(v) for v in value]
        if isinstance(value, dict): return {k:expand(v) for k,v in value.items()}
        return value
    return expand(json.loads(Path(filename).read_text()))
