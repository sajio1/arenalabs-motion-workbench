"""Fixed shape planning; indices refer to original observations, never edited targets."""
import hashlib
import json
from pathlib import Path
import numpy as np

POLICY = 'egoallo-window-300-overlap-60-frame-noise-v2-candidate'

def plan(length, size=300, overlap=60, first_step=None):
    if length < 1 or not 0 < overlap < size:
        raise ValueError('Invalid sequence/window size')
    start = 0
    while True:
        valid = min(size, length - start)
        yield start, valid
        if start + valid == length:
            break
        step = first_step if start == 0 and first_step is not None else size-overlap
        if not 0 < step <= size-overlap:
            raise ValueError('Invalid test seam offset')
        start += step

def indices(start, valid, size=300):
    return start + np.minimum(np.arange(size), valid-1)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)
