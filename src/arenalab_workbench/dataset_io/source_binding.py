"""Match historical output inputs to a current complete source, without decoding.

Code fingerprints intentionally do not participate: a new release does not
change the recorded input bytes of an otherwise valid historical result.
"""
import json
from pathlib import Path

import numpy as np


def verify_saved_source(robot, spec, row):
    from .export import asset_path
    from ..motion_backend import cached_sha
    from ..motion_sources import _clock, _npz

    # Independent observations need their own original-byte binding. Existing
    # historical receipts contain normalized observation hashes only; source-key
    # lookup remains the safe path for those recordings.
    if row.get('observations_source') or Path(row.get('source', '')).resolve() != Path(spec['path']).resolve():
        raise ValueError('Historical independent-observation source binding is unavailable')
    validation = json.loads(asset_path(robot['playback_url']).with_name('validation.json').read_text())
    work, hashes = validation.get('source_work'), validation.get('source_hashes')
    if not work or not hashes:
        output = (Path(validation['native_output_dir'])/'motion.npz').resolve().parent
        request = json.loads((output/'request.json').read_text())
        if request.get('robot_id') != robot['id'] or Path(request['output_dir']).resolve() != output:
            raise ValueError('Producing request does not match selected robot output')
        work, hashes = request['source_work'], request['source_sha256']
    work = Path(work)
    def checked(name):
        path = work/name
        if not hashes.get(name) or cached_sha(path) != hashes[name]:
            raise ValueError('Saved source receipt is unavailable or changed')
        return path
    saved = json.loads(checked('input_manifest.json').read_text())
    actual = cached_sha(spec['path'])
    if saved.get('source_sha256') != actual or spec.get('sha256', actual) != actual:
        raise ValueError('Saved result used different original source bytes')
    previous = saved.get('source_spec', saved.get('motion_source'))
    if not isinstance(previous, dict):
        raise ValueError('Historical complete source descriptor is unavailable')
    semantic = lambda value: {k:v for k,v in value.items() if k not in {'path','objects_path','sha256'}}
    if semantic(previous) != semantic(spec):
        raise ValueError('Saved result used a different complete source descriptor')
    object_hash = cached_sha(spec['objects_path']) if spec.get('objects_path') else None
    if object_hash != hashes.get('objects.npz'):
        raise ValueError('Saved result used different source objects')
    if object_hash: checked('objects.npz')
    times, indices, _ = _clock(_npz(spec['path']), spec)
    with np.load(checked('native_source.npz'), allow_pickle=False) as native:
        if (not np.array_equal(native['timestamps_s'], times)
                or not np.array_equal(native['source_frame_indices'], indices)):
            raise ValueError('Saved result used a different source clock')
    if row.get('frames', len(times)) != len(times) or saved.get('frames') != len(times):
        raise ValueError('Saved result used a different source window')
