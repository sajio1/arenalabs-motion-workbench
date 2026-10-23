"""Provenance changes must never reuse a human asset with stale inputs."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
from arenalab_workbench.smpl_library import digest, identity, verified


def test_smpl_library():
    base=dict(dataset='EgoDex',source_sha256='source',frames=300,fps=30,
              floor_z_m=0.,height_calibration=None,source_window=None,policy='shared-v1')
    assert identity(base)==identity(dict(reversed(list(base.items()))))
    for change in [dict(floor_z_m=-.2),dict(frames=450),dict(source_sha256='other'),
                   dict(source_window={'start_frame':90}),dict(policy='shared-v2'),
                   dict(height_calibration={'minimum_height_cm':80})]:
        assert identity(base)!=identity({**base,**change}),change
    with TemporaryDirectory() as folder:
        root=Path(folder);names=['canonical.npz','smplh_motion.npz','egoallo_smplh.npz','observations.npz']
        for name in names:(root/name).write_bytes(b'immutable-'+name.encode())
        (root/'prior-ready.json').write_text(json.dumps(dict(raw_sha256={name:digest(root/name) for name in names})))
        verified(root)
        (root/'canonical.npz').write_bytes(b'changed')
        try:verified(root)
        except ValueError as error:assert 'changed' in str(error)
        else:raise AssertionError('Modified original prior was accepted')
    print('PASS: height/source/window/policy changes invalidate assets; modified priors rejected')
