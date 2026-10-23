"""Native SMPL-H contract for every EgoAllo body path; no model conversion."""
from pathlib import Path
import json
import numpy as np

SCHEMA = 'arenalab.human-reference.v1'
SMPLH_POLICY = 'egoallo-smplh-native-only-v2'

def requested_family(dataset):
    return 'smplh'

def family_for_work(work):
    work = Path(work)
    manifest_path = work / 'input_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get('human_model', 'smplh') != 'smplh':
        raise ValueError('Retired body family: regenerate this workspace as native SMPL-H')
    descriptor = work / 'human_reference.json'
    if descriptor.exists():
        value = json.loads(descriptor.read_text())
        if value.get('schema') != SCHEMA or value.get('family') != 'smplh':
            raise ValueError('Native SMPL-H reference required; converted body caches are retired')
    elif (work / 'smplx_motion.npz').exists():
        raise ValueError('Legacy converted body cache is retired; export native SMPL-H in a fresh workspace')
    return 'smplh'

def motion_path(work):
    family_for_work(work)
    return Path(work) / 'smplh_motion.npz'

def load_reference(work):
    with np.load(motion_path(work)) as saved:
        result = {key: saved[key] for key in saved.files}
    if 'human_model' in result and str(result['human_model']) != 'smplh':
        raise ValueError('Body family differs from native SMPL-H contract')
    result['human_model'] = np.array('smplh')
    return result

def support_model(work):
    family_for_work(work)
    with np.load(Path(work) / 'model.npz', allow_pickle=True) as saved:
        return {key: saved[key] for key in ('v_template', 'shapedirs', 'weights')}

def artifact_names(work):
    work = Path(work)
    names = ['canonical.npz', motion_path(work).name, 'observations.npz',
             'source_hands.npz', 'operation-accuracy/source-targets.npz',
             'human_reference.json']
    return [name for name in names if (work / name).exists()]
