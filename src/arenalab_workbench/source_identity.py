"""Resolve registered complete sources identically for planning and execution.

Identity is computed locally from original bytes and private model dependencies.
No model, source payload or derived body is uploaded by these helpers.
"""
import copy
import hashlib
import json
from pathlib import Path

from .settings import paths


def absolute_spec(spec, base):
    if spec is None:
        return None
    result = copy.deepcopy(spec)
    if not isinstance(result, dict) or not result.get('path'):
        raise ValueError('Registered motion source requires an explicit local path')
    for key in ('path', 'objects_path'):
        if result.get(key):
            path = Path(result[key])
            result[key] = str((path if path.is_absolute() else Path(base) / path).resolve())
    return result


def registered_source(scene_id, *, row=None, config_path=None):
    """Return the selected observation row and complete source, if registered.

    Uploads override configured clips; calibrated variants retain their own
    observations. Gripper execution consumes source.json, so any contradictory
    registry declaration is an error rather than a fallback to reconstruction.
    """
    from .dataset_io.storage import entry
    from .height_registry import variant, public_entry
    from .gripper_registry import sources
    uploaded = entry(scene_id)
    custom = variant(scene_id)
    inherited = entry(custom['original_scene_id']) if custom and not uploaded else None
    recording = uploaded or inherited or {}
    config = Path(config_path or paths.viewer_config)
    if row is None:
        if uploaded:
            row = uploaded.get('source_row', {})
        elif custom and custom.get('source_row'):
            row = custom['source_row']
        else:
            public = public_entry(scene_id) or {}
            clips = json.loads(config.read_text()).get('clips', []) if config.is_file() else []
            row = next((c for c in clips if c['id'] == public.get('source_catalog_id', scene_id)), {})
    row = copy.deepcopy(row)
    base = config.parent
    if recording.get('source_row', {}).get('source'):
        base = Path(recording['source_row']['source']).resolve().parent
    candidates = [absolute_spec(s, base) for s in (
        recording.get('motion_source'), row.get('motion_source'),
        recording.get('manifest', {}).get('motion_source')) if s is not None]
    if candidates and any(s != candidates[0] for s in candidates[1:]):
        raise ValueError('Conflicting registered complete motion sources')
    supplied = candidates[0] if candidates else None
    gripper = sources().get(scene_id)
    if gripper:
        work = Path(gripper['work']).resolve()
        metadata = json.loads((work / 'source.json').read_text())
        actual = absolute_spec(metadata.get('motion_source'), work)
        if supplied is not None and actual != supplied:
            raise ValueError('Registered complete source disagrees with gripper source.json; attach an immutable workspace revision')
        supplied = actual
        row.update(work=str(work), source=str(work / 'source.npz'))
        row.update({key:metadata[key] for key in ('frames', 'fps') if key in metadata})
    if supplied is not None:
        row['motion_source'] = supplied
    return row, supplied


def robot_source_identity(spec, row=None):
    """Bind full robot state, clock, objects, observations and local model assets."""
    if not spec or spec.get('format') != 'robot_joint_state':
        return None
    from .motion_sources import sha256
    from .motion_backend import robot_asset_identity
    from .robot_motion_source import source_robot
    digest = sha256(spec['path'])
    if spec.get('sha256') and digest != spec['sha256']:
        raise ValueError('Original robot motion hash changed')
    model = source_robot(spec)
    identity = dict(policy='complete-robot-source-identity-v2', spec=spec,
                    state_sha256=digest, source_robot=model,
                    model_assets=robot_asset_identity(model['xml']))
    if spec.get('objects_path'):
        identity['objects_sha256'] = sha256(spec['objects_path'])
    row = row or {}
    identity['clock'] = {key:row.get(key) for key in ('source_window', 'frames', 'fps')}
    identity['observations'] = {key:sha256(row[key]) for key in ('source', 'observations_source') if row.get(key)}
    return identity


def identity_digest(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
