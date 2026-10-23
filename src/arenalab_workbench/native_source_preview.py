"""On-demand native source FK previews, without neural inference or refinement."""
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import tempfile
import threading

import numpy as np

from .settings import paths

POLICY = 'native-source-preview-v1'
_BUILD_SLOTS = threading.BoundedSemaphore(2)


def _sha(path):
    from .motion_sources import sha256
    return sha256(path)


def _identity(spec, row):
    from .source_identity import robot_source_identity
    source = robot_source_identity(spec, row)
    if source is None:
        raise ValueError('On-demand source preview requires native robot joint states')
    modules = ('arenalab_workbench.native_source_preview', 'arenalab_workbench.source_preview',
               'arenalab_workbench.source_objects', 'arenalab_workbench.robot_motion_source',
               'arenalab_workbench.source_robot_registry', 'arenalab_workbench.motion_sources',
               'arenalab_motion.prior', 'arenalab_motion.asset_cache', 'greenwich_pipelines.viewer.export')
    code = {name:_sha(importlib.import_module(name).__file__) for name in modules}
    versions = {name:str(getattr(importlib.import_module(name), '__version__', 'unknown'))
                for name in ('numpy', 'scipy', 'mujoco', 'trimesh')}
    return dict(policy=POLICY, source=source, code=code, versions=versions)


def _render(spec, output, robot):
    """Only named-state loading, native geometry FK and original object copying."""
    import mujoco as mj
    import trimesh
    from .robot_motion_source import read_state
    from .source_objects import load_source_objects
    from greenwich_pipelines.viewer.export import visible_geoms, mesh_scene
    target, qpos, times, indices = read_state(spec, robot)
    model = target['model']; data = mj.MjData(model)
    geometry = mesh_scene(model, visible_geoms(model))
    parts = [(i, geometry.geometry['geom_'+str(i)]) for i in visible_geoms(model)
             if 'geom_'+str(i) in geometry.geometry]
    if not parts:
        raise ValueError('Source model has no visible native geometry')
    visual = trimesh.Scene()
    for i, (_, mesh) in enumerate(parts):
        visual.add_geometry(mesh, node_name='geom_'+str(i), geom_name='geom_'+str(i))
    positions = np.empty((len(times),len(parts),3))
    rotations = np.empty((len(times),len(parts),3,3))
    ids = [i for i, _ in parts]
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mj.mj_kinematics(model, data)
        positions[frame] = data.geom_xpos[ids]
        rotations[frame] = data.geom_xmat[ids].reshape(-1,3,3)
    output = Path(output); output.mkdir()
    (output/'source_robot.glb').write_bytes(visual.export(file_type='glb'))
    np.savez_compressed(output/'source_robot_playback.npz', positions=positions, rotations=rotations,
                        timestamps_s=times, source_frame_indices=indices, exact_fps=np.array(30.))
    if spec.get('objects_path'):
        load_source_objects(spec['objects_path'], times)
        shutil.copyfile(spec['objects_path'], output/'objects.npz')
    provenance = dict(source_format='robot_joint_state', source_robot_id=spec['robot_id'],
                      source_sha256=_sha(spec['path']), source_video_role=spec.get('video_role', 'preview-only'),
                      source_model_glb_sha256=_sha(output/'source_robot.glb'),
                      source_visual_poses_sha256=_sha(output/'source_robot_playback.npz'))
    for key, field in (('source_evidence','evidence'), ('source_stage','source_stage')):
        if spec.get(field) is not None:
            provenance[key] = spec[field]
    (output/'source_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


def _files(root):
    return {p.relative_to(root).as_posix():_sha(p) for p in sorted(Path(root).rglob('*')) if p.is_file()}


def _preview_hashes(preview, destination):
    pairs = [(preview['model_url'],preview['model_sha256']),
             (preview['playback_url'],preview['playback_sha256'])]
    for obj in preview.get('objects', []):
        pairs.extend(((obj['mesh_url'],obj['mesh_sha256']), (obj['poses_url'],obj['poses_sha256'])))
    result = {}
    for url, expected in pairs:
        if not isinstance(url,str) or not url.startswith('/assets/'):
            raise ValueError('Source preview requires local platform assets')
        path = (paths.store/url.removeprefix('/assets/')).resolve()
        if not path.is_relative_to(Path(destination).resolve()) or _sha(path) != expected:
            raise ValueError('Source preview asset integrity check failed')
        result[url] = expected
    return result


def source_preview(scene_id, *, workspace=None):
    """Return only public preview metadata; callers authorize the recording first.

    `workspace` is the server-resolved owner hash, never a request-selected scope.
    Public and private asset namespaces remain separate even for identical inputs.
    """
    from .source_identity import registered_source, identity_digest
    from .source_preview import export_source_preview
    from .gpu_scheduler import resource_lock
    from .dataset_io.storage import mark_private
    row, spec = registered_source(scene_id)
    identity = _identity(spec, row)
    scope = hashlib.sha256(('workspace:'+workspace).encode()).hexdigest() if workspace else 'public'
    key = identity_digest(identity)
    cache = paths.workbench/'source-preview-cache'/scope/key
    destination = paths.store/'manipulation4d/source-previews'/scope/key
    with resource_lock('source-preview:'+scope+':'+key):
        if workspace:
            mark_private(destination.parent, workspace)
        if cache.is_dir():
            receipt = json.loads((cache/'receipt.json').read_text())
            if receipt.get('identity') != identity or receipt.get('identity_sha256') != key:
                raise ValueError('Source preview cache identity mismatch')
            if _files(cache/'source') != receipt['source_files']:
                raise ValueError('Source preview cache contents changed')
            if _preview_hashes(receipt['source_motion'],destination) != receipt['preview_files']:
                raise ValueError('Source preview asset inventory changed')
            current_row, current_spec = registered_source(scene_id)
            if _identity(current_spec,current_row) != identity:
                raise ValueError('Registered source changed while reading its preview')
            return dict(recording_id=scene_id, source_motion=receipt['source_motion'], cache_hit=True)
        cache.parent.mkdir(parents=True,exist_ok=True)
        with _BUILD_SLOTS, tempfile.TemporaryDirectory(prefix='.'+key[:16]+'.',dir=cache.parent) as temporary:
            tmp = Path(temporary)
            _render(spec,tmp/'source',identity['source']['source_robot'])
            preview = export_source_preview(tmp/'source',destination)
            current_row, current_spec = registered_source(scene_id)
            if _identity(current_spec,current_row) != identity:
                raise ValueError('Registered source or local decoder changed during preview preparation')
            receipt = dict(identity=identity, identity_sha256=key, source_motion=preview,
                           source_files=_files(tmp/'source'), preview_files=_preview_hashes(preview,destination))
            (tmp/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
            tmp.rename(cache)
        return dict(recording_id=scene_id, source_motion=preview, cache_hit=False)
