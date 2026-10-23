"""Selected-output orchestration; numerical kernels are owned by the motion backend."""
import copy
import hashlib
import json
import os
from pathlib import Path

from arenalab_workbench.gpu_scheduler import write, merge_scene
from arenalab_workbench.output_modalities import resolve_outputs, plan_outputs

STAGES = {'trajectory': 'motion', 'point_cloud': 'scene', 'contact': 'contact', 'inpaint_fov': 'fov'}


def selection_for(scene_id, robot_id):
    from arenalab_workbench.source_requirements import input_requirements
    raw = os.environ.get('ARENA_OUTPUT_MODALITIES')
    requested = None if raw is None else json.loads(raw)
    capabilities = input_requirements(scene_id, robot_id)['output_capabilities']
    selection = resolve_outputs(requested, capabilities)
    plan = plan_outputs(selection['effective_outputs'], os.environ.get('ARENA_TRAJECTORY_SCALE', 'preserve'))
    return selection, plan


def filtered_scene(scene, selection, ready):
    """Publish selected artifacts only, keeping the worker scene intact for reuse."""
    result = copy.deepcopy(scene)
    selected = selection['effective_outputs']
    completed = [name for name in selected if ready.get(STAGES[name])]
    if 'trajectory' not in completed:
        result['robots'] = []
        result.pop('source_body', None)
        result.pop('source_robot', None)
        result.pop('hand_motion_audit_url', None)
    if 'point_cloud' not in completed:
        for key in ('cloud', 'segmented_instances', 'source_camera'):
            result.pop(key, None)
        result['objects'] = []
        result['geometry'] = {key: value for key, value in result.get('geometry', {}).items()
                              if key in ('ground', 'source_trajectory_edited')}
        result['geometry']['status'] = 'pending' if 'point_cloud' in selected else 'not_requested'
    if 'inpaint_fov' not in completed:
        for key in ('patch_videos','patch_unavailable','patch_kinds','patch_depth_policy'):
            result.pop(key, None)
    if 'contact' not in completed:
        result['tactile'] = {'status': 'unavailable', 'reason': 'Contact generation pending' if 'contact' in selected else 'Contact was not requested'}
    result['output_modalities'] = selected
    result['completed_outputs'] = completed
    result['output_status'] = {name: 'completed' if name in completed else 'pending' for name in selected}
    result.setdefault('generation_status',{}).update(status='completed' if len(completed)==len(selected) else 'running',
                                                    output_modalities=selected,completed_outputs=completed)
    return result


def publish_scene(worker_path, public_path, selection, ready):
    scene = json.loads(Path(worker_path).read_text())
    write(public_path, filtered_scene(scene, selection, ready))


def apply_scene_scale(source_work, robot, scene_path, gripper=False):
    """Cloud-only morphology matching uses the existing CPU fit, never retargeting."""
    import numpy as np
    from arenalab_motion.robot_metadata import ensure_robot_metadata
    from arenalab_motion.scaling import fit_scale
    from arenalab_workbench.motion_backend import normalized_source
    from arenalab_workbench.source_display import source_scene_metadata, GRIPPER_WORLD_TO_DISPLAY, MANIPULATION_WORLD_TO_DISPLAY
    normalized = normalized_source(source_work)
    with np.load(normalized / 'native_source.npz', allow_pickle=False) as data:
        source = {key: data[key] for key in data.files}
    row = ensure_robot_metadata(robot['pipeline_row'])
    scaling = fit_scale(row, source)
    numerical = dict(trajectory_scale='match_robot', scale_transform=dict(
        scale=scaling['scale'], anchor_xyz_m=scaling['anchor_world_m'], world_up='z'))
    metadata = source_scene_metadata(None, numerical, GRIPPER_WORLD_TO_DISPLAY if gripper else MANIPULATION_WORLD_TO_DISPLAY)
    merge_scene(scene_path, lambda scene: scene.update(metadata))
    return scaling


def native_cache_valid(destination, robot_id, source_work=None, source_sha256=None):
    """Validate immutable native outputs before skipping any numerical worker."""
    from arenalab_workbench.motion_backend import cache_matches, sha
    folder = Path(destination) / ('robot-' + robot_id)
    try:
        report = json.loads((folder / 'validation.json').read_text())
        if not cache_matches(report):
            return False
        if not all((folder / name).is_file() for name in ('model.glb', 'playback.json')):
            return False
        output = Path(report['native_output_dir'])
        if not all((output / name).is_file() for name in ('motion.npz', 'gpu_qpos.npz', 'report.json')):
            return False
        priors = report.get('raw_sha256')
        if not priors or not all(Path(name).is_file() and sha(name) == digest for name, digest in priors.items()):
            return False
        if report.get('native_qpos_sha256') and sha(output/'gpu_qpos.npz') != report['native_qpos_sha256']:
            return False
        if source_sha256 is not None and report.get('source_sha256') != source_sha256:
            return False
        if source_work is not None:
            from arenalab_workbench.motion_backend import normalized_source, SOURCE_FILES
            normalized = normalized_source(source_work)
            hashes = {name: sha(normalized / name) for name in SOURCE_FILES if (normalized / name).is_file()}
            recorded = report.get('source_hashes')
            if recorded is None:
                # Gripper exports keep the immutable request beside the numerical report.
                request = json.loads((output/'report.json').resolve().with_name('request.json').read_text())
                recorded = request['source_sha256']
            if recorded != hashes:
                return False
        scene = json.loads((Path(destination) / 'scene.json').read_text())
        return any(robot.get('id') == robot_id for robot in scene.get('robots', []))
    except (OSError, KeyError, ValueError):
        return False


def render_key(destination, robot_id):
    destination = Path(destination)
    scene = json.loads((destination / 'scene.json').read_text())
    geometry = dict(geometry=scene.get('geometry', {}), source_transform=scene.get('source_transform'),
                    source_world_to_display=scene.get('source_world_to_display'))
    return hashlib.sha256((destination / ('robot-' + robot_id) / 'playback.json').read_bytes()
                          + json.dumps(geometry, sort_keys=True).encode()).hexdigest()


def prepare_contact_source(public, row, root):
    """Stage only the existing observed hand stream, without any body/model assets."""
    import h5py
    import numpy as np
    from arenalab_workbench.motion_backend import sha
    from arenalab_workbench.gpu_scheduler import resource_lock
    source = Path(row['source'])
    video = row['video']
    identity = dict(policy='source-observed-hand-contact-v1', source_sha256=sha(source),
                    video_sha256=sha(video['path']), video=video, window=row.get('source_window'),
                    frames=int(row['frames']))
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output = Path(root) / key
    with resource_lock('contact-observations:' + key):
        receipt = output / 'observations-ready.json'
        if receipt.is_file():
            saved = json.loads(receipt.read_text())
            if saved['identity'] != identity or not all(sha(output/name) == digest for name,digest in saved['sha256'].items()):
                raise ValueError('Immutable contact observations changed')
            return output
        output.mkdir(parents=True, exist_ok=True)
        n = identity['frames']; start = int((row.get('source_window') or {}).get('start_frame', 0))
        C = np.array([[1.,0,0],[0,0,-1],[0,1,0]])
        points=[]; confidence=[]
        with h5py.File(source, 'r') as data:
            for side in ('left','right'):
                names=[side+'Hand']+[side+'Thumb'+part for part in ('Knuckle','IntermediateBase','IntermediateTip','Tip')]
                names += [side+finger+'Finger'+part for finger in ('Index','Middle','Ring','Little') for part in ('Knuckle','IntermediateBase','IntermediateTip','Tip')]
                points.append(np.stack([data['transforms/'+name][start:start+n,:3,3] for name in names], 1) @ C.T)
                confidence.append(np.stack([data['confidences/'+name][start:start+n] for name in names], 1))
        points=np.stack(points,1); confidence=np.stack(confidence,1)
        if points.shape != (n,2,21,3) or confidence.shape != (n,2,21) or not np.isfinite(points).all() or not np.isfinite(confidence).all():
            raise ValueError('Contact requires finite observed 21-landmark hands on the selected clock')
        np.savez_compressed(output/'source_hands.npz',points=points,confidence=confidence,source_to_zup=C)
        write(output/'input_manifest.json',dict(dataset=public['dataset'],clip=public['id'],source=str(source),
              source_window=row.get('source_window'),frames=n,fps=30,video=video,
              source_sha256=identity['source_sha256'],source_trajectory_edited=False,
              fingers='Original observed 21 landmarks per hand; body inference was not used'))
        write(receipt,dict(identity=identity,sha256={name:sha(output/name) for name in ('source_hands.npz','input_manifest.json')}))
    return output


def require_point_cloud(scene):
    if not scene.get('cloud'):
        raise ValueError('Requested point cloud was not produced: ' + str(scene.get('geometry',{}).get('reason','missing cloud artifact')))


def require_fov(scene, robot_id):
    reason = scene.get('patch_unavailable', {}).get(robot_id)
    if reason or scene.get('patch_kinds', {}).get(robot_id) == 'native_preview' or not scene.get('patch_videos', {}).get(robot_id):
        raise ValueError('Requested inpaint/FOV was not produced: ' + str(reason or 'no source-camera composite artifact'))


def publish_failure(public_path, selection, ready, error):
    def update(scene):
        completed = [name for name in selection['effective_outputs'] if ready.get(STAGES[name])]
        scene['output_status'] = {name: 'completed' if name in completed else 'failed' for name in selection['effective_outputs']}
        scene.setdefault('generation_status',{}).update(status='failed',reason=str(error),completed_outputs=completed)
    merge_scene(public_path,update)
