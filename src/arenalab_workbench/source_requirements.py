"""Source routing and explicit morphology choices shared by API and workers."""
from pathlib import Path
import json
import os
import numpy as np

POLICY = 'unified-provided-body-or-three-point-native-h-v1'
MISMATCH_RATIO = .8


def _robot_height(row):
    target_height = None
    try:
        import mujoco as mj
        from greenwich_pipelines.viewer.export import visible_geoms
        m = mj.MjModel.from_xml_path(str(row['xml'])); d = mj.MjData(m); mj.mj_forward(m,d)
        extrema=[]
        for g in visible_geoms(m):
            rot=d.geom_xmat[g].reshape(3,3); pos=d.geom_xpos[g]
            if m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
                mid=m.geom_dataid[g]; a=m.mesh_vertadr[mid]; n=m.mesh_vertnum[mid]
                z=(m.mesh_vert[a:a+n]@rot.T+pos)[:,2]; extrema += [float(z.min()),float(z.max())]
            else:
                # Conservative neutral bounding box is used only for a UI prompt.
                half=float(np.abs(rot[2])@m.geom_size[g]);extrema += [pos[2]-half,pos[2]+half]
        if extrema: target_height=float(max(extrema)-min(extrema))
    except (ImportError, ValueError, KeyError, OSError):
        return None
    return target_height


def motion_preflight(clip, robot, source_work=None):
    from arenalab_workbench.settings import paths
    spec = clip.get('motion_source') or clip.get('manifest', {}).get('motion_source')
    from .robot_motion_source import is_robot, source_robot
    robot_source = is_robot(spec)
    source_height = clip.get('source_height_m') or (spec or {}).get('height_m')
    if robot_source:
        source_height = _robot_height(source_robot(spec))
    reason = 'Local source robot neutral geometry' if robot_source else 'Declared source neutral height'
    if source_work and not robot_source:
        source_work = Path(source_work)
        mesh = source_work / 'source_mesh.npz'
        if mesh.is_file():
            with np.load(mesh, allow_pickle=False) as d:
                v = d['vertices_world_m']
                source_height = float(np.quantile(np.max(v[..., 2], axis=1)-np.min(v[..., 2], axis=1), .95))
                reason = 'Supplied body mesh vertical extent; posture-dependent estimate'
    if not source_height and not robot_source:
        family = (spec or {}).get('family', 'smplh')
        model = Path(os.environ.get('ARENA_'+family.upper()+'_MODEL', str(paths.egoallo/'model.npz') if family=='smplh' else ''))
        if model.is_file():
            with np.load(model, allow_pickle=False) as d:
                v = np.asarray(d['v_template'], float)
                source_height = float(np.ptp(v[:, 1]))
                reason = 'Configured human model neutral-height estimate; actual scale computed from body and robot limbs'
    target_height = _robot_height(robot.get('pipeline_row',robot))
    available = bool(source_height and target_height and np.isfinite([source_height,target_height]).all() and min(source_height,target_height)>0)
    ratio = target_height/source_height if available else None
    mismatch = bool(available and (ratio<MISMATCH_RATIO or ratio>1/MISMATCH_RATIO))
    morphology=dict(available=available,source_height_m=source_height,target_height_m=target_height,
                    scale_factor=ratio,large_mismatch=mismatch,threshold=MISMATCH_RATIO,
                    estimate=True,reason=reason,
                    scaling_method='8-limb uniform similarity about first pelvis floor projection',
                    scaling_scope='Source body, operation trajectories and source scene together; robot geometry unchanged')
    return dict(source_policy='supplied-robot' if robot_source else 'supplied-body' if spec else 'three-point-egoallo',source_policy_version=POLICY,
                morphology=morphology,scale_choice_required=mismatch,
                generated_human_model=None if spec else 'smplh',pipeline_version='resident-skeleton-surface-wrist-v3')


def resolve_scale(choice, preflight):
    if choice not in (None,'preserve','match_robot'):raise ValueError('Unknown trajectory scale mode')
    if choice is None and preflight.get('scale_choice_required'):
        raise ValueError('SCALE_CHOICE_REQUIRED: choose preserve or match_robot explicitly')
    if choice=='match_robot' and not preflight.get('morphology',{}).get('available'):
        raise ValueError('SCALE_ESTIMATE_UNAVAILABLE: source and robot morphology are required')
    return choice or 'preserve'


def _hdf_source_stream(row, require_hands=False):
    """Inspect dataset headers and window coverage without reading pose arrays."""
    source=Path(row.get('source',''))
    if not source.is_file() or source.suffix.lower() not in {'.h5','.hdf5'}:return False
    try:
        import h5py
        with h5py.File(source,'r') as data:
            required=['transforms/camera'];confidence=[]
            if require_hands:
                for side in ('left','right'):
                    names=[side+'Hand']+[side+'Thumb'+part for part in ('Knuckle','IntermediateBase','IntermediateTip','Tip')]
                    names += [side+finger+'Finger'+part for finger in ('Index','Middle','Ring','Little') for part in ('Knuckle','IntermediateBase','IntermediateTip','Tip')]
                    required += ['transforms/'+name for name in names]
                    confidence += ['confidences/'+name for name in names]
            if not all(name in data for name in required+confidence) or 'camera/intrinsic' not in data:return False
            start=int((row.get('source_window') or {}).get('start_frame',0))
            count=int(row.get('frames',data['transforms/camera'].shape[0]-start))
            if start<0 or count<1:return False
            stop=start+count
            if any(len(data[name].shape)!=3 or data[name].shape[1:]!=(4,4) or data[name].shape[0]<stop for name in required):return False
            if any(len(data[name].shape)!=1 or data[name].shape[0]<stop for name in confidence):return False
            shape=data['camera/intrinsic'].shape
            return shape==(3,3) # Existing contact/CV readers use one fixed calibration.
    except (ImportError,OSError,ValueError,TypeError,IndexError,AttributeError):return False


def observed_hand_stream(row):
    """Check the actual adapter input; an end-effector label is not evidence."""
    return _hdf_source_stream(row,require_hands=True)


def input_requirements(scene_id, robot_id=None, config_path=None):
    from arenalab_workbench.height_registry import default_id, variant, public_entry
    from arenalab_workbench.gripper_registry import sources
    from arenalab_workbench.settings import paths
    from arenalab_workbench.dataset_io.storage import entry as uploaded_entry
    resolved=default_id(scene_id) or scene_id; entry=public_entry(resolved)
    if entry is None:raise ValueError('Recording is not connected')
    gripper_sources=sources()
    hand=('robotiq-2f-140' if resolved in gripper_sources else entry.get('end_effector','none' if entry['dataset']=='Nymeria' else 'wuji'))
    result=dict(recording_id=resolved,hand_id=hand,requires_initial_height=entry['dataset']=='UMI' and not variant(resolved))
    upload=uploaded_entry(resolved)
    cfg=json.loads(Path(config_path or paths.viewer_config).read_text())
    custom=variant(resolved)
    row=dict(upload['source_row'] if upload else custom['source_row'] if custom and custom.get('source_row') else next((c for c in cfg['clips'] if c['id']==entry.get('source_catalog_id',resolved)),{}))
    if upload:
        row['manifest']=upload.get('manifest',{})
        if upload.get('motion_source'):row['motion_source']=upload['motion_source']
    if resolved in gripper_sources:
        source_work=Path(gripper_sources[resolved]['work'])
        metadata=json.loads((source_work/'source.json').read_text())
        if metadata.get('motion_source'):row['motion_source']=metadata['motion_source']
        camera_path=Path(metadata.get('head_video') or '')
        if not camera_path.is_absolute():camera_path=source_work/camera_path
        row.update(work=str(source_work),source=str(source_work/'source.npz'),video={'path':str(camera_path)} if camera_path.is_file() else None)
    else:
        row['has_observed_hands']=observed_hand_stream(row) if hand=='wuji' else False
    video=row.get('video') or {}
    row['camera_input_registered']=bool(video.get('path') and Path(video['path']).is_file())
    if row['camera_input_registered'] and Path(row.get('source','')).suffix.lower() in {'.h5','.hdf5'}:
        row['camera_input_registered']=_hdf_source_stream(row)
    # A supplied-body attachment alone has no camera poses for CV. Validated
    # uploads with observations include camera/intrinsics in their input bundle.
    if upload and upload.get('public',{}).get('uploaded') and upload.get('motion_source') and row['camera_input_registered']:
        observed=Path(row.get('observations_source',row.get('source','')))
        if observed.suffix=='.npz':
            try:
                with np.load(observed,allow_pickle=False) as data:
                    if 'camera' not in data or 'intrinsic' not in data:row['camera_input_registered']=False
                    else:
                        camera_shape=data['camera'].shape;intrinsic_shape=data['intrinsic'].shape
                        frames=int(row.get('frames',0))
                        row['camera_input_registered']=(frames>0 and camera_shape==(frames,4,4)
                            and intrinsic_shape==(3,3))
            except (OSError,ValueError,TypeError):row['camera_input_registered']=False
    from arenalab_workbench.output_modalities import source_capabilities
    result['output_capabilities']=source_capabilities({**entry,'end_effector':hand},row,gripper=resolved in gripper_sources)
    if robot_id:
        from arenalab_workbench.robot_registry import roster
        robot=roster(config_path).get(robot_id)
        if not robot:raise ValueError('Robot is unavailable')
        result.update(motion_preflight(row,robot))
    return result
