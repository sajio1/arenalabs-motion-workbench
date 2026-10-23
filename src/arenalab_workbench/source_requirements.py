"""Source routing and explicit morphology choices shared by API and workers."""
from pathlib import Path
import json
import os
import numpy as np

POLICY = 'unified-provided-body-or-three-point-native-h-v1'
MISMATCH_RATIO = .8


def motion_preflight(clip, robot, source_work=None):
    from arenalab_workbench.settings import paths
    spec = clip.get('motion_source') or clip.get('manifest', {}).get('motion_source')
    source_height = clip.get('source_height_m') or (spec or {}).get('height_m')
    reason = 'Declared source neutral height'
    if source_work:
        source_work = Path(source_work)
        mesh = source_work / 'source_mesh.npz'
        if mesh.is_file():
            with np.load(mesh, allow_pickle=False) as d:
                v = d['vertices_world_m']
                source_height = float(np.quantile(np.max(v[..., 2], axis=1)-np.min(v[..., 2], axis=1), .95))
                reason = 'Supplied body mesh vertical extent; posture-dependent estimate'
    if not source_height:
        family = (spec or {}).get('family', 'smplh')
        model = Path(os.environ.get('ARENA_'+family.upper()+'_MODEL', str(paths.egoallo/'model.npz') if family=='smplh' else ''))
        if model.is_file():
            with np.load(model, allow_pickle=False) as d:
                v = np.asarray(d['v_template'], float)
                source_height = float(np.ptp(v[:, 1]))
                reason = 'Configured human model neutral-height estimate; actual scale computed from body and robot limbs'
    target_height = None
    row = robot.get('pipeline_row', robot)
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
        reason += '; robot neutral geometry is unavailable'
    available = bool(source_height and target_height and np.isfinite([source_height,target_height]).all() and min(source_height,target_height)>0)
    ratio = target_height/source_height if available else None
    mismatch = bool(available and (ratio<MISMATCH_RATIO or ratio>1/MISMATCH_RATIO))
    morphology=dict(available=available,source_height_m=source_height,target_height_m=target_height,
                    scale_factor=ratio,large_mismatch=mismatch,threshold=MISMATCH_RATIO,
                    estimate=True,reason=reason,
                    scaling_method='8-limb uniform similarity about first pelvis floor projection',
                    scaling_scope='Source body, operation trajectories and source scene together; robot geometry unchanged')
    return dict(source_policy='supplied-body' if spec else 'three-point-egoallo',source_policy_version=POLICY,
                morphology=morphology,scale_choice_required=mismatch,
                generated_human_model=None if spec else 'smplh',pipeline_version='resident-skeleton-surface-wrist-v3')


def resolve_scale(choice, preflight):
    if choice not in (None,'preserve','match_robot'):raise ValueError('Unknown trajectory scale mode')
    if choice is None and preflight.get('scale_choice_required'):
        raise ValueError('SCALE_CHOICE_REQUIRED: choose preserve or match_robot explicitly')
    if choice=='match_robot' and not preflight.get('morphology',{}).get('available'):
        raise ValueError('SCALE_ESTIMATE_UNAVAILABLE: source and robot morphology are required')
    return choice or 'preserve'


def input_requirements(scene_id, robot_id=None, config_path=None):
    from arenalab_workbench.height_registry import default_id, variant, public_entry
    from arenalab_workbench.gripper_job import sources
    from arenalab_workbench.settings import paths
    from arenalab_workbench.dataset_io.storage import entry as uploaded_entry
    resolved=default_id(scene_id) or scene_id; entry=public_entry(resolved)
    if entry is None:raise ValueError('Recording is not connected')
    hand=('robotiq-2f-140' if resolved in sources() else entry.get('end_effector','none' if entry['dataset']=='Nymeria' else 'wuji'))
    result=dict(recording_id=resolved,hand_id=hand,requires_initial_height=entry['dataset']=='UMI' and not variant(resolved))
    if robot_id:
        from arenalab_workbench.robot_registry import roster
        robot=roster(config_path).get(robot_id)
        if not robot:raise ValueError('Robot is unavailable')
        upload=uploaded_entry(resolved)
        cfg=json.loads(Path(config_path or paths.viewer_config).read_text())
        row=upload['source_row'] if upload else next((c for c in cfg['clips'] if c['id']==entry.get('source_catalog_id',resolved)),{})
        if upload: row={**row, 'manifest':upload.get('manifest',{})}
        result.update(motion_preflight(row,robot))
    return result
