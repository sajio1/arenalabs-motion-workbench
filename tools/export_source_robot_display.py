"""Export original robot FK for display; no retargeting, inference or pose fitting."""
from pathlib import Path
import hashlib
import json
import numpy as np


def export(source, row, robot_id, destination, store):
    import mujoco as mj
    from scipy.spatial.transform import Rotation
    from alphamotion.engine.descriptor import build_from_mjcf
    from greenwich_pipelines.viewer.export import visible_geoms, mesh_scene
    from arenalab_workbench.source_preview import _immutable_bytes
    source, destination, store = map(Path, (source, destination, store))
    destination.resolve().relative_to(store.resolve())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with np.load(source, allow_pickle=False) as d:
        q, roots, rot6d = d['q'], d['root_t'], d['rot6d']
        names = d['joint_names'].tolist(); expected = d['world_position_cm']/100
        fps = float(d['fps'])
    if fps != 30 or any(not np.isfinite(v).all() for v in (q, roots, rot6d, expected)):
        raise ValueError('Source must be finite, original 30 Hz robot states')
    spec, _, _, bindings, _ = build_from_mjcf(row['xml'], row['body'])
    model = mj.MjModel.from_xml_path(row['xml']); data = mj.MjData(model)
    geoms = visible_geoms(model); scene = mesh_scene(model, geoms)
    if len(scene.geometry) != len(geoms):raise ValueError('Unsupported source visual geometry')
    # Use contiguous geometry indices, as in the target robot playback format.
    import trimesh
    output = trimesh.Scene()
    for i, g in enumerate(geoms):output.add_geometry(scene.geometry[f'geom_{g}'],node_name=f'geom_{i}',geom_name=f'geom_{i}')
    glb = output.export(file_type='glb'); model_hash = hashlib.sha256(glb).hexdigest()
    folder = destination/(digest[:16]+'-'+model_hash[:16])
    _immutable_bytes(destination/'models'/model_hash/'model.glb',glb)
    C = np.array([[0.,1,0],[0,0,1],[1,0,0]])
    r = rot6d[:,0]; x = r[:,:3]/np.linalg.norm(r[:,:3],axis=1,keepdims=True)
    y = r[:,3:]-(x*r[:,3:]).sum(1,keepdims=True)*x; y/=np.linalg.norm(y,axis=1,keepdims=True)
    rotations = C.T@np.stack((x,y,np.cross(x,y)),axis=-1)@C
    body_ids = [model.body(name).id for name in names]
    positions=[]; orientations=[]; max_error=0.
    joint_values={name:q[:,names.index(spec.joint_names[j]),axis] for j,items in enumerate(bindings) for axis,name in enumerate(items)}
    hinges={model.joint(j).name for j in range(model.njnt) if model.jnt_type[j]==mj.mjtJoint.mjJNT_HINGE}
    if set(joint_values)!=hinges:raise ValueError('Source joint coverage does not match the local model')
    for f in range(len(q)):
        data.qpos[:]=model.qpos0;data.qpos[:3]=roots[f]/100@C
        data.qpos[3:7]=Rotation.from_matrix(rotations[f]).as_quat()[[3,0,1,2]]
        for name,values in joint_values.items():data.qpos[model.jnt_qposadr[model.joint(name).id]]=values[f]
        mj.mj_forward(model,data)
        max_error=max(max_error,float(np.max(np.linalg.norm(data.xpos[body_ids]@C.T-expected[f],axis=-1))))
        positions.append(data.geom_xpos[geoms].copy());orientations.append(data.geom_xmat[geoms].reshape(-1,3,3).copy())
    if max_error>1e-6:raise ValueError(f'Original-source FK mismatch: {max_error} m')
    motion=dict(fps=fps,frames=len(q),positions=np.asarray(positions).tolist(),quaternions=Rotation.from_matrix(np.asarray(orientations).reshape(-1,3,3)).as_quat().reshape(len(q),len(geoms),4).tolist())
    payload=(json.dumps(motion,separators=(',',':'),allow_nan=False)+'\n').encode()
    playback_hash=_immutable_bytes(folder/'playback.json',payload)
    url=lambda p:'/assets/'+p.resolve().relative_to(store.resolve()).as_posix()
    result=dict(source_family='robot',source_robot_id=robot_id,world_up='z',frames=len(q),fps=fps,
                model_url=url(destination/'models'/model_hash/'model.glb'),playback_url=url(folder/'playback.json'),
                source_sha256=digest,model_sha256=model_hash,playback_sha256=playback_hash,source_fk_max_error_m=max_error,
                source_trajectory_modified=False,retargeted=False)
    _immutable_bytes(folder/'manifest.json',(json.dumps(result,indent=2)+'\n').encode())
    return result
