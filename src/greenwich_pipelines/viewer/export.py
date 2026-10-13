"""Export actual MuJoCo visual geoms without fitting or grounding trajectories."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation

C = np.array([[0.,1,0],[0,0,1],[1,0,0]])  # vendor Z-up -> descriptor Y-up, metres

def array_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False),encoding='utf-8')

def visible_geoms(model):
    import mujoco as mj
    candidates=[i for i in range(model.ngeom) if model.geom_bodyid[i]>0 and model.geom_rgba[i,3]>.01 and model.geom_type[i]!=mj.mjtGeom.mjGEOM_PLANE]
    # Prefer the model's visual mesh over its duplicate collision primitive.
    mesh_bodies={model.geom_bodyid[i] for i in candidates if model.geom_type[i]==mj.mjtGeom.mjGEOM_MESH}
    return [i for i in candidates if model.geom_type[i]==mj.mjtGeom.mjGEOM_MESH or model.geom_bodyid[i] not in mesh_bodies]

def mesh_scene(model,ids):
    import mujoco as mj, trimesh
    scene=trimesh.Scene()
    for i in ids:
        typ=model.geom_type[i];size=model.geom_size[i]
        if typ==mj.mjtGeom.mjGEOM_MESH:
            m=model.geom_dataid[i];a=model.mesh_vertadr[m];n=model.mesh_vertnum[m];fa=model.mesh_faceadr[m];fn=model.mesh_facenum[m]
            mesh=trimesh.Trimesh(model.mesh_vert[a:a+n].copy(),model.mesh_face[fa:fa+fn].copy(),process=False)
        elif typ==mj.mjtGeom.mjGEOM_BOX:mesh=trimesh.creation.box(extents=size*2)
        elif typ==mj.mjtGeom.mjGEOM_SPHERE:mesh=trimesh.creation.icosphere(subdivisions=2,radius=size[0])
        elif typ==mj.mjtGeom.mjGEOM_CAPSULE:mesh=trimesh.creation.capsule(radius=size[0],height=size[1]*2,count=[12,12])
        elif typ==mj.mjtGeom.mjGEOM_CYLINDER:mesh=trimesh.creation.cylinder(radius=size[0],height=size[1]*2,sections=20)
        elif typ==mj.mjtGeom.mjGEOM_ELLIPSOID:
            mesh=trimesh.creation.icosphere(subdivisions=2);mesh.vertices*=size
        else:continue
        rgba=np.clip(model.geom_rgba[i]*255,0,255).astype(np.uint8)
        mesh.visual.vertex_colors=np.tile(rgba,(len(mesh.vertices),1))
        scene.add_geometry(mesh,node_name=f'geom_{i}',geom_name=f'geom_{i}')
    return scene

def robot_model(row,folder):
    import mujoco as mj
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    model=mj.MjModel.from_xml_path(str(row['xml']));ids=visible_geoms(model)
    mesh_scene(model,ids).export(folder/'model.glb')
    data=mj.MjData(model);mj.mj_forward(model,data)
    p=data.geom_xpos[ids]@C.T;r=C@data.geom_xmat[ids].reshape(-1,3,3)
    quat=Rotation.from_matrix(r).as_quat()
    array_json(folder/'neutral.json',dict(geom_ids=ids,positions=p.tolist(),quaternions=quat.tolist(),body_positions=(data.xpos@C.T).tolist(),parents=model.body_parentid.tolist()))
    return model,ids

def motion_payload(row,path,pipeline):
    import mujoco as mj
    from greenwich_umi_proof.native_visual_fk import descriptor_visual_fk
    model=mj.MjModel.from_xml_path(str(row['xml']));ids=visible_geoms(model)
    with np.load(path,allow_pickle=False) as saved:d={k:saved[k].copy() for k in saved.files}
    if pipeline=='gento':
        data=mj.MjData(model);positions=[];rotations=[];bp=[]
        for q in d['qpos']:
            data.qpos[:]=q;mj.mj_forward(model,data)
            positions.append(data.geom_xpos[ids].copy());rotations.append(data.geom_xmat[ids].reshape(-1,3,3).copy());bp.append(data.xpos.copy())
        positions=np.asarray(positions)@C.T;rotations=C@np.asarray(rotations)
        skeleton=np.asarray(bp)@C.T;parents=model.body_parentid.tolist()
        targets=np.asarray(d['targets'])@C.T
    else:
        a=d['rot6d'][:,0];x=a[:,:3];x=x/np.linalg.norm(x,axis=-1,keepdims=True)
        y=a[:,3:]-np.sum(x*a[:,3:],axis=-1,keepdims=True)*x;y/=np.linalg.norm(y,axis=-1,keepdims=True)
        R=np.stack((x,y,np.cross(x,y)),-1)
        # Descriptor rotations use a changed basis on BOTH sides. Native FK
        # expects vendor coordinates, not a Y-up rotation on a Z-up mesh.
        positions,rotations=descriptor_visual_fk(row['xml'],row['body'],d['joint_names'].tolist(),d['q'],ids,d['root_t']@C/100,C.T@R@C)
        positions=positions@C.T;rotations=C@rotations
        skeleton=d.get('world_position_cm',d.get('joints_yup_cm'))/100
        if 'parents' in d:parents=d['parents'].tolist()
        else:
            from alphamotion.engine.descriptor import build_from_mjcf
            parents=build_from_mjcf(row['xml'],row['body'])[0].parents.tolist()
        targets=d.get('input_positions_yup_cm',d.get('source_positions_cm',skeleton*100))/100
    T=len(positions)
    if not all(np.isfinite(v).all() for v in (positions,rotations,skeleton,targets)):raise ValueError('Nonfinite playback')
    from .tracking import tracking_payload
    tracking=tracking_payload(row,d,pipeline,model)
    return dict(schema='greenwich.playback.v1',fps=30,frames=T,duration=(T-1)/30,unit='m',axis='y-up',geom_ids=ids,
                positions=positions.astype(np.float32).tolist(),quaternions=Rotation.from_matrix(rotations.reshape(-1,3,3)).as_quat().reshape(T,len(ids),4).astype(np.float32).tolist(),
                skeleton=skeleton.astype(np.float32).tolist(),parents=parents,targets=targets.astype(np.float32).tolist(),quality_status='not_evaluated',**({'tracking':tracking} if tracking else {}))


def join_playback_segments(segments):
    """Preserve independent window outputs in chronological playback order."""
    iterator=iter(segments)
    try:result=next(iterator)
    except StopIteration:raise ValueError('No motion segments')
    boundaries=[0];total=result['frames']
    for segment in iterator:
        for key in ('fps','geom_ids','parents','unit','axis'):
            if result.get(key)!=segment.get(key):raise ValueError('Incompatible playback segment '+key)
        boundaries.append(total)
        for key in ('positions','quaternions','skeleton','targets'):result[key].extend(segment[key])
        if bool(result.get('tracking'))!=bool(segment.get('tracking')):raise ValueError('Incompatible segment tracking')
        if result.get('tracking'):
            for key in ('roles','reference','unit','axis','thresholds'):
                if result['tracking'][key]!=segment['tracking'][key]:raise ValueError('Incompatible tracking '+key)
            for key in ('actual_positions','target_positions','position_error_mm','orientation_error_deg'):
                result['tracking'][key].extend(segment['tracking'][key])
        total+=segment['frames']
    result.update(frames=total,duration=(total-1)/result['fps'],window_boundaries=boundaries)
    return result
