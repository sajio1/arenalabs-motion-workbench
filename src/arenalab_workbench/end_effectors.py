"""Shared Wuji visual attachment and physical operation-point FK.

Robot identity never selects solver behavior. Fixed axes come from robot metadata;
the visual flange origin is inferred from proximal native hand geometry and labeled
as an estimate until hardware calibration is supplied.
"""
from pathlib import Path
import os,json,xml.etree.ElementTree as ET
import numpy as np,mujoco as mj
from scipy.spatial.transform import Rotation
from alphamotion.viz.kinematics import visual_mesh_geom_ids
from greenwich_umi_proof.interactive_viewer import _sample_greenwich_head

os.environ['WUJI_TELEMETRY_DISABLE']='1'

def hand_angles(source_path):
    from wuji_sdk import RetargetSession,HandModel,Handedness
    source=np.load(source_path)['points'];angles=[]
    for si,side in enumerate([Handedness.Left,Handedness.Right]):
        session=RetargetSession.for_hand(HandModel.WujiHand,side=side)
        angles.append(np.stack([np.asarray(session.step(np.asarray(p-p[0],np.float32))) for p in source[:,si]]))
    return np.stack(angles,1)

def mount_frame(model,data,row,side,ids):
    bid=model.body(row['endpoints'][side+'_wrist']).id
    frame=row['palm_frames'][side+'_wrist'];distal=np.asarray(frame['distal'],float);distal/=np.linalg.norm(distal)
    normal=np.asarray(frame['normal'],float);normal-=distal*np.dot(normal,distal);normal/=np.linalg.norm(normal)
    rotation=np.stack([normal,np.cross(distal,normal),distal],1)
    candidates=[]
    for geom in ids:
        if model.geom_type[geom]!=mj.mjtGeom.mjGEOM_MESH:continue
        owner=model.body(int(model.geom_bodyid[geom])).name.lower();name=model.mesh(int(model.geom_dataid[geom])).name.lower();text=owner+' '+name
        if side not in text or not any(word in text for word in ['hand','palm']):continue
        if any(word in text for word in ['finger','thumb']):continue
        mid=model.geom_dataid[geom];a=model.mesh_vertadr[mid];n=model.mesh_vertnum[mid]
        world=model.mesh_vert[a:a+n]@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom]
        local=(world-data.xpos[bid])@data.xmat[bid].reshape(3,3)
        candidates.extend(local@distal)
    if 'hand_mounts' in row and side in row['hand_mounts']:
        origin=np.asarray(row['hand_mounts'][side]['origin_m'],float);provenance='explicit physical mounting metadata'
    elif candidates:
        origin=distal*float(np.min(candidates));provenance='estimated proximal native palm-mesh extent along declared distal axis'
    else:
        origin=np.zeros(3);provenance='wrist-link origin; native palm flange extent unavailable'
    return bid,origin,rotation,provenance

def operation_local_points(row,angles,wuji_folder,frames):
    """Original Wuji FK/TCP component without requiring a body-motion result.

    These wrist-local Z-up metre offsets are identical to export_wuji's offsets;
    body refinement can therefore run first without a legacy body solve.
    """
    if angles is None:return np.zeros((frames,2,1,3))
    if np.asarray(angles).shape!=(frames,2,20):raise ValueError('Expected Wuji angles [frames,2,20]')
    model=mj.MjModel.from_xml_path(row['xml']);md=mj.MjData(model);mj.mj_forward(model,md)
    ids=list(visual_mesh_geom_ids(model));result=[]
    for si,side in enumerate(['left','right']):
        _,offset,mount,_=mount_frame(model,md,row,side,ids)
        path=Path(wuji_folder)/'urdf'/f'{side}.urdf'
        hand=mj.MjModel.from_xml_path(str(path));hd=mj.MjData(hand);tree=ET.parse(path)
        tip_offsets=[np.fromstring(tree.find(f"joint[@name='{side}_finger{i}_tip_fixed']/origin").attrib['xyz'],sep=' ') for i in range(1,6)]
        side_points=[]
        for t in range(frames):
            for finger in range(5):
                for joint in range(4):
                    jid=hand.joint(f'{side}_finger{finger+1}_joint{joint+1}').id
                    hd.qpos[hand.jnt_qposadr[jid]]=angles[t,si,finger*4+joint]
            mj.mj_forward(hand,hd)
            palm=np.mean([hd.xpos[hand.body(f'{side}_finger{i}_link1').id] for i in range(2,6)],0)
            tips=[]
            for i in range(1,6):
                bid=hand.body(f'{side}_finger{i}_link4').id
                tips.append(hd.xpos[bid]+hd.xmat[bid].reshape(3,3)@tip_offsets[i-1])
            side_points.append(np.r_[palm[None],tips,np.mean(tips[:2],0)[None]]@mount.T+offset)
        result.append(side_points)
    return np.stack(result,1)

def native_world_visuals(model,qpos_path):
    """Exact native qpos world FK, including declared non-mesh visual primitives."""
    import trimesh
    ids=[g for g in range(model.ngeom) if model.geom_bodyid[g]>0 and model.geom_group[g] in (1,2) and model.geom_rgba[g,3]>0]
    if not ids:ids=[g for g in range(model.ngeom) if model.geom_bodyid[g]>0 and not model.geom_contype[g] and not model.geom_conaffinity[g] and model.geom_rgba[g,3]>0]
    if not ids:
        from greenwich_pipelines.viewer.export import visible_geoms
        ids=visible_geoms(model)
    if not ids:raise ValueError('Native visual metadata is required')
    with np.load(qpos_path) as saved:q=np.asarray(saved['qpos'],float)
    if q.ndim!=2 or q.shape[1]!=model.nq or not np.isfinite(q).all():raise ValueError('Native visual qpos mismatch')
    data=mj.MjData(model);positions=[];rotations=[]
    for pose in q:
        data.qpos[:]=pose;mj.mj_forward(model,data)
        positions.append(data.geom_xpos[ids].copy());rotations.append(data.geom_xmat[ids].reshape(-1,3,3).copy())
    meshes=[]
    for g in ids:
        kind=model.geom_type[g];size=model.geom_size[g]
        if kind==mj.mjtGeom.mjGEOM_MESH:
            mid=int(model.geom_dataid[g]);a,n=int(model.mesh_vertadr[mid]),int(model.mesh_vertnum[mid]);b,f=int(model.mesh_faceadr[mid]),int(model.mesh_facenum[mid])
            vertices=model.mesh_vert[a:a+n].copy();faces=model.mesh_face[b:b+f].copy()
        else:
            if kind==mj.mjtGeom.mjGEOM_CYLINDER:mesh=trimesh.creation.cylinder(radius=size[0],height=2*size[1],sections=64)
            elif kind==mj.mjtGeom.mjGEOM_CAPSULE:mesh=trimesh.creation.capsule(radius=size[0],height=2*size[1],count=[24,24])
            elif kind==mj.mjtGeom.mjGEOM_BOX:mesh=trimesh.creation.box(extents=2*size)
            elif kind==mj.mjtGeom.mjGEOM_SPHERE:mesh=trimesh.creation.icosphere(subdivisions=3,radius=size[0])
            elif kind==mj.mjtGeom.mjGEOM_ELLIPSOID:
                mesh=trimesh.creation.icosphere(subdivisions=3,radius=1);mesh.vertices*=size
            else:raise ValueError(f'Native visual primitive type {kind} requires explicit tessellation')
            vertices=mesh.vertices;faces=mesh.faces
        meshes.append((vertices,faces,model.geom_rgba[g].copy()))
    return ids,np.asarray(positions),np.asarray(rotations),meshes

def export_wuji(row,motion_path,angles,wuji_folder,output,native_qpos_path=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    model=mj.MjModel.from_xml_path(row['xml']);md=mj.MjData(model);mj.mj_forward(model,md);ids=list(visual_mesh_geom_ids(model))
    if native_qpos_path is None:
        pos,quat,meshes,*_=_sample_greenwich_head(Path(motion_path),row['xml'],row['body'],rotation_key='rot6d',world_position_key='world_position_cm',floor_y_cm=0.)
    else:
        ids,pos,native_rot,meshes=native_world_visuals(model,native_qpos_path)
    C=np.array([[0,1,0],[0,0,1],[1,0,0]],float)
    if native_qpos_path is None:
        origin=np.load(motion_path)['root_t'][0]/100@C;pos=pos.astype(float);pos[...,:2]+=origin[:2]
        rot=Rotation.from_quat(quat.reshape(-1,4)[:,[1,2,3,0]]).as_matrix().reshape(len(pos),-1,3,3)
    else:rot=native_rot
    n=len(pos);allpos=[];allrot=[];allmesh=[];removed=[];mounts=[];operation_local=[];retained=[]
    for k,g in enumerate(ids):
        meshname=model.mesh(int(model.geom_dataid[g])).name if model.geom_type[g]==mj.mjtGeom.mjGEOM_MESH else ''
        text=(model.body(int(model.geom_bodyid[g])).name+' '+meshname).lower()
        if (native_qpos_path is None or angles is not None) and any(word in text for word in ['hand','palm','finger','thumb','dummy']):removed.append(text);continue
        allpos.append(pos[:,k]);allrot.append(rot[:,k]);allmesh.append(meshes[k]);retained.append(g)
    for si,side in enumerate(['left','right']):
        if angles is None:
            operation_local.append(np.zeros((n,1,3)))
            continue
        bid,offset,mount,provenance=mount_frame(model,md,row,side,ids)
        geom=next(g for g in ids if model.geom_bodyid[g]==bid);k=ids.index(geom)
        localrot=Rotation.from_quat(model.geom_quat[geom][[1,2,3,0]]).as_matrix()
        wristrot=rot[:,k]@localrot.T;wristpos=pos[:,k]-np.einsum('tij,j->ti',wristrot,model.geom_pos[geom])
        handpos=wristpos+np.einsum('tij,j->ti',wristrot,offset);handrot=wristrot@mount
        path=Path(wuji_folder)/'urdf'/f'{side}.urdf';hand=mj.MjModel.from_xml_path(str(path));hd=mj.MjData(hand);hids=list(visual_mesh_geom_ids(hand));tree=ET.parse(path)
        tip_offsets=[np.fromstring(tree.find(f"joint[@name='{side}_finger{i}_tip_fixed']/origin").attrib['xyz'],sep=' ') for i in range(1,6)]
        positions=[];rotations=[];localpoints=[]
        for t in range(n):
            for finger in range(5):
                for joint in range(4):
                    jid=hand.joint(f'{side}_finger{finger+1}_joint{joint+1}').id
                    hd.qpos[hand.jnt_qposadr[jid]]=angles[t,si,finger*4+joint]
            mj.mj_forward(hand,hd)
            positions.append(handpos[t]+hd.geom_xpos[hids]@handrot[t].T);rotations.append(handrot[t]@hd.geom_xmat[hids].reshape(-1,3,3))
            palm=np.mean([hd.xpos[hand.body(f'{side}_finger{i}_link1').id] for i in range(2,6)],0)
            tips=[]
            for i in range(1,6):
                finger=hand.body(f'{side}_finger{i}_link4').id;tips.append(hd.xpos[finger]+hd.xmat[finger].reshape(3,3)@tip_offsets[i-1])
            localpoints.append(np.r_[palm[None],tips,np.mean(tips[:2],0)[None]]@mount.T+offset)
        operation_local.append(localpoints);positions=np.asarray(positions);rotations=np.asarray(rotations)
        for k,g in enumerate(hids):
            mid=hand.geom_dataid[g];a=hand.mesh_vertadr[mid];b=hand.mesh_vertnum[mid];c=hand.mesh_faceadr[mid];e=hand.mesh_facenum[mid]
            allmesh.append((hand.mesh_vert[a:a+b].copy(),hand.mesh_face[c:c+e].copy(),np.array([.60,.72,.64,1.])))
            allpos.append(positions[:,k]);allrot.append(rotations[:,k])
        mounts.append(dict(side=side,parent=row['endpoints'][side+'_wrist'],origin_m=offset.tolist(),rotation=mount.tolist(),provenance=provenance,hardware_calibrated=False))
    payload=dict(positions=np.stack(allpos,1),rotations=np.stack(allrot,1),frames=n)
    if native_qpos_path is not None:payload.update(native_body_geom_ids=np.asarray(retained),native_qpos_path=str(Path(native_qpos_path).resolve()),native_world_fk=True)
    for i,(v,f,c) in enumerate(allmesh):payload[f'v{i}']=v;payload[f'f{i}']=f;payload[f'c{i}']=c
    np.savez_compressed(output/'robot-render.npz',**payload)
    if angles is not None:
        np.savez_compressed(output/'wuji_angles.npz',angles=angles,fps=30,provenance='EgoDex original 21-point trajectories directly mapped to Wuji')
    (output/'hand-mounts.json').write_text(json.dumps(mounts,indent=2));(output/'dummy-removal.json').write_text(json.dumps(dict(removed_meshes=removed),indent=2))
    return np.stack(operation_local,1)
