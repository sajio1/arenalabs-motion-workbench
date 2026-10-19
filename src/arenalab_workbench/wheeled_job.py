"""Shared mobile backend for observed wrists, palms, and gripper centers."""
from arenalab_workbench.settings import paths
from pathlib import Path
import json,sys
import numpy as np,mujoco as mj,trimesh
from scipy.spatial.transform import Rotation as R
from alphamotion.viz.kinematics import visual_mesh_geom_ids
from greenwich_umi_proof.palm_frames import palm_frame
from arenalab_workbench.end_effectors import hand_angles
from arenalab_workbench.gripper_export import tool_fk
from arenalab_workbench.gripper_retarget import GRIPPER
from arenalab_workbench.scene_job import save
from arenalab_workbench.wheeled_body import solve,POLICY

C=np.array([[0.,1,0],[0,0,1],[1,0,0]])

def mesh_geometry(model,ids):
    output=[]
    for g in ids:
        mid=model.geom_dataid[g];a=model.mesh_vertadr[mid];b=model.mesh_vertnum[mid];c=model.mesh_faceadr[mid];d=model.mesh_facenum[mid]
        mesh=trimesh.Trimesh(model.mesh_vert[a:a+b],model.mesh_face[c:c+d],process=False)
        mesh.visual.vertex_colors=[180,190,182,255];output.append(mesh)
    return output

def run(robot,bodywork,dest,hand,gripper_work=None):
    bodywork=Path(bodywork);dest=Path(dest);rid=robot['id'];row=robot['pipeline_row'];out=bodywork/'mobile'/POLICY/rid;out.mkdir(parents=True,exist_ok=True)
    meta=json.loads(Path(row['metadata']).read_text());model=mj.MjModel.from_xml_path(row['xml'])
    # Native solver is Z-up. Match the existing source adapter's DISPLAY basis,
    # not AlphaMotion's internal canonical basis.
    C=__import__('arenalab_workbench.gripper_source', fromlist=['*']).C if gripper_work else np.array([[1.,0,0],[0,0,1],[0,-1,0]])
    obs=np.load(bodywork/'observations.npz');n=len(obs['wrist_position']);targets=np.load(bodywork/'operation-accuracy/source-targets.npz')['points'][:,:,0]
    frames=[next(f for f in meta['end_effector_frames'] if f['link']==w) for w in meta['wrists']]
    semantic_local=np.array([f['rotation'] for f in frames]);mounts=semantic_local[:,:,[2,1,0]].copy();mounts[:,:,1]*=-1
    origins=np.array([f['origin_m'] for f in frames]);ids=[model.body(w).id for w in meta['wrists']]
    hand_meshes=[];hand_positions=[];hand_rotations=[];joint_names=[];joint_values=[]
    if hand=='none':
        offsets=np.zeros((n,2,3));desired=palm_frame(obs['palm_forward'],obs['palm_normal'])@semantic_local.transpose(0,2,1)
    elif hand=='robotiq-2f-140':
        src=np.load(Path(gripper_work)/'source.npz');desc=json.loads((GRIPPER/'description.json').read_text());lut=desc['width_lookup'];width=np.array([x['width'] for x in lut]);q=np.array([x['q'] for x in lut]);values=src['width_m']
        if np.any(values<0) or np.any(values>width.max()+1e-6):raise ValueError('Source width exceeds target gripper range')
        angles=np.interp(values,width[::-1],q[::-1]);offsets=np.empty((n,2,3));desired=src['tool_R_zup']@mounts.transpose(0,2,1)
        visuals=[]
        for node in desc['nodes']:
            for v in node['visuals']:
                mesh=trimesh.load(GRIPPER/v['mesh'],force='mesh') if 'mesh' in v else trimesh.creation.box(v['box'])
                mesh.apply_scale(v.get('scale',[1,1,1]));T=np.eye(4);T[:3,:3]=R.from_euler('xyz',v['rpy']).as_matrix();T[:3,3]=v['xyz'];mesh.apply_transform(T);mesh.visual.vertex_colors=(np.array(v['color'])*255).astype('u1');visuals.append((node['name'],mesh))
        for side in range(2):
            pp=[];rr=[]
            for t in range(n):
                poses=tool_fk(desc,angles[t,side]);pads=[]
                for jaw in ['left','right']:
                    T=poses[jaw+'_inner_finger_pad'];p=T[:3,3].copy();normal=T[:3,2];p+=normal*.00375*(-np.sign(normal[1]*p[1]));pads.append(p)
                from arenalab_workbench.gripper_reference import housing_reference
                reference,_=housing_reference(GRIPPER,desc)
                offsets[t,side]=origins[side]+mounts[side]@reference
                pp.append([origins[side]+mounts[side]@poses[name][:3,3] for name,_ in visuals]);rr.append([mounts[side]@poses[name][:3,:3] for name,_ in visuals])
            hand_meshes.append([m.copy() for _,m in visuals]);hand_positions.append(np.array(pp));hand_rotations.append(np.array(rr))
        joint_names=['left_gripper_finger_joint','right_gripper_finger_joint'];joint_values=angles.tolist()
    elif hand=='wuji':
        angles=hand_angles(bodywork/'source_hands.npz');offsets=np.empty((n,2,3));desired=palm_frame(obs['palm_forward'],obs['palm_normal'])@semantic_local.transpose(0,2,1)
        for si,side in enumerate(['left','right']):
            handmodel=mj.MjModel.from_xml_path(str(bodywork/'wuji'/'urdf'/f'{side}.urdf'));data=mj.MjData(handmodel);geoms=list(visual_mesh_geom_ids(handmodel));pp=[];rr=[]
            for t in range(n):
                for finger in range(5):
                    for joint in range(4):data.qpos[handmodel.jnt_qposadr[handmodel.joint(f'{side}_finger{finger+1}_joint{joint+1}').id]]=angles[t,si,finger*4+joint]
                mj.mj_forward(handmodel,data);palm=np.mean([data.xpos[handmodel.body(f'{side}_finger{i}_link1').id] for i in range(2,6)],0)
                offsets[t,si]=origins[si]+mounts[si]@palm;pp.append(origins[si]+data.geom_xpos[geoms]@mounts[si].T);rr.append(mounts[si]@data.geom_xmat[geoms].reshape(-1,3,3))
            hand_meshes.append(mesh_geometry(handmodel,geoms));hand_positions.append(np.array(pp));hand_rotations.append(np.array(rr))
            joint_names.extend(f'{side}_finger{f+1}_joint{j+1}' for f in range(5) for j in range(4))
        joint_values=angles.reshape(n,40).tolist()
    else:raise ValueError('Unmapped end effector')
    result=solve(row,bodywork,out,targets,desired,offsets);motion=np.load(result)
    native=[g for g in visual_mesh_geom_ids(model) if not any(word in (model.body(int(model.geom_bodyid[g])).name+' '+model.mesh(int(model.geom_dataid[g])).name).lower() for word in ['dummy','hand','finger','palm','thumb'])]
    meshes=mesh_geometry(model,native);allpos=[];allrot=[];actual=[];disagreement=[]
    for t in range(n):
        data=mj.MjData(model);data.qpos[:]=motion['qpos'][t];mj.mj_forward(model,data)
        pp=list(data.geom_xpos[native]);rr=list(data.geom_xmat[native].reshape(-1,3,3));aa=[]
        for si,b in enumerate(ids):
            rot=data.xmat[b].reshape(3,3);pos=data.xpos[b];aa.append(pos+rot@offsets[t,si])
            if hand!='none':pp.extend(pos+hand_positions[si][t]@rot.T);rr.extend(rot@hand_rotations[si][t])
        actual.append(aa);allpos.append(pp);allrot.append(rr);disagreement.append(np.max(np.abs(data.xpos[1:]-motion['positions'][t,1:])))
    if max(disagreement)>1e-6:raise ValueError('Independent native mobile FK mismatch')
    for m in hand_meshes:meshes.extend(m)
    folder=dest/('robot-'+rid);folder.mkdir(parents=True,exist_ok=True);scene=trimesh.Scene()
    for i,mesh in enumerate(meshes):scene.add_geometry(mesh,node_name=f'geom_{i}',geom_name=f'geom_{i}')
    scene.export(folder/'model.glb');actual=np.array(actual);error=np.linalg.norm(actual-targets,axis=-1)*1000
    def ancestors(b):
        chain=[]
        while b>0:chain.append(b);b=int(model.body_parentid[b])
        return chain+[0]
    paths=[ancestors(b) for b in ids];common=next(b for b in paths[0] if b in paths[1]);branches=[p[p.index(common)-1] for p in paths]
    selected=[i for i,gid in enumerate(native) if any(b in ancestors(int(model.geom_bodyid[gid])) for b in branches)]+list(range(len(native),len(meshes)))
    save(folder/'render-layout.json',dict(arm_and_gripper_geoms=selected))
    names=[model.joint(j).name for j in range(1,model.njnt)]+joint_names
    values=motion['qpos'][:,7:] if hand=='none' else np.c_[motion['qpos'][:,7:],joint_values]
    units=['m' if model.jnt_type[j]==mj.mjtJoint.mjJNT_SLIDE else 'rad' for j in range(1,model.njnt)]+['rad']*len(joint_names)
    save(folder/'playback.json',dict(fps=30,frames=n,positions=(np.array(allpos)@C.T).tolist(),quaternions=R.from_matrix((C@np.array(allrot)).reshape(-1,3,3)).as_quat().reshape(n,len(meshes),4).tolist(),
        tracking=dict(actual=(actual@C.T).tolist(),target=(targets@C.T).tolist(),error_mm=error.tolist(),orientation_deg=np.rad2deg(motion['orientation_error_rad']).tolist(),reference='Gripper housing center' if hand=='robotiq-2f-140' else 'Wrist' if hand=='none' else 'Palm'),
        joints=dict(names=names,values=values.tolist(),unit='mixed',units=units,representation='Native hinge radians / slide meters; wheel actuation unavailable'),base=dict(xy_yaw=np.c_[motion['qpos'][:,:2],R.from_quat(motion['qpos'][:,[4,5,6,3]]).as_euler('xyz')[:,2]].tolist())))
    base='/assets/'+dest.relative_to(str(paths.store)).as_posix()
    entry=dict(encoder_policy=__import__('greenwich_pipelines.input_policy',fromlist=['POLICY']).POLICY,id=rid,label=robot['label'],hand=hand,experimental=True,model_url=base+'/robot-'+rid+'/model.glb',playback_url=base+'/robot-'+rid+'/playback.json',reference='gripper' if hand=='robotiq-2f-140' else 'wrist' if hand=='none' else 'palm',quality=dict(mean_mm=float(error.mean()),max_mm=float(error.max())))
    from arenalab_workbench.gpu_scheduler import merge_scene
    def update(manifest):
        manifest['robots']=[r for r in manifest['robots'] if r['id']!=rid]+[entry];manifest.setdefault('patch_unavailable',{})[rid]='Mobile native motion ready; calibrated source-FOV composite not generated'
        manifest['geometry']['ground']=dict(height_m=float(motion['ground_height_m']),estimated=True,policy='source-sole-plane-v1')
        manifest['geometry']['mobile_base']='Planar kinematic base trajectory; fixed vendor wheel joints do not supply wheel actuation'
        manifest['generation_status']=dict(status='running',phase='Robot ready; completing modalities',robot_id=rid,hand_id=hand)
    merge_scene(dest/'scene.json',update)
    report=json.loads((out/'wheeled-validation.json').read_text());report.update(encoder_policy=__import__('greenwich_pipelines.input_policy',fromlist=['POLICY']).POLICY,hand=hand,display_from_zup=C.tolist(),independent_fk_max_error_m=max(disagreement))
    if gripper_work:
        import hashlib
        report.update(policy=__import__('arenalab_workbench.gripper_job', fromlist=['POLICY']).POLICY,head_calibration=json.loads((Path(gripper_work)/'source.json').read_text()).get('head_calibration'),solver_policy=POLICY,source_sha256=hashlib.sha256((Path(gripper_work)/'source.npz').read_bytes()).hexdigest())
    else:
        report.update(source_id=json.loads((bodywork/'input_manifest.json').read_text())['clip'],support_policy=__import__('arenalab_workbench.support_reference', fromlist=['*']).POLICY)
    save(folder/'validation.json',report)
    return base+'/scene.json'

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('robot');p.add_argument('bodywork',type=Path);p.add_argument('dest',type=Path);p.add_argument('hand');p.add_argument('--gripper-work',type=Path);a=p.parse_args()
    from arenalab_workbench.robot_registry import roster
    if a.gripper_work:
        from arenalab_workbench.gripper_body import run as recover
        a.bodywork=recover(a.gripper_work)
    run(roster()[a.robot],a.bodywork,a.dest,a.hand,a.gripper_work)
