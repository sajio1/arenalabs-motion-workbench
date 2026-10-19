"""Native FK export with a real mimic-joint gripper; no display pose repairs."""
from arenalab_workbench.settings import paths
from pathlib import Path
import json,sys
import numpy as np,mujoco as mj,trimesh,torch
from scipy.spatial.transform import Rotation as R
from alphamotion.engine.nets.rotations import rot6d_to_matrix
from alphamotion.viz.kinematics import visual_mesh_geom_ids
from greenwich_pipelines.pipelines.nymeria.embodiment_exact import load_robot
from arenalab_workbench.gripper_source import C,ROOT,save,digest
from arenalab_workbench.gripper_retarget import GRIPPER
from arenalab_workbench.gripper_body import body_path

def tool_fk(desc,q):
 poses={desc['root']:np.eye(4)};pending=desc['joints'].copy()
 while pending:
  ready=[j for j in pending if j['parent'] in poses]
  if not ready:raise ValueError('Invalid gripper topology')
  for j in ready:
   T=np.eye(4);T[:3,:3]=R.from_euler('xyz',j['rpy']).as_matrix();T[:3,3]=j['xyz']
   if j['type']=='revolute':
    mimic=j['mimic'];v=q if mimic is None else q*float(mimic.get('multiplier',1))+float(mimic.get('offset',0));T[:3,:3]=T[:3,:3]@R.from_rotvec(np.array(j['axis'])*v).as_matrix()
   poses[j['child']]=poses[j['parent']]@T;pending.remove(j)
 return poses

def run(work,rid,dest):
 meta=json.loads((Path(work)/'source.json').read_text())
 work=Path(work);out=work/rid;dest=Path(dest);folder=dest/('robot-'+rid);folder.mkdir(parents=True,exist_ok=True);row=json.loads((out/'robot.json').read_text());attach=json.loads((out/'attachment.json').read_text());src=np.load(work/'source.npz');tool=np.load(out/'tool.npz');body=load_robot(row);m=mj.MjModel.from_xml_path(row['xml']);ids=list(visual_mesh_geom_ids(m));native=[];meshes=[]
 for g in ids:
  text=(m.body(int(m.geom_bodyid[g])).name+' '+m.mesh(int(m.geom_dataid[g])).name).lower()
  if any(k in text for k in ['hand','finger','palm','dummy','thumb']):continue
  native.append(g);mid=m.geom_dataid[g];a=m.mesh_vertadr[mid];c=m.mesh_faceadr[mid];mesh=trimesh.Trimesh(m.mesh_vert[a:a+m.mesh_vertnum[mid]],m.mesh_face[c:c+m.mesh_facenum[mid]],process=False);mesh.visual.vertex_colors=[180,190,182,255];meshes.append(mesh)
 desc=json.loads((GRIPPER/'description.json').read_text());visuals=[]
 for node in desc['nodes']:
  for v in node['visuals']:
   mesh=trimesh.load(GRIPPER/v['mesh'],force='mesh') if 'mesh' in v else trimesh.creation.box(v['box']);mesh.apply_scale(v.get('scale',[1,1,1]));T=np.eye(4);T[:3,:3]=R.from_euler('xyz',v['rpy']).as_matrix();T[:3,3]=v['xyz'];mesh.apply_transform(T);mesh.visual.vertex_colors=(np.array(v['color'])*255).astype('u1');visuals.append((node['name'],mesh))
 for side in range(2):meshes.extend(mesh.copy() for _,mesh in visuals)
 motion=np.load(out/'motion.npz');n=len(motion['q']);Rs=rot6d_to_matrix(torch.tensor(motion['rot6d'])).numpy();root=int(np.flatnonzero(np.array(body.spec.parents)<0)[0]);pos=[];rot=[];actual=[];toolrot=[];joint_values=[];fkerr=[]
 names=[name for group in body.qnames for name in group]+['left_gripper_finger_joint','right_gripper_finger_joint'];wristids=[m.body(row['endpoints'][s+'_wrist']).id for s in ['left','right']]
 for t in range(n):
  d=mj.MjData(m);d.qpos[:]=m.qpos0;d.qpos[:3]=motion['root_t'][t]@C/100;d.qpos[3:7]=R.from_matrix(C.T@Rs[t,root]@C).as_quat()[[3,0,1,2]];jv=[]
  for j,group in enumerate(body.qnames):
   for k,name in enumerate(group):d.qpos[m.jnt_qposadr[m.joint(name).id]]=motion['q'][t,j,k];jv.append(float(motion['q'][t,j,k]))
  mj.mj_forward(m,d);pp=list(d.geom_xpos[native]@C.T);rr=list(C@d.geom_xmat[native].reshape(-1,3,3));aa=[];tr=[]
  for si,bid in enumerate(wristids):
   wr=d.xmat[bid].reshape(3,3);wp=d.xpos[bid];tr.append(wr@tool['mount_rotation'][si]);mountp=wp+wr@tool['mount_origin'][si];mountR=wr@tool['mount_rotation'][si];poses=tool_fk(desc,float(tool['angles'][t,si]))
   referencep=mountp+mountR@tool['reference_in_mount']
   if np.linalg.norm(referencep-(wp+wr@tool['offset_native'][t,si]))>1e-6:raise ValueError('Gripper housing center differs from solver reference')
   aa.append(referencep)
   for name,_ in visuals:
    T=poses[name];pp.append(C@(mountp+mountR@T[:3,3]));rr.append(C@mountR@T[:3,:3])
  actual.append(aa);toolrot.append(tr);pos.append(pp);rot.append(rr);joint_values.append(jv+tool['angles'][t].tolist())
  for j,name in enumerate(body.spec.joint_names):
   bid=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,name)
   if bid>=0:fkerr.append(np.linalg.norm(d.xpos[bid]@C.T-motion['world_position_cm'][t,j]/100))
 if max(fkerr)>1e-6:raise ValueError('Independent native FK mismatch')
 actual=np.array(actual);err=np.linalg.norm(actual-src['tcp_zup'],axis=-1)*1000;angle=np.rad2deg(R.from_matrix((np.array(toolrot).transpose(0,1,3,2)@src['tool_R_zup']).reshape(-1,3,3)).magnitude()).reshape(n,2)
 scene=trimesh.Scene()
 for i,mesh in enumerate(meshes):scene.add_geometry(mesh,node_name=f'geom_{i}',geom_name=f'geom_{i}')
 scene.export(folder/'model.glb')
 def ancestors(b):
  path=[]
  while b>0:path.append(b);b=int(m.body_parentid[b])
  return path+[0]
 paths=[ancestors(i) for i in wristids];common=next(i for i in paths[0] if i in paths[1]);branches=[p[p.index(common)-1] for p in paths];selected=[i for i,g in enumerate(native) if any(b in ancestors(int(m.geom_bodyid[g])) for b in branches)]+list(range(len(native),len(meshes)))
 save(folder/'render-layout.json',dict(arm_and_gripper_geoms=selected))
 rotation=np.array(rot);payload=dict(fps=30,frames=n,positions=np.array(pos).tolist(),quaternions=R.from_matrix(rotation.reshape(-1,3,3)).as_quat().reshape(n,len(meshes),4).tolist(),tracking=dict(actual=(actual@C.T).tolist(),target=(src['tcp_zup']@C.T).tolist(),error_mm=err.tolist(),orientation_deg=angle.tolist(),reference='Fixed-housing center → '+attach['reference']),joints=dict(names=names,values=joint_values,unit='rad',representation='Native robot + actual gripper linkage angle'))
 save(folder/'playback.json',payload);base='/assets/'+dest.relative_to(str(paths.store)).as_posix();entry=dict(encoder_policy=__import__('greenwich_pipelines.input_policy',fromlist=['POLICY']).POLICY,id=rid,label=attach['label'],hand='robotiq-2f-140',model_url=base+'/robot-'+rid+'/model.glb',playback_url=base+'/robot-'+rid+'/playback.json?v='+digest(folder/'playback.json')[:16],reference='gripper',quality=dict(mean_mm=float(err.mean()),max_mm=float(err.max())))
 from arenalab_workbench.gpu_scheduler import merge_scene
 def update(manifest):
  manifest['geometry']['head_reference']=json.loads((body_path(work)/'input_manifest.json').read_text()).get('head_reference_audit');manifest['robots']=[r for r in manifest['robots'] if r['id']!=rid]+[entry];manifest['geometry']['ground']=dict(height_m=float(motion['ground_height_m']),estimated=True,policy='source-sole-plane-v1',provenance=meta['floor_provenance'] if meta.get('height_calibration') else 'Reconstructed SMPL-X soles; not recorded source robot feet',height_calibration=meta.get('height_calibration'))
 merge_scene(dest/'scene.json',update)
 report=dict(source_sha256=digest(work/'source.npz'),source_trajectory_edited=False,independent_fk_max_error_m=max(fkerr),tcp_error_mm=entry['quality'],orientation_p95_deg=np.percentile(angle,95,axis=0).tolist(),mounts=attach['mounts'],source_reference=attach['reference'],head_calibration=meta.get('head_calibration'),physical_validation=False,reference_metadata=attach['reference_metadata'],encoder_policy=__import__('greenwich_pipelines.input_policy',fromlist=['POLICY']).POLICY,policy='gripper-head-two-ee-smplx-housing-v3');save(folder/'validation.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':run(*sys.argv[1:])
