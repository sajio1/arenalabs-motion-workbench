"""Native FK export with a real mimic-joint gripper; no display pose repairs."""
from arenalab_workbench.settings import paths
from pathlib import Path
import json,sys
import numpy as np,mujoco as mj,trimesh
from scipy.spatial.transform import Rotation as R
from alphamotion.viz.kinematics import visual_mesh_geom_ids
from greenwich_pipelines.pipelines.nymeria.embodiment_exact import load_robot
from arenalab_workbench.gripper_source import C,ROOT,save,digest
from arenalab_workbench.gripper_retarget import GRIPPER,result_path,source_human_metadata
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

def native_motion_state(model, path, frames):
 """Read authoritative world qpos; export scalar values by native addresses."""
 with np.load(path,allow_pickle=False) as saved:
  qpos=np.asarray(saved['qpos'],dtype=np.float64).copy()
  fps=float(saved['fps'])
 if qpos.shape!=(frames,model.nq) or not np.isfinite(qpos).all() or fps!=30:
  raise ValueError('Expected finite native world qpos [frames,nq] at 30 Hz')
 free=np.flatnonzero(model.jnt_type==mj.mjtJoint.mjJNT_FREE)
 if len(free)!=1 or int(model.jnt_qposadr[free[0]])!=0:
  raise ValueError('Native gripper export requires one floating root at qpos 0')
 if not np.allclose(np.linalg.norm(qpos[:,3:7],axis=-1),1.,atol=1e-6,rtol=0):
  raise ValueError('Native root quaternion must be unit length; no display normalization')
 ids=np.flatnonzero(np.isin(model.jnt_type,[mj.mjtJoint.mjJNT_HINGE,mj.mjtJoint.mjJNT_SLIDE]))
 names=[model.joint(int(j)).name for j in ids]
 if len(set(names))!=len(names) or any(not n for n in names):
  raise ValueError('Native scalar joints require distinct names')
 return qpos,names,qpos[:,model.jnt_qposadr[ids]].copy()

def run(work,rid,dest):
 meta=json.loads((Path(work)/'source.json').read_text())
 work=Path(work);out=result_path(work,rid);dest=Path(dest);folder=dest/('robot-'+rid);folder.mkdir(parents=True,exist_ok=True);row=json.loads((out/'robot.json').read_text());attach=json.loads((out/'attachment.json').read_text());src=np.load(work/'source.npz');tool=np.load(out/'tool.npz');body=load_robot(row);m=mj.MjModel.from_xml_path(row['xml']);ids=list(visual_mesh_geom_ids(m));native=[];meshes=[]
 for g in ids:
  text=(m.body(int(m.geom_bodyid[g])).name+' '+m.mesh(int(m.geom_dataid[g])).name).lower()
  if any(k in text for k in ['hand','finger','palm','dummy','thumb']):continue
  native.append(g);mid=m.geom_dataid[g];a=m.mesh_vertadr[mid];c=m.mesh_faceadr[mid];mesh=trimesh.Trimesh(m.mesh_vert[a:a+m.mesh_vertnum[mid]],m.mesh_face[c:c+m.mesh_facenum[mid]],process=False);mesh.visual.vertex_colors=[180,190,182,255];meshes.append(mesh)
 desc=json.loads((GRIPPER/'description.json').read_text());visuals=[]
 for node in desc['nodes']:
  for v in node['visuals']:
   mesh=trimesh.load(GRIPPER/v['mesh'],force='mesh') if 'mesh' in v else trimesh.creation.box(v['box']);mesh.apply_scale(v.get('scale',[1,1,1]));T=np.eye(4);T[:3,:3]=R.from_euler('xyz',v['rpy']).as_matrix();T[:3,3]=v['xyz'];mesh.apply_transform(T);mesh.visual.vertex_colors=(np.array(v['color'])*255).astype('u1');visuals.append((node['name'],mesh))
 for side in range(2):meshes.extend(mesh.copy() for _,mesh in visuals)
 motion=np.load(out/'motion.npz',allow_pickle=False);n=len(motion['world_position_cm']);pos=[];rot=[];actual=[];toolrot=[];joint_values=[];fkerr=[]
 native_path=out/'gpu_qpos.npz';qpos,native_names,native_values=native_motion_state(m,native_path,n)
 names=native_names+['left_gripper_finger_joint','right_gripper_finger_joint'];wristids=[m.body(row['endpoints'][s+'_wrist']).id for s in ['left','right']]
 if len(set(names))!=len(names):raise ValueError('Native and added gripper joint names conflict')
 for t in range(n):
  d=mj.MjData(m);d.qpos[:]=qpos[t];jv=native_values[t].tolist()
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
 targets=np.load(out/'source-targets.npz')['points'][:,:,0]
 numerical=json.loads((out/'report.json').read_text())
 human=source_human_metadata(body_path(work))
 encoder_policy=numerical.get('encoder_policy','supplied-full-body-all-visible-no-source-decode-v1')
 actual=np.array(actual);err=np.linalg.norm(actual-targets,axis=-1)*1000;angle=np.rad2deg(R.from_matrix((np.array(toolrot).transpose(0,1,3,2)@src['tool_R_zup']).reshape(-1,3,3)).magnitude()).reshape(n,2)
 scene=trimesh.Scene()
 for i,mesh in enumerate(meshes):scene.add_geometry(mesh,node_name=f'geom_{i}',geom_name=f'geom_{i}')
 scene.export(folder/'model.glb')
 def ancestors(b):
  path=[]
  while b>0:path.append(b);b=int(m.body_parentid[b])
  return path+[0]
 arm_paths=[ancestors(i) for i in wristids];common=next(i for i in arm_paths[0] if i in arm_paths[1]);branches=[p[p.index(common)-1] for p in arm_paths];selected=[i for i,g in enumerate(native) if any(b in ancestors(int(m.geom_bodyid[g])) for b in branches)]+list(range(len(native),len(meshes)))
 save(folder/'render-layout.json',dict(arm_and_gripper_geoms=selected))
 rotation=np.array(rot);payload=dict(fps=30,frames=n,positions=np.array(pos).tolist(),quaternions=R.from_matrix(rotation.reshape(-1,3,3)).as_quat().reshape(n,len(meshes),4).tolist(),tracking=dict(actual=(actual@C.T).tolist(),target=(targets@C.T).tolist(),error_mm=err.tolist(),orientation_deg=angle.tolist(),reference='Fixed-housing center → '+attach['reference']),joints=dict(names=names,values=joint_values,unit='rad',representation='Native robot + actual gripper linkage angle'))
 save(folder/'playback.json',payload);base='/assets/'+dest.relative_to(str(paths.store)).as_posix();entry=dict(trajectory_scale=numerical.get('trajectory_scale','preserve'),solver=numerical.get('solver',{}),motion_policy_version='resident-skeleton-surface-wrist-v3',encoder_policy=encoder_policy,id=rid,label=attach['label'],hand='robotiq-2f-140',model_url=base+'/robot-'+rid+'/model.glb',playback_url=base+'/robot-'+rid+'/playback.json?v='+digest(folder/'playback.json')[:16],reference='gripper',quality=dict(mean_mm=float(err.mean()),max_mm=float(err.max())))
 from arenalab_workbench.gpu_scheduler import merge_scene
 from arenalab_workbench.motion_backend import normalized_source
 from arenalab_workbench.source_preview import export_source_preview
 from arenalab_workbench.source_display import source_scene_metadata
 source_metadata=source_scene_metadata(export_source_preview(normalized_source(body_path(work)),dest),numerical,C)
 def update(manifest):
  manifest.update(source_metadata)
  manifest['geometry']['head_reference']=json.loads((body_path(work)/'input_manifest.json').read_text()).get('head_reference_audit');manifest['robots']=[r for r in manifest['robots'] if r['id']!=rid]+[entry];manifest['geometry']['ground']=dict(height_m=float(motion['ground_height_m']),estimated=True,policy='source-sole-plane-v1',provenance=meta['floor_provenance'] if meta.get('height_calibration') else human['body_prior']+'; source mesh sole plane, not recorded robot feet',height_calibration=meta.get('height_calibration'))
 merge_scene(dest/'scene.json',update)
 report=dict(motion_backend='gpu_hybrid',motion_backend_fingerprint=numerical.get('motion_backend_fingerprint'),trajectory_scale=numerical.get('trajectory_scale','preserve'),solver=numerical.get('solver',{}),native_output_dir=str(out.resolve()),**human,native_qpos_path=str(native_path.resolve()),native_qpos_sha256=digest(native_path),raw_sha256=numerical.get('raw_sha256',{}),source_sha256=digest(work/'source.npz'),source_trajectory_edited=numerical.get('trajectory_scale')=='match_robot',independent_fk_max_error_m=max(fkerr),tcp_error_mm=entry['quality'],orientation_p95_deg=np.percentile(angle,95,axis=0).tolist(),mounts=attach['mounts'],source_reference=attach['reference'],head_calibration=meta.get('head_calibration'),physical_validation=False,reference_metadata=attach['reference_metadata'],encoder_policy=encoder_policy,policy='unified-gpu-hybrid-gripper-housing-v1');save(folder/'validation.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':run(*sys.argv[1:])
