"""Supplied body or native SMPL-H reconstruction -> one GPU refiner with gripper TCP metadata."""
from arenalab_workbench.settings import paths
from pathlib import Path
import os,json,sys,subprocess,time,hashlib
from arenalab_workbench.motion_backend import normalized_source,build_request,run_gpu_motion,backend_fingerprint,trajectory_scale
import numpy as np,mujoco as mj
from greenwich_pipelines.pipelines.nymeria.embodiment_exact import load_robot
from greenwich_pipelines.pipelines.nymeria.input import endpoint_indices
from alphamotion.viz.kinematics import visual_mesh_geom_ids
from arenalab_workbench.end_effectors import mount_frame
from arenalab_workbench.gripper_reference import housing_reference
from arenalab_workbench.gripper_source import ROOT,digest,save
from arenalab_workbench.gripper_body import run as body_run
GRIPPER=ROOT/'grippers/robotiq-2f-140'
def result_path(work,rid):
 return Path(work)/'unified-motion'/trajectory_scale()/rid
def source_human_metadata(bodywork):
 """Report the actual prepared source; never label supplied X as H inference."""
 bodywork=Path(bodywork)
 def read(name):
  path=bodywork/name
  return json.loads(path.read_text()) if path.is_file() else {}
 manifest=read('input_manifest.json');ready=read('prior-ready.json');human=read('human_reference.json')
 supplied=manifest.get('motion_source')
 if supplied:
  family=str(supplied['family']).lower();route='supplied_body'
  evidence=supplied.get('evidence','supplied')
  prior='Provided complete '+family.upper()+'; body reconstruction bypassed'
 else:
  family=human.get('family',ready.get('human_model',manifest.get('human_model','unknown')))
  route=ready.get('body_route','three-point-egoallo' if family=='smplh' else 'unknown')
  evidence='reconstructed' if route=='three-point-egoallo' else 'unknown'
  prior='Head and two grippers -> native '+family.upper() if route=='three-point-egoallo' else 'Source body provenance unavailable'
 return dict(human_model=family,human_policy=ready.get('policy',human.get('policy',manifest.get('policy','unknown'))),
             body_route=route,body_prior=prior,source_body_evidence=evidence)

def run(work,rid):
 work=Path(work);bodywork=body_run(work);src=np.load(work/'source.npz');meta=json.loads((work/'source.json').read_text());n=len(src['tcp_zup']);cfg=json.loads(paths.viewer_config.read_text());rows={r['id']:r for r in cfg['robots']};reg=paths.store / 'uploaded/registry.json'
 if reg.exists():rows.update({r['id']:r for r in json.loads(reg.read_text())})
 robot=rows[rid];row=robot['pipeline_row'].copy();body=load_robot(row);ids=endpoint_indices(body.spec,row);row['endpoints']={k:body.spec.joint_names[i] for k,i in zip(['left_wrist','right_wrist','left_ankle','right_ankle'],ids)}
 m=mj.MjModel.from_xml_path(row['xml']);d=mj.MjData(m);mj.mj_forward(m,d);geoms=list(visual_mesh_geom_ids(m));mounts=[mount_frame(m,d,row,s,geoms) for s in ['left','right']]
 desc=json.loads((GRIPPER/'description.json').read_text());lut=desc['width_lookup'];widths=np.array([a['width'] for a in lut]);width=src['width_m']
 if np.any(width<0) or np.any(width>widths.max()+1e-6):raise ValueError('Source opening exceeds physical target gripper range; no silent clamp')
 reference,reference_metadata=housing_reference(GRIPPER,desc)
 angles=np.interp(width,widths[::-1],[a['q'] for a in lut][::-1]);local=np.broadcast_to(np.array([m[1]+m[2]@reference for m in mounts])[None],(n,2,3)).copy();desired=np.stack([src['tool_R_zup'][:,s]@mounts[s][2].T for s in range(2)],1)
 out=result_path(work,rid);out.mkdir(parents=True,exist_ok=True);save(out/'robot.json',row);np.savez_compressed(out/'tool.npz',angles=angles,reference_in_mount=reference,offset_native=local,mount_origin=np.array([a[1] for a in mounts]),mount_rotation=np.array([a[2] for a in mounts]),target_tool_R=src['tool_R_zup'])
 save(out/'attachment.json',dict(robot=rid,label=robot['label'],mounts=[dict(side=s,origin_m=mounts[i][1].tolist(),rotation=mounts[i][2].tolist(),provenance=mounts[i][3],hardware_calibrated=False) for i,s in enumerate(['left','right'])],reference=meta['target_reference'],source_sha256=digest(work/'source.npz'),physical_tool='Robotiq 2F-140 fixed housing center',target_reference='housing_center',reference_metadata=reference_metadata))
 from arenalab_motion.robot_metadata import ensure_robot_metadata
 row=ensure_robot_metadata(row)
 source=normalized_source(bodywork)
 local_file=out/'operation-points.npz'
 np.savez_compressed(local_file,points=local[:,:,None],target_rotation_zup=desired)
 # Immutable numerical versions are separate from physical gripper attachments.
 identity=hashlib.sha256((backend_fingerprint()+digest(local_file)+digest(source/'native_source.npz')).encode()).hexdigest()[:24]
 version=out/'versions'/identity
 if version.exists():version=version.with_name(identity+'-retry-'+str(time.time_ns()))
 request=build_request(rid,row,source,version,'tcp',local_file)
 with (out/'refine.log').open('w') as log:
  report,priors=run_gpu_motion(request,env=os.environ.copy(),cwd=bodywork,stdout=log,stderr=subprocess.STDOUT)
 for name in ('motion.npz','gpu_qpos.npz','source-targets.npz','report.json'):
  destination=out/name
  if destination.is_symlink():destination.unlink()
  elif destination.exists():destination.rename(destination.with_name(name+'.previous-'+str(time.time_ns())))
  destination.symlink_to(version/name)
 save(out/'provenance.json',dict(source_sha256=digest(work/'source.npz'),original_priors=priors,**source_human_metadata(bodywork),trajectory_scale=trajectory_scale(),source_trajectory_edited=trajectory_scale()=='match_robot',scale_transform=report.get('scale_transform')))
 print('GRIPPER_COMPLETE',work.name,rid,flush=True)
if __name__=='__main__':run(*sys.argv[1:])
