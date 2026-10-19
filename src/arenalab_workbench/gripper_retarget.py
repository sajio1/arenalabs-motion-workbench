"""Reuse SMPL-X -> AlphaMotion -> operation refinement with gripper TCP metadata."""
from arenalab_workbench.settings import paths
from pathlib import Path
import os,json,sys,subprocess,time
from arenalab_workbench.refinement_runtime import run_refinement
import numpy as np,mujoco as mj
from greenwich_pipelines.pipelines.nymeria.embodiment_exact import load_robot
from greenwich_pipelines.pipelines.nymeria.input import endpoint_indices
from alphamotion.viz.kinematics import visual_mesh_geom_ids
from arenalab_workbench.end_effectors import mount_frame
from arenalab_workbench.gripper_reference import housing_reference
from arenalab_workbench.gripper_source import ROOT,digest,save
from arenalab_workbench.gripper_body import run as body_run
GRIPPER=ROOT/'grippers/robotiq-2f-140'
def run(work,rid):
 work=Path(work);bodywork=body_run(work);src=np.load(work/'source.npz');meta=json.loads((work/'source.json').read_text());n=len(src['tcp_zup']);cfg=json.loads(paths.viewer_config.read_text());rows={r['id']:r for r in cfg['robots']};reg=paths.store / 'uploaded/registry.json'
 if reg.exists():rows.update({r['id']:r for r in json.loads(reg.read_text())})
 robot=rows[rid];row=robot['pipeline_row'].copy();body=load_robot(row);ids=endpoint_indices(body.spec,row);row['endpoints']={k:body.spec.joint_names[i] for k,i in zip(['left_wrist','right_wrist','left_ankle','right_ankle'],ids)}
 m=mj.MjModel.from_xml_path(row['xml']);d=mj.MjData(m);mj.mj_forward(m,d);geoms=list(visual_mesh_geom_ids(m));mounts=[mount_frame(m,d,row,s,geoms) for s in ['left','right']]
 desc=json.loads((GRIPPER/'description.json').read_text());lut=desc['width_lookup'];widths=np.array([a['width'] for a in lut]);width=src['width_m']
 if np.any(width<0) or np.any(width>widths.max()+1e-6):raise ValueError('Source opening exceeds physical target gripper range; no silent clamp')
 reference,reference_metadata=housing_reference(GRIPPER,desc)
 angles=np.interp(width,widths[::-1],[a['q'] for a in lut][::-1]);local=np.broadcast_to(np.array([m[1]+m[2]@reference for m in mounts])[None],(n,2,3)).copy();desired=np.stack([src['tool_R_zup'][:,s]@mounts[s][2].T for s in range(2)],1)
 out=work/rid;out.mkdir(exist_ok=True);save(out/'robot.json',row);np.savez_compressed(out/'tool.npz',angles=angles,reference_in_mount=reference,offset_native=local,mount_origin=np.array([a[1] for a in mounts]),mount_rotation=np.array([a[2] for a in mounts]),target_tool_R=src['tool_R_zup'])
 save(out/'attachment.json',dict(robot=rid,label=robot['label'],mounts=[dict(side=s,origin_m=mounts[i][1].tolist(),rotation=mounts[i][2].tolist(),provenance=mounts[i][3],hardware_calibrated=False) for i,s in enumerate(['left','right'])],reference=meta['target_reference'],source_sha256=digest(work/'source.npz'),physical_tool='Robotiq 2F-140 fixed housing center',target_reference='housing_center',reference_metadata=reference_metadata))
 save(bodywork/(rid+'-robot.json'),[row]);np.savez_compressed(bodywork/'operation-accuracy'/f'{rid}-local-points.npz',points=local[:,:,None],target_rotation_zup=desired)
 job=dict(schema='greenwich.job.v1',id='gripper_'+rid,pipeline='fullbody',source='sources.json',robots=rid+'-robot.json',output=rid+'-result',runtime=str(paths.config / 'runtime.cloud.json'),fps=30);save(bodywork/(rid+'-job.json'),job)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES','3'),EGL_DEVICE_ID=os.environ.get('EGL_DEVICE_ID','3'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',PYTHONPATH=paths.pythonpath(),ARENA_EGO_WORKDIR=str(bodywork),ARENA_SOURCE_WORK=str(bodywork),ARENA_OUTPUTDIR=str(out),ARENA_OPERATION_MODE='tcp')
 from greenwich_pipelines.input_policy import POLICY as ENCODER_POLICY
 env['ARENA_FRAME_CACHE_DIR']=str(bodywork/'frame-stream'/rid)
 previous=bodywork/(rid+'-result')
 originals=list(previous.rglob('prime_original.npz')) if previous.exists() else []
 if previous.exists() and (not originals or any(str(np.load(p).get('encoder_policy','legacy-full'))!=ENCODER_POLICY for p in originals)):
  previous.rename(bodywork/(rid+'-result.legacy-'+str(time.time_ns())))
 if not previous.exists():
  with (out/'model.log').open('w') as log:subprocess.run([sys.executable,'-m','greenwich_pipelines.cli','run',str(bodywork/(rid+'-job.json'))],cwd=bodywork,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 priors={str(p):digest(p) for p in (bodywork/(rid+'-result')).rglob('prime_original.npz')}
 if not priors:raise ValueError('Missing original AlphaMotion prior')
 with (out/'refine.log').open('w') as log:run_refinement([sys.executable,str(Path(__file__).with_name('refine_operation.py')),rid],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 assert all(digest(k)==v for k,v in priors.items());save(out/'provenance.json',dict(source_sha256=digest(work/'source.npz'),original_priors=priors,body_prior='Head and two grippers -> EgoAllo -> SMPL-X',source_trajectory_edited=False));print('GRIPPER_COMPLETE',work.name,rid,flush=True)
if __name__=='__main__':run(*sys.argv[1:])
