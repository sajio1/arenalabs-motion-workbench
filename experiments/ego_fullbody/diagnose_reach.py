from pathlib import Path
import json,numpy as np,torch,mujoco
from greenwich_pipelines.pipelines.nymeria.embodiment_exact import load_robot
from greenwich_pipelines.pipelines.nymeria.physical_fk import PhysicalFK
from body_frames import BodyFrames
root=Path(__import__("os").environ.get("ARENA_EGO_WORKDIR", str(Path(__file__).resolve().parent)));C=np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
human=np.load(root/'smplx_motion.npz');hp=human['joints']@C.T;obs=np.load(root/'observations.npz')['wrist_position']@C.T
for robot in ['h2','a3']:
 row=json.loads((root/(robot+'-robot.json')).read_text())[0];body=load_robot(row);fk=PhysicalFK(body,'cpu',torch.float64);names=list(body.spec.joint_names)
 ids=[names.index(row['endpoints'][r]) for r in ['left_wrist','right_wrist','left_ankle','right_ankle']];frames=BodyFrames(body,fk,ids[:2],ids[2:])
 model=mujoco.MjModel.from_xml_path(row['xml']);j=model.body(row['endpoints']['left_wrist']).id;arm=[]
 while j!=model.body(names[frames.chest]).id:
  for k in range(model.body_jntadr[j],model.body_jntadr[j]+model.body_jntnum[j]):arm.append((model.joint(k).name,np.rad2deg(model.jnt_range[k]).round(1).tolist()))
  j=model.body_parentid[j]
 d=np.load(root/'observed-refinement'/robot/'motion.npz');p=d['world_position_cm']/100
 q=torch.zeros((1,len(fk.active_indices)),dtype=torch.float64);R,neutral=fk(q,torch.eye(3,dtype=torch.float64)[None]);neutral=neutral.numpy()[0]
 td=json.loads((root/'observed-refinement'/robot/'torso-validation.json').read_text());diff=np.array(td['versions']['after']['pitch_deg'])-np.array(td['source_pitch_deg']);t=int(np.argmax(diff))
 print(json.dumps(dict(robot=robot,arm_joints=arm,neutral_shoulder_wrist_distance_m=np.linalg.norm(neutral[frames.shoulders]-neutral[ids[:2]],axis=-1).tolist(),worst_extra_tilt_frame=t,source_tilt=td['source_pitch_deg'][t],robot_tilt=td['versions']['after']['pitch_deg'][t],source_pelvis_height=float(hp[t,0,1]),robot_pelvis_height=float(p[t,frames.pelvis,1]),source_shoulders=hp[t,[16,17]].tolist(),robot_shoulders=p[t,frames.shoulders].tolist(),observed_wrists=obs[t].tolist(),robot_knees_deg=np.rad2deg(d['knee_flexion_rad'][t]).tolist(),source_knees_deg=np.rad2deg(d['source_knee_flexion_rad'][t]).tolist()),indent=2))
