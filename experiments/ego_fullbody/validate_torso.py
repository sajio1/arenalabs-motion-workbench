"""Compare unchanged v3 and torso-prior candidates in anatomical frames."""
from pathlib import Path
import os,json,sys
import numpy as np,torch
from greenwich_pipelines.pipelines.nymeria.embodiment_exact import load_robot
from greenwich_pipelines.pipelines.nymeria.physical_fk import PhysicalFK
from alphamotion.engine.nets.rotations import rot6d_to_matrix
from body_frames import BodyFrames,angle
root=Path(os.environ.get('ARENA_EGO_WORKDIR',Path(__file__).resolve().parent));label=sys.argv[1]
torch.set_num_threads(2)
row=json.loads((root/(label+'-robot.json')).read_text())[0];body=load_robot(row);fk=PhysicalFK(body,'cpu',torch.float64)
ids=[list(body.spec.joint_names).index(row['endpoints'][r]) for r in ('left_wrist','right_wrist','left_ankle','right_ankle')]
frames=BodyFrames(body,fk,ids[:2],ids[2:]);C=np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
human=np.load(root/'smplx_motion.npz');chest=torch.tensor(C@human['global_rotation'][:,9],dtype=torch.float64);pelvis=torch.tensor(C@human['global_rotation'][:,0],dtype=torch.float64);relative=pelvis.transpose(-1,-2)@chest
def stats(x):
 x=np.asarray(x);return dict(mean=float(x.mean()),p95=float(np.percentile(x,95)),max=float(x.max()))
def pitch(R):
 R=R.numpy();up=R[:,:,1];heading=R[:,:,2].copy();heading[:,1]=0;heading/=np.linalg.norm(heading,axis=-1,keepdims=True)
 return np.rad2deg(np.arctan2((up*heading).sum(-1),up[:,1]))
report=dict(frames=frames.metadata,source_pitch_deg=pitch(chest).tolist(),versions={})
for version,folder in [('before',os.environ.get('ARENA_BASELINE_FOLDER','observed-refinement-v5')),('after','observed-refinement')]:
 d=np.load(root/folder/label/'motion.npz');R=rot6d_to_matrix(torch.tensor(d['rot6d'],dtype=torch.float64));actual,rel,pel=frames(R)
 # Compare saved rotation fields with actual physical-axis forward kinematics.
 q=torch.tensor(d['q'],dtype=torch.float64).flatten(1)[:,fk.active_indices];check,_=fk(q,R[:,fk.root]);native,_,_=frames(check)
 result=dict(chest_error_deg=stats(angle(actual,chest).numpy()*180/np.pi),relative_error_deg=stats(angle(rel,relative).numpy()*180/np.pi),pitch_deg=pitch(actual).tolist(),pitch_abs_error_deg=stats(np.abs(pitch(actual)-pitch(chest))),saved_vs_physical_chest_deg=stats(angle(native,actual).numpy()*180/np.pi),root_vertical_acceleration_m_s2=stats(np.abs(np.diff(d['root_t'][:,1],n=2)*9)),joint_acceleration_rad_s2=stats(np.abs(np.diff(d['q'],n=2,axis=0)*900)),wrist_position_cm=stats(d['wrist_error_m']*100),wrist_orientation_deg=stats(d['wrist_error_deg']))
 report['versions'][version]=result
folder=root/'observed-refinement'/label
(folder/'torso-validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps({v:{k:value for k,value in r.items() if k!='pitch_deg'} for v,r in report['versions'].items()},indent=2))
