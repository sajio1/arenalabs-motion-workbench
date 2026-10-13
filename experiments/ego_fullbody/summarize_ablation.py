from pathlib import Path
import json,numpy as np,torch
from greenwich_pipelines.pipelines.nymeria.embodiment_exact import load_robot
from greenwich_pipelines.pipelines.nymeria.physical_fk import PhysicalFK
from alphamotion.engine.nets.rotations import rot6d_to_matrix
from body_frames import BodyFrames,angle
root=Path(__import__("os").environ.get("ARENA_EGO_WORKDIR", str(Path(__file__).resolve().parent)));row=json.loads((root/'h2-robot.json').read_text())[0];body=load_robot(row);fk=PhysicalFK(body,'cpu',torch.float64);names=list(body.spec.joint_names);ids=[names.index(row['endpoints'][r]) for r in ['left_wrist','right_wrist','left_ankle','right_ankle']];frames=BodyFrames(body,fk,ids[:2],ids[2:]);C=np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
human=np.load(root/'smplx_motion.npz');target=torch.tensor(C@human['global_rotation'][:,9],dtype=torch.float64)
report={}
files=[root/'observed-refinement-v5/h2/motion.npz']+list((root/'reach-diagnostics').glob('*/h2/motion.npz'))
for f in files:
 d=np.load(f);R=rot6d_to_matrix(torch.tensor(d['rot6d'],dtype=torch.float64));ch,rel,pel=frames(R);e=angle(ch,target).numpy()*180/np.pi;name='baseline-v5' if f.parent.parent.name=='observed-refinement-v5' else f.parent.parent.name
 report[name]=dict(chest_error_mean_deg=float(e.mean()),chest_error_max_deg=float(e.max()),wrist_max_cm=float(d['wrist_error_m'].max()*100),wrist_max_deg=float(d['wrist_error_deg'].max()),knee_deviation_mean_deg=float(np.rad2deg(np.abs(d['knee_flexion_rad']-d['source_knee_flexion_rad'])).mean()),root_mean_y=float(d['root_t'][:,1].mean()/100),minimum_foot_cm=float(d['sole_surface_height_cm'].min()))
(root/'reach-diagnostics/summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
