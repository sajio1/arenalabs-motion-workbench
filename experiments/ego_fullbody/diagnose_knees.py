"""Locate loss of flexion across human, decoder, joint projection and refinement."""
from pathlib import Path
import os,sys,json
import numpy as np,torch
from greenwich_pipelines.pipelines.nymeria.embodiment_exact import load_robot
from greenwich_pipelines.pipelines.nymeria.physical_fk import PhysicalFK
from alphamotion.engine.nets.rotations import rot6d_to_matrix
from greenwich_motion_sdk.native_projection import project
root=Path(os.environ.get('ARENA_EGO_WORKDIR',Path(__file__).resolve().parent));label=sys.argv[1]
torch.set_num_threads(2)
row=json.loads((root/(label+'-robot.json')).read_text())[0];body=load_robot(row);fk=PhysicalFK(body,'cpu',torch.float64)
d=np.load(next((root/(label+'-result')).rglob('motion.npz')))
raw=rot6d_to_matrix(torch.tensor(d['raw_rot6d'],dtype=torch.float64));rawp=fk.positions(raw).numpy()
rest=fk.positions(torch.eye(3,dtype=torch.float64)[None,None].expand(1,body.spec.J,3,3)).numpy()[0]
chains=[]
for side in ['left','right']:
 j=list(body.spec.joint_names).index(row['endpoints'][side+'_ankle']);chain=[]
 while j!=fk.root:
  if not chain or np.linalg.norm(rest[j]-rest[chain[-1]])>.08:chain.append(j)
  j=int(body.spec.parents[j])
 chains.append(chain[:3])
def angles(p,chains):
 results=[]
 for a,b,c in chains:
  u=p[:,a]-p[:,b];v=p[:,b]-p[:,c]
  results.append(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(u,v),axis=-1),(u*v).sum(-1))))
 return np.stack(results,1)
source=angles(np.load(root/'smplx_motion.npz')['joints'],[[7,4,1],[8,5,2]])
q,_,_=project(torch.tensor(d['raw_rot6d'],dtype=torch.float64),body.spec,torch.tensor(body.dof,dtype=torch.float64),torch.tensor(body.rest,dtype=torch.float64))
_,projection=fk(q.flatten(1)[:,fk.active_indices],raw[:,fk.root])
stages={'smplx':source,'alphamotion_raw':angles(rawp,chains),'physical_projection':angles(projection.numpy(),chains),'old_refinement':angles(d['world_position_cm'],chains)}
new=root/'observed-refinement'/label/'motion.npz'
if new.exists():stages['new_refinement']=angles(np.load(new)['world_position_cm'],chains)
report={}
for stage,a in stages.items():
 report[stage]=dict(mean_deg=a.mean(0).tolist(),range_deg=np.stack([a.min(0),a.max(0)]).tolist(),p95_deg=np.percentile(a,95,axis=0).tolist(),mean_abs_difference_to_smplx_deg=np.abs(a-source).mean(0).tolist(),timeseries_deg=a.tolist())
out=root/'observed-refinement'/label/'knee-diagnosis.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2))
print(json.dumps({stage:{k:v for k,v in r.items() if k!='timeseries_deg'} for stage,r in report.items()},indent=2))
