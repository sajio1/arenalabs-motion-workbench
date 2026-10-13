"""Check merged descriptor chest rotations against actual native body frames."""
from pathlib import Path
import json,os,sys,numpy as np,torch,mujoco as mj
from alphamotion.engine.descriptor import build_from_mjcf
from alphamotion.embodiment.mjcf_build import build_merge
from alphamotion.engine.nets.rotations import rot6d_to_matrix
root=Path(os.environ.get('ARENA_EGO_WORKDIR',Path(__file__).resolve().parent));label=sys.argv[1]
row=json.loads((root/(label+'-robot.json')).read_text())[0];model=mj.MjModel.from_xml_path(row['xml']);data=mj.MjData(model)
spec,dof,rest,qnames,_=build_from_mjcf(row['xml'],row['body']);keep,frames=build_merge(model)
folder=Path(os.environ.get('ARENA_OUTPUTDIR',root/'observed-refinement'/label));d=np.load(folder/'motion.npz');meta=json.loads((folder/'report.json').read_text())['body_frames'];ci=list(spec.joint_names).index(meta['chest']);native_id=frames[keep[ci]]
R=rot6d_to_matrix(torch.tensor(d['rot6d'],dtype=torch.float64)).numpy();C=np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
err=[]
for t in range(len(R)):
 data.qpos[:]=model.qpos0
 for i,qs in enumerate(qnames):
  for s,name in enumerate(qs):data.qpos[model.jnt_qposadr[model.joint(name).id]]=d['q'][t,i,s]
 mj.mj_forward(model,data)
 delta=(C.T@R[t,0]@C)@data.xmat[frames[keep[0]]].reshape(3,3).T
 actual=C@delta@data.xmat[native_id].reshape(3,3)@C.T
 error=actual@R[t,ci].T;err.append(np.rad2deg(np.arccos(np.clip((np.trace(error)-1)/2,-1,1))))
report=dict(descriptor_node=meta['chest'],native_chest_frame=model.body(native_id).name,rotation_difference_max_deg=float(max(err)))
(folder/'native-chest-validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
