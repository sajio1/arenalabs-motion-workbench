from pathlib import Path
import json,numpy as np,mujoco as mj
from alphamotion.engine.descriptor import build_from_mjcf
from alphamotion.embodiment.mjcf_build import build_merge
root=Path(__import__("os").environ.get("ARENA_EGO_WORKDIR", str(Path(__file__).resolve().parent)))
for robot in ['h2','a3']:
 row=json.loads((root/(robot+'-robot.json')).read_text())[0];model=mj.MjModel.from_xml_path(row['xml']);data=mj.MjData(model)
 spec,dof,rest,qnames,_=build_from_mjcf(row['xml'],row['body']);keep,frames=build_merge(model)
 d=np.load(root/'observed-refinement'/robot/'motion.npz');meta=json.loads((root/'observed-refinement'/robot/'report.json').read_text())['body_frames'];ci=list(spec.joint_names).index(meta['chest']);print(robot,'chest node',ci,meta['chest'],'native frame',model.body(frames[keep[ci]]).name)
 for i,qs in enumerate(qnames):
  for s,name in enumerate(qs):
   if 'waist' in name or 'wrist_pitch' in name:
    j=model.joint(name).id;print(name,'q range',np.rad2deg([d['q'][:,i,s].min(),d['q'][:,i,s].max()]).round(2),'at67',round(np.rad2deg(d['q'][67,i,s]),2),'limit',np.rad2deg(model.jnt_range[j]).round(1))
