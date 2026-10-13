from pathlib import Path
import json,numpy as np,mujoco
from alphamotion.engine.descriptor import build_from_mjcf
from alphamotion.embodiment.mjcf_build import build_merge
root=Path(__import__("os").environ.get("ARENA_EGO_WORKDIR", str(Path(__file__).resolve().parent)))
for robot in ['h2','a3']:
 row=json.loads((root/(robot+'-robot.json')).read_text())[0]
 model=mujoco.MjModel.from_xml_path(row['xml'])
 meta=json.loads((root/'observed-refinement'/robot/'report.json').read_text())['body_frames']
 spec,_,_,_,_=build_from_mjcf(row['xml'],row['body']);keep,frames=build_merge(model)
 b=frames[keep[list(spec.joint_names).index(meta['chest'])]]
 end=frames[keep[list(spec.joint_names).index(meta['pelvis'])]];chain=[]
 while b!=end:
  for j in range(model.body_jntadr[b],model.body_jntadr[b]+model.body_jntnum[b]):
   chain.append(dict(joint=model.joint(j).name,axis=model.jnt_axis[j].tolist(),range_deg=np.rad2deg(model.jnt_range[j]).tolist()))
  b=model.body_parentid[b]
 print(robot,json.dumps(chain))
