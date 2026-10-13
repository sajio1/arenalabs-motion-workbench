from pathlib import Path
import json,numpy as np,mujoco
root=Path(__import__("os").environ.get("ARENA_EGO_WORKDIR", str(Path(__file__).resolve().parent)))
for robot in ['h2','a3']:
 row=json.loads((root/(robot+'-robot.json')).read_text())[0]
 model=mujoco.MjModel.from_xml_path(row['xml'])
 meta=json.loads((root/'observed-refinement'/robot/'report.json').read_text())['body_frames']
 b=model.body(meta['chest']).id;end=model.body(meta['pelvis']).id;chain=[]
 while b!=end:
  for j in range(model.body_jntadr[b],model.body_jntadr[b]+model.body_jntnum[b]):
   chain.append(dict(joint=model.joint(j).name,axis=model.jnt_axis[j].tolist(),range_deg=np.rad2deg(model.jnt_range[j]).tolist()))
  b=model.body_parentid[b]
 print(robot,json.dumps(chain))
