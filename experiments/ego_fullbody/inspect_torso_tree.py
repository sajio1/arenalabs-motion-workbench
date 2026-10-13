from pathlib import Path
import json,mujoco as mj,numpy as np
from greenwich_pipelines.pipelines.nymeria.embodiment_exact import load_robot
root=Path(__import__("os").environ.get("ARENA_EGO_WORKDIR", str(Path(__file__).resolve().parent)));row=json.loads((root/'h2-robot.json').read_text())[0];body=load_robot(row);model=mj.MjModel.from_xml_path(row['xml']);names=list(body.spec.joint_names)
for endpoint in ['left_wrist','right_wrist']:
 j=names.index(row['endpoints'][endpoint]);chain=[]
 while j>=0:chain.append(names[j]);j=int(body.spec.parents[j])
 print('DESCRIPTOR',endpoint,chain)
 j=model.body(row['endpoints'][endpoint]).id;chain=[]
 while j>0:chain.append(model.body(j).name);j=int(model.body_parentid[j])
 print('NATIVE',endpoint,chain)
for j in range(model.njnt):
 name=model.joint(j).name
 if any(k in name for k in ['waist','torso']):print('JOINT',j,name,'body',model.body(model.jnt_bodyid[j]).name,'axis',model.jnt_axis[j],'range',np.rad2deg(model.jnt_range[j]))
for b in range(model.nbody):
 name=model.body(b).name
 if any(k in name for k in ['waist','torso','shoulder','head']):print('BODY',b,name,'parent',model.body(model.body_parentid[b]).name,'geom',model.body_geomnum[b])
for e in range(model.neq):print('EQUALITY',model.eq_type[e],model.eq_obj1id[e],model.eq_obj2id[e],model.eq_data[e])
