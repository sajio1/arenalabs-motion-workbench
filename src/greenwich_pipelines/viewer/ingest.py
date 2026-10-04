"""Persist a common humanoid descriptor and the identical model used for FK."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from ..robot_semantics import resolve_endpoints,ROLES
from .export import array_json

def ingest_humanoid(urdf,folder,body,report):
    import mujoco as mj
    from alphamotion.engine.descriptor import build_from_mjcf
    folder=Path(folder).resolve();native=folder/'model.xml'
    if report.get('mimic_joints'):raise ValueError('Mimic joints require explicit mechanical coupling constraints')
    original=mj.MjModel.from_xml_path(str(urdf))
    mj.mj_saveLastXML(str(native),original)
    # URDF describes the physical assembly, often omitting a world free joint.
    tree=ET.parse(native);root=tree.getroot();bases=root.findall('./worldbody/body')
    if len(bases)!=1:raise ValueError('Expected one connected robot root')
    base=bases[0];added=False
    if not any(j.get('type')=='free' for j in root.findall('.//joint')) and not root.findall('.//freejoint'):
        if base.findall('joint'):raise ValueError('Root has constrained joints; base topology needs confirmation')
        ET.SubElement(base,'freejoint',name='uploaded_world_root');added=True
    tree.write(native,encoding='utf-8',xml_declaration=True)
    model=mj.MjModel.from_xml_path(str(native));spec,dof,rest,qn,_=build_from_mjcf(native,body)
    indices=resolve_endpoints(spec);endpoints=dict(zip(ROLES,[spec.joint_names[i] for i in indices]))
    if added:
        position=np.fromstring(base.get('pos','0 0 0'),sep=' ')
        position[2]+=float(spec.rest_offsets[0,1])/100
        base.set('pos',' '.join(map(str,position)));tree.write(native,encoding='utf-8',xml_declaration=True)
        model=mj.MjModel.from_xml_path(str(native))
    data=mj.MjData(model);mj.mj_forward(model,data);palms={};details={}
    for role,i in zip(ROLES[:2],indices[:2]):
        bid=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,spec.joint_names[i]);position=data.xpos[bid]
        parent=int(model.body_parentid[bid])
        while parent>0 and np.linalg.norm(position-data.xpos[parent])<1e-5:parent=int(model.body_parentid[parent])
        distal=position-data.xpos[parent]
        if np.linalg.norm(distal)<1e-5:raise ValueError(f'{role}: no physical distal direction; frame confirmation required')
        distal/=np.linalg.norm(distal)
        # Neutral palmar normal faces inward. Mesh thickness supplies a more
        # informative plane when the endpoint owns a non-isotropic hand mesh.
        inward=np.array([0.,-1. if role.startswith('left') else 1.,0.]);normal=inward.copy();method='neutral inward normal and distal chain'
        for g in np.flatnonzero(model.geom_bodyid==bid):
            if model.geom_type[g]!=mj.mjtGeom.mjGEOM_MESH:continue
            mid=model.geom_dataid[g];a=model.mesh_vertadr[mid];n=model.mesh_vertnum[mid]
            v=model.mesh_vert[a:a+n].astype(float);values,vectors=np.linalg.eigh(np.cov(v.T));candidate=data.geom_xmat[g].reshape(3,3)@vectors[:,0]
            if values[1]>0 and values[0]/values[1]<.5 and abs(candidate@distal)<.7 and abs(candidate@inward)>.3:
                normal=candidate*np.sign(candidate@inward);method='mesh thickness normal and distal chain';break
        normal-=distal*(normal@distal)
        if np.linalg.norm(normal)<1e-5:raise ValueError(f'{role}: degenerate palm frame; frame confirmation required')
        normal/=np.linalg.norm(normal);R=data.xmat[bid].reshape(3,3)
        palms[role]=dict(distal=(R.T@distal).tolist(),normal=(R.T@normal).tolist());details[role]=method
    row=dict(name=body,body=body,xml=str(native),model=str(native),endpoints=endpoints,palm_frames=palms)
    # Keep physical metadata inspectable and persistent, not a named robot solver.
    array_json(folder/'descriptor.json',dict(schema='arenalab.robot.v1',**row,frame_provenance=details))
    np.savez_compressed(folder/'descriptor.npz',parents=spec.parents,rest_offsets=spec.rest_offsets,joint_names=np.asarray(spec.joint_names),dof=dof,rest=rest)
    report.update(status='ready',conversion_ready=True,floating_root_added=added,endpoints=endpoints,palm_frames=palms,frame_provenance=details,message='Humanoid descriptor ready. Wrist frames inferred from neutral physical geometry; inspect axes when accurate TCP calibration is required.')
    return row
