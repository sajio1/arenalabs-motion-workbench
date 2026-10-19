"""Verify source finger articulation against actual Wuji FK, excluding wrist motion."""
from arenalab_workbench.settings import paths
from pathlib import Path
import json,hashlib,os,sys,xml.etree.ElementTree as ET
import numpy as np,mujoco as mj

ROOT=Path(os.environ['ARENA_SOURCE_WORK'])
identifier=sys.argv[1]
DEST=Path(os.environ['ARENA_EXPORT_DIR'])
source_path=ROOT/'source_hands.npz';source=np.load(source_path)['points']
angles=np.load(Path(os.environ.get('ARENA_RETARGET_WORK',str(ROOT/'retarget')))/f'operation-accuracy/palm/{identifier}/wuji_angles.npz')['angles']
assert angles.shape==(len(source),2,20) and np.isfinite(angles).all()

def bends(points):
    edges=np.diff(points,axis=-2);norm=np.linalg.norm(edges,axis=-1,keepdims=True)
    if np.any(norm<1e-7):raise ValueError('Degenerate finger bone')
    direction=edges/norm
    return np.degrees(np.arccos(np.clip(np.sum(direction[...,:-1,:]*direction[...,1:,:],axis=-1),-1,1))).sum(-1)
source_curl=bends(source[:,:,1:].reshape(len(source),2,5,4,3));robot_curl=[];local_tip=[];limits=[]
for si,side in enumerate(['left','right']):
    path=ROOT/f'wuji/urdf/{side}.urdf';model=mj.MjModel.from_xml_path(str(path));data=mj.MjData(model);tree=ET.parse(path)
    offsets=[np.fromstring(tree.find(f"joint[@name='{side}_finger{finger}_tip_fixed']/origin").attrib['xyz'],sep=' ') for finger in range(1,6)]
    positions=[];qs=[]
    for t in range(len(source)):
        for finger in range(1,6):
            for joint in range(1,5):
                jid=model.joint(f'{side}_finger{finger}_joint{joint}').id
                data.qpos[model.jnt_qposadr[jid]]=angles[t,si,(finger-1)*4+joint-1]
        mj.mj_forward(model,data);frame=[]
        for finger in range(1,6):
            ids=[model.body(f'{side}_finger{finger}_link{joint}').id for joint in [2,3,4]]
            tip=data.xpos[ids[-1]]+data.xmat[ids[-1]].reshape(3,3)@offsets[finger-1]
            frame.append(np.r_[data.xpos[ids],tip[None]])
        positions.append(frame)
    positions=np.asarray(positions);robot_curl.append(bends(positions));local_tip.append(positions[:,:,-1])
    for finger in range(1,6):
        for joint in range(1,5):
            jid=model.joint(f'{side}_finger{finger}_joint{joint}').id;limits.append(model.jnt_range[jid].tolist())
robot_curl=np.stack(robot_curl,1);local_tip=np.stack(local_tip,1);rows=[]
for si,side in enumerate(['left','right']):
    for finger,name in enumerate(['thumb','index','middle','ring','little']):
        a=source_curl[:,si,finger];b=robot_curl[:,si,finger]
        corr=float(np.corrcoef(a,b)[0,1]) if min(a.std(),b.std())>1e-5 else None
        rows.append(dict(side=side,finger=name,source_curl_range_deg=float(np.ptp(a)),robot_curl_range_deg=float(np.ptp(b)),curl_correlation=corr,maximum_joint_range_deg=float(np.ptp(angles[:,si,finger*4:finger*4+4],axis=0).max()*180/np.pi),local_tip_bounding_diagonal_mm=float(np.linalg.norm(np.ptp(local_tip[:,si,finger],axis=0))*1000)))
report=dict(source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),frames=len(source),fps=30,method='Source anatomical finger bends versus native Wuji FK, excluding palm movement',verified_robots=[identifier],finger_motion=rows,source_curl_deg=source_curl.tolist(),robot_curl_deg=robot_curl.tolist(),limitations=['Bend correlation is not object contact accuracy'])
report['frozen']=bool(np.max(np.ptp(angles,axis=0))<=.01)
(DEST/('hand-motion-audit-'+identifier+'.json')).write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['source_curl_deg','robot_curl_deg']},indent=2))

from arenalab_workbench.gpu_scheduler import merge_scene
merge_scene(DEST/'scene.json',lambda scene:scene.setdefault('hand_motion_audit_urls',{}).update({identifier:'/assets/'+(DEST/('hand-motion-audit-'+identifier+'.json')).relative_to(str(paths.store)).as_posix()}))
