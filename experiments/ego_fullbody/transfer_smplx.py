"""An explicit pose transfer with fitted constant shape, not renamed SMPL-H vertices."""
from pathlib import Path
import json,sys,hashlib
import numpy as np,torch,smplx
from scipy.spatial.transform import Rotation
from smplx.joint_names import JOINT_NAMES

from greenwich_motion_sdk.motion import MotionClip
from egoallo.fncsmpl import SmplhModel

root=Path(__import__("os").environ.get("ARENA_EGO_WORKDIR", str(Path(__file__).resolve().parent)))
d=np.load(root/'egoallo_smplh.npz');n=len(d['root_pose']);torch.set_num_threads(8)
bh=d['betas'].mean(0);h=SmplhModel.load(root/'model.npz').with_shape(torch.tensor(bh))
target=np.concatenate([np.zeros((1,3)),h.joints_zero.numpy()])
x=np.load(root/'SMPLX_NEUTRAL.npz',allow_pickle=True)
reg=x['J_regressor'];reg=reg.toarray() if hasattr(reg,'toarray') else reg
base=reg@x['v_template'];shapes=np.einsum('jv,vck->jck',reg,x['shapedirs'][...,:100])
mapping=list(range(22))+list(range(25,55))
A=(shapes-shapes[:1])[mapping].reshape(-1,100)
b=(target-(base-base[:1])[mapping]).ravel()
# Shared constant shape, regularized in model shape units. Poses remain unchanged.
beta=np.linalg.solve(A.T@A+np.eye(100)*1e-5,A.T@b).astype(np.float32)
rest=base+np.einsum('jck,k->jc',shapes,beta)
params=dict(model_path=str(root/'SMPLX_NEUTRAL.npz'),gender='neutral',ext='npz',num_betas=100,use_pca=False,flat_hand_mean=True,batch_size=1)
model=smplx.SMPLX(**params).cuda()
q=d['local_quats'];aa=Rotation.from_quat(q.reshape(-1,4)[:,[1,2,3,0]]).as_rotvec().reshape(n,51,3)
orient=Rotation.from_quat(d['root_pose'][:,[1,2,3,0]]).as_rotvec()
trans=d['root_pose'][:,4:]-rest[0]
verts=[];joints=[]
with torch.no_grad():
 for start in range(0,n,15):
    s=slice(start,min(start+15,n));k=len(aa[s]);T=lambda a:torch.tensor(a,dtype=torch.float32,device='cuda')
    out=model(betas=T(np.broadcast_to(beta,(k,100)).copy()),global_orient=T(orient[s]),transl=T(trans[s]),body_pose=T(aa[s,:21].reshape(k,63)),left_hand_pose=T(aa[s,21:36].reshape(k,45)),right_hand_pose=T(aa[s,36:51].reshape(k,45)),jaw_pose=T(np.zeros((k,3))),leye_pose=T(np.zeros((k,3))),reye_pose=T(np.zeros((k,3))),expression=T(np.zeros((k,10))))
    verts.append(out.vertices.cpu().numpy());joints.append(out.joints.cpu().numpy())
verts=np.concatenate(verts);joints=np.concatenate(joints)
parents=model.parents.cpu().numpy();local=np.concatenate([orient[:,None],aa[:,:21],np.zeros((n,3,3)),aa[:,21:]],1)
localR=Rotation.from_rotvec(local.reshape(-1,3)).as_matrix().reshape(n,55,3,3)
globalR=np.empty_like(localR);offset=np.zeros((55,3));world=np.empty((n,55,3))
for j,p in enumerate(parents):
 if p<0:globalR[:,j]=localR[:,j];world[:,j]=d['root_pose'][:,4:]
 else:
    offset[j]=rest[j]-rest[p];globalR[:,j]=globalR[:,p]@localR[:,j];world[:,j]=world[:,p]+np.einsum('tij,j->ti',globalR[:,p],offset[j])
assert np.max(np.abs(world-joints[:,:55]))<1e-5
C=np.array([[0,1,0],[0,0,1],[1,0,0.]])
aliases={'pelvis':'Hips','head':'Head','left_hip':'LeftLeg','right_hip':'RightLeg','left_knee':'LeftShin','right_knee':'RightShin','left_ankle':'LeftFoot','right_ankle':'RightFoot','left_foot':'LeftToeBase','right_foot':'RightToeBase','left_shoulder':'LeftArm','right_shoulder':'RightArm','left_elbow':'LeftForeArm','right_elbow':'RightForeArm','left_wrist':'LeftHand','right_wrist':'RightHand'}
def anatomical_name(name):
 if name in aliases:return aliases[name]
 side,_,distal=name.partition('_')
 if side in ('left','right') and distal.startswith(('index','middle','pinky','ring','thumb')):return side.title()+'Hand'+distal.title()
 return ''.join(p.title() for p in name.split('_'))
names=[anatomical_name(name) for name in JOINT_NAMES[:55]]
provenance=dict(source='EgoAllo head/wrist reconstruction',human_model='SMPL-X',transfer='constant-shape joint fit and shared local pose; approximation',fingers='inferred prior; no Nymeria finger observations',fps=30,source_sha256=hashlib.sha256((root/'egoallo_smplh.npz').read_bytes()).hexdigest())
# EgoAllo world is Z-up, but SMPL joint LOCAL frames and offsets remain Y-up.
# Transform the world side only; conjugation would rotate the rest skeleton sideways.
MotionClip(names,parents,offset*100,C@globalR,world@C.T*100,d['timestamps_s'],provenance).save(root/'canonical.npz')
neutral=x['v_template']+np.einsum('vck,k->vc',x['shapedirs'][...,:100],beta)
height=((rest[[7,10,8,11],1]-neutral[:,1].min())*100).tolist()
(root/'sources.json').write_text(json.dumps(dict(schema='alphamotion.sources.v1',sources=[dict(id='nymeria_'+json.loads((root/'input_manifest.json').read_text()).get('clip','motion'),format='canonical',path='canonical.npz',roles=['Hips','Head','LeftHand','RightHand','LeftFoot','RightFoot'],foot_pairs=[['LeftFoot','LeftToeBase'],['RightFoot','RightToeBase']],foot_landmark_heights_cm=height)]),indent=2))
np.savez_compressed(root/'smplx_motion.npz',vertices=verts,faces=model.faces,joints=joints,betas=beta,local_axis_angle=local,root_translation=trans,global_rotation=globalR,timestamps_s=d['timestamps_s'])
targetworld=np.concatenate([d['root_pose'][:,None,4:],d['joint_poses'][...,4:]],1)
err=np.linalg.norm(world[:,mapping]-targetworld,axis=-1)*1000
obs=np.load(root/'observations.npz');wrist=np.linalg.norm(world[:,[20,21]]-obs['wrist_position'],axis=-1)*1000
report=dict(frames=n,transfer_joint_error_mm=dict(mean=float(err.mean()),p95=float(np.percentile(err,95)),max=float(err.max())),source_wrist_error_mm=dict(mean=float(wrist.mean()),p95=float(np.percentile(wrist,95)),max=float(wrist.max())),betas_time_std_max=float(d['betas'].std(0).max()),shape_fit_rest_mm=float(np.linalg.norm((rest-rest[:1])[mapping]-target,axis=-1).mean()*1000),fingers=provenance['fingers'])
(root/'human_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
