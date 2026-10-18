"""Replace finger local rotations only; keep accepted body and wrists immutable."""
from pathlib import Path
import os,json,hashlib
import numpy as np,torch,smplx
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
root=Path(os.environ['ARENA_EGO_WORKDIR']);out=root/'video-hands';d=dict(np.load(root/'smplx_motion.npz'));raw=np.load(out/'reviewed-mano.npz');n=len(d['vertices']);aa=d['local_axis_angle'].copy();original=aa.copy()
torch.set_num_threads(4)
def continuous(q):
 for i in range(1,len(q)):
  q[i]*=np.where((q[i]*q[i-1]).sum(-1,keepdims=True)<0,-1,1)
 return q
weights=np.zeros((n,2));angles=[]
for side,sl in [(0,slice(25,40)),(1,slice(40,55))]:
 base=continuous(Rotation.from_rotvec(aa[:,sl].reshape(-1,3)).as_quat().reshape(n,15,4));valid=raw['status'][:,side]>0
 candidate=base.copy()
 candidate[valid]=Rotation.from_rotvec(raw['hand_pose'][valid,side].reshape(-1,3)).as_quat().reshape(-1,15,4)
 candidate*=np.where((candidate*base).sum(-1,keepdims=True)<0,-1,1)
 w=np.where(raw['status'][:,side]==2,1.,np.where(valid,.7,0.))
 # Soft temporal transitions. Unseen intervals return to the inferred body prior.
 w=gaussian_filter1d(w,2);weights[:,side]=w
 delta=np.where(valid[:,None,None],candidate-base,0.)
 denom=gaussian_filter1d(valid.astype(float),2)
 smooth=gaussian_filter1d(delta,2,axis=0)/np.maximum(denom[:,None,None],1e-6)
 q=base+w[:,None,None]*smooth;q/=np.linalg.norm(q,axis=-1,keepdims=True)
 q=continuous(q);q=gaussian_filter1d(q,1,axis=0);q/=np.linalg.norm(q,axis=-1,keepdims=True)
 aa[:,sl]=Rotation.from_quat(q.reshape(-1,4)).as_rotvec().reshape(n,15,3)
 angle=(Rotation.from_rotvec(original[:,sl].reshape(-1,3)).inv()*Rotation.from_rotvec(aa[:,sl].reshape(-1,3))).magnitude();angles.append(float(np.rad2deg(angle).max()))
modelpath=root/'SMPLX_NEUTRAL.npz'
if not modelpath.exists():modelpath=root.parent/'SMPLX_NEUTRAL.npz'
model=smplx.SMPLX(model_path=str(modelpath),gender='neutral',ext='npz',num_betas=100,use_pca=False,flat_hand_mean=True,batch_size=1).cuda()
T=lambda x:torch.tensor(x,dtype=torch.float32,device='cuda')
vs=[];js=[]
with torch.no_grad():
 for start in range(0,n,15):
  sl=slice(start,min(start+15,n));k=len(aa[sl]);z=T(np.zeros((k,3)))
  pred=model(betas=T(np.broadcast_to(d['betas'],(k,100)).copy()),global_orient=T(aa[sl,0]),transl=T(d['root_translation'][sl]),body_pose=T(aa[sl,1:22].reshape(k,63)),left_hand_pose=T(aa[sl,25:40].reshape(k,45)),right_hand_pose=T(aa[sl,40:55].reshape(k,45)),jaw_pose=z,leye_pose=z,reye_pose=z,expression=T(np.zeros((k,10))))
  vs.append(pred.vertices.cpu().numpy());js.append(pred.joints.cpu().numpy())
vertices=np.concatenate(vs);joints=np.concatenate(js);error=float(np.abs(joints[:,:25]-d['joints'][:,:25]).max())
assert error<1e-5,error
globalR=np.empty_like(d['global_rotation']);localR=Rotation.from_rotvec(aa.reshape(-1,3)).as_matrix().reshape(n,55,3,3)
for j,p in enumerate(model.parents.cpu().numpy()):globalR[:,j]=localR[:,j] if p<0 else globalR[:,p]@localR[:,j]
d.update(vertices=vertices,joints=joints,local_axis_angle=aa,global_rotation=globalR,finger_image_weight=weights,finger_estimate_status=raw['status'],finger_provenance='WiLoR associated RGB prediction blended with prior for occlusion; not measured fingers')
np.savez_compressed(out/'smplx_motion.npz',**d)
report=dict(body_joint_max_difference_m=error,body_local_rotations_unchanged=bool(np.array_equal(aa[:,:25],original[:,:25])),source_human_sha256=hashlib.sha256((root/'smplx_motion.npz').read_bytes()).hexdigest(),max_finger_rotation_change_deg=angles,method='SO3 quaternion blending; reviewed image estimate weight 1, missing weight 0; temporal sigma 2 frames; no body re-solve',raw_observation_claim=False)
(out/'finger-transfer-validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
