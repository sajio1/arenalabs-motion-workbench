from pathlib import Path
import os,json,time
from egoallo_runtime import gpu_environment
os.environ.update(gpu_environment())
import jax
if jax.default_backend() != 'gpu':
    raise RuntimeError('EgoAllo requires CUDA JAX; CPU fallback is disabled')
print('JAX_BACKEND',jax.default_backend(),jax.devices(),flush=True)
import numpy as np,torch
from scipy.spatial.transform import Rotation
from egoallo import fncsmpl,fncsmpl_extensions
from egoallo.inference_utils import load_denoiser
from egoallo.sampling import run_sampling_with_stitching,quadratic_ts
if len(quadratic_ts()) != 31:
    raise RuntimeError("Expected the original 30-step EgoAllo schedule")
from egoallo.hand_detection_structs import AriaHandWristPoseWrtWorld,CorrespondedAriaHandWristPoseDetections

root=Path(__import__("os").environ.get("ARENA_EGO_WORKDIR", str(Path(__file__).resolve().parent)))
torch.set_num_threads(8);torch.manual_seed(20261013)
device=torch.device('cuda:0');d=np.load(root/'observations.npz')
body=fncsmpl.SmplhModel.load(root/'model.npz').to(device)
offset=fncsmpl_extensions.get_T_head_cpf(body.with_shape(torch.zeros(16,device=device))).cpu().numpy()[4:]
cpf_pos=d['head_position']  # Source optical camera proxy; no guessed translation
cpf=np.concatenate([Rotation.from_matrix(d['head_rotation']).as_quat()[:,[3,0,1,2]],cpf_pos],-1)
T=lambda a:torch.as_tensor(a,dtype=torch.float32,device=device)
hands=[];n=len(d['timestamps_s'])
for i in range(2):
    hands.append(AriaHandWristPoseWrtWorld(confidence=T(np.ones(n)),wrist_position=T(d['wrist_position'][:,i]),palm_position=T(d['palm_position'][:,i]),wrist_normal=T(d['palm_normal'][:,i]),palm_normal=T(d['palm_normal'][:,i]),indices=torch.arange(n,device=device)))
model=load_denoiser(root/'egoallo_checkpoint_april13/checkpoints_3000000').to(device)
tic=time.time()
with torch.no_grad():
    traj=run_sampling_with_stitching(model,body,'aria_wrist_only',True,True,T(cpf),float(d['floor_z']),None,CorrespondedAriaHandWristPoseDetections(*hands),1,device,guidance_verbose=False)
    posed=traj.apply_to_body(body)
    rootpose=fncsmpl_extensions.get_T_world_root_from_cpf_pose(posed,T(cpf[1:]))
    posed=posed.with_new_T_world_root(rootpose)
    mesh=posed.lbs()
    arr=lambda x:x.detach().cpu().numpy()[0]
    np.savez_compressed(root/'egoallo_smplh.npz',root_pose=arr(rootpose),local_quats=arr(posed.local_quats),joint_poses=arr(posed.Ts_world_joint),vertices=arr(mesh.verts),faces=body.faces.cpu().numpy(),betas=arr(traj.betas),contacts=arr(traj.contacts),cpf=cpf[1:],timestamps_s=d['timestamps_s'])
print('EGOALLO_COMPLETE_SECONDS',time.time()-tic,flush=True)
