"""Use the existing body runtime for a committed block, with original frame ids."""
from pathlib import Path
import os,sys,json,runpy
import numpy as np,torch
from scipy.spatial.transform import Rotation
from egoallo import fncsmpl,fncsmpl_extensions
from arenalab_workbench.human_reference import family_for_work

root=Path(sys.argv[1]);folder=Path(sys.argv[2]);start=int(sys.argv[3]);end=int(sys.argv[4])
torch.set_num_threads(2);device=torch.device('cuda:0');T=lambda a:torch.as_tensor(a,dtype=torch.float32,device=device)
data=np.load(folder/'committed-input.npz');obs=np.load(root/'observations.npz');count=end-start
cpf=np.concatenate([Rotation.from_matrix(obs['head_rotation']).as_quat()[:,[3,0,1,2]],obs['head_position']],-1)
body=fncsmpl.SmplhModel.load(root/'model.npz').to(device)
with torch.no_grad():
    posed=body.with_shape(T(data['betas'])).with_pose(T(np.tile([1.,0,0,0,0,0,0],(count,1))),T(data['local_quats']))
    rp=fncsmpl_extensions.get_T_world_root_from_cpf_pose(posed,T(cpf[start+1:end+1]));posed=posed.with_new_T_world_root(rp)
    native=dict(root_pose=rp.cpu().numpy(),joint_poses=posed.Ts_world_joint.cpu().numpy(),local_quats=data['local_quats'],betas=np.broadcast_to(data['betas'],(count,16)),timestamps_s=obs['timestamps_s'][start:end])
    family_for_work(root)
    native.update(vertices=posed.lbs().verts.cpu().numpy(),faces=body.faces.cpu().numpy())
    np.savez_compressed(folder/'egoallo_smplh.npz',**native)
os.environ['ARENA_EGO_WORKDIR']=str(folder)
np.savez_compressed(folder/'observations.npz',wrist_position=obs['wrist_position'][start:end],timestamps_s=obs['timestamps_s'][start:end])
runpy.run_path(str(Path(__file__).with_name('emit_smplh.py')),run_name='__main__')
