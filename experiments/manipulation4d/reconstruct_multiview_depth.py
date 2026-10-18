"""Reconstruct the same source with current multiview metric model on GPU 3 only."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','3');os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import time,json,hashlib,cv2,h5py,numpy as np,torch
from depth_anything_3.api import DepthAnything3
ROOT=Path('/home/ubuntu/arenalab-experiments/manipulation4d');EXP=ROOT.parent/'egodex-threepoint';manifest=json.loads((EXP/'input_manifest.json').read_text());source=Path(manifest['source']);srcvideo=source.with_suffix('.mp4')
cap=cv2.VideoCapture(str(srcvideo));images=[];ids=[];frame=0
while True:
 ok,image=cap.read()
 if not ok:break
 if frame%2==0:images.append(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));ids.append(frame)
 frame+=1
cap.release();ids=np.array(ids)
with h5py.File(source) as f:K=f['camera/intrinsic'][()];E=np.linalg.inv(f['transforms/camera'][()][ids])
model_name='depth-anything/DA3NESTED-GIANT-LARGE-1.1';print('LOAD',model_name,flush=True);model=DepthAnything3.from_pretrained(model_name).to('cuda');print('LOADED',flush=True)
out={k:[] for k in ['depth','conf','images','intrinsics','extrinsics']};anchors=np.linspace(0,len(ids)-1,9).astype(int);t=time.time()
for start in range(0,len(ids),16):
 core=np.arange(start,min(start+16,len(ids)));batch=np.unique(np.r_[core,anchors]);p=model.inference([images[i] for i in batch],extrinsics=E[batch].astype('float32'),intrinsics=np.repeat(K[None],len(batch),0).astype('float32'),process_res=504,align_to_input_ext_scale=True,ref_view_strategy='saddle_balanced');sel=np.searchsorted(batch,core)
 for key in out:out[key].append(getattr(p,'processed_images' if key=='images' else key)[sel])
 print('DEPTH',start,len(core),time.time()-t,flush=True)
np.savez_compressed(ROOT/'depth-multiview-v2.npz',**{k:np.concatenate(v) for k,v in out.items()},ids=ids)
report=dict(model=model_name,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),frames=len(ids),fps=15,inference_seconds=time.time()-t,physical_gpu=int(os.environ.get('CUDA_VISIBLE_DEVICES','3')),source_trajectory_edited=False,camera_pose_max_difference=float(np.max(abs(np.concatenate(out['extrinsics'])-E[:,:3]))),post_scale_applied=False)
(ROOT/'depth-multiview-v2.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
