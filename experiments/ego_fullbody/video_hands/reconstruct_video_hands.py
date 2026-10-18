"""RGB-derived MANO estimates; retain missing-frame masks and all raw candidates."""
from pathlib import Path
import os,sys,json,inspect,time
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
import numpy as np
for name,value in [('bool',bool),('int',int),('float',float),('complex',complex),('object',object),('str',str),('unicode',str)]:
 if name not in np.__dict__:setattr(np,name,value)
if not hasattr(inspect,'getargspec'):inspect.getargspec=inspect.getfullargspec
import cv2,torch
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
root=Path(os.environ['ARENA_EGO_WORKDIR']);out=root/'video-hands';out.mkdir(exist_ok=True)
torch.set_num_threads(4)
pipe=WiLorHandPose3dEstimationPipeline(device=torch.device('cuda'),dtype=torch.float32,verbose=False,wilor_pretrained_dir='/home/ubuntu/arenalab-experiments/egoallo/video-hand-weights')
cap=cv2.VideoCapture(str(root/'00-source.mp4'));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
assert abs(fps-30)<1e-4
stride=int(os.environ.get('HAND_FRAME_STRIDE','1'))
scores=np.zeros((n,2));pose=np.full((n,2,15,3),np.nan);verts=np.full((n,2,778,3),np.nan);joints=np.full((n,2,21,3),np.nan);xy=np.full((n,2,21,2),np.nan);orient=np.full((n,2,3),np.nan);betas=np.full((n,2,10),np.nan)
candidates=[];previews=[];start=time.time()
for f in range(n):
 ok,im=cap.read()
 if not ok:raise RuntimeError(('short video',f,n))
 if f%stride:continue
 image=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
 # Detector and reconstruction follow upstream RGB demo. Preserve detector scores.
 det=pipe.hand_detector(image,conf=.35,verbose=False)[0].boxes.data.cpu().numpy()
 b=[];s=[];cs=[]
 for side in [0,1]:
  choices=det[det[:,5]==side]
  if len(choices):
   best=choices[np.argmax(choices[:,4])];b.append(best[:4]);s.append(side);cs.append(float(best[4]))
 result=pipe.predict_with_bboxes(image,np.asarray(b),np.asarray(s)) if b else []
 for item,score in zip(result,cs):
  si=int(item['is_right']);d=item['wilor_preds'];scores[f,si]=score
  pose[f,si]=d['hand_pose'].reshape(15,3);verts[f,si]=d['pred_vertices'].reshape(778,3);joints[f,si]=d['pred_keypoints_3d'].reshape(21,3);xy[f,si]=d['pred_keypoints_2d'].reshape(21,2);orient[f,si]=d['global_orient'].reshape(3);betas[f,si]=d['betas'].reshape(10)
  candidates.append(dict(frame=f,side=si,score=score,bbox=item['hand_bbox']))
  for p in xy[f,si]:cv2.circle(im,tuple(np.round(p).astype(int)),2,(0,230,100) if si==0 else (0,160,255),-1)
  x,y=map(int,item['hand_bbox'][:2]);cv2.putText(im,('L' if si==0 else 'R')+f' {score:.2f}',(max(0,x),max(20,y)),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
 if f%30==0 or (result and len(previews)<12):
  cv2.putText(im,f'Frame {f}',(20,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
  previews.append(cv2.resize(im,(320,320)))
 if f%30==0:print('FRAME',f,'DETECTIONS',len(result),'SECONDS',time.time()-start,flush=True)
cap.release()
np.savez_compressed(out/'raw-mano.npz',confidence=scores,hand_pose=pose,vertices=verts,joints=joints,image_xy=xy,global_orient=orient,betas=betas,faces=pipe.wilor_model.mano.faces,fps=fps)
(out/'detections.json').write_text(json.dumps(candidates,indent=2))
report=dict(frames=n,stride=stride,detected_per_hand=(scores>0).sum(0).tolist(),above_05=(scores>=.5).sum(0).tolist(),bilateral_frames=int((scores>=.5).all(1).sum()),seconds=time.time()-start,provenance='WiLoR monocular RGB estimate; not Nymeria finger ground truth')
(out/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
while len(previews)%4:previews.append(np.zeros_like(previews[0]))
cv2.imwrite(str(out/'detections.jpg'),np.vstack([np.hstack(previews[i:i+4]) for i in range(0,len(previews),4)]))
