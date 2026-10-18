"""Known-camera multiview triangulation audit; no source trajectories modified."""
from pathlib import Path
import json,cv2,numpy as np
from prepare_objects import OUT,depth,mask_at
ROOT=Path('/home/ubuntu/arenalab-experiments/manipulation4d');I=depth['images'];K=depth['intrinsics'];E=depth['extrinsics'];T,H,W=depth['depth'].shape
masks={n:np.load(OUT/f'{n}-masks.npz')['masks'] for n in ['hand','arm','cup','lid']}
sift=cv2.SIFT_create(nfeatures=3500,contrastThreshold=.015);features={}
for i in range(0,T,4):
 human=(mask_at(masks['hand'],i)>0)|(mask_at(masks['arm'],i)>0);human=cv2.dilate(human.astype('u1'),np.ones((9,9),np.uint8))>0
 kp,d=sift.detectAndCompute(cv2.cvtColor(I[i],cv2.COLOR_RGB2GRAY),(~human).astype('u1')*255);features[i]=(np.array([k.pt for k in kp],dtype='float32'),d)
allp=[];allc=[];alluv=[];alld=[];allf=[];isobj=[];pairrows=[];bf=cv2.BFMatcher()
for i,(ui,di) in features.items():
 for offset in [4,12,24,48]:
  j=i+offset
  if j not in features:continue
  uj,dj=features[j];matches=[a for a,b in bf.knnMatch(di,dj,k=2) if a.distance<.7*b.distance]
  if len(matches)<5:continue
  a=ui[[m.queryIdx for m in matches]];b=uj[[m.trainIdx for m in matches]]
  p=cv2.triangulatePoints(K[i]@E[i],K[j]@E[j],a.T,b.T);p=(p[:3]/p[3:]).T
  ca=p@E[i,:3,:3].T+E[i,:3,3];cb=p@E[j,:3,:3].T+E[j,:3,3]
  pa=ca@K[i].T;pb=cb@K[j].T;err=np.maximum(np.linalg.norm(pa[:,:2]/pa[:,2:]-a,axis=1),np.linalg.norm(pb[:,:2]/pb[:,2:]-b,axis=1))
  origins=[-E[t,:3,:3].T@E[t,:3,3] for t in [i,j]];ra=p-origins[0];rb=p-origins[1];angle=np.degrees(np.arccos(np.clip(np.sum(ra*rb,axis=1)/np.linalg.norm(ra,axis=1)/np.linalg.norm(rb,axis=1),-1,1)))
  valid=(err<.45)&(angle>1.5)&(ca[:,2]>.08)&(cb[:,2]>.08)&(ca[:,2]<4)&(cb[:,2]<4)&np.isfinite(p).all(1)
  # Independent third-frame image correspondence, not depth agreement.
  mid=4*round((i+j)/8)
  if mid in [i,j] or mid not in features:continue
  um,dm=features[mid];mapm={m.queryIdx:m.trainIdx for m,n in bf.knnMatch(di,dm,k=2) if m.distance<.7*n.distance}
  cm=p@E[mid,:3,:3].T+E[mid,:3,3];pm=cm@K[mid].T;pm=pm[:,:2]/pm[:,2:];third=np.array([np.linalg.norm(pm[t]-um[mapm[m.queryIdx]]) if m.queryIdx in mapm else 999 for t,m in enumerate(matches)])
  valid&=(third<.8)&(cm[:,2]>.08)
  if not valid.any():continue
  p=p[valid];a=a[valid];ca=ca[valid];uv=np.rint(a).astype(int);ys,xs=uv[:,1],uv[:,0];pred=depth['depth'][i,ys,xs]
  obj=(mask_at(masks['cup'],i)[ys,xs]>0)|(mask_at(masks['lid'],i)[ys,xs]>0)
  allp.extend(p);allc.extend(I[i,ys,xs]);alluv.extend(a);alld.extend(np.c_[pred,ca[:,2]]);allf.extend([i]*len(p));isobj.extend(obj)
  pairrows.append(dict(i=i,j=j,count=len(p),baseline_m=float(np.linalg.norm(origins[0]-origins[1])),reprojection_median_px=float(np.median(err[valid])),third_view_median_px=float(np.median(third[valid]))))
p=np.asarray(allp);d=np.asarray(alld);obj=np.asarray(isobj,dtype=bool)
np.savez_compressed(ROOT/'multiview-triangulation.npz',points=p,colors=np.asarray(allc,dtype='u1'),uv=np.asarray(alluv),depth=d,frames=np.asarray(allf),objects=obj)
report=dict(points=len(p),background_points=int((~obj).sum()),pairs=pairrows)
for name,keep in [('all',np.ones(len(p),bool)),('background',~obj),('objects',obj)]:
 if keep.sum():
  e=abs(d[keep,0]-d[keep,1])*1000;ratio=d[keep,1]/d[keep,0];report[name]=dict(median_error_mm=float(np.median(e)),p95_error_mm=float(np.percentile(e,95)),median_ratio=float(np.median(ratio)),ratio_p10_p90=np.percentile(ratio,[10,90]).tolist())
(ROOT/'multiview-triangulation.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='pairs'},indent=2),flush=True)
