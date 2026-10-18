"""Prepare category-independent clear object views in the depth model's exact image grid."""
from pathlib import Path
import json,cv2,numpy as np
from prepare_objects import ROOT,OUT,depth,frame_ids,mask_at,camera_points
folder=ROOT/'asset-completion-v3';folder.mkdir(exist_ok=True);rows=[]
hand=np.load(OUT/'hand-masks.npz')['masks'];arm=np.load(OUT/'arm-masks.npz')['masks']
for kind in ['cup','lid']:
 labels=np.load(OUT/f'{kind}-masks.npz')['masks'];best=None
 for i in range(len(frame_ids)):
  masks=mask_at(labels,i);human=(mask_at(hand,i)>0)|(mask_at(arm,i)>0);near=cv2.dilate(human.astype('u1'),np.ones((17,17),'u1'))>0
  for ident in np.unique(masks):
   if ident==0:continue
   mask=masks==ident;area=int(mask.sum());clear=float((mask&~near).sum()/max(area,1));score=area*clear*clear
   if best is None or score>best[0]:best=(score,i,int(ident))
 score,i,ident=best;dest=folder/kind;dest.mkdir(exist_ok=True);mask=(mask_at(labels,i)==ident).astype('u1')*255
 cv2.imwrite(str(dest/'image.png'),cv2.cvtColor(depth['images'][i],cv2.COLOR_RGB2BGR));cv2.imwrite(str(dest/'mask.png'),mask);np.save(dest/'camera-pointmap.npy',camera_points(i).astype('f4'))
 row=dict(kind=kind,instance_id=ident,keyframe=i,source_frame=int(frame_ids[i]),camera_extrinsics=depth['extrinsics'][i].tolist(),folder=str(dest),selection='Largest visible area times squared unoccluded fraction; same rule for all categories; exact processed RGB/mask/pointmap grid',score=score)
 (dest/'input.json').write_text(json.dumps(row,indent=2));rows.append(row)
(folder/'selected-objects.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
