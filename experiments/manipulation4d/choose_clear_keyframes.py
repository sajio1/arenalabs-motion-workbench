"""Choose less hand-occluded object views with one category-independent rule."""
from pathlib import Path
import json
import cv2,numpy as np
from prepare_objects import ROOT,OUT,depth,frame_ids,camera_points,depth_calibration

DEST=Path('/opt/arenalab/store/manipulation4d/pilot');rows=[]
hand=np.load(OUT/'hand-masks.npz')['masks']>0
for row in json.loads((OUT/'selected-objects.json').read_text()):
    track=json.loads((DEST/f"{row['kind']}-{row['instance_id']}-track.json").read_text())
    labels=np.load(OUT/f"{row['kind']}-masks.npz")['masks'];scores=[]
    for i,frame in enumerate(track):
        mask=(labels[i]==frame['segmentation_instance_id']).astype('u1');area=int(mask.sum())
        if area<100:scores.append(0.);continue
        adjacent=cv2.dilate(hand[i].astype('u1'),np.ones((31,31),np.uint8))>0
        clear=float((mask[~adjacent]>0).sum()/area)
        scores.append(area*clear*clear)
    key=int(np.argmax(scores));folder=ROOT/'clearviews'/f"{row['kind']}-{row['instance_id']}";folder.mkdir(parents=True,exist_ok=True)
    image=cv2.imread(str(ROOT.parent/'da3/no-hands/frames'/f'{key:05d}.jpg'))
    mask=(labels[key]==track[key]['segmentation_instance_id']).astype('u1')*255
    cv2.imwrite(str(folder/'image.png'),image);cv2.imwrite(str(folder/'mask.png'),mask);np.save(folder/'camera-pointmap.npy',camera_points(key).astype('float32'))
    nextrow={**row,'keyframe':key,'source_frame':int(frame_ids[key]),'keyframe_seconds':float(frame_ids[key]/30),'camera_extrinsics':depth['extrinsics'][key].tolist(),'folder':str(folder),'selection':'maximum mask area times squared fraction outside a 15-pixel hand neighborhood; shared across object categories'}
    nextrow['pointmap_depth_calibration']=depth_calibration
    (folder/'input.json').write_text(json.dumps(nextrow,indent=2));rows.append(nextrow)
    print(row['kind'],key,int(frame_ids[key]),scores[key],flush=True)
(ROOT/'clearviews/selected-objects.json').write_text(json.dumps(rows,indent=2))
