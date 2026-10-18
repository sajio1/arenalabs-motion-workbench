"""Choose interacted instances from measured source hand proximity, then keyframes."""
from pathlib import Path
import cv2, h5py, json, os
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path('/home/ubuntu/arenalab-experiments/manipulation4d')
OUT=ROOT/'cup-lids'
BASE=Path('/home/ubuntu/arenalab-experiments/da3/no-hands')
depth_path=Path(os.environ.get('ARENA_DEPTH_FILE',str(BASE/'depth-sequence.npz')))
depth_calibration=json.loads((ROOT/'depth-calibration.json').read_text()) if depth_path==ROOT/'depth-calibrated.npz' else None
with np.load(depth_path) as saved:
    depth={key:saved[key] for key in saved.files}
manifest=json.loads(Path('/home/ubuntu/arenalab-experiments/egodex-threepoint/input_manifest.json').read_text())
with h5py.File(manifest['source']) as f:
    intrinsic=f['camera/intrinsic'][()]
    hands=np.stack([f['transforms/'+s+'Hand'][:,:3,3] for s in ['left','right']],1)
frame_ids=depth['ids']; H,W=depth['depth'].shape[1:]
yy,xx=np.mgrid[:H,:W]
homogeneous=np.stack([xx,yy,np.ones_like(xx)],-1)
def mask_at(labels,i):
    k=intrinsic.copy();k[:2]*=.5
    return cv2.warpPerspective(labels[i],depth['intrinsics'][i]@np.linalg.inv(k),(W,H),flags=cv2.INTER_NEAREST)
def camera_points(i):
    return (homogeneous@np.linalg.inv(depth['intrinsics'][i]).T)*depth['depth'][i,...,None]
def world_points(i):
    e=depth['extrinsics'][i]
    return (camera_points(i)-e[:3,3])@e[:3,:3]
def prepare():
    selected=[]
    for kind in ['cup','lid']:
        labels=np.load(OUT/f'{kind}-masks.npz')['masks']; instances=[]
        for object_id in np.unique(labels):
            if object_id==0:continue
            proximities=[];areas=[]
            for i,source_id in enumerate(frame_ids):
                mask=mask_at(labels,i)==object_id;valid=mask&np.isfinite(depth['depth'][i])&(depth['depth'][i]>.05)
                cloud=world_points(i)[valid]
                areas.append(int(valid.sum()))
                proximities.append(float(cKDTree(cloud).query(hands[source_id])[0].min()) if len(cloud)>20 else float('inf'))
            finite=np.asarray(proximities)[np.isfinite(proximities)]
            score=float(np.percentile(finite,15)) if len(finite) else float('inf')
            instances.append(dict(id=int(object_id),proximity_score_m=score,areas=areas,proximities=proximities))
        if not instances:continue
        candidate=min(instances,key=lambda x:x['proximity_score_m'])
        i=int(np.argmax(candidate['areas']))
        folder=OUT/f'{kind}-{candidate["id"]}';folder.mkdir(exist_ok=True)
        image=cv2.imread(str(BASE/'frames'/f'{i:05d}.jpg'))
        mask=(labels[i]==candidate['id']).astype(np.uint8)*255
        cv2.imwrite(str(folder/'image.png'),image);cv2.imwrite(str(folder/'mask.png'),mask)
        audit=image.copy();audit[mask>0]=(0,210,100);cv2.imwrite(str(folder/'mask-preview.jpg'),cv2.addWeighted(image,.65,audit,.35,0))
        np.save(folder/'camera-pointmap.npy',camera_points(i).astype('float32'))
        row=dict(kind=kind,instance_id=candidate['id'],keyframe=i,source_frame=int(frame_ids[i]),keyframe_seconds=float(frame_ids[i]/30),selection='lowest 15th-percentile source-wrist distance to observed object surface; largest visible mask keyframe',proximity_score_m=candidate['proximity_score_m'],rankings=[{'id':x['id'],'proximity_score_m':x['proximity_score_m']} for x in instances],camera_extrinsics=depth['extrinsics'][i].tolist(),pointmap='DA3 estimated camera-frame XYZ',folder=str(folder))
        (folder/'input.json').write_text(json.dumps(row,indent=2));selected.append(row)
    (OUT/'selected-objects.json').write_text(json.dumps(selected,indent=2));print(json.dumps(selected,indent=2))

if __name__=="__main__":
    prepare()
