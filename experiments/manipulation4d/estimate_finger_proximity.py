"""Independent per-finger proximity diagnostic; not contact probability or pressure."""
from pathlib import Path
import json, numpy as np
from scipy.spatial import cKDTree
from prepare_objects import OUT, depth, frame_ids, mask_at, world_points

dest=Path('/opt/arenalab/store/manipulation4d/reconstruction-v2')
source=np.load('/home/ubuntu/arenalab-experiments/egodex-threepoint/source_hands.npz')
S=source['source_to_zup'];tips=source['points'][:,:,[4,8,12,16,20]]@S
confidence=source['confidence'][:,:,[4,8,12,16,20]]
labels={kind:np.load(OUT/f'{kind}-masks.npz')['masks'] for kind in ['cup','lid','hand']}
values=[];distances=[]
for i,source_id in enumerate(frame_ids):
    mask=((mask_at(labels['cup'],i)>0)|(mask_at(labels['lid'],i)>0))&(mask_at(labels['hand'],i)==0)
    valid=mask&np.isfinite(depth['depth'][i])&(depth['depth'][i]>.05)
    points=world_points(i)[valid]
    if len(points)<20:
        values.append([[None]*5 for _ in range(2)]);distances.append([[None]*5 for _ in range(2)]);continue
    distance=cKDTree(points).query(tips[source_id].reshape(-1,3))[0].reshape(2,5)
    scores=(np.exp(-.5*(distance/.02)**2)*confidence[source_id]).tolist()
    for side in range(2):
        for finger in range(5):
            if not np.isfinite(confidence[source_id,side,finger]) or confidence[source_id,side,finger]<=0:scores[side][finger]=None
    values.append(scores);distances.append((distance*1000).tolist())
payload=dict(fps=15,source_frame_ids=frame_ids.tolist(),fingers=['thumb','index','middle','ring','little'],likelihood=values,nearest_visible_surface_mm=distances,method='confidence-weighted fingertip proximity to SAM 3.1 object masks lifted through DA3 estimated depth',distance_scale_m=.02,measured=False,force_available=False,limitations=['No force, pressure, friction or hidden contact inference','Source hand and depth errors affect estimates','A missing visible surface is not evidence of no contact'])
(dest/'finger-proximity-v2.json').write_text(json.dumps(payload,separators=(',',':')))
scene=json.loads((dest/'scene.json').read_text());scene['tactile']['proximity_url']='/assets/manipulation4d/reconstruction-v2/finger-proximity-v2.json'
tmp=dest/'scene-proximity.tmp';tmp.write_text(json.dumps(scene,separators=(',',':')));tmp.replace(dest/'scene.json');print('FINGER_PROXIMITY_COMPLETE')
