"""Compare monocular scene depth with unchanged visible source hand landmarks."""
from pathlib import Path
import json
import cv2, h5py, numpy as np
from prepare_objects import OUT, depth, frame_ids, mask_at

EXP=Path('/home/ubuntu/arenalab-experiments/egodex-threepoint')
source=np.load(EXP/'source_hands.npz');points=source['points']@source['source_to_zup']
hands=np.load(OUT/'hand-masks.npz')['masks'];rows=[]
for i,source_id in enumerate(frame_ids):
    e=depth['extrinsics'][i];k=depth['intrinsics'][i];mask=mask_at(hands,i)>0
    camera=points[source_id]@e[:3,:3].T+e[:3,3]
    projected=camera@k.T;uv=np.rint(projected[...,:2]/np.maximum(projected[...,2:],1e-8)).astype(int)
    for side in range(2):
        for joint in range(1,21):
            x,y=uv[side,joint];h,w=mask.shape
            if not(2<=x<w-2 and 2<=y<h-2) or source['confidence'][source_id,side,joint]<.5 or not mask[y,x]:continue
            local=depth['depth'][i,y-1:y+2,x-1:x+2];valid=local[mask[y-1:y+2,x-1:x+2]&np.isfinite(local)]
            if not len(valid):continue
            rows.append(dict(frame=int(source_id),side=side,joint=joint,predicted=float(np.median(valid)),source=float(camera[side,joint,2])))
x=np.asarray([r['predicted'] for r in rows]);y=np.asarray([r['source'] for r in rows]);ratio=y/x
report=dict(anchors=len(rows),median_source_over_predicted_depth=float(np.median(ratio)),ratio_p10_p90=np.percentile(ratio,[10,90]).tolist(),depth_error_mm=dict(median=float(np.median(abs(y-x))*1000),p95=float(np.percentile(abs(y-x),95)*1000)),rows=rows,provenance='Source hand joints are unmodified; scene depth is estimated; surface-to-skeleton offset remains uncertain')
manifest=json.loads((EXP/'input_manifest.json').read_text())
with h5py.File(manifest['source']) as f:
    original=np.linalg.inv(f['transforms/camera'][()][frame_ids])[:,:3]
report['camera_world_to_view_max_difference']=float(np.max(abs(original-depth['extrinsics'])))
(OUT/'depth-anchors.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
