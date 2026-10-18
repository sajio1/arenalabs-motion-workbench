"""Check inferred geometry against observed silhouettes across the source video.

The residual of a trimmed ICP fit alone cannot validate hidden geometry. These
checks use held-out views and keep the depth/mask uncertainty explicit.
"""
from pathlib import Path
import json
import cv2, numpy as np, trimesh
from prepare_objects import OUT, depth, frame_ids, mask_at

DEST=Path('/opt/arenalab/store/manipulation4d/pilot')
H,W=depth['depth'].shape[1:]
scene=json.loads((DEST/'scene.json').read_text())
reports=[]
for obj in scene['objects']:
    mesh=trimesh.load(DEST/Path(obj['mesh_url']).name,force='mesh',process=False)
    sample,_=trimesh.sample.sample_surface(mesh,160000,seed=42)
    frames=json.loads((DEST/Path(obj['track_url']).name).read_text())
    labels=np.load(OUT/f"{obj['label']}-masks.npz")['masks']
    rows=[]
    for i,frame in enumerate(frames):
        if frame['status']!='observed':continue
        transform=np.asarray(frame['matrix']);e=depth['extrinsics'][i];k=depth['intrinsics'][i]
        world=sample@transform[:3,:3].T+transform[:3,3]
        camera=world@e[:3,:3].T+e[:3,3]
        projected=camera@k.T
        uv=np.rint(projected[:,:2]/np.maximum(projected[:,2:],1e-8)).astype(int)
        valid=(camera[:,2]>.03)&(uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H)
        uv=uv[valid];z=camera[valid,2]
        nearest=np.full(H*W,np.inf);np.minimum.at(nearest,uv[:,1]*W+uv[:,0],z);nearest=nearest.reshape(H,W)
        # A source foreground occluder is not evidence against a hidden mesh.
        rendered=np.isfinite(nearest)&(nearest<=depth['depth'][i]+.015)
        mask=mask_at(labels,i)==frame['segmentation_instance_id']
        tolerance=cv2.dilate(mask.astype('u1'),np.ones((3,3),np.uint8))>0
        intersection=int((rendered&mask).sum());n=int(rendered.sum())
        if n<20:continue
        rows.append(dict(source_frame=int(frame_ids[i]),precision=float((rendered&tolerance).sum()/n),recall=float(intersection/max(1,mask.sum())),rendered_pixels=n,mask_pixels=int(mask.sum())))
    precision=np.asarray([row['precision'] for row in rows])
    report=dict(id=obj['id'],frames=rows,median_silhouette_precision=float(np.median(precision)),p10_silhouette_precision=float(np.percentile(precision,10)),method='Visible projected surface samples vs SAM masks; one pixel boundary tolerance; DA3 occlusion estimate',depth_tolerance_m=.015,ground_truth=False)
    reports.append(report)
    print(obj['id'],json.dumps({key:value for key,value in report.items() if key!='frames'}),flush=True)
(DEST/'shape-validation.json').write_text(json.dumps(reports,indent=2))
