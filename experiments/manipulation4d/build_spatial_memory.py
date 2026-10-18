"""Causal static-background memory; moving hands/arms/objects never accumulate.

This is estimated RGB-D fusion, not SLAM or a claim of metric scene accuracy.
Each voxel needs two distinct observations. Its geometry is frozen at birth;
later free-space evidence can retire it. Seeking never reveals future voxels.
"""
from pathlib import Path
import json
import cv2,h5py,numpy as np
from prepare_objects import ROOT,OUT,depth,frame_ids,mask_at,world_points

DEST=Path('/opt/arenalab/store/manipulation4d/pilot');VOXEL=.015
labels={name:np.load(OUT/f'{name}-masks.npz')['masks'] for name in ['hand','arm','cup','lid']}
H,W=depth['depth'].shape[1:];yy,xx=np.mgrid[:H,:W];subsample=(yy%2==0)&(xx%2==0)
candidates={};active={};records=[];counts=[];excluded=[]
for i,source_id in enumerate(frame_ids):
    human=mask_at(labels['hand'],i)>0;human|=mask_at(labels['arm'],i)>0
    human=cv2.dilate(human.astype('u1'),np.ones((13,13),np.uint8))>0
    objects=(mask_at(labels['cup'],i)>0)|(mask_at(labels['lid'],i)>0)
    objects=cv2.dilate(objects.astype('u1'),np.ones((7,7),np.uint8))>0
    allowed=~(human|objects)&np.isfinite(depth['depth'][i])&(depth['depth'][i]>.05)&(depth['depth'][i]<5)
    excluded.append(int((human|objects).sum()))
    # Retire foreground ghosts only when a later static ray sees free space.
    if active:
        keys=list(active);ids=np.asarray([active[key] for key in keys]);points=np.asarray([records[j]['position'] for j in ids])
        e=depth['extrinsics'][i];camera=points@e[:3,:3].T+e[:3,3];uv=camera@depth['intrinsics'][i].T;uv=np.rint(uv[:,:2]/np.maximum(uv[:,2:],1e-8)).astype(int)
        visible=(camera[:,2]>.03)&(uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H)
        for n in np.flatnonzero(visible):
            x,y=uv[n]
            if not allowed[y,x]:continue
            observed=depth['depth'][i,y,x];record=records[int(ids[n])]
            conflict=camera[n,2]<observed-max(.03,.06*observed)
            record['conflicts']=record['conflicts']+1 if conflict else 0
            if record['conflicts']>=2:
                record['last']=int(source_id);active.pop(keys[n],None);candidates.pop(keys[n],None)
    valid=allowed&subsample;points=world_points(i)[valid];colors=depth['images'][i][valid].astype(float)
    cells=np.floor(points/VOXEL).astype(np.int32);unique,inverse=np.unique(cells,axis=0,return_inverse=True);n=np.bincount(inverse)
    p=np.column_stack([np.bincount(inverse,weights=points[:,axis])/n for axis in range(3)])
    c=np.column_stack([np.bincount(inverse,weights=colors[:,axis])/n for axis in range(3)])
    for key,point,color in zip(map(tuple,unique),p,c):
        if key in active:continue
        if key not in candidates:candidates[key]=(point,color);continue
        previous=candidates.pop(key);index=len(records)
        records.append(dict(position=(previous[0]+point)*.5,color=np.rint((previous[1]+color)*.5).astype('u1'),first=int(source_id),last=np.iinfo(np.uint32).max,conflicts=0));active[key]=index
    counts.append(len(active));print('SPATIAL_MEMORY',i,len(active),len(records),flush=True)
name='spatial-memory-v1.bin'
with (DEST/name).open('wb') as stream:
    def block(array):
        stream.write(b'\0'*((-stream.tell())%4));offset=stream.tell();stream.write(array.tobytes());return offset
    data=dict(url='/assets/manipulation4d/pilot/'+name,count=len(records),positions=block(np.asarray([r['position'] for r in records],dtype='<f4')),colors=block(np.asarray([r['color'] for r in records],dtype='u1')),first_frames=block(np.asarray([r['first'] for r in records],dtype='<u4')),last_frames=block(np.asarray([r['last'] for r in records],dtype='<u4')),voxel_m=VOXEL,minimum_observations=2,source_frame_ids=frame_ids.tolist(),active_points=counts,estimated=True,provenance='Causal observed background fusion; SAM hands, arms, cups and lids excluded; geometry fixed at the second observation; no future-view geometry')
scene=json.loads((DEST/'scene.json').read_text());scene['spatial_memory']=data
source=json.loads((ROOT.parent/'egodex-threepoint/input_manifest.json').read_text())['source']
with h5py.File(source) as f:
    intrinsic=f['camera/intrinsic'][()].copy();intrinsic[:2]*=.5
    scene['source_camera']=dict(poses=f['transforms/camera'][()].tolist(),intrinsic=intrinsic.tolist(),width=960,height=540,fps=30,provenance='Original source camera-to-world poses and exact resized source intrinsics')
scene['hand_motion_audit_url']='/assets/manipulation4d/pilot/hand-motion-audit.json'
(DEST/'scene.json').write_text(json.dumps(scene,separators=(',',':')))
report=dict(voxels=len(records),active_points=counts,source_frames=frame_ids.tolist(),bytes=(DEST/name).stat().st_size,voxel_m=VOXEL,exclusion_pixels=excluded,excluded_classes=['hand','arm','cup','lid'],geometry_available_before_observation=False,static_only=True,depth_ground_truth=False,world='Original source world, Y up, metres')
(DEST/'spatial-memory-validation.json').write_text(json.dumps(report,indent=2));print('MEMORY_READY',len(records),flush=True)
