"""Export visible object triangle surfaces without inventing hidden geometry.

Depth is still a DA3 estimate. Each frame uses unchanged camera/world coordinates;
triangles crossing a mask boundary or a depth discontinuity are omitted.
"""
from pathlib import Path
import json
import cv2, numpy as np
from prepare_objects import OUT, depth, frame_ids, mask_at, world_points

DEST=Path('/opt/arenalab/store/manipulation4d/pilot')
scene=json.loads((DEST/'scene.json').read_text());H,W=depth['depth'].shape[1:]
hands=np.load(OUT/'hand-masks.npz')['masks'];report=[]
for obj in scene['objects']:
    labels=np.load(OUT/f"{obj['label']}-masks.npz")['masks']
    track=json.loads((DEST/Path(obj['track_url']).name).read_text());frames=[]
    name=f"{obj['id']}-surfaces.bin"
    with (DEST/name).open('wb') as stream:
        def block(array):
            stream.write(b'\0'*((-stream.tell())%4));offset=stream.tell();stream.write(array.tobytes());return offset
        for i,source_id in enumerate(frame_ids):
            z=depth['depth'][i];instances=mask_at(labels,i);mask=(instances>0)&(mask_at(hands,i)==0)&np.isfinite(z)&(z>.05)
            # Remove one uncertain mask-boundary pixel shared by all objects.
            mask=cv2.erode(mask.astype('u1'),np.ones((3,3),np.uint8))>0
            ids=np.full((H,W),-1,np.int32);ids[mask]=np.arange(int(mask.sum()))
            a,b,c,d=ids[:-1,:-1],ids[:-1,1:],ids[1:,:-1],ids[1:,1:]
            faces=[]
            for corners in [(a,b,c),(b,d,c)]:
                grid=np.stack(corners,-1);valid=(grid>=0).all(-1)
                local_ids=instances[mask];safe=np.maximum(grid,0)
                if len(local_ids):valid&=(local_ids[safe]==local_ids[safe[...,0:1]]).all(-1)
                faces.append(grid[valid])
            faces=np.concatenate(faces) if faces else np.empty((0,3),np.int32)
            positions=world_points(i)[mask].astype('<f4');colors=depth['images'][i][mask].astype('u1')
            if len(faces):
                lengths=np.linalg.norm(positions[faces]-positions[faces[:,[1,2,0]]],axis=-1)
                # 3x local diagonal pixel footprint at the observed range.
                footprint=z[mask][faces].mean(1)*np.hypot(1/depth['intrinsics'][i,0,0],1/depth['intrinsics'][i,1,1])
                faces=faces[(lengths<=3*footprint[:,None]).all(1)]
            boxes=[]
            for instance in np.unique(instances[mask]):
                points=positions[instances[mask]==instance]
                if len(points)>=3:boxes.append(dict(id=int(instance),bounds=[points.min(0).tolist(),points.max(0).tolist()]))
            frames.append(dict(source_frame=int(source_id),time=float(source_id/30),vertices=len(positions),faces=len(faces),positions=block(positions),colors=block(colors),indices=block(faces.astype('<u4')),bounds=[positions.min(0).tolist(),positions.max(0).tolist()] if len(positions) else None,boxes=boxes))
    obj['observed_surface']=dict(url='/assets/manipulation4d/pilot/'+name,fps=15,frames=frames,all_instances=True,provenance='All visible SAM 3.1 instances triangulated from DA3-estimated depth; visible surfaces only; no backside completion; segmentation IDs are not guaranteed persistent physical IDs')
    report.append(dict(id=obj['id'],frames_with_surface=sum(f['faces']>0 for f in frames),total_frames=len(frames),bytes=(DEST/name).stat().st_size,source_trajectory_edited=False,hidden_surfaces_generated=False))
(DEST/'scene.json').write_text(json.dumps(scene,separators=(',',':')))
(DEST/'surface-validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
