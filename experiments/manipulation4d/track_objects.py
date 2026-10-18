"""Rigid visible-surface registration of inferred assets against observed depth.

No source trajectory, background, camera pose or object scale is edited. Lost
observations are reported; a previous pose is never presented as a new measurement.
"""
from pathlib import Path
import json
import cv2, numpy as np, trimesh
from scipy.spatial import cKDTree
from prepare_objects import OUT, depth, frame_ids, mask_at, world_points, depth_calibration

DEST=Path('/opt/arenalab/store/manipulation4d/pilot')
URL='/assets/manipulation4d/pilot/'
rng=np.random.default_rng(42)

def rigid_fit(a,b):
    ac,bc=a.mean(0),b.mean(0)
    u,_,v=np.linalg.svd((a-ac).T@(b-bc));rot=v.T@u.T
    if np.linalg.det(rot)<0:v[-1]*=-1;rot=v.T@u.T
    delta=np.eye(4);delta[:3,:3]=rot;delta[:3,3]=bc-rot@ac
    return delta

def visible_samples(points,e,k,mask,shape):
    camera=points@e[:3,:3].T+e[:3,3]
    uv=camera@k.T;uv=uv[:,:2]/np.maximum(uv[:,2:],1e-8)
    xy=np.rint(uv).astype(int);h,w=shape
    valid=(camera[:,2]>.03)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
    ids=np.flatnonzero(valid)
    ids=ids[mask[xy[ids,1],xy[ids,0]]]
    if not len(ids):return ids,xy
    pixel=xy[ids,1]*w+xy[ids,0]
    # Retain the visible front sample per pixel, avoiding fitting hidden surfaces.
    order=np.lexsort((camera[ids,2],pixel));_,first=np.unique(pixel[order],return_index=True)
    return ids[order[first]],xy

def fit(samples,initial,observed,e,k,mask,diameter):
    tree=cKDTree(observed);pose=initial.copy();details=None
    for _ in range(8):
        points=samples@pose[:3,:3].T+pose[:3,3]
        visible,_=visible_samples(points,e,k,mask,mask.shape)
        if len(visible)<40:return None,{'status':'lost','reason':'too little visible mesh support'}
        a=points[visible];distance,index=tree.query(a)
        threshold=min(float(np.percentile(distance,75)),diameter*.35)
        keep=distance<=max(threshold,.002)
        if keep.sum()<40:return None,{'status':'lost','reason':'insufficient geometric correspondences'}
        pose=rigid_fit(a[keep],observed[index[keep]])@pose
        details=dict(status='observed',inliers=int(keep.sum()),visible_samples=len(visible),surface_rmse_m=float(np.sqrt(np.mean(distance[keep]**2))))
    return pose,details

def browser_mesh(mesh):
    if len(mesh.faces)<=80000:return mesh.copy()
    voxel=float(mesh.extents.max()/128)
    cells=np.floor((mesh.vertices-mesh.bounds[0])/voxel).astype(int)
    _,inverse=np.unique(cells,axis=0,return_inverse=True);counts=np.bincount(inverse)
    vertices=np.column_stack([np.bincount(inverse,weights=mesh.vertices[:,j])/counts for j in range(3)])
    faces=inverse[mesh.faces];faces=faces[(faces[:,0]!=faces[:,1])&(faces[:,1]!=faces[:,2])&(faces[:,0]!=faces[:,2])]
    reduced=trimesh.Trimesh(vertices,faces,process=False)
    colors=mesh.visual.vertex_colors
    reduced.visual.vertex_colors=np.column_stack([np.bincount(inverse,weights=colors[:,j])/counts for j in range(4)]).astype(np.uint8)
    return reduced

objects=[];reports=[]
hand=np.load(OUT/'hand-masks.npz')['masks']
for row in json.loads((OUT/'selected-objects.json').read_text()):
    folder=Path(row['folder']);mesh=trimesh.load(folder/'world.glb',force='mesh',process=False)
    if depth_calibration:
        e=depth['extrinsics'][row['keyframe']];camera_center=-e[:3,:3].T@e[:3,3]
        mesh.vertices=camera_center+depth_calibration['scale']*(mesh.vertices-camera_center)
    samples,_=trimesh.sample.sample_surface(mesh,40000,seed=42)
    labels=np.load(OUT/f'{row["kind"]}-masks.npz')['masks'];n=len(frame_ids);diameter=float(np.linalg.norm(mesh.extents))
    observations=[];masks=[];associated_ids=[row['instance_id']]*n
    label_frames=[mask_at(labels,i) for i in range(n)]
    hand_frames=[mask_at(hand,i)>0 for i in range(n)]
    world_frames=[world_points(i) for i in range(n)]
    # SAM can assign a new instance ID after occlusion. Associate only a nearby
    # same-class observed surface; absence remains explicit rather than interpolated.
    key=row['keyframe']
    for direction in [1,-1]:
        active_id=row['instance_id'];previous_center=np.median(world_frames[key][label_frames[key]==active_id],axis=0)
        for i in range(key+direction,n if direction>0 else -1,direction):
            candidates=[]
            for object_id in np.unique(label_frames[i]):
                if object_id==0:continue
                valid=(label_frames[i]==object_id)&~hand_frames[i]
                cloud=world_frames[i][valid]
                if len(cloud)<40:continue
                center=np.median(cloud,axis=0);distance=float(np.linalg.norm(center-previous_center))
                candidates.append((int(object_id),distance,center))
            existing=next((c for c in candidates if c[0]==active_id),None)
            best=existing or (min(candidates,key=lambda c:c[1]) if candidates else None)
            if best is not None and (existing is not None or best[1]<diameter*.4):
                active_id=best[0];previous_center=best[2];associated_ids[i]=active_id
            else:associated_ids[i]=-1
    for i in range(n):
        mask=(label_frames[i]==associated_ids[i])&~hand_frames[i]
        valid=mask&np.isfinite(depth['depth'][i])&(depth['depth'][i]>.05)
        cloud=world_frames[i][valid]
        if len(cloud)>8000:cloud=cloud[rng.choice(len(cloud),8000,replace=False)]
        observations.append(cloud);masks.append(mask)
    frames=[None]*n;poses=[None]*n;key=row['keyframe']
    pose,info=fit(samples,np.eye(4),observations[key],depth['extrinsics'][key],depth['intrinsics'][key],masks[key],diameter)
    if pose is None:raise RuntimeError(f'Cannot register keyframe for {row["kind"]}: {info}')
    poses[key]=pose;frames[key]=info
    for direction in [1,-1]:
        previous=pose.copy()
        for i in range(key+direction,n if direction>0 else -1,direction):
            if len(observations[i])<40:frames[i]=dict(status='lost',reason='object unobserved');continue
            # Observation-derived reinitialization is shared across objects.
            # It only places the inferred mesh; observed source points are immutable.
            reinitialized=pose.copy()
            reinitialized[:3,3]+=np.median(observations[i],axis=0)-np.median(observations[key],axis=0)
            candidates=[]
            for initial,origin in [(previous,'previous observation'),(reinitialized,'current segmented depth centroid')]:
                result,details=fit(samples,initial,observations[i],depth['extrinsics'][i],depth['intrinsics'][i],masks[i],diameter)
                if result is not None:
                    details['initialization']=origin;candidates.append((result,details))
            if candidates:
                result,details=min(candidates,key=lambda x:x[1]['surface_rmse_m'])
            else:
                result,details=None,dict(status='lost',reason='both temporal and depth-centroid registration failed')
            frames[i]=details
            if result is not None:poses[i]=result;previous=result
    for i in range(n):
        frame=frames[i];frame.update(source_frame=int(frame_ids[i]),time=float(frame_ids[i]/30),segmentation_instance_id=int(associated_ids[i]))
        if poses[i] is not None:
            frame['matrix']=poses[i].tolist();vertices=mesh.bounding_box.vertices@poses[i][:3,:3].T+poses[i][:3,3]
            frame['bounds']=[vertices.min(0).tolist(),vertices.max(0).tolist()]
    stem=f'{row["kind"]}-{row["instance_id"]}';reduced=browser_mesh(mesh);reduced.export(DEST/f'{stem}.glb')
    (DEST/f'{stem}-track.json').write_text(json.dumps(frames,separators=(',',':')))
    residuals=[f['surface_rmse_m'] for f in frames if f['status']=='observed']
    report=dict(id=stem,model='SAM 3D Objects',keyframe_source_frame=row['source_frame'],source_vertices=len(mesh.vertices),browser_vertices=len(reduced.vertices),browser_faces=len(reduced.faces),visible_frames=len(residuals),total_frames=n,surface_rmse_mm=dict(mean=float(np.mean(residuals)*1000),p95=float(np.percentile(residuals,95)*1000),max=float(max(residuals)*1000)),registration='visible-surface trimmed rigid ICP, fixed inferred scale',segmentation_ids_used=sorted(set(i for i in associated_ids if i>0)),ground_truth_accuracy=False,unobservable='axis-symmetric yaw and hidden surfaces are not verified',source_trajectory_edited=False)
    report['scene_depth_calibration']=depth_calibration
    (folder/'tracking-validation.json').write_text(json.dumps(report,indent=2));reports.append(report)
    objects.append(dict(id=stem,label=row['kind'],mesh_url=URL+f'{stem}.glb',track_url=URL+f'{stem}-track.json',inferred=True,semantic_label=1 if row['kind']=='cup' else 2,quality=report))
    print('OBJECT_TRACKED',json.dumps(report),flush=True)
scene=json.loads((DEST/'scene.json').read_text());scene['objects']=objects;(DEST/'scene.json').write_text(json.dumps(scene,separators=(',',':')))
(DEST/'object-validation.json').write_text(json.dumps(reports,indent=2));print('TRACKING_COMPLETE',flush=True)
