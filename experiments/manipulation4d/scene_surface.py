"""One static surface from unchanged camera/depth observations, not point stacking.

TSDF averaging/carving merges observations before extracting its zero crossing.
The returned vertices are displayed as points, never as a completed object mesh.
All settings are shared across sources and robots. Raw depth/cameras stay intact.
"""
import os,time
os.environ.setdefault('CUDA_VISIBLE_DEVICES','3')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import cv2,numpy as np

MIN_VOXEL=.004
MIN_TRUNCATION=.04

def fuse_background(depth,background):
    import open3d as o3d
    started=time.monotonic();z=np.asarray(depth['depth']);T,H,W=z.shape
    original_cameras=np.asarray(depth['extrinsics']).copy()
    masks=[]
    for i in range(T):
        valid=np.asarray(background[i],bool)&np.isfinite(z[i])&(z[i]>.05)
        if valid.any():
            valid&=depth['conf'][i]>=np.percentile(depth['conf'][i][valid],25)
        # Exclude uncertain depth boundaries and foreground-mask margins from
        # persistent static fusion. Per-frame observed foreground is unaffected.
        valid=cv2.erode(valid.astype('u1'),np.ones((3,3),'u1'))>0
        hi=cv2.dilate(z[i].astype('f4'),np.ones((3,3),'u1'))
        lo=cv2.erode(z[i].astype('f4'),np.ones((3,3),'u1'))
        valid&=(hi-lo)<np.maximum(.02,.03*z[i]);masks.append(valid)
    masks=np.asarray(masks)
    # Do not oversample distant room surfaces into a huge volume finer than the
    # observations. This common camera-resolution rule is independent of robot,
    # clip identity and action; it changes map sampling, never source scale/poses.
    pitches=[float(np.median(z[i][masks[i]]))/float(np.sqrt(depth['intrinsics'][i,0,0]*depth['intrinsics'][i,1,1])) for i in range(T) if masks[i].any()]
    voxel=max(MIN_VOXEL,float(np.median(pitches))) if pitches else MIN_VOXEL
    truncation=max(MIN_TRUNCATION,10*voxel)
    volume=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel,sdf_trunc=truncation,color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,depth_sampling_stride=2)
    views=np.arange(T)
    for i in views:
        valid=masks[i]
        if not valid.any():continue
        image=np.ascontiguousarray(depth['images'][i],dtype='u1')
        observed=np.where(valid,z[i],0).astype('f4')
        rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(image),o3d.geometry.Image(observed),depth_scale=1.,depth_trunc=float(observed.max()+truncation),convert_rgb_to_intensity=False)
        k=depth['intrinsics'][i];intrinsic=o3d.camera.PinholeCameraIntrinsic(W,H,k[0,0],k[1,1],k[0,2],k[1,2])
        e=np.eye(4);e[:3]=depth['extrinsics'][i,:3]
        volume.integrate(rgbd,intrinsic,e)
        print('TSDF',int(i),T,flush=True)
    mesh=volume.extract_triangle_mesh();p=np.asarray(mesh.vertices);rgb=np.asarray(mesh.vertex_colors)
    # A voxel occupancy union still has multiple depth layers. Here the samples
    # come only from the fused field's zero crossing. Reject its unsupported /
    # free-space-conflicting remnants instead of retaining every depth cluster.
    support=np.zeros(len(p),int);conflicts=np.zeros(len(p),int);checks=np.unique(np.linspace(0,T-1,min(T,64)).astype(int))
    for i in checks:
        e=depth['extrinsics'][i];q=p@e[:3,:3].T+e[:3,3];uv=q@depth['intrinsics'][i].T
        uv=np.rint(uv[:,:2]/np.maximum(q[:,2:],1e-8)).astype(int)
        inside=(q[:,2]>.05)&(uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H);ix=np.flatnonzero(inside);x,y=uv[ix].T
        visible=masks[i,y,x];ix=ix[visible];x,y=uv[ix].T
        observed=z[i,y,x];difference=q[ix,2]-observed;tolerance=np.maximum(voxel*4,.025*observed)
        support[ix[abs(difference)<=tolerance]]+=1
        conflicts[ix[difference < -np.maximum(truncation,tolerance)]]+=1
    minimum=min(5,len(checks));keep=(support>=minimum)&(support>=4*conflicts)
    selected=p[keep].astype('<f4');colors=np.clip(np.rint(rgb[keep]*255),0,255).astype('u1')
    assert np.array_equal(original_cameras,depth['extrinsics'])
    report=dict(method='Open3D 0.19 TSDF zero-crossing surface + free-space consistency',sampling_policy='one median projected source-depth pixel; minimum 4 mm',voxel_m=voxel,truncation_m=truncation,integration_frames=views.tolist(),validation_frames=checks.tolist(),minimum_observations=minimum,extracted_points=len(p),retained_points=len(selected),rejected_points=int((~keep).sum()),support_median=float(np.median(support[keep])) if keep.any() else 0.,seconds=time.monotonic()-started,source_cameras_unchanged=True,source_trajectory_edited=False,whole_clip_offline=True,unobserved_geometry_generated=False)
    return selected,colors,report

def write_background(path,url,depth,background,block):
    points,colors,report=fuse_background(depth,background)
    if not len(points):return None,report
    T=len(depth['depth'])
    with path.open('wb') as stream:
        memory=dict(url=url,count=len(points),positions=block(stream,points),colors=block(stream,colors),first_frames=block(stream,np.zeros(len(points),'<u4')),last_frames=block(stream,np.full(len(points),np.iinfo('u4').max,dtype='<u4')),source_frame_ids=depth['ids'].tolist(),active_points=[len(points)]*T,voxel_m=report['voxel_m'],mode='full_clip',estimated=True,surface_fusion=report)
    return memory,report
