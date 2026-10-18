"""Current-frame point clouds with semantic instances; no historical background.

No object mesh, hand skeleton, frustum, source trajectory registration or manual
object selection. Bounds describe visible estimated depth, not complete assets.
"""
from pathlib import Path
import json,sys
import cv2,numpy as np

work=Path(sys.argv[1]);manifest=json.loads((work/'manifest.json').read_text());row=manifest['source'];public=manifest['public'];dest=Path(manifest['destination']);url=manifest['url']
with np.load(work/'depth.npz') as saved:depth={key:saved[key] for key in saved.files}
prep=json.loads((work/'preparation.json').read_text());semantic=json.loads((work/'semantic.json').read_text());prompts=json.loads((work/'prompts.json').read_text());T,H,W=depth['depth'].shape
source_intrinsic=np.load(work/'cameras.npz')['intrinsic'] if (work/'cameras.npz').exists() else (np.load(work/'camera-intrinsics.npz')['intrinsic'] if (work/'camera-intrinsics.npz').exists() else None)
# Nymeria inference preserves raw Xsens Z-up. Robot playback exports [x,z,-y].
# Apply the identical lossless display basis to clouds, boxes and optical poses.
display_from_source=np.eye(4)
if (public['dataset']=='Nymeria' or (public.get('uploaded') and public.get('end_effector')=='none')) and (work/'cameras.npz').exists():
    display_from_source[:3,:3]=[[1,0,0],[0,0,1],[0,-1,0]]
yy,xx=np.mgrid[:H,:W];pixel=np.stack([xx,yy,np.ones_like(xx)],-1)
def world(i):
    e=depth['extrinsics'][i];p=(pixel@np.linalg.inv(depth['intrinsics'][i]).T)*depth['depth'][i,...,None]
    return ((p-e[:3,3])@e[:3,:3])@display_from_source[:3,:3].T
def mask_at(mask,i):
    # DA3 upper_bound_resize resizes both dimensions; it does not warp images
    # to the predicted intrinsics. Masks follow processed_images exactly.
    return cv2.resize(mask,(W,H),interpolation=cv2.INTER_NEAREST)
labels={};human=np.zeros((T,H,W),bool);instances=[];mapping={};nextlabel=1;warnings=[]
if not semantic['objects']:warnings.append('VLM proposed no manipulable objects; this is not evidence that none are present')
for item in prompts:
    key,category=item[:2]
    saved=np.load(work/'masks'/f'{key}-masks.npz');frames=saved['frame_indices'];raw=saved['masks'];byindex={int(f):m for f,m in zip(frames,raw)}
    if len(raw)!=len(frames) or sorted(byindex)!=list(range(T)):
        raise ValueError(f'SAM/DA3 frame coverage mismatch: {key}')
    if 'source_frame_indices' in saved and not np.array_equal(saved['source_frame_indices'],depth['ids'][frames]):
        raise ValueError(f'SAM/DA3 source timestamp mismatch: {key}')
    mask=np.stack([mask_at(byindex.get(i,np.zeros((540,960),dtype='u2')),i) for i in range(T)])
    if key.startswith('human-'):human|=mask>0;continue
    labels[key]=mask
    ids=np.unique(mask);ids=ids[ids>0]
    if not len(ids):warnings.append(f'No mask found for proposed category: {category}')
    for instance in ids:
        if nextlabel>254:raise ValueError('Too many semantic instances for the cloud format')
        mapping[key,int(instance)]=nextlabel;instances.append(dict(id=f'{key}-{int(instance)}',label=category,semantic_label=nextlabel,estimated=True));nextlabel+=1
human=np.stack([cv2.dilate(m.astype('u1'),np.ones((7,7),'u1'))>0 for m in human]);cloudframes=[];tracks=[];counts={o['id']:[] for o in instances}
def block(f,a):
    f.write(b'\0'*((-f.tell())%4));offset=f.tell();f.write(a.tobytes());return offset
with (dest/'cloud.bin').open('wb') as stream:
    for i in range(T):
        sem=np.zeros((H,W),'u1')
        for key,mask in labels.items():
            for instance in np.unique(mask[i]):
                if instance:sem[mask[i]==instance]=mapping[key,int(instance)]
        valid=np.isfinite(depth['depth'][i])&(depth['depth'][i]>.05)&~human[i];sample=valid.copy();sample[1::2,:]=False;sample[:,1::2]=False
        xyz=world(i);boxes=[]
        for obj in instances:
            points=xyz[valid&(sem==obj['semantic_label'])];counts[obj['id']].append(len(points))
            if len(points)>5:boxes.append(dict(id=obj['id'],label=obj['label'],semantic_label=obj['semantic_label'],points=len(points),bounds=[np.percentile(points,1,axis=0).tolist(),np.percentile(points,99,axis=0).tolist()]))
        cloudframes.append(dict(source_frame=int(depth['ids'][i]),time=float(depth['ids'][i]/30),count=int(sample.sum()),positions=block(stream,xyz[sample].astype('<f4')),colors=block(stream,depth['images'][i][sample].astype('u1')),labels=block(stream,sem[sample])))
        tracks.append(dict(time=float(depth['ids'][i]/30),source_frame=int(depth['ids'][i]),instances=boxes))
camera=None;measured_camera=(work/'cameras.npz').exists()
if (work/'cameras.npz').exists():
    saved=np.load(work/'cameras.npz');camera=dict(poses=(display_from_source@saved['poses']).tolist(),intrinsic=saved['intrinsic'].tolist(),width=960,height=540,fps=30)
else:
    homogeneous=np.repeat(np.eye(4)[None],T,0);homogeneous[:,:3]=depth['extrinsics']
    poses=np.linalg.inv(homogeneous);poses=poses[np.minimum(np.arange(row['frames'])//2,T-1)]
    camera=dict(poses=poses.tolist(),intrinsic=depth['intrinsics'][0].tolist(),width=W,height=H,fps=30,estimated=True,provenance='DA3 estimated camera poses for scene display only; not calibrated to source robot trajectories')
audit=dict(policy='sam31-da3-semantic-v2',discovery_policy=semantic.get('policy'),model='SAM 3.1 Object Multiplex',automatic_object_proposals=True,source_frame_alignment_verified=True,pixel_space='Prepared camera images, shared by SAM and DA3',objects=[{**o,'visible_frames':sum(x>0 for x in counts[o['id']]),'maximum_pixels':max(counts[o['id']],default=0)} for o in instances],warnings=warnings,mask_categories=[p[1] for p in prompts if not p[0].startswith('human-')],all_objects_guaranteed=False,source_trajectory_edited=False)
(dest/'segmentation.json').write_text(json.dumps(audit,indent=2))
scene=dict(id=public['id'],title=public['title'],dataset=public['dataset'],end_effector=public.get('end_effector'),uploaded=public.get('uploaded',False),fps=30,frames=row['frames'],duration=(row['frames']-1)/30,world='Source world' if measured_camera else 'Estimated camera world · scale/registration unverified',video=public['video'],cloud=dict(url=url+'cloud.bin',frames=cloudframes,fps=15),robots=[],objects=[],segmented_instances=dict(fps=15,frames=tracks,categories=semantic['objects'],audit_url=url+'segmentation.json'),semantic=semantic,tactile=dict(status='unavailable',reason='No tactile sensor stream; contact inference not generated by the scene worker'),wrist_videos=[],geometry=dict(depth=json.loads((work/'depth.json').read_text()),segmentation=audit,body_status='Scene reconstruction complete; body/hand conversion is a separate stage',background='Current video frame only; no retained or fused background',source_sha256=manifest['source_sha256'],source_trajectory_edited=False))
if (work/'camera-calibration.json').exists():
    calibration=json.loads((work/'camera-calibration.json').read_text());scene['geometry']['camera_calibration']=calibration;scene['world']='Source world · estimated camera calibration';camera.update(estimated=True,provenance=calibration['extrinsic_provenance'])
scene['geometry']['scene_registration']='recorded' if measured_camera else 'unregistered'
if (work/'camera-calibration.json').exists():scene['geometry']['scene_registration']='calibrated'
scene['geometry']['display_from_source']=display_from_source.tolist()
if (work/'lens-calibration.json').exists():scene['geometry']['lens_calibration']=json.loads((work/'lens-calibration.json').read_text())
if camera:scene['source_camera']=camera
from gpu_scheduler import resource_lock, write
with resource_lock('manifest:'+str((dest/'scene.json').resolve())):
    if (dest/'scene.json').exists():
        previous=json.loads((dest/'scene.json').read_text())
        for key in ['robots','patch_videos','patch_kinds','patch_unavailable','tactile','hand_motion_audit_url','hand_motion_audit_urls','generation_status']:
            if key in previous:scene[key]=previous[key]
        if previous.get('geometry',{}).get('ground'):scene['geometry']['ground']=previous['geometry']['ground']
    write(dest/'scene.json',scene)
print(json.dumps(audit),flush=True)
