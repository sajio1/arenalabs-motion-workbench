"""Versioned multiview scene export; immutable robot/source artifacts are reused.
Static fusion uses only repeatedly observed pixels; no unobserved geometry is invented.
"""
from pathlib import Path
import json,hashlib,cv2,numpy as np
from scipy.spatial import cKDTree
from prepare_objects import OUT,depth,frame_ids,mask_at,world_points
ROOT=Path('/home/ubuntu/arenalab-experiments/manipulation4d');DEST=Path('/opt/arenalab/store/manipulation4d/reconstruction-v2');DEST.mkdir(exist_ok=True)
URL='/assets/manipulation4d/reconstruction-v2/';scene=json.loads((DEST/'scene.json' if (DEST/'scene.json').exists() else Path('/opt/arenalab/store/manipulation4d/pilot/scene.json')).read_text());T,H,W=depth['depth'].shape
labels={key:np.load(OUT/f'{key}-masks.npz')['masks'] for key in ['hand','arm','cup','lid']};masks=[{key:mask_at(value,i) for key,value in labels.items()} for i in range(T)]
humans=[];background=[]
for i,m in enumerate(masks):
 human=cv2.dilate(((m['hand']>0)|(m['arm']>0)).astype('u1'),np.ones((7,7),'u1'))>0
 objects=cv2.dilate(((m['cup']>0)|(m['lid']>0)).astype('u1'),np.ones((5,5),'u1'))>0
 humans.append(human);background.append(~(human|objects)&np.isfinite(depth['depth'][i])&(depth['depth'][i]>.05))
background=np.asarray(background);yy,xx=np.mgrid[:H,:W];views=np.arange(0,T,4);pool=[];rgb=[];support_rows=[];support_birth=[]
for i in views:
 valid=background[i].copy();valid[1::2,:]=False;valid[:,1::2]=False
 valid&=depth['conf'][i]>np.percentile(depth['conf'][i][background[i]],25)
 p=world_points(i)[valid];color=depth['images'][i][valid];agree=np.zeros(len(p),int);last=np.full(len(p),-1,int);first=np.full(len(p),T,int)
 for j in views:
  q=p@depth['extrinsics'][j,:3,:3].T+depth['extrinsics'][j,:3,3];uv=q@depth['intrinsics'][j].T;uv=np.rint(uv[:,:2]/np.maximum(q[:,2:],1e-8)).astype(int)
  inside=(q[:,2]>.05)&(uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H);ix=np.flatnonzero(inside);x,y=uv[ix].T
  ok=background[j,y,x]&(abs(q[ix,2]-depth['depth'][j,y,x])<np.maximum(.012,.025*q[ix,2]))&(np.abs(depth['images'][j,y,x].astype(float)-color[ix]).mean(1)<24)
  ix=ix[ok];agree[ix]+=1;first[ix]=np.minimum(first[ix],j);last[ix]=j
 keep=agree>=5;pool.append(p[keep]);rgb.append(color[keep]);support_rows.append(agree[keep]);support_birth.append(first[keep]);print('FUSE',int(i),int(keep.sum()),flush=True)
p=np.concatenate(pool);color=np.concatenate(rgb);support=np.concatenate(support_rows);first=np.concatenate(support_birth)
if len(p)<1000:raise ValueError('Insufficient mutually consistent background geometry')
# Select the highest-supported actually observed point per voxel, rather than averaging separated surfaces.
order=np.argsort(-support,kind='stable');cells=np.floor(p[order]/.004).astype(int);_,ix=np.unique(cells,axis=0,return_index=True);ix=order[ix];p=p[ix].astype('<f4');color=color[ix];support=support[ix]
with (DEST/'background.bin').open('wb') as f:
 def block(a):
  f.write(b'\0'*((-f.tell())%4));offset=f.tell();f.write(a.tobytes());return offset
 memory=dict(url=URL+'background.bin',count=len(p),positions=block(p),colors=block(color.astype('u1')),first_frames=block(np.zeros(len(p),dtype='<u4')),last_frames=block(np.full(len(p),np.iinfo(np.uint32).max,dtype='<u4')),voxel_m=.004,minimum_observations=5,source_frame_ids=frame_ids.tolist(),active_points=[len(p)]*T,mode='full_clip',estimated=True,provenance='Static background from the whole clip. Each point agrees in at least five source views in depth and color; source hands, arms, cups and lids excluded. Not causal online SLAM; unseen regions remain unknown.')
scene['spatial_memory']=memory
# Per-frame foreground and optional unfused background, with original camera coordinates.
cloudframes=[]
with (DEST/'cloud.bin').open('wb') as f:
 def block(a):
  f.write(b'\0'*((-f.tell())%4));offset=f.tell();f.write(a.tobytes());return offset
 for i,m in enumerate(masks):
  semantic=np.zeros((H,W),'u1');semantic[m['cup']>0]=1;semantic[m['lid']>0]=2
  valid=np.isfinite(depth['depth'][i])&(depth['depth'][i]>.05)&~humans[i];valid[1::2,:]=False;valid[:,1::2]=False
  xyz=world_points(i)[valid].astype('<f4');color=depth['images'][i][valid].astype('u1');lab=semantic[valid]
  cloudframes.append(dict(source_frame=int(frame_ids[i]),time=float(frame_ids[i]/30),count=len(xyz),positions=block(xyz),colors=block(color),labels=block(lab)))
scene['cloud'].update(url=URL+'cloud.bin',frames=cloudframes)
# Visible instance surfaces: cut depth edges and ignore implausible long thin components.
for obj in scene['objects']:
 frames=[];kind=obj['label'];name=obj['id']+'-surfaces.bin'
 with (DEST/name).open('wb') as f:
  def block(a):
   f.write(b'\0'*((-f.tell())%4));offset=f.tell();f.write(a.tobytes());return offset
  for i,m in enumerate(masks):
   z=depth['depth'][i];instances=m[kind];mask=cv2.erode(((instances>0)&~humans[i]).astype('u1'),np.ones((3,3),'u1'))>0
   ids=np.full((H,W),-1,'i4');ids[mask]=np.arange(mask.sum());a,b,c,d=ids[:-1,:-1],ids[:-1,1:],ids[1:,:-1],ids[1:,1:];faces=[];local=instances[mask]
   for corners in [(a,b,c),(b,d,c)]:
    grid=np.stack(corners,-1);valid=(grid>=0).all(-1)
    if len(local):valid&=(local[np.maximum(grid,0)]==local[np.maximum(grid[...,0:1],0)]).all(-1)
    faces.append(grid[valid])
   faces=np.concatenate(faces);points=world_points(i)[mask].astype('<f4');colors=depth['images'][i][mask].astype('u1')
   if len(faces):
    lengths=np.linalg.norm(points[faces]-points[faces[:,[1,2,0]]],axis=-1);footprint=z[mask][faces].mean(1)*np.hypot(1/depth['intrinsics'][i,0,0],1/depth['intrinsics'][i,1,1]);faces=faces[(lengths<2.5*footprint[:,None]).all(1)]
   boxes=[]
   for instance in np.unique(instances[mask]):
    subset=points[local==instance]
    if len(subset)>5:boxes.append(dict(id=int(instance),bounds=[np.percentile(subset,1,axis=0).tolist(),np.percentile(subset,99,axis=0).tolist()]))
   frames.append(dict(source_frame=int(frame_ids[i]),time=float(frame_ids[i]/30),vertices=len(points),faces=len(faces),positions=block(points),colors=block(colors),indices=block(faces.astype('<u4')),bounds=[points.min(0).tolist(),points.max(0).tolist()] if len(points) else None,boxes=boxes))
 obj['observed_surface']=dict(url=URL+name,fps=15,frames=frames,all_instances=True,provenance='Current metric multiview depth; visible surfaces only, no invented backsides')
# Independent sparse triangulation is never used as a fitting target.
a=np.load(ROOT/'multiview-triangulation.npz');uv=np.rint(a['uv']).astype(int);f=a['frames'];pred=depth['depth'][f,uv[:,1],uv[:,0]];truth=a['depth'][:,1];old=a['depth'][:,0]
metrics=lambda error:dict(median_mm=float(np.median(abs(error))*1000),p95_mm=float(np.percentile(abs(error),95)*1000))
report=dict(depth_model=json.loads((ROOT/'depth-multiview-v2.json').read_text()),triangulation_check=dict(points=len(truth),before=metrics(old-truth),after=metrics(pred-truth),not_used_for_fitting=True,caveat='Sparse three-view correspondences, limited baseline and matching uncertainty; not dense ground truth'),fused_background_points=len(p),support_min=int(support.min()),support_median=float(np.median(support)),source_trajectory_edited=False,whole_clip_offline=True,unobserved_geometry_generated=False)
scene['geometry'].pop('depth_scale_calibration',None);scene['geometry']['multiview_reconstruction']=report;scene['geometry']['background_completion']='Only repeatedly observed surfaces are fused. Never-observed backside regions remain unknown.'
scene['completion_available']=False
scene['reconstruction_validation']=report
# Old visual contact proximity used the previous scene and cannot be reused as a new measurement.
contact=scene.get('tactile',{});keep_video_contact=False
if contact.get('status')=='video_estimate' and contact.get('url'):
    payload=json.loads((Path('/opt/arenalab/store')/contact['url'].removeprefix('/assets/')).read_text())
    video_path=Path('/opt/arenalab/store')/scene['video']['url'].removeprefix('/assets/')
    hands_path=ROOT.parent/'egodex-threepoint/source_hands.npz'
    keep_video_contact=payload.get('depth_independent',False) and payload.get('source_video_sha256')==hashlib.sha256(video_path.read_bytes()).hexdigest() and payload.get('source_hands_sha256')==hashlib.sha256(hands_path.read_bytes()).hexdigest()
if not keep_video_contact:scene['tactile']=dict(status='unavailable',provenance='Depth-dependent proximity estimates invalidated by reconstruction; no tactile sensor data')
(DEST/'scene.json').write_text(json.dumps(scene,separators=(',',':')));(DEST/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
