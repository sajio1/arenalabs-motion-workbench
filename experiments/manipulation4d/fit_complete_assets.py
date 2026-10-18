"""Fit completed category prototypes to observed instances; fixed per-instance scale.
The generated shape is immutable; fitted assets and poses are separate estimates.
"""
from pathlib import Path
import json,hashlib,cv2,numpy as np,trimesh
from scipy.spatial import cKDTree
from prepare_objects import ROOT,OUT,depth,frame_ids,mask_at,world_points
DEST=Path('/opt/arenalab/store/manipulation4d/reconstruction-v2');URL='/assets/manipulation4d/reconstruction-v2/';rows=json.loads((ROOT/'asset-completion-v3/selected-objects.json').read_text());rng=np.random.default_rng(42);H,W=depth['depth'].shape[1:]
hands=np.load(OUT/'hand-masks.npz')['masks'];arms=np.load(OUT/'arm-masks.npz')['masks'];human=[cv2.dilate(((mask_at(hands,i)>0)|(mask_at(arms,i)>0)).astype('u1'),np.ones((3,3),'u1'))>0 for i in range(len(frame_ids))];world=[world_points(i) for i in range(len(frame_ids))]
def transform_fit(a,b,reference):
 ac,bc=a.mean(0),b.mean(0);cov=(a-ac).T@(b-bc);cov+=reference.T*np.sum((a-ac)**2) # shared SO(3) regularization around the input orientation
 u,_,v=np.linalg.svd(cov);r=v.T@u.T
 if np.linalg.det(r)<0:v[-1]*=-1;r=v.T@u.T
 return r,bc-r@ac

def project(points,i):
 c=points@depth['extrinsics'][i,:3,:3].T+depth['extrinsics'][i,:3,3];uv=c@depth['intrinsics'][i].T;uv=np.rint(uv[:,:2]/np.maximum(uv[:,2:],1e-8)).astype(int);inside=(c[:,2]>.05)&(uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H)
 return uv,c[:,2],inside

def fit(sample,rot,t,scale,obs,mask,i,steps=12):
 tree=cKDTree(obs);r=rot.copy();t=t.copy()
 for _ in range(steps):
  p=sample@r.T*scale+t;uv,z,inside=project(p,i);ids=np.flatnonzero(inside)
  if not len(ids):return None
  pix=(uv[ids,1]//2)*(W//2+1)+uv[ids,0]//2;order=np.lexsort((z[ids],pix));_,first=np.unique(pix[order],return_index=True);ids=ids[order[first]]
  allowed=cv2.dilate((mask|human[i]).astype('u1'),np.ones((5,5),'u1'))>0;ids=ids[allowed[uv[ids,1],uv[ids,0]]]
  if len(ids)<30:return None
  distance,nearest=tree.query(p[ids]);keep=distance<=np.percentile(distance,65)
  if keep.sum()<20:return None
  reference=rot@r.T
  dr,dt=transform_fit(p[ids][keep],obs[nearest[keep]],reference);r=dr@r;t=dr@t+dt
 p=sample@r.T*scale+t;uv,z,inside=project(p,i);ids=np.flatnonzero(inside);within=mask[uv[ids,1],uv[ids,0]];distance=tree.query(p[ids])[0];precision=float(within.mean()) if len(ids) else 0
 # Screen contour agreement discourages a hidden-surface fit from drifting outside the source object.
 dmask=cv2.distanceTransform((~mask).astype('u1'),cv2.DIST_L2,3);screen=float(np.mean(np.minimum(dmask[uv[ids,1],uv[ids,0]],15))) if len(ids) else 15
 residual=float(np.sqrt(np.mean(np.sort(distance)[:max(20,int(len(distance)*.65))]**2)))
 obs_dist=cKDTree(p).query(obs)[0];coverage_residual=float(np.sqrt(np.mean(np.sort(obs_dist)[:max(20,int(len(obs_dist)*.85))]**2)))
 target_xy=np.argwhere(mask)[:,::-1][::2];screen_in=float(np.minimum(cKDTree(uv[ids].astype(float)).query(target_xy)[0],20).mean()) if len(ids) and len(target_xy) else 20
 score=.5*(residual+coverage_residual)+screen*.002+screen_in*.003
 return r,t,dict(surface_rmse_m=residual,observed_coverage_rmse_m=coverage_residual,projected_precision=precision,projected_coverage_distance_px=screen_in,score=score)

completed=[];reports=[]
for row in rows:
 kind=row['kind'];folder=Path(row['folder'])/'isolated';mesh=trimesh.load(folder/'world.glb',force='mesh',process=False);prior_sha=hashlib.sha256((folder/'world.glb').read_bytes()).hexdigest();center=mesh.bounding_box.centroid.copy();mesh.vertices-=center;sample,_=trimesh.sample.sample_surface(mesh,6000,seed=42)
 # Preserve a compact mesh while keeping a genuinely closed backside.
 if len(mesh.faces)>80000:
  cells=np.floor((mesh.vertices-mesh.bounds[0])/(mesh.extents.max()/128)).astype(int);_,inverse=np.unique(cells,axis=0,return_inverse=True);count=np.bincount(inverse);vertices=np.column_stack([np.bincount(inverse,weights=mesh.vertices[:,j])/count for j in range(3)]);faces=inverse[mesh.faces];faces=faces[(faces[:,0]!=faces[:,1])&(faces[:,1]!=faces[:,2])&(faces[:,0]!=faces[:,2])];reduced=trimesh.Trimesh(vertices,faces,process=True);reduced.visual.vertex_colors=[232,218,189,255]
 else:reduced=mesh.copy()
 # Export original complete mesh too; browser mesh simplification can change topology at tiny seams.
 full_name=f'{kind}-complete-v3-full.glb';mesh.export(DEST/full_name);name=f'{kind}-complete-v3.glb';reduced.export(DEST/name)
 class_masks=np.load(OUT/f'{kind}-masks.npz')['masks'];masks=[mask_at(class_masks,i) for i in range(len(frame_ids))];identifiers=sorted({int(n) for m in masks for n in np.unique(m) if n>0});frames=[dict(source_frame=int(f),time=float(f/30),instances=[]) for f in frame_ids];instances=[]
 proto_mask=(masks[row['keyframe']]==row['instance_id'])&~human[row['keyframe']];proto_obs=world[row['keyframe']][proto_mask];proto_median=np.median(proto_obs,axis=0)
 for ident in identifiers:
  observed=[];valid_masks=[];scores=[]
  for i in range(len(frame_ids)):
   mask=(masks[i]==ident)&~human[i];mask=cv2.erode(mask.astype('u1'),np.ones((3,3),'u1'))>0;obs=world[i][mask];obs=obs[np.isfinite(obs).all(1)];observed.append(obs[::max(1,len(obs)//4000)]);valid_masks.append(mask);scores.append(len(obs))
  key=int(np.argmax(scores))
  if scores[key]<60:continue
  obs=observed[key];t0=center+np.median(obs,axis=0)-proto_median
  # Shared observation-derived scale candidates. Never robot-, action- or frame-specific tuning.
  spans=np.ptp(obs,axis=0);proto_spans=np.maximum(np.ptp(proto_obs,axis=0),.001);base=float(np.clip(np.median(spans/proto_spans),.4,2.5));candidates=[]
  for scale in base*np.array([.8,1,1.25]):
   result=fit(sample,np.eye(3),t0,scale,obs,valid_masks[key],key,18)
   if result is not None:candidates.append((scale,result))
  if not candidates:continue
  scale,result=min(candidates,key=lambda row:row[1][2]['score']);r,t,details=result;solved={key:(r,t,details)}
  for direction in [-1,1]:
   prev_r,prev_t=r.copy(),t.copy();prev_center=np.median(obs,axis=0)
   for i in range(key+direction,len(frame_ids) if direction>0 else -1,direction):
    obs_i=observed[i]
    if len(obs_i)<40:continue
    now_center=np.median(obs_i,axis=0);guess=prev_t+now_center-prev_center;fitted=fit(sample,prev_r,guess,scale,obs_i,valid_masks[i],i)
    if fitted is None:continue
    prev_r,prev_t,info=fitted;prev_center=now_center;solved[i]=fitted
  for i,(r,t,info) in solved.items():
   matrix=np.eye(4);matrix[:3,:3]=r*scale;matrix[:3,3]=t;bounds=reduced.bounding_box.vertices@matrix[:3,:3].T+t
   frames[i]['instances'].append(dict(id=ident,matrix=matrix.tolist(),bounds=[bounds.min(0).tolist(),bounds.max(0).tolist()],status='estimated',observed=True,**info))
  instances.append(dict(segmentation_id=ident,scale_estimated=float(scale),keyframe=int(frame_ids[key]),tracked_frames=len(solved),keyframe_fit=details));print('COMPLETE_INSTANCE',kind,ident,len(solved),scale,details,flush=True)
 trackname=f'{kind}-complete-v5-track.json';(DEST/trackname).write_text(json.dumps(frames,separators=(',',':')))
 info=dict(kind=kind,model='SAM 3D Objects',raw_prior_sha256=prior_sha,full_asset_url=URL+full_name,browser_asset_url=URL+name,full_vertices=len(mesh.vertices),full_faces=len(mesh.faces),full_watertight=bool(mesh.is_watertight),browser_vertices=len(reduced.vertices),browser_faces=len(reduced.faces),backside='Generated estimate, not observed truth',instances=instances,source_trajectory_edited=False)
 completed.append(dict(kind=kind,model_url=URL+name,full_asset_url=URL+full_name,track_url=URL+trackname,fps=15,estimated=True));reports.append(info)
scene=json.loads((DEST/'scene.json').read_text());scene['completed_assets']=completed;scene['completion_available']=True;scene['completion_default']=True;scene['geometry']['complete_assets']='SAM 3D full meshes with generated backsides; instance pose and scale fitted to source observations; not contact ground truth';(DEST/'scene-complete-v5-candidate.json').write_text(json.dumps(scene,separators=(',',':')));(DEST/'complete-assets-v5-validation.json').write_text(json.dumps(reports,indent=2));print('ASSETS_COMPLETE',json.dumps(reports),flush=True)
