from pathlib import Path
import numpy as np,json,hashlib
D=Path('/opt/arenalab/store');scene=json.loads((D/'manipulation4d/reconstruction-v2/scene.json').read_text());memory=scene['spatial_memory'];cloud=scene['cloud'];cam=scene['source_camera'];mb=(D/memory['url'].removeprefix('/assets/')).read_bytes();cb=(D/cloud['url'].removeprefix('/assets/')).read_bytes();history=np.frombuffer(mb,'<f4',memory['count']*3,memory['positions']).reshape(-1,3);k=np.array(cam['intrinsic']);W=256;H=round(W*cam['height']/cam['width']);rows=[]
def project(points,view):
 c=points@view[:3,:3].T+view[:3,3];uv=c@k.T;uv=uv[:,:2]/np.maximum(c[:,2:],1e-8);uv*=np.array([W/cam['width'],H/cam['height']]);uv=np.floor(uv).astype(int);inside=(c[:,2]>.03)&(uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H);return c[:,2],uv,inside
for index in [0,66,132]:
 f=cloud['frames'][index];xyz=np.frombuffer(cb,'<f4',f['count']*3,f['positions']).reshape(-1,3);labels=np.frombuffer(cb,'u1',f['count'],f['labels']);view=np.linalg.inv(cam['poses'][f['source_frame']]);z,uv,inside=project(xyz,view);dep=np.full(H*W,np.inf);nearest_bg=np.zeros(H*W,bool)
 ids=np.flatnonzero(inside);expanded=[]
 for dy in [-1,0,1]:
  for dx in [-1,0,1]:
   q=uv[ids]+[dx,dy];ok=(q[:,0]>=0)&(q[:,0]<W)&(q[:,1]>=0)&(q[:,1]<H);expanded.extend(zip((q[ok,1]*W+q[ok,0]).tolist(),ids[ok].tolist()))
 for pixel,j in expanded:
  if z[j]<dep[pixel]:dep[pixel]=z[j];nearest_bg[pixel]=labels[j]==0
 hz,huv,hi=project(history,view);observed=np.zeros(len(history),bool);ix=np.flatnonzero(hi);pix=huv[ix,1]*W+huv[ix,0];observed[ix]=nearest_bg[pix]&(abs(dep[pix]-hz[ix])<np.maximum(.012,.025*hz[ix]));rows.append(dict(source_frame=f['source_frame'],current_points=int(f['count']),current_background_points=int((labels==0).sum()),history_visible_matching=int(observed.sum()),history_dimmed=int((~observed).sum()),history_out_of_view=int((~hi).sum())))
 assert observed.any() and (~observed).any()
p=json.loads((D/scene['tactile']['proximity_url'].removeprefix('/assets/')).read_text());a=np.asarray(p['likelihood'],float);assert a.shape==(133,2,5) and np.nanmin(a)>=0 and np.nanmax(a)<=1;assert np.nanmax(np.std(a,axis=2))>.02
report=dict(visibility=rows,finger_frames=133,distinct_finger_scores=True,pressure_measured=False,scene_cloud_sha256=hashlib.sha256(cb).hexdigest(),history_sha256=hashlib.sha256(mb).hexdigest());(D/'manipulation4d/reconstruction-v2/display-diagnostics-validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
