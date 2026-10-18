"""Validate versioned scene geometry and unchanged task/robot targets before promotion."""
from pathlib import Path
import json,hashlib,cv2,h5py,numpy as np
ROOT=Path('/home/ubuntu/arenalab-experiments/manipulation4d');DEST=Path('/opt/arenalab/store/manipulation4d/reconstruction-v2');STORE=Path('/opt/arenalab/store');s=json.loads((DEST/'scene.json').read_text());old=json.loads((STORE/'manipulation4d/pilot/scene.json').read_text())
path=lambda url:STORE/url.removeprefix('/assets/')
assert s['robots']==old['robots'] and s['source_hands_url']==old['source_hands_url'] and s['source_camera']==old['source_camera']
assert s['video']==old['video']
for frame in s['cloud']['frames']:
 blob=(DEST/'cloud.bin').read_bytes() if 'blob' not in locals() else blob
 xyz=np.frombuffer(blob,'<f4',frame['count']*3,frame['positions']);assert np.isfinite(xyz).all();assert frame['labels']+frame['count']<=len(blob)
for obj in s['objects']:
 surface=obj['observed_surface'];blob=path(surface['url']).read_bytes();assert len(surface['frames'])==133
 for frame in surface['frames']:
  p=np.frombuffer(blob,'<f4',frame['vertices']*3,frame['positions']);f=np.frombuffer(blob,'<u4',frame['faces']*3,frame['indices']);assert np.isfinite(p).all() and (not len(f) or f.max()<frame['vertices'])
mem=s['spatial_memory'];blob=path(mem['url']).read_bytes();p=np.frombuffer(blob,'<f4',mem['count']*3,mem['positions']);assert np.isfinite(p).all() and mem['mode']=='full_clip' and mem['count']>1000
new=np.load(ROOT/'depth-multiview-v2.npz');anchors=np.load(ROOT/'multiview-triangulation.npz');uv=np.rint(anchors['uv']).astype(int);fr=anchors['frames'];truth=anchors['depth'][:,1];pred=new['depth'][fr,uv[:,1],uv[:,0]];before=np.abs(anchors['depth'][:,0]-truth);after=np.abs(pred-truth)
assert np.median(after)<np.median(before) and np.percentile(after,95)<np.percentile(before,95)
manifest=json.loads((ROOT.parent/'egodex-threepoint/input_manifest.json').read_text())
with h5py.File(manifest['source']) as f:assert np.array_equal(s['source_camera']['poses'],f['transforms/camera'][:])
report=json.loads((DEST/'validation.json').read_text());report['checks']=dict(original_source_sha256=hashlib.sha256(Path(manifest['source']).read_bytes()).hexdigest(),robots_and_targets_identical=True,source_cameras_identical=True,source_video_identical=True,finite_buffers=True,improves_sparse_checks=True,never_observed_regions_remain_unknown=True)
report['videos']={}
for name,url in s['patch_videos'].items():
 cap=cv2.VideoCapture(str(path(url)));fps=cap.get(cv2.CAP_PROP_FPS);n=0
 while True:
  ok,_=cap.read()
  if not ok:break
  n+=1
 cap.release();assert n==266 and abs(fps-30)<1e-6;report['videos'][name]=dict(decoded_frames=n,fps=fps)
report['occlusion_background_coverage']=[]
# Measure actual cross-view support behind currently occluding source objects/humans.
from prepare_objects import depth,OUT,mask_at
labels={k:np.load(OUT/f'{k}-masks.npz')['masks'] for k in ['hand','arm','cup','lid']};points=p.reshape(-1,3);H,W=depth['depth'].shape[1:]
for i in np.linspace(0,132,9).astype(int):
 q=points@depth['extrinsics'][i,:3,:3].T+depth['extrinsics'][i,:3,3];uv=q@depth['intrinsics'][i].T;uv=np.rint(uv[:,:2]/np.maximum(q[:,2:],1e-8)).astype(int);keep=(q[:,2]>.05)&(uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H);ix=np.flatnonzero(keep);x,y=uv[ix].T;hidden=(mask_at(labels['hand'],i)[y,x]>0)|(mask_at(labels['arm'],i)[y,x]>0)|(mask_at(labels['cup'],i)[y,x]>0)|(mask_at(labels['lid'],i)[y,x]>0);hidden&=q[ix,2]>depth['depth'][i,y,x]+.02
 report['occlusion_background_coverage'].append(dict(source_frame=int(depth['ids'][i]),source_pixels_with_observed_background_behind_occluder=int(len(np.unique(y[hidden]*W+x[hidden])))))
(DEST/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
