"""Independent source-fingertip proximity to visible object points, not pressure."""
from arenalab_workbench.settings import paths
from pathlib import Path
import json,os,shutil
import numpy as np
from scipy.spatial import cKDTree
SOURCE=Path(os.environ['ARENA_SOURCE_WORK']);DEST=Path(os.environ['ARENA_EXPORT_DIR']);scene=json.loads((DEST/'scene.json').read_text());store=paths.store;cloud=scene['cloud'];buffer=(store/cloud['url'].removeprefix('/assets/')).read_bytes() if cloud.get('url') else None;hands=np.load(SOURCE/'source_hands.npz');S=hands['source_to_zup'];tips=hands['points'][:,:,[4,8,12,16,20]]@S;confidence=hands['confidence'][:,:,[4,8,12,16,20]];values=[];distances=[]
for frame_index,frame in enumerate(cloud['frames']):
 if cloud.get('format')=='depth-rgb-v1':
  from arenalab_workbench.dataset_io.portable_reader import geometry_frame,points_from_geometry
  value=points_from_geometry(geometry_frame(store/cloud['geometry_url'].removeprefix('/assets/'),frame_index));xyz=value['xyz'];labels=value['instance_ids']
 else:
  n=frame['count'];xyz=np.frombuffer(buffer,'<f4',n*3,frame['positions']).reshape(n,3);labels=np.frombuffer(buffer,'u1',n,frame['labels'])
 objects=xyz[(labels>0)&np.isfinite(xyz).all(1)];index=frame['source_frame']
 if len(objects)<20:values.append([[None]*5,[None]*5]);distances.append([[None]*5,[None]*5]);continue
 distance=cKDTree(objects).query(tips[index].reshape(-1,3))[0].reshape(2,5);scores=np.exp(-.5*(distance/.02)**2)*confidence[index];scores=np.where(confidence[index]>0,scores,np.nan);values.append([[float(v) if np.isfinite(v) else None for v in hand] for hand in scores]);distances.append((distance*1000).tolist())
payload=dict(fps=cloud['fps'],source_frame_ids=[f['source_frame'] for f in cloud['frames']],likelihood=values,nearest_visible_surface_mm=distances,fingers=['thumb','index','middle','ring','little'],method='Independent source fingertip distance to current-frame segmented object points; confidence-weighted Gaussian proximity, 20 mm scale',measured=False,force_available=False,calibrated_probability=False,limitations=['Estimated depth and source finger error affect indices','Proximity is not verified contact, pressure or force','Only currently observed object surfaces are used'])
(DEST/'finger-proximity.json').write_text(json.dumps(payload,separators=(',',':')))
video_path=DEST/'video-contact.json'
if video_path.exists():
 video=json.loads(video_path.read_text());known=[v for row in video['likelihood'] for hand in row for v in hand if v is not None]
 if known and max(known)-min(known)<1e-8 and len(known)==len(video['likelihood'])*10:
  raw=DEST/'video-contact-unfiltered.json'
  if not raw.exists():shutil.copy2(video_path,raw)
  video['raw_model_scores']=video['likelihood'];video['likelihood']=[[[None]*5,[None]*5] for _ in video['likelihood']];video['quality_status']='unreliable_constant_digit_scores';video['quality_reason']='Video model returned the same score for every digit in every sample; non-discriminating estimates are withheld'
  for observation in video['observations']:observation['status']='unreliable';observation['evidence']=video['quality_reason']
  video_path.write_text(json.dumps(video,separators=(',',':')))
from arenalab_workbench.gpu_scheduler import merge_scene
merge_scene(DEST/'scene.json',lambda scene:scene.setdefault('tactile',{}).update(proximity_url='/assets/'+(DEST/'finger-proximity.json').relative_to(store).as_posix(),reason='Per-digit geometry proximity; video contact is an estimate, not measured pressure.'))

known=[v for row in values for hand in row for v in hand if v is not None];print(json.dumps(dict(samples=len(values),known=len(known),min=min(known,default=None),max=max(known,default=None),pressure=False)),flush=True)
