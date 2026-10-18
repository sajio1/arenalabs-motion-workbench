"""Explicit floor calibration, first-height UMI defaults and temporary previews."""
from pathlib import Path
import json,time,uuid
import numpy as np
from fastapi import HTTPException,Request
from pydantic import BaseModel,Field
from gripper_source import ROOT
REGISTRY=ROOT/'height-previews.json'
DEFAULTS=ROOT/'height-defaults.json'
CATALOG=Path('/opt/arenalab/store/manipulation4d/catalog.json')

def read(path,default=None):return json.loads(path.read_text()) if path.exists() else ({} if default is None else default)
def variants():return read(REGISTRY)
def variant(sid):return variants().get(sid)
def default_id(sid):return read(DEFAULTS).get(sid)
def public_entry(sid):
 v=variant(sid)
 from dataset_io.storage import entry as uploaded_entry
 upload=uploaded_entry(sid)
 if upload:return upload['public']
 return v['public'] if v else next((x for x in read(CATALOG,[]) if x['id']==sid),None)
def source_data(sid):
 from gripper_job import sources
 v=variant(sid);original=v['original_scene_id'] if v else sid;entry=public_entry(original)
 if not entry:raise HTTPException(404,'Recording unavailable')
 g=sources().get(original)
 if g:
  work=Path(g['work']);src=np.load(work/'source.npz');points=src['tcp_zup'];floor=float(src['floor_z']);row=None
 else:
  from body_prior import folder,prepare
  from scene_job import resolve_clip
  work=folder(original)
  if not (work/'observations.npz').exists():prepare(original)
  src=np.load(work/'observations.npz');points=src['palm_position'];floor=float(src['floor_z']);_,row=resolve_clip(original)
 if v:floor=float(v['calibration']['floor_z_m'])
 return original,entry,work,src,points,floor,g,row

class HeightRequest(BaseModel):
 scene_id:str
 minimum_height_cm:float=Field(ge=0,le=300,allow_inf_nan=False)

def attach_height_routes(app,lock):
 from scene_job import save
 import threading
 def sweeper():
  while True:
   try:
    with lock:expire_temporary()
   except Exception:
    import logging;logging.exception('Temporary height preview cleanup failed')
   time.sleep(3600)
 threading.Thread(target=sweeper,daemon=True,name='height-preview-cleanup').start()
 @app.get('/trajectory-height/{scene_id}')
 def preview(scene_id:str,request:Request):
  from dataset_io.routes import require_scene
  require_scene(scene_id,request)
  original,entry,work,src,points,floor,g,row=source_data(scene_id)
  missing=entry['dataset']=='UMI'
  return dict(points=points.tolist(),fps=30,minimum_height_cm=float((points[:,:,2].min()-floor)*100),floor_z_m=floor,requires_initial_height=missing and not default_id(original) and not variant(scene_id),default_scene_id=default_id(original),default_scene_url=(variant(default_id(original)) or {}).get('public',{}).get('scene_url'),reference='Both source gripper references' if g else 'Both source wrist references' if entry['dataset']=='Nymeria' else 'Both source palm references')
 @app.post('/trajectory-height')
 def apply(body:HeightRequest,http:Request):
  from dataset_io.routes import require_scene
  require_scene(body.scene_id,http)
  request=body
  original,entry,work,src,points,floor,g,row=source_data(request.scene_id)
  with lock:
   sid='height-'+uuid.uuid4().hex;dest=Path('/opt/arenalab/store/manipulation4d/height-previews')/sid;dest.mkdir(parents=True)
   from dataset_io.storage import mark_private
   mark_private(dest,entry.get('owner'))
   floor=float(points[:,:,2].min())-request.minimum_height_cm/100
   calibration=dict(original_scene_id=original,minimum_height_cm=request.minimum_height_cm,floor_z_m=floor,estimated=True,policy='user-source-floor-v2')
   eligible=entry['dataset']=='UMI' and not default_id(original)
   public={**entry,'id':sid,'title':entry['title']+' · '+str(request.minimum_height_cm)+' cm','temporary':True,'height_calibration':calibration};public.pop('scene_url',None)
   record=dict(original_scene_id=original,public=public,calibration=calibration,created_at=time.time(),temporary=True,eligible_first=eligible,destination=str(dest),body_work=str(ROOT/'height-body'/sid),source_row=row)
   if g:
    from gripper_job import sources,REGISTRY as GRIPPERS
    target=ROOT/'height-grippers'/sid;target.mkdir(parents=True);payload={k:src[k].copy() for k in src.files};payload['floor_z']=np.array(floor)
    # Re-read the original when the user adjusts an already-custom preview.
    raw=np.load(work/'source.npz');payload={k:raw[k].copy() for k in raw.files};payload['floor_z']=np.array(floor)
    np.savez_compressed(target/'source.npz',**payload)
    meta=read(work/'source.json');meta.update(height_calibration=calibration,floor_provenance='User-declared trajectory minimum height; shared floor across robots',estimated_floor_z_m=floor)
    save(target/'source.json',meta);registry=sources();registry[sid]=dict(work=str(target),destination=str(dest));save(GRIPPERS,registry)
   elif entry.get('scene_url'):
    scene=read(Path('/opt/arenalab/store')/entry['scene_url'].removeprefix('/assets/'))
    if not scene.get('geometry',{}).get('scene_work'):scene=None
    if scene is not None:
     scene.update(id=sid,title=public['title'],robots=[],patch_videos={},patch_unavailable={},temporary=True)
     scene['geometry']['ground']=dict(height_m=floor,estimated=True,height_calibration=calibration)
     scene.pop('generation_status',None);save(dest/'scene.json',scene);public['scene_url']='/assets/'+(dest/'scene.json').relative_to('/opt/arenalab/store').as_posix()
   all_rows=variants();all_rows[sid]=record;save(REGISTRY,all_rows)
   return public

def complete(sid,scene_url):
 """Called under API catalog lock. Only first successful missing-height result persists."""
 from scene_job import save
 record=variant(sid)
 if not record:return False
 records=variants();defaults=read(DEFAULTS);original=record['original_scene_id']
 if record['eligible_first'] and not defaults.get(original):defaults[original]=sid;save(DEFAULTS,defaults)
 keep=defaults.get(original)==sid
 record['temporary']=not keep;record['public'].update(scene_url=scene_url,temporary=not keep);records[sid]=record;save(REGISTRY,records)
 scene_path=Path('/opt/arenalab/store')/scene_url.removeprefix('/assets/');scene=read(scene_path);scene['temporary']=not keep;save(scene_path,scene)
 if keep:
  catalog=read(CATALOG,[])
  for entry in catalog:
   if entry['id']==original:entry.update(scene_url=scene_url,height_calibration=record['calibration'])
  save(CATALOG,catalog)
 return True

def expire_temporary():
 """Remove generated preview artifacts after 24 h; never remove source recordings."""
 import shutil
 from scene_job import save
 from gripper_job import sources,REGISTRY as GRIPPERS
 now=time.time();records=variants();grippers=sources();changed=False
 active=set()
 for p in (ROOT/'queue').glob('*/state.json'):
  if read(p).get('status') in ('running','queued'):active.add(read(p.with_name('job.json')).get('scene_id'))
 for sid,row in list(records.items()):
  if not row['temporary'] or sid in active or now-row['created_at']<86400:continue
  dest=Path(row['destination']);url=row['public'].get('scene_url')
  if url:dest=Path('/opt/arenalab/store')/url.removeprefix('/assets/')
  if dest.exists():
   scene=read(dest/'scene.json')
   if scene.get('id')==sid and (Path('/opt/arenalab/store/manipulation4d/height-previews') in dest.resolve().parents or Path('/opt/arenalab/store/manipulation4d/generated') in dest.resolve().parents):shutil.rmtree(dest)
  for path,parent in [(Path(row['body_work']),ROOT/'height-body'),(ROOT/'height-grippers'/sid,ROOT/'height-grippers')]:
   if path.exists() and parent.resolve() in path.resolve().parents:shutil.rmtree(path)
  records.pop(sid);grippers.pop(sid,None);changed=True
 if changed:save(REGISTRY,records);save(GRIPPERS,grippers)
