"""User-declared UMI floor calibration; immutable poses and versioned outputs."""
from pathlib import Path
import hashlib,json,threading
import numpy as np
from pydantic import BaseModel,Field
from fastapi import HTTPException
from gripper_source import ROOT,digest
from gripper_job import sources,REGISTRY

class HeightRequest(BaseModel):
 scene_id:str
 minimum_height_cm:float=Field(ge=0,le=300,allow_inf_nan=False)

def calibrated_payload(source,height_cm):
 payload={k:source[k].copy() for k in source.files}
 floor=float(np.min(payload['tcp_zup'][:,:,2]))-float(height_cm)/100
 payload['floor_z']=np.array(floor)
 return payload,floor

def attach_height_routes(app,lock):
 def read(scene_id):
  row=sources().get(scene_id)
  if not row:raise HTTPException(404,'Source unavailable')
  work=Path(row['work']);meta=json.loads((work/'source.json').read_text())
  if meta['dataset']!='UMI':raise HTTPException(400,'Height calibration is only available for UMI')
  return work,meta,np.load(work/'source.npz')
 @app.get('/umi-height/{scene_id}')
 def preview(scene_id:str):
  work,meta,src=read(scene_id);xyz=src['tcp_zup'];floor=float(src['floor_z'])
  return dict(points=xyz.tolist(),fps=30,minimum_height_cm=float((xyz[:,:,2].min()-floor)*100),floor_z_m=floor,calibration=meta.get('height_calibration'),reference=meta['target_reference'])
 @app.post('/umi-height')
 def apply(request:HeightRequest):
  from workbench_api import save
  with lock:
   work,meta,src=read(request.scene_id)
   original=meta.get('height_calibration',{}).get('original_scene_id',request.scene_id)
   base=sources()[original];basework=Path(base['work']);raw=np.load(basework/'source.npz');base_meta=json.loads((basework/'source.json').read_text())
   identity=dict(original_scene_id=original,source_sha256=digest(basework/'source.npz'),minimum_height_cm=request.minimum_height_cm,policy='user-umi-floor-v1')
   key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16];sid=original+'-height-'+key
   target=ROOT/'gripper-height'/sid;dest=Path('/opt/arenalab/store/manipulation4d/gripper-height')/sid
   payload,floor=calibrated_payload(raw,request.minimum_height_cm)
   if not (target/'source.npz').exists():
    target.mkdir(parents=True,exist_ok=True);np.savez_compressed(target/'source.npz',**payload)
    for k in raw.files:
     if k!='floor_z':assert np.array_equal(raw[k],payload[k],equal_nan=True),k
    base_meta.update(height_calibration=dict(**identity,floor_z_m=floor,estimated=True),floor_provenance='User-entered minimum gripper trajectory height; floor estimate shared across target robots',estimated_floor_z_m=floor)
    save(target/'source.json',base_meta)
   registry=sources();registry[sid]=dict(work=str(target),destination=str(dest));save(REGISTRY,registry)
   catalog_path=Path('/opt/arenalab/store/manipulation4d/catalog.json');catalog=json.loads(catalog_path.read_text());original_entry=next(x for x in catalog if x['id']==original)
   entry=next((x for x in catalog if x['id']==sid),None)
   if entry is None:
    entry={**original_entry,'id':sid,'title':original_entry['title']+' · '+str(request.minimum_height_cm)+' cm','height_calibration':identity};entry.pop('scene_url',None);catalog.append(entry);save(catalog_path,catalog)
   return entry
