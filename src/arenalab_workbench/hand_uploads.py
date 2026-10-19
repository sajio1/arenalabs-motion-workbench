"""Persist and inspect end-effector packages without inventing mounting frames."""
from pathlib import Path
import hashlib,json,threading,uuid,xml.etree.ElementTree as ET
from fastapi import UploadFile,File,HTTPException
from greenwich_pipelines.viewer.uploads import unpack_robot,MAX_BYTES
from greenwich_pipelines.viewer.export import robot_model

def attach_hand_routes(app,storage):
    storage=Path(storage);registry=storage/'end-effectors/registry.json';lock=threading.Lock()
    def rows():return json.loads(registry.read_text()) if registry.exists() else []
    @app.get('/hands')
    def hands():
        return [dict(id='none',label='No hand',status='ready',supported=True),dict(id='wuji',label='Wuji',status='ready',supported=True),dict(id='robotiq-2f-140',label='Robotiq 2F-140',status='ready',supported=True),*rows()]
    @app.post('/hands')
    async def upload_hand(file:UploadFile=File(...)):
        content=await file.read(MAX_BYTES+1);digest=hashlib.sha256(content).hexdigest()
        with lock:
            existing=next((r for r in rows() if r['package_sha256']==digest),None)
        if existing:return existing
        ident='hand_'+uuid.uuid4().hex[:12];folder=storage/'end-effectors'/ident
        try:
            xml,label,report=unpack_robot(content,file.filename or '',folder)
            root=ET.parse(xml).getroot();joints=[j for j in root.findall('joint') if j.get('type')!='fixed']
            if not 1<=len(joints)<=80:raise ValueError('An end effector must have 1–80 movable joints')
            row=dict(id=ident,label=label,status='needs_mapping',supported=False,package_sha256=digest,
                report={**report,'movable_joints':[j.get('name') for j in joints],'required':['Left/right mounting transform and palm/TCP reference','Source-finger or gripper joint mapping'],'message':'Saved. Attach this hand after supplying its physical mount and retarget mapping; no automatic attachment is claimed.'})
            if report['missing_meshes']:row['status']='missing_assets'
            else:
                try:
                    robot_model(dict(xml=str(xml)),folder)
                    row.update(model_url=f'/assets/end-effectors/{ident}/model.glb',neutral_url=f'/assets/end-effectors/{ident}/neutral.json')
                except Exception:
                    row['status']='invalid_model';row['report']['message']='Saved URDF could not be loaded. Check mesh files and joint definitions.'
            with lock:
                current=rows();existing=next((r for r in current if r['package_sha256']==digest),None)
                if existing:return existing
                registry.parent.mkdir(parents=True,exist_ok=True);temporary=registry.with_suffix('.tmp');temporary.write_text(json.dumps([*current,row],indent=2));temporary.replace(registry)
            return row
        except (ValueError,ET.ParseError) as error:raise HTTPException(400,str(error)) from error
