"""Current platform API. Thin adapters preserve the deployed numerical pipeline."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
from urllib.parse import quote, urlsplit

from fastapi import FastAPI, Request, HTTPException, Header, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from . import __version__
from .bridge import alias
from .services import PlatformServices
from .operator import attach_operator
from .media import attach_media
from .contracts import RunRequest, BatchRequest, JobView, PlanView, Contract
from .store import Store, Conflict

PREFIX = '/api/v1'
COOKIE = 'arena_workspace'
TERMINAL = {'succeeded','failed','cancelled','interrupted','outcome_unknown'}
OPERATIONS = {
    'manipulation': dict(target='workbench',path='/generate',status='/jobs/',stages=[
        'source adaptation','native SMPL-H where required','robot retargeting',
        'shared refinement','scene perception where available','source FOV / preview','contact where available']),
    'locomotion': dict(target='viewer',path='/api/convert',status='/api/jobs/',stages=[
        'source adaptation','fullbody retargeting','existing locomotion refinement','playback']),
    'export': dict(target='workbench',path='/exports',status='/exports/',stages=[
        'validate generated result','named-joint motion and models','modalities and provenance','archive']),
}

def owner(request):
    token=request.cookies.get(COOKIE)
    if not token:raise HTTPException(401,'Create a workspace with POST /api/v1/sessions; retain its cookie')
    return hashlib.sha256(token.encode()).hexdigest()

def safe_asset(root, url):
    parsed=urlsplit(url)
    if parsed.scheme or parsed.netloc or not parsed.path.startswith('/assets/'):
        raise HTTPException(409,'Result does not reference a platform asset')
    relative=parsed.path.removeprefix('/assets/')
    if '\\' in relative or ':' in relative or '..' in relative.split('/'):
        raise HTTPException(409,'Invalid asset reference')
    p=(root/relative).resolve()
    if not p.is_relative_to(root.resolve()):raise HTTPException(409,'Asset is outside the store')
    return p

def sanitize(value):
    """Only retain public result fields from heterogeneous legacy job records."""
    allowed={'status','phase','error','url','scene_url','robot_id','robot_ready',
             'cache_hit','cached','download_url','frames','fps','bytes','queue_position',
             'seconds','elapsed_seconds','body_ready','scene_ready','fov_ready','contact_ready',
             'sha256','format','robot_motion','revision','warnings','progress'}
    return {k:v for k,v in value.items() if k in allowed}

def public_job(row):
    result=json.loads(row['result'])
    if json.loads(row['request'])['workflow']=='export' and row['status']=='succeeded':
        result['download_url']=PREFIX+'/jobs/'+row['id']+'/download'
    return JobView(id=row['id'],workflow=json.loads(row['request'])['workflow'],status=row['status'],
                   created_at=row['created'],updated_at=row['updated'],backend_job_id=row['backend_id'],
                   result=result,error=json.loads(row['error']) if row['error'] else None,
                   cache_hit=bool(result.get('cache_hit') or result.get('cached')),
                   status_url=PREFIX+'/jobs/'+row['id'],events_url=PREFIX+'/jobs/'+row['id']+'/events')

def create_api(viewer, workbench, *, storage, database, operator_root=None, bridge=None):
    storage=Path(storage).resolve()
    registry=Store(database)
    services=bridge or PlatformServices(viewer,workbench)
    app=FastAPI(title='ArenaLabs Platform API',version=__version__,
                description='Shared platform operations for the website and external clients. Native SMPL-H, existing shared refinement and numerical policies are unchanged. Recalibration UI and live physics rollout are not part of this release.')
    app.state.registry=registry
    app.state.services=services
    app.state.bridge=services  # Compatibility for callers inspecting bound application instances.

    @app.middleware('http')
    async def boundary(request, call_next):
        if request.method not in {'GET','HEAD','OPTIONS'}:
            origin=request.headers.get('origin')
            # The established nginx rewrites the origin and host to loopback.
            trusted_proxy = request.client and request.client.host in {'127.0.0.1','::1'} and origin in {'http://localhost:8765','http://127.0.0.1:8765'}
            if origin and not trusted_proxy and urlsplit(origin).netloc != request.headers.get('host'):
                return JSONResponse({'type':'about:blank','title':'Cross-origin request denied','status':403},status_code=403,media_type='application/problem+json')
            # Compatibility session creation is the only mutation without a workspace.
            if request.url.path.rstrip('/').split('/')[-1] != 'sessions':
                if not request.cookies.get(COOKIE):
                    return JSONResponse({'type':'about:blank','title':'Workspace required','status':401,'detail':'POST /api/v1/sessions first'},status_code=401,media_type='application/problem+json')
        response=await call_next(request)
        response.headers.setdefault('Cache-Control','private, no-store')
        response.headers['X-Arena-API-Version']=__version__
        return response

    @app.exception_handler(HTTPException)
    async def http_error(request,error):
        return JSONResponse({'type':'about:blank','title':'Request rejected','status':error.status_code,
                             'detail':error.detail,'trace_id':uuid.uuid4().hex},status_code=error.status_code,
                            headers=error.headers,media_type='application/problem+json')

    @app.exception_handler(RequestValidationError)
    async def invalid(request,error):
        errors=[{'loc':e['loc'],'type':e['type'],'msg':e['msg']} for e in error.errors()]
        return JSONResponse({'type':'about:blank','title':'Invalid input','status':422,'field_errors':errors},status_code=422,media_type='application/problem+json')

    @app.exception_handler(Conflict)
    async def conflict(request,error):
        return JSONResponse({'type':'about:blank','title':'Conflict','status':409,'detail':str(error)},status_code=409,media_type='application/problem+json')

    @app.get('/health',tags=['platform'])
    def health():
        return {'status':'ok','api_version':__version__,'scheduler':'existing platform workers',
                'job_registry':'sqlite-wal','calibration_workflow':'deferred'}

    @app.get('/capabilities',tags=['platform'])
    def capabilities():
        return {'api_version':__version__,'workflows':OPERATIONS,'body_model':'native-smplh',
                'fps':30,'foot_objective':'soft guide; reasonable grounding; residual sliding accepted',
                'recalibration_ui':False,'live_physics_rollout':False,
                'independent_stage_execution':False,
                'stage_access':'Existing numerical stages execute inside the established workflow. Result modalities are separately readable.',
                'cancellation':'Not available in the current schedulers; API explicitly rejects it.',
                'resource_scope':'Workspace-owned recordings/jobs/exports; robot and hand registries retain current shared scope.',
                'recovery':'Persisted API identity. Legacy worker restart may interrupt jobs; no blind resubmission.',
                'website_api':'v1',
                'internal_dispatch':'direct application service bindings',
                'operator_replay':bool(operator_root and (Path(operator_root)/'library/catalog.json').is_file())}

    # Direct aliases keep existing streaming and typed validation. No new calibration flow.
    for old,new,method,tag in [
        ('/library','/sessions','POST','workspace'),
        ('/uploads/template','/recordings/template','GET','recordings'),
        ('/uploads','/recordings/import','POST','recordings'),
        ('/robots','/robots','GET','robots'),
        ('/hands','/end-effectors','GET','robots'),
        ('/hands','/end-effectors','POST','robots'),
        ('/trajectory-height/{scene_id}','/recordings/{scene_id}/height-preview','GET','existing-height'),
        ('/trajectory-height','/height-adjustments','POST','existing-height'),
        ('/geometry/frame','/geometry/frame','GET','scene'),
    ]:
        # Session creation reuses the read-library handler that issues the cookie.
        alias(app,workbench,old,new,'GET' if new=='/sessions' else method,tag)
        if new=='/sessions':app.routes[-1].methods={'POST'}
    for old,new,method,tag in [
        ('/api/catalog/{page}','/catalogs/{page}','GET','catalogs'),
        ('/api/source-video/{clip_id}','/recordings/{clip_id}/source-video','GET','recordings'),
        ('/api/robots/{page}','/robots/import/{page}','POST','robots'),
        ('/api/robots/{robot_id}/metadata','/robots/{robot_id}/metadata','PATCH','robots'),
        ('/api/robots/{robot_id}','/robots/{robot_id}','DELETE','robots'),
    ]:alias(app,viewer,old,new,method,tag)

    async def recording_rows(request,workflow):
        return await services.recordings(request,workflow)

    @app.get('/recordings',tags=['recordings'])
    async def recordings(request:Request,workflow:str=Query('manipulation',pattern='^(manipulation|locomotion)$'),
                         limit:int=Query(50,ge=1,le=500),offset:int=Query(0,ge=0),q:str=''):
        rows=await recording_rows(request,workflow)
        if q:rows=[r for r in rows if q.casefold() in (r.get('title','')+' '+r['id']).casefold()]
        return {'items':rows[offset:offset+limit],'total':len(rows),'offset':offset,'limit':limit}

    async def check(body,request):
        owner(request)
        rows=await recording_rows(request,'locomotion' if body.workflow=='locomotion' else 'manipulation')
        source=next((r for r in rows if r['id']==body.recording_id),None)
        # Height variants are authorized by the existing generation endpoint.
        if source is None and not (body.workflow!='locomotion' and body.recording_id.startswith('height-')):
            raise HTTPException(404,'Recording unavailable in this workspace')
        if body.workflow=='locomotion':
            robots=(await services.catalog(request,page='locomotion'))['robots']
            ready='conversion_ready'
        else:
            robots=await services.robots(request);ready='supported'
        robot=next((r for r in robots if r['id']==body.robot_id),None)
        if not robot:raise HTTPException(404,'Robot unavailable')
        if body.workflow!='export' and not robot.get(ready):raise HTTPException(409,'Robot metadata is not ready')
        if body.workflow=='manipulation':
            requirements=await services.input_requirements(body.recording_id,request)
            # Dataset labels cannot distinguish GRAIL six-pose recordings from
            # measured real-robot grippers. The queue owns those semantics.
            expected=(requirements or {}).get('hand_id',(source or {}).get('end_effector'))
            if expected is not None and body.hand_id!=expected:
                raise HTTPException(409,'Source requires hand_id='+expected)
            if requirements and requirements.get('requires_initial_height'):
                raise HTTPException(409,'Set the first trajectory height before generating UMI')
        return source,robot

    @app.post('/plans',response_model=PlanView,tags=['workflows'])
    async def plan(body:RunRequest,request:Request):
        source,robot=await check(body,request)
        return PlanView(executable=True,request=body,operation=body.workflow,stages=OPERATIONS[body.workflow]['stages'],
                        policy={'selection':'current deployed pipeline','refiner_signature':robot.get('refiner_signature'),
                                'body_model':'native-smplh','route':'resolved by current worker using source measurements and target topology'},
                        limitations=['Metadata preflight only; worker validates numerical inputs and route.',
                                     'No implicit standalone reconstruction, physics rollout or new calibration.',
                                     'Plans are previews; execution rechecks current inputs and records backend result.'])

    async def refresh(row,request):
        if row['status'] in TERMINAL:return row
        if not row['backend_id']:
            if time.time()-row['updated']>120:
                return registry.update(row['id'],status='outcome_unknown',error={'code':'DISPATCH_OUTCOME_UNKNOWN','detail':'Submission outcome was not recorded. Inspect backend before starting a new request.'})
            return row
        spec=json.loads(row['request']);op=OPERATIONS[spec['workflow']]
        try:raw=await services.status(spec['workflow'],row['backend_id'],request)
        except HTTPException as exc:
            if exc.status_code==404:
                return registry.update(row['id'],status='interrupted',error={'code':'BACKEND_JOB_UNAVAILABLE','detail':'Current scheduler lost or removed this job. No automatic resubmission.'})
            raise
        status={'completed':'succeeded','interrupted':'interrupted','failed':'failed','queued':'queued','running':'running'}.get(raw.get('status'),'running')
        error={'code':'BACKEND_FAILED','detail':raw.get('error','Current workflow did not complete')} if status in {'failed','interrupted'} else None
        return registry.update(row['id'],status=status,result=sanitize(raw),error=error)

    async def submit(body,request,key):
        who=owner(request)
        if not 1<=len(key)<=160:raise HTTPException(422,'Idempotency-Key must contain 1–160 characters')
        row,fresh=registry.create(who,key,body.model_dump())
        if not fresh:return public_job(await refresh(row,request))
        op=OPERATIONS[body.workflow]
        payload={'robot_id':body.robot_id,('clip_id' if body.workflow=='locomotion' else 'scene_id'):body.recording_id}
        if body.workflow=='manipulation':payload['hand_id']=body.hand_id
        try:
            await check(body,request)
            raw=await services.submit(body.workflow,request,payload)
        except HTTPException as exc:
            registry.update(row['id'],status='failed',error={'code':'SUBMISSION_REJECTED','http_status':exc.status_code,'detail':exc.detail})
            raise
        except Exception:
            import logging
            logging.exception('Platform dispatch outcome unknown: job=%s operation=%s',row['id'],body.workflow)
            registry.update(row['id'],status='outcome_unknown',error={'code':'DISPATCH_OUTCOME_UNKNOWN','detail':'Inspect backend before resubmitting'})
            raise HTTPException(502,'Submission outcome unknown; query this idempotency key again')
        result=sanitize(raw)
        status='succeeded' if raw.get('status')=='completed' else raw.get('status','queued')
        backend_id=raw.get('id')
        if status=='succeeded' and not backend_id:result['cache_hit']=True
        return public_job(registry.update(row['id'],status=status,result=result,backend_id=backend_id))

    @app.post('/runs',response_model=JobView,status_code=202,tags=['workflows'])
    async def run(body:RunRequest,request:Request,idempotency_key:str=Header(...,alias='Idempotency-Key')):
        return await submit(body,request,idempotency_key)

    @app.post('/batches',status_code=202,tags=['workflows'])
    async def batch(body:BatchRequest,request:Request,idempotency_key:str=Header(...,alias='Idempotency-Key')):
        if not 1<=len(idempotency_key)<=128:raise HTTPException(422,'Batch key must contain 1–128 characters')
        registry.claim_batch(owner(request),idempotency_key,body.model_dump())
        # Each item has a stable identity. A partial batch is safe to submit again.
        items=[]
        for index,item in enumerate(body.items):
            item_key='batch:'+hashlib.sha256((idempotency_key+':'+str(index)).encode()).hexdigest()
            try:items.append({'index':index,'job':(await submit(item,request,item_key)).model_dump()})
            except (HTTPException,Conflict) as exc:items.append({'index':index,'error':{'status':getattr(exc,'status_code',409),'detail':getattr(exc,'detail',str(exc))}})
        return {'items':items,'submission':'partial' if any('error' in r for r in items) else 'accepted'}

    def owned(ident,request):
        row=registry.get(ident,owner(request))
        if not row:raise HTTPException(404,'Job unavailable')
        return row

    @app.get('/jobs',tags=['jobs'])
    def jobs(request:Request,limit:int=Query(50,ge=1,le=200),offset:int=Query(0,ge=0)):
        return {'items':[public_job(r) for r in registry.list(owner(request),limit,offset)],'status_semantics':'Last observed status; GET individual job refreshes from its scheduler.'}

    @app.get('/jobs/{ident}',response_model=JobView,tags=['jobs'])
    async def job(ident:str,request:Request):return public_job(await refresh(owned(ident,request),request))

    @app.post('/jobs/{ident}/retries',response_model=JobView,status_code=202,tags=['jobs'])
    async def retry(ident:str,request:Request,idempotency_key:str=Header(...,alias='Idempotency-Key')):
        row=await refresh(owned(ident,request),request)
        if row['status'] not in {'failed','interrupted'}:raise HTTPException(409,'Only confirmed failed/interrupted jobs can be retried')
        if row['key']==idempotency_key:raise HTTPException(409,'Use a new key for a deliberate retry')
        return await submit(RunRequest.model_validate_json(row['request']),request,idempotency_key)

    @app.post('/jobs/{ident}/cancel',tags=['jobs'])
    async def cancel(ident:str,request:Request):
        owned(ident,request)
        raise HTTPException(409,'Existing schedulers do not support safe cancellation of shared stages. No process was killed.')

    @app.get('/jobs/{ident}/events',tags=['jobs'])
    async def events(ident:str,request:Request,after:int=Query(0,ge=0),last_event_id:str|None=Header(None)):
        row=owned(ident,request)
        if last_event_id:
            try:after=max(after,int(last_event_id))
            except ValueError:raise HTTPException(422,'Invalid Last-Event-ID')
        async def stream():
            cursor=after
            while not await request.is_disconnected():
                current=await refresh(registry.get(ident,row['owner']),request)
                records=registry.events(ident,cursor)
                for event in records:
                    cursor=event['seq']
                    yield 'id: '+str(cursor)+'\nevent: job.status\ndata: '+event['value']+'\n\n'
                if current['status'] in TERMINAL and len(records)<100:break
                yield ': heartbeat\n\n'
                await asyncio.sleep(2)
        return StreamingResponse(stream(),media_type='text/event-stream',headers={'X-Accel-Buffering':'no','Cache-Control':'no-store'})

    @app.get('/jobs/{ident}/download',tags=['exports'])
    async def download(ident:str,request:Request):
        row=await refresh(owned(ident,request),request)
        if json.loads(row['request'])['workflow']!='export' or row['status']!='succeeded':raise HTTPException(409,'Export is not ready')
        # The legacy download independently checks workspace ownership.
        return await services.download(row['backend_id'],request)

    async def result_manifest(ident,request):
        row=await refresh(owned(ident,request),request)
        result=json.loads(row['result']);url=result.get('scene_url') or result.get('url')
        if not url:raise HTTPException(409,'No playback result has been published yet')
        path=safe_asset(storage,url)
        if not path.is_file():raise HTTPException(409,'Published result is no longer available')
        manifest=await run_in_threadpool(lambda:json.loads(path.read_text()))
        return row,path,manifest

    @app.get('/jobs/{ident}/playback',tags=['results'])
    async def playback(ident:str,request:Request):
        row,path,manifest=await result_manifest(ident,request)
        spec=json.loads(row['request'])
        selected=next((r for r in manifest.get('robots',[]) if
                       r.get('id')==spec['robot_id'] or r.get('registry_robot_id')==spec['robot_id']),None)
        solver=(selected or {}).get('solver',{})
        validation={}
        if selected and selected.get('playback_url'):
            validation_path=safe_asset(storage,selected['playback_url']).with_name('validation.json')
            if validation_path.is_file():
                validation=await run_in_threadpool(lambda:json.loads(validation_path.read_text()))
        support=validation.get('solver',{}).get('contact_audit')
        assessment={'operation_completed':row['status']=='succeeded',
                    'solver_converged':solver.get('converged'),
                    'quality':(selected or {}).get('quality'),
                    'quality_status':manifest.get('quality_status'),
                    'routing':(selected or {}).get('routing'),
                    'metric_reference':(selected or {}).get('reference'),
                    'support':support,
                    'support_metric_source':'refiner report; not an independent physical measurement' if support else None,
                    'reconstruction_model':validation.get('human_model'),
                    'original_priors_unchanged':validation.get('original_priors_unchanged'),
                    'physics_validation':'not_run',
                    'acceptance':'not_evaluated',
                    'warnings':(['SOLVER_NOT_CONVERGED'] if solver.get('converged') is False else [])}
        return {'job_id':ident,'manifest':manifest,'manifest_url':'/assets/'+path.relative_to(storage).as_posix(),
                'selected_robot_id':spec['robot_id'],'assessment':assessment,
                'revision':hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
                'semantics':'Current published scene; may contain progressively generated modalities.',
                'physics_status':'not_run'}

    def public_artifact(row):
        return {k:row[k] for k in ('id','sha256','size','kind')} | {'download_url':PREFIX+'/artifacts/'+row['id']+'/content'}

    @app.get('/jobs/{ident}/artifacts',tags=['results'])
    async def artifacts(ident:str,request:Request):
        row,path,manifest=await result_manifest(ident,request)
        spec=json.loads(row['request']);files=[(path,'playback_manifest')]
        folder=path.parent/('robot-'+spec['robot_id'])
        for name,kind in [('playback.json','robot_motion'),('model.glb','robot_visual'),('validation.json','quality_report')]:
            candidate=folder/name
            if candidate.is_file():files.append((candidate,kind))
        # Public asset references only, never internal model or source filesystem paths.
        for robot in manifest.get('robots',[]):
            if robot.get('id')==spec['robot_id']:
                for key,value in robot.items():
                    if isinstance(value,str) and value.startswith('/assets/'):
                        candidate=safe_asset(storage,value)
                        if candidate.is_file():files.append((candidate,key))
        items=[];seen=set()
        for candidate,kind in files:
            if candidate in seen:continue
            seen.add(candidate)
            record=await run_in_threadpool(registry.artifact,row['owner'],candidate,kind)
            items.append(public_artifact(record))
        return {'items':items,'semantics':'Content-identified references. If a legacy path is overwritten, download returns 409 rather than different bytes.'}

    @app.get('/artifacts/{ident}',tags=['results'])
    def artifact(ident:str,request:Request):
        row=registry.get_artifact(ident,owner(request))
        if not row:raise HTTPException(404,'Artifact unavailable')
        return public_artifact(row)

    @app.get('/artifacts/{ident}/content',tags=['results'])
    async def content(ident:str,request:Request):
        row=registry.get_artifact(ident,owner(request))
        if not row:raise HTTPException(404,'Artifact unavailable')
        path=Path(row['path']).resolve()
        if not path.is_relative_to(storage) or not path.is_file():raise HTTPException(404,'Artifact unavailable')
        # Re-check legacy private markers on each read, not only at registration.
        for parent in [path.parent,*path.parents]:
            marker=parent/'.workspace.json'
            if marker.is_file() and json.loads(marker.read_text()).get('owner')!=owner(request):raise HTTPException(404,'Artifact unavailable')
            if parent==storage:break
        stat=path.stat()
        if (stat.st_size,stat.st_mtime_ns)!=(row['size'],row['mtime']):
            latest=await run_in_threadpool(registry.artifact,row['owner'],path,row['kind'])
            if latest['sha256']!=row['sha256']:raise HTTPException(409,'Artifact revision changed; obtain the current result')
        etag='"'+row['sha256']+'"'
        if request.headers.get('if-none-match')==etag:
            from fastapi import Response
            return Response(status_code=304,headers={'ETag':etag})
        return FileResponse(path,filename=path.name,headers={'ETag':etag,'Cache-Control':'private, max-age=300','Vary':'Cookie'})

    attach_media(app, storage)
    attach_operator(app, operator_root)

    return app
