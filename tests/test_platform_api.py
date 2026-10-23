import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pytest
from fastapi import FastAPI,Request,Response,HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from arena_api.app import create_api
from arena_api.store import Store,Conflict

class Generate(BaseModel):
    scene_id:str
    robot_id:str
    hand_id:str='none'
    trajectory_scale:str|None=None
    output_modalities:list[str]|None=None

def fixtures(tmp_path, database=None):
    viewer=FastAPI();workbench=FastAPI();calls=[];states={}
    storage=tmp_path/'assets';storage.mkdir(exist_ok=True)
    dest=storage/'scene';dest.mkdir(exist_ok=True)
    (dest/'scene.json').write_text(json.dumps({'id':'clip','robots':[{'id':'x2','playback_url':'/assets/scene/robot-x2/playback.json'}]}))
    robot=dest/'robot-x2';robot.mkdir(exist_ok=True)
    (robot/'playback.json').write_text('{"frames":3}')
    @workbench.get('/library')
    def library(request:Request,response:Response):
        if not request.cookies.get('arena_workspace'):response.set_cookie('arena_workspace','test-owner',httponly=True)
        return [{'id':'clip','title':'Clip','dataset':'Nymeria','end_effector':'none'},
                {'id':'cached','title':'Cached','dataset':'Nymeria','end_effector':'none'}]
    @workbench.get('/robots')
    def robots():return [{'id':'x2','supported':True,'refiner_signature':'unchanged'}]
    @workbench.post('/generate')
    def generate(body:Generate):
        calls.append(body.model_dump())
        if body.scene_id=='cached':return {'status':'completed','scene_url':'/assets/scene/scene.json'}
        ident='backend-'+str(len(calls));states[ident]={'id':ident,'status':'queued'}
        return states[ident]
    @workbench.get('/jobs/{ident}')
    def status(ident:str):
        if ident not in states:raise HTTPException(404,'Gone')
        return states[ident]
    @workbench.post('/exports')
    def export(body:Generate):
        calls.append(body.model_dump());states['export-1']={'id':'export-1','status':'queued'}
        return states['export-1']
    @workbench.get('/exports/{ident}')
    def export_status(ident:str):return states[ident]
    @viewer.get('/api/catalog/{page}')
    def catalog(page:str):return {'robots':[{'id':'x2','conversion_ready':True}], 'clips':[{'id':'clip','page':page}]}
    @viewer.post('/api/convert')
    def convert(body:dict):
        calls.append(body);return {'status':'completed','url':'/assets/scene/robot-x2/playback.json','cached':True}
    @viewer.get('/api/jobs/{ident}')
    def locomotion_status(ident:str):return states[ident]
    @workbench.get('/exports/{ident}/download')
    def export_download(ident:str):
        from fastapi.responses import Response
        return Response(b'PK\x03\x04',media_type='application/zip')
    async def stub(request:Request):return {'path':request.url.path}
    for path,methods in [('/uploads/template',['GET']),('/uploads',['POST']),('/hands',['GET','POST']),
                         ('/trajectory-height/{scene_id}',['GET']),('/trajectory-height',['POST']),('/geometry/frame',['GET'])]:
        workbench.add_api_route(path,stub,methods=methods)
    for path,methods in [('/api/source-video/{clip_id}',['GET']),('/api/robots/{page}',['POST']),
                         ('/api/robots/{robot_id}/metadata',['PATCH']),('/api/robots/{robot_id}',['DELETE'])]:
        viewer.add_api_route(path,stub,methods=methods)
    app=create_api(viewer,workbench,storage=storage,database=database or tmp_path/'api.sqlite')
    client=TestClient(app);client.cookies.set('arena_workspace','test-owner');client.post('/sessions')
    return client,calls,states,storage,app

def spec(**changes):return {'workflow':'manipulation','recording_id':'clip','robot_id':'x2','hand_id':'none',**changes}
def post(client,body=None,key='test'):return client.post('/runs',json=body or spec(),headers={'Idempotency-Key':key})

def test_idempotency_and_conflicting_inputs(tmp_path):
    c,calls,states,_,_=fixtures(tmp_path)
    first=post(c);again=post(c)
    assert first.status_code==202 and again.json()['id']==first.json()['id']
    assert len(calls)==1
    conflict=post(c,spec(recording_id='cached'))
    assert conflict.status_code==409
    assert conflict.headers['content-type'].startswith('application/problem+json')

def test_reused_completed_cache_has_job_identity(tmp_path):
    c,calls,_,_,_=fixtures(tmp_path)
    result=post(c,spec(recording_id='cached')).json()
    assert result['status']=='succeeded' and result['cache_hit'] and result['id']
    assert c.get('/jobs/'+result['id']).json()['id']==result['id']
    assert len(calls)==1

def test_explicit_hand_and_no_algorithm_overrides(tmp_path):
    c,calls,_,_,_=fixtures(tmp_path)
    body=spec();body.pop('hand_id')
    assert post(c,body).status_code==422
    assert post(c,spec(foot_weight=99)).status_code==422
    assert post(c,spec(hand_id='wuji')).status_code==409
    assert not calls

def test_workspace_origin_and_private_job(tmp_path):
    c,_,_,_,app=fixtures(tmp_path)
    ident=post(c).json()['id']
    outsider=TestClient(app)
    assert post(outsider).status_code==401
    outsider.cookies.set('arena_workspace','other')
    assert outsider.get('/jobs/'+ident).status_code==404
    assert c.post('/runs',json=spec(),headers={'Idempotency-Key':'bad','Origin':'https://evil.example'}).status_code==403

def test_status_events_and_loss_of_legacy_scheduler(tmp_path):
    c,_,states,_,_=fixtures(tmp_path)
    result=post(c).json();bid=result['backend_job_id'];ident=result['id']
    states[bid]={'status':'completed','scene_url':'/assets/scene/scene.json','native_work_dir':'/private/model'}
    value=c.get('/jobs/'+ident).json()
    assert value['status']=='succeeded' and value['physics_status']=='not_run'
    assert 'native_work_dir' not in value['result']
    events=c.get('/jobs/'+ident+'/events')
    assert 'event: job.status' in events.text and 'succeeded' in events.text
    last=int([line[4:] for line in events.text.splitlines() if line.startswith('id: ')][-1])
    assert c.get('/jobs/'+ident+'/events',headers={'Last-Event-ID':str(last)}).text==''
    second=post(c,key='another').json();del states[second['backend_job_id']]
    assert c.get('/jobs/'+second['id']).json()['status']=='interrupted'

def test_retry_and_no_fake_cancel(tmp_path):
    c,calls,states,_,_=fixtures(tmp_path)
    first=post(c).json()
    assert c.post('/jobs/'+first['id']+'/cancel').status_code==409
    assert c.post('/jobs/'+first['id']+'/retries',headers={'Idempotency-Key':'new'}).status_code==409
    states[first['backend_job_id']]={'status':'failed','error':'Numerical failure'}
    retry=c.post('/jobs/'+first['id']+'/retries',headers={'Idempotency-Key':'retry'})
    assert retry.status_code==202 and retry.json()['id']!=first['id'] and len(calls)==2

def test_persistence_across_api_restart(tmp_path):
    c,calls,_,_,_=fixtures(tmp_path)
    first=post(c,spec(recording_id='cached')).json()
    c2,calls2,_,_,_=fixtures(tmp_path)
    assert post(c2,spec(recording_id='cached')).json()['id']==first['id']
    assert not calls2

def test_artifact_integrity_access_and_conditional_download(tmp_path):
    c,_,_,storage,app=fixtures(tmp_path)
    ident=post(c,spec(recording_id='cached')).json()['id']
    assert c.get('/jobs/'+ident+'/playback').json()['manifest']['id']=='clip'
    items=c.get('/jobs/'+ident+'/artifacts').json()['items']
    motion=next(r for r in items if r['kind']=='robot_motion')
    url='/artifacts/'+motion['id']+'/content'
    first=c.get(url);assert first.status_code==200
    assert c.get(url,headers={'If-None-Match':first.headers['etag']}).status_code==304
    assert c.get(url,headers={'Range':'bytes=0-4'}).status_code==206
    outsider=TestClient(app);outsider.cookies.set('arena_workspace','other')
    assert outsider.get(url).status_code==404
    (storage/'scene/robot-x2/playback.json').write_text('{"frames":456}')
    assert c.get(url).content==first.content
    updated=c.get('/jobs/'+ident+'/artifacts').json()['items']
    next_motion=next(r for r in updated if r['kind']=='robot_motion')
    assert next_motion['id']!=motion['id']
    assert c.get('/artifacts/'+next_motion['id']+'/content').json()=={'frames':456}
    assert c.get(url).content==first.content

def test_contact_subset_keeps_existing_download_bytes(tmp_path):
    c,_,_,storage,app=fixtures(tmp_path)
    install_output_preflight(app)
    scene=storage/'scene/scene.json'
    manifest=json.loads(scene.read_text())
    manifest['tactile']={'url':'/assets/scene/video-contact.json'}
    scene.write_text(json.dumps(manifest))
    contact=storage/'scene/video-contact.json'
    data={'likelihood':[[[None]*5,[None]*5]],'quality_status':'unreliable_constant_digit_scores'}
    contact.write_text(json.dumps(data,indent=2))
    all_job=post(c,spec(recording_id='cached',output_modalities=['trajectory','point_cloud','contact','inpaint_fov']),key='all').json()['id']
    all_items=c.get('/jobs/'+all_job+'/artifacts').json()['items']
    old=next(r for r in all_items if r['kind']=='contact')
    old_url='/artifacts/'+old['id']+'/content'
    original=c.get(old_url)
    # Contact-only may reformat a shared JSON without changing its values.
    contact.write_text(json.dumps(data,separators=(',',':')))
    subset=post(c,spec(recording_id='cached',output_modalities=['contact']),key='contact').json()['id']
    new_items=c.get('/jobs/'+subset+'/artifacts').json()['items']
    current=next(r for r in new_items if r['kind']=='contact')
    assert current['id']!=old['id']
    assert not any(r['kind']=='robot_motion' for r in new_items)
    assert c.get(old_url).status_code==200
    assert c.get(old_url).content==original.content
    assert c.get('/artifacts/'+current['id']+'/content').json()==data
    assert c.get(old_url,headers={'If-None-Match':original.headers['etag']}).status_code==304

def test_existing_legacy_download_is_frozen_on_first_read(tmp_path):
    import hashlib
    c,_,_,storage,app=fixtures(tmp_path)
    source=storage/'scene/robot-x2/playback.json'
    original=source.read_bytes();sha=hashlib.sha256(original).hexdigest()
    owner=hashlib.sha256(c.cookies.get('arena_workspace').encode()).hexdigest()
    ident=hashlib.sha256((owner+'\0'+str(source)+'\0'+sha).encode()).hexdigest()
    stat=source.stat()
    with app.state.registry.connect() as db:
        db.execute('INSERT INTO artifacts VALUES(?,?,?,?,?,?,?)',
                   (ident,owner,str(source),sha,stat.st_size,stat.st_mtime_ns,'robot_motion'))
    url='/artifacts/'+ident+'/content'
    assert c.get(url).content==original
    assert app.state.registry.get_artifact(ident,owner)['path']!=str(source)
    source.unlink()
    assert c.get(url).content==original

def test_aliases_and_openapi_contract(tmp_path):
    c,_,_,_,_=fixtures(tmp_path)
    schema=c.get('/openapi.json').json()
    assert 'post' in schema['paths']['/sessions']
    assert '/recordings/import' in schema['paths'] and '/robots/{robot_id}/metadata' in schema['paths']
    assert schema['paths']['/runs']['post']['requestBody']['content']['application/json']['schema']['$ref'].endswith('/RunRequest')
    assert c.get('/capabilities').json()['independent_stage_execution'] is False
    assert c.get('/simulation-episodes').status_code==503

def test_batch_retry_does_not_duplicate_items(tmp_path):
    c,calls,_,_,_=fixtures(tmp_path)
    body={'items':[spec(),spec(recording_id='cached')]};headers={'Idempotency-Key':'batch'}
    first=c.post('/batches',json=body,headers=headers)
    second=c.post('/batches',json=body,headers=headers)
    assert first.status_code==202 and len(calls)==2
    assert [r['job']['id'] for r in first.json()['items']]==[r['job']['id'] for r in second.json()['items']]
    assert c.post('/batches',json={'items':[spec()]},headers=headers).status_code==409

def test_nginx_loopback_origin_and_mounted_routes(tmp_path):
    c,_,_,_,app=fixtures(tmp_path)
    root=FastAPI();root.mount('/api/v1',app)
    client=TestClient(root,base_url='http://localhost',client=('127.0.0.1',9001))
    response=client.post('/api/v1/sessions',headers={'Origin':'http://localhost:8765'})
    assert response.status_code==200 and client.cookies.get('arena_workspace')
    assert client.get('/api/v1/openapi.json').status_code==200

def test_operator_manifest_and_asset_bounds(tmp_path):
    c,_,_,_,app=fixtures(tmp_path)
    # Reuse the fake backend apps via bridge; Operator remains static/read-only.
    root=tmp_path/'operator';(root/'library').mkdir(parents=True)
    (root/'library/movie.bin').write_bytes(b'0123456789')
    (root/'library/catalog.json').write_text(json.dumps({'cases':[
        {'id':'ok','motion':'library/movie.bin'}, {'id':'bad','motion':'../secret'}]}))
    api=create_api(app.state.bridge.viewer,app.state.bridge.workbench,storage=tmp_path/'assets',database=tmp_path/'operator.sqlite',operator_root=root)
    client=TestClient(api)
    assert client.get('/simulation-episodes').json()['total']==2
    assert client.get('/simulation-episodes/ok/playback').json()['mode']=='recorded_replay'
    assert client.get('/simulation-episodes/ok/assets/motion').content==b'0123456789'
    assert client.get('/simulation-episodes/bad/assets/motion').status_code==409

def test_unknown_dispatch_is_not_resubmitted(tmp_path):
    c,calls,_,_,app=fixtures(tmp_path)
    import hashlib
    row,_=app.state.registry.create(hashlib.sha256(b'test-owner').hexdigest(),'lost',spec())
    with app.state.registry.connect() as db:db.execute('UPDATE jobs SET updated=0 WHERE id=?',(row['id'],))
    response=post(c,key='lost')
    assert response.json()['status']=='outcome_unknown' and not calls

def test_terminal_status_cannot_regress_from_delayed_poll(tmp_path):
    store=Store(tmp_path/'terminal.sqlite');row,_=store.create('owner','key',spec())
    store.update(row['id'],status='succeeded',result={'url':'/assets/a'})
    assert store.update(row['id'],status='running')['status']=='succeeded'

def test_store_concurrent_claim_is_atomic(tmp_path):
    store=Store(tmp_path/'concurrent.sqlite')
    with ThreadPoolExecutor(max_workers=8) as pool:
        results=list(pool.map(lambda _:store.create('owner','key',spec()),range(16)))
    assert sum(fresh for _,fresh in results)==1
    assert len({row['id'] for row,_ in results})==1
    with pytest.raises(Conflict):store.create('owner','key',spec(robot_id='other'))

def test_locomotion_and_export_use_same_job_envelope(tmp_path):
    c,_,states,_,_=fixtures(tmp_path)
    loco=post(c,{'workflow':'locomotion','recording_id':'clip','robot_id':'x2'},key='loco').json()
    assert loco['workflow']=='locomotion' and loco['status']=='succeeded'
    export=post(c,{'workflow':'export','recording_id':'clip','robot_id':'x2'},key='export').json()
    states[export['backend_job_id']]={'status':'completed','download_url':'/api/4d/exports/export-1/download'}
    result=c.get('/jobs/'+export['id']+'/download',follow_redirects=False)
    assert result.status_code==200 and result.content==b'PK\x03\x04'


def test_export_readiness_receipt_reaches_public_api(tmp_path):
    c,_,states,_,_=fixtures(tmp_path)
    job=post(c,dict(workflow='export',recording_id='clip',robot_id='x2')).json()
    receipt=dict(schema='arenalabs-dataset-readiness-v1',ready=True,
                 scope='complete_kinematic_reference_data',frames=3,
                 dynamics_validated=False,controller_validated=False)
    states[job['backend_job_id']]=dict(status='completed',ready=True,readiness=receipt,
                                     path='/private/dataset.zip',format='native-reference')
    public=c.get('/jobs/'+job['id']).json()
    assert public['status']=='succeeded'
    assert public['result']['ready'] is True
    assert public['result']['readiness']==receipt
    assert 'path' not in public['result']
    assert public['physics_status']=='not_run'


def install_preflight(app, *, available=True, mismatch=True):
    seen=[]
    def preflight(scene_id,request,robot_id=None,workflow='manipulation'):
        seen.append((scene_id,robot_id,workflow))
        return {'hand_id':'none','source_policy':'provided-fullbody',
                'source_body_family':'smplx',
                'source_policy_version':'source-v1','motion_policy_version':'hybrid-v1',
                'scale_choice_required':mismatch,
                'morphology':{'available':available,'source_height_m':1.8,'target_height_m':1.2,
                              'scale_factor':2/3,'large_mismatch':mismatch,'threshold':.2,
                              'private_model_path':'/never/expose/model'}}
    app.state.services.workbench.state.input_requirements=preflight
    app.state.services.viewer.state.motion_preflight=preflight
    return seen


def test_size_preflight_requires_explicit_choice_without_dispatch(tmp_path):
    c,calls,_,_,app=fixtures(tmp_path);seen=install_preflight(app)
    result=c.post('/plans',json=spec())
    assert result.status_code==200
    plan=result.json()
    assert plan['executable'] is False and plan['required_choice']=='trajectory_scale'
    assert plan['choices']==['preserve','match_robot']
    assert plan['policy']['source_policy']=='provided-fullbody'
    assert plan['policy']['body_model']=='provided'
    assert plan['policy']['generated_body_model'] is None
    assert plan['policy']['source_body_family']=='smplx'
    assert 'private_model_path' not in plan['policy']['morphology']
    rejected=post(c)
    assert rejected.status_code==409 and rejected.json()['detail']['code']=='SCALE_CHOICE_REQUIRED'
    assert not calls and seen==[('clip','x2','manipulation')]*2


@pytest.mark.parametrize('mode',['preserve','match_robot'])
@pytest.mark.parametrize('workflow',['manipulation','locomotion'])
def test_explicit_scale_reaches_both_workers_and_job_results(tmp_path,mode,workflow):
    c,calls,states,_,app=fixtures(tmp_path);seen=install_preflight(app)
    body=spec(workflow=workflow,trajectory_scale=mode)
    if workflow=='locomotion':body.pop('hand_id')
    plan=c.post('/plans',json=body).json();assert plan['executable']
    result=post(c,body).json()
    assert calls[-1]['trajectory_scale']==mode and result['trajectory_scale']==mode
    assert result['result']['source_policy_version']=='source-v1'
    assert seen[-1]==('clip','x2',workflow)
    if workflow=='manipulation':
        states[result['backend_job_id']]={'status':'completed','scene_url':'/assets/scene/scene.json'}
        result=c.get('/jobs/'+result['id']).json()
        assert result['result']['source_policy_version']=='source-v1' and result['trajectory_scale']==mode


def test_scale_choice_is_part_of_idempotency_identity(tmp_path):
    c,calls,_,_,app=fixtures(tmp_path);install_preflight(app)
    assert post(c,spec(trajectory_scale='preserve')).status_code==202
    assert post(c,spec(trajectory_scale='match_robot')).status_code==409
    assert len(calls)==1
    assert post(c,spec(trajectory_scale='match_robot'),key='other-scale').status_code==202
    assert len(calls)==2


def test_unknown_size_never_auto_scales_and_cannot_claim_match(tmp_path):
    c,calls,_,_,app=fixtures(tmp_path);install_preflight(app,available=False,mismatch=False)
    plan=c.post('/plans',json=spec()).json()
    assert plan['executable'] and not plan['policy']['morphology']['available']
    assert plan['policy']['trajectory_scale']=='preserve'
    assert post(c).status_code==202 and calls[0]['trajectory_scale']=='preserve'
    plan=c.post('/plans',json=spec(trajectory_scale='match_robot')).json()
    assert not plan['executable'] and plan['policy']['scale_estimate_unavailable']
    rejected=post(c,spec(trajectory_scale='match_robot'),key='match')
    assert rejected.status_code==409 and rejected.json()['detail']['code']=='SCALE_ESTIMATE_UNAVAILABLE'
    assert len(calls)==1


def test_old_preflight_signature_and_invalid_overrides(tmp_path):
    c,calls,_,_,app=fixtures(tmp_path)
    app.state.services.workbench.state.input_requirements=lambda scene_id,request:{'hand_id':'none'}
    assert post(c).status_code==202 and calls[0]['trajectory_scale']=='preserve'
    assert post(c,spec(trajectory_scale='automatic'),key='invalid').status_code==422
    assert post(c,spec(hand_tolerance_mm=8),key='tolerance').status_code==422
    assert post(c,spec(source_mode='reconstruct'),key='source').status_code==422


def test_export_keeps_saved_scale_selection_without_new_generation_preflight(tmp_path):
    c,calls,_,_,app=fixtures(tmp_path);install_preflight(app,available=False,mismatch=True)
    body={'workflow':'export','recording_id':'clip','robot_id':'x2','trajectory_scale':'match_robot'}
    assert c.post('/plans',json=body).json()['executable']
    assert post(c,body).status_code==202 and calls[-1]['trajectory_scale']=='match_robot'


def install_output_preflight(app, available=('trajectory','point_cloud','contact','inpaint_fov')):
    def requirements(scene_id,request,robot_id=None,workflow='manipulation'):
        from arenalab_workbench.output_modalities import OUTPUTS
        return {'hand_id':'none','output_capabilities':{'available_outputs':list(available),
            'outputs':{name:{'available':name in available,'reason':'Registered source' if name in available else 'No registered source RGB/hand stream'} for name in OUTPUTS}}}
    app.state.services.workbench.state.input_requirements=requirements
    return requirements


@pytest.mark.parametrize('selection',[['trajectory'],['point_cloud'],['contact'],['inpaint_fov'],['point_cloud','contact'],['inpaint_fov','trajectory']])
def test_output_selection_dispatch_identity_and_progress_retention(tmp_path,selection):
    c,calls,states,_,app=fixtures(tmp_path);install_output_preflight(app)
    body=spec(output_modalities=selection)
    plan=c.post('/plans',json=body).json()
    assert plan['executable'] and set(plan['outputs']['effective'])==set(selection)
    result=post(c,body).json();backend=result['backend_job_id']
    assert set(calls[-1]['output_modalities'])==set(selection)
    assert result['output_modalities']==plan['outputs']['effective']
    # Same logical set, different ordering, same idempotency identity.
    assert post(c,spec(output_modalities=list(reversed(selection)))).json()['id']==result['id']
    states[backend]={'status':'running','scene_ready':True,'scene_url':'/assets/scene/scene.json','completed_outputs':['point_cloud']}
    partial=c.get('/jobs/'+result['id']).json()
    assert partial['result']['effective_outputs']==plan['outputs']['effective']
    assert partial['result']['dependency_plan']==plan['outputs']['dependencies']
    states[backend]={'status':'failed','error':'Later stage failed'}
    failed=c.get('/jobs/'+result['id']).json()
    assert failed['result']['scene_ready'] and failed['result']['scene_url']=='/assets/scene/scene.json'
    assert failed['status']=='failed' and failed['result']['completed_outputs']==['point_cloud']
    assert c.get('/jobs/'+result['id']+'/playback').status_code==200


def test_outputs_are_deliverables_not_dependency_expansion(tmp_path):
    c,calls,_,_,app=fixtures(tmp_path);install_output_preflight(app)
    fov=c.post('/plans',json=spec(output_modalities=['inpaint_fov'])).json()['outputs']
    assert fov['effective']==['inpaint_fov'] and fov['dependencies']['retarget'] and fov['dependencies']['scene']
    contact=c.post('/plans',json=spec(output_modalities=['contact'])).json()['outputs']
    assert contact['dependencies']['contact'] and not contact['dependencies']['body'] and not contact['dependencies']['retarget']
    assert not contact['dependencies']['scene'] and not contact['dependencies']['proximity']
    assert not calls


def test_unavailable_outputs_and_invalid_selections_do_not_dispatch(tmp_path):
    c,calls,_,_,app=fixtures(tmp_path);install_output_preflight(app,('trajectory',))
    plan=c.post('/plans',json=spec(output_modalities=['inpaint_fov'])).json()
    assert not plan['executable'] and plan['outputs']['unavailable']==['inpaint_fov']
    reject=post(c,spec(output_modalities=['inpaint_fov']))
    assert reject.status_code==409 and reject.json()['detail']['code']=='OUTPUT_MODALITY_UNAVAILABLE'
    for value in [[],['trajectory','trajectory'],['unknown']]:
        assert post(c,spec(output_modalities=value)).status_code==422
    assert not calls
    default=post(c,key='default').json()
    assert default['result']['requested_outputs'] is None and default['output_modalities']==['trajectory']
    assert calls[-1]['output_modalities'] is None


def test_output_variants_have_distinct_identity_and_export_keeps_variant(tmp_path):
    c,calls,_,_,app=fixtures(tmp_path);install_output_preflight(app)
    one=post(c,spec(output_modalities=['trajectory']))
    assert one.status_code==202
    assert post(c,spec(output_modalities=['point_cloud'])).status_code==409
    exported=post(c,dict(workflow='export',recording_id='clip',robot_id='x2',output_modalities=['point_cloud','trajectory']),key='export-output').json()
    assert exported['workflow']=='export' and calls[-1]['output_modalities']==['trajectory','point_cloud']
    plan=c.post('/plans',json=dict(workflow='export',recording_id='clip',robot_id='x2',output_modalities=['trajectory'])).json()
    assert plan['outputs']['dependencies']=={} and 'existing generation variant' in plan['outputs']['semantics']


def test_artifacts_follow_public_urls_and_hide_internal_trajectory(tmp_path):
    c,_,states,store,app=fixtures(tmp_path);install_output_preflight(app)
    internal=store/'private-dependency';internal.mkdir();(internal/'playback.json').write_text('{}');(internal/'validation.json').write_text('{}')
    cloud=store/'points.bin';cloud.write_bytes(b'cloud')
    manifest={'id':'clip','robots':[{'id':'x2','playback_url':'/assets/private-dependency/playback.json'}],
              'cloud':{'url':'/assets/points.bin'}}
    (store/'scene/scene.json').write_text(json.dumps(manifest))
    for outputs in [['point_cloud'],['trajectory']]:
        job=post(c,spec(output_modalities=outputs),key=outputs[0]).json();states[job['backend_job_id']]={'status':'completed','scene_url':'/assets/scene/scene.json'}
        artifacts=c.get('/jobs/'+job['id']+'/artifacts').json()['items'];kinds={a['kind'] for a in artifacts}
        if outputs==['point_cloud']:assert kinds=={'playback_manifest','point_cloud'}
        else:
            assert kinds=={'playback_manifest','robot_motion','quality_report'}
            motion=next(a for a in artifacts if a['kind']=='robot_motion')
            assert c.get(motion['download_url'].removeprefix('/api/v1')).content==b'{}'


def test_session_is_lightweight_and_preserves_workspace(tmp_path):
    c,_,_,_,app=fixtures(tmp_path)
    c.cookies.clear()
    async def no_catalog(*args):raise AssertionError('Session must not read the catalogue')
    app.state.services.recordings=no_catalog
    response=c.post('/sessions')
    assert response.json()=={'ready':True}
    token=c.cookies.get('arena_workspace');assert token and len(token)>30
    assert c.post('/sessions').json()=={'ready':True}
    assert c.cookies.get('arena_workspace')==token


def test_recording_pagination_search_and_individual_lookup(tmp_path):
    c,_,_,_,app=fixtures(tmp_path)
    from types import SimpleNamespace
    async def rows(request,workflow):
        private=request.cookies.get('arena_workspace')=='test-owner'
        return [dict(id=f'r{i}',title=f'Task {i}',family='pick' if i%2 else 'place',dataset='HiPhi' if i%2 else 'GRAIL') for i in range(1100 if private else 1)]
    app.state.services.recordings=rows
    page=c.get('/recordings',params={'dataset':'HiPhi','q':'pick','limit':50,'offset':500}).json()
    assert page['total']==550 and len(page['items'])==50 and page['items'][0]['id']=='r1001'
    assert page['datasets']=={'HiPhi':550,'GRAIL':550}
    assert c.get('/recordings/r1099').json()['id']=='r1099'
    c.cookies.clear()
    assert c.get('/recordings/r1099').status_code==404
    assert c.get('/recordings',params={'dataset':'HiPhi'}).json()['total']==0
