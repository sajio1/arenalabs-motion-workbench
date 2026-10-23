"""Small contract checks for API cutover; no inference or end-to-end acceptance."""
import hashlib
import json
from fastapi import FastAPI
from fastapi.testclient import TestClient
from arena_api.operator import attach_operator
from arena_api.media import attach_media
from test_platform_api import fixtures, post


def test_progress_revision_survives_shared_service(tmp_path):
    client, _, states, _, _ = fixtures(tmp_path)
    job = post(client).json()
    states[job['backend_job_id']] = {'status': 'running', 'robot_ready': True,
        'scene_url': '/assets/scene/scene.json', 'revision': 7, 'phase': 'FOV rendering', 'scene_ready': True}
    current = client.get('/jobs/' + job['id']).json()
    assert current['result']['revision'] == 7 and current['result']['robot_ready']
    assert current['result']['phase'] == 'FOV rendering'


def test_source_hand_semantics_come_from_queue_not_dataset_label(tmp_path):
    client, _, _, _, app = fixtures(tmp_path)
    app.state.services.workbench.state.input_requirements = lambda scene, request: {
        'hand_id': 'robotiq-2f-140', 'requires_initial_height': False}
    spec = {'workflow':'manipulation','recording_id':'clip','robot_id':'x2','hand_id':'robotiq-2f-140'}
    assert client.post('/plans',json=spec).status_code==200
    assert client.post('/plans',json={**spec,'hand_id':'none'}).status_code==409
    app.state.services.workbench.state.input_requirements = lambda scene, request: {
        'hand_id': 'robotiq-2f-140', 'requires_initial_height': True}
    response=client.post('/plans',json=spec)
    assert response.status_code==409 and 'height' in response.json()['detail']


def test_media_boundary_range_and_private_cache(tmp_path):
    (tmp_path/'public').mkdir(); (tmp_path/'public/motion.bin').write_bytes(b'123456789')
    (tmp_path/'private').mkdir(); (tmp_path/'private/motion.bin').write_bytes(b'secret')
    (tmp_path/'private/.workspace.json').write_text(json.dumps({'owner': hashlib.sha256(b'mine').hexdigest()}))
    app=FastAPI();attach_media(app,tmp_path);client=TestClient(app)
    response=client.get('/media/public/motion.bin',headers={'Range':'bytes=2-4'})
    assert response.status_code==206 and response.content==b'345'
    assert client.head('/media/public/motion.bin').status_code==200
    assert client.get('/media/private/motion.bin').status_code==404
    client.cookies.set('arena_workspace','mine')
    response=client.get('/media/private/motion.bin')
    assert response.content==b'secret' and 'no-store' in response.headers['cache-control']
    assert client.get('/media/private/.workspace.json').status_code==404
    assert client.get('/media/inputs/weights.json').status_code==404


def test_operator_cameras_gzip_revision_and_review(tmp_path):
    (tmp_path/'library').mkdir();(tmp_path/'cache').mkdir()
    (tmp_path/'cache/motion.bin.gz').write_bytes(b'\x1f\x8bdata')
    (tmp_path/'library/left.mp4').write_bytes(b'video')
    rows={'cases':[{'id':'one','motion':'cache/motion.bin.gz','mode':'physx'}]}
    (tmp_path/'library/catalog.json').write_text(json.dumps(rows))
    (tmp_path/'library/repair-candidates.json').write_text(json.dumps(rows))
    (tmp_path/'library/real-cameras.json').write_text(json.dumps({'views':{'one':{'head':None,'left':{'url':'library/left.mp4','offset':.125}}}}))
    app=FastAPI();attach_operator(app,tmp_path);client=TestClient(app)
    row=client.get('/simulation-episodes?collection=1').json()['items'][0]
    assert row['cameras']['left']['offset']==.125 and 'head' not in row['cameras']
    url=row['motion'].removeprefix('/api/v1')
    response=client.get(url)
    assert response.content==b'\x1f\x8bdata' and 'content-encoding' not in response.headers
    (tmp_path/'cache/motion.bin.gz').write_bytes(b'changed data')
    assert client.get(url).status_code==409
    assert client.get('/simulation-episodes?collection=../../secret').status_code==404
