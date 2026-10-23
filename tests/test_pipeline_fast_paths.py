import gzip
import hashlib
import json
from pathlib import Path
import os
import socket
import struct
import array
import threading
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_shared_models_preserve_bytes_and_private_permissions(tmp_path,monkeypatch):
    from arenalab_workbench import gpu_scheduler
    monkeypatch.setattr(gpu_scheduler,'ROOT',tmp_path/'scheduler')
    from arenalab_workbench.model_assets import publish_model
    from arena_api.media import attach_media
    raw=b'glTF'+b'original robot mesh\0'*10000
    public=publish_model(raw,tmp_path)
    assert publish_model(raw,tmp_path)==public
    owner=hashlib.sha256(b'customer').hexdigest()
    private=publish_model(raw,tmp_path,owner)
    assert public!=private
    app=FastAPI();attach_media(app,tmp_path);client=TestClient(app)
    url='/media/'+public.removeprefix('/assets/')
    response=client.get(url,headers={'accept-encoding':'gzip'})
    assert response.content==raw and response.headers['content-encoding']=='gzip'
    assert 'immutable' in response.headers['cache-control']
    assert client.get(url,headers={'accept-encoding':'gzip;q=0'}).headers.get('content-encoding') is None
    assert client.get(url,headers={'accept-encoding':'gzip','if-none-match':response.headers['etag']}).status_code==304
    protected='/media/'+private.removeprefix('/assets/')
    assert client.get(protected).status_code==404
    client.cookies.set('arena_workspace','customer')
    response=client.get(protected)
    assert response.content==raw and response.headers['cache-control']=='private, no-store'


def test_historical_scene_alias_does_not_modify_original_and_invalidates(tmp_path,monkeypatch):
    from arenalab_workbench import gpu_scheduler
    monkeypatch.setattr(gpu_scheduler,'ROOT',tmp_path/'scheduler')
    from arenalab_workbench.model_assets import prepare_existing,rewrite_scene_models
    folder=tmp_path/'manipulation4d/old';folder.mkdir(parents=True)
    model=folder/'model.glb';model.write_bytes(b'model')
    original={'robots':[{'model_url':'/assets/manipulation4d/old/model.glb'}]}
    scene=folder/'scene.json';scene.write_text(json.dumps(original));before=scene.read_bytes()
    assert prepare_existing(tmp_path)['models']==1
    rewritten=rewrite_scene_models(json.loads(scene.read_text()),tmp_path)
    assert '/robot-models/' in rewritten['robots'][0]['model_url']
    assert scene.read_bytes()==before
    model.write_bytes(b'new model')
    assert rewrite_scene_models(json.loads(scene.read_text()),tmp_path)==original


def test_geometry_cache_invalidation_includes_included_mesh_and_semantics(tmp_path):
    from arenalab_motion.asset_cache import robot_key
    mesh=tmp_path/'body.stl';mesh.write_bytes(b'old mesh')
    xml=tmp_path/'robot.xml';xml.write_text('<mujoco><asset><mesh file="body.stl"/></asset></mujoco>')
    row={'xml':str(xml),'semantic_metadata':{'root':'pelvis'}}
    old=robot_key(row);assert robot_key(row)==old
    mesh.write_bytes(b'new mesh content');assert robot_key(row)!=old
    assert robot_key(row)!=robot_key({**row,'semantic_metadata':{'root':'torso'}})


def test_worker_socket_transfers_and_releases_real_lease(tmp_path):
    from arenalab_workbench.motion_worker import exchange,recv_exact
    import fcntl
    address=tmp_path/'worker.sock';lock=tmp_path/'gpu-0.lock'
    errors=[]
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server,lock.open('w') as lease:
        server.bind(str(address));server.listen();fcntl.flock(lease,fcntl.LOCK_EX)
        def worker():
            try:
                connection,_=server.accept()
                with connection:
                    header,ancillary,_,_=connection.recvmsg(4,socket.CMSG_SPACE(4))
                    fds=array.array('i');fds.frombytes(ancillary[0][2]);received=fds[0]
                    payload=json.loads(recv_exact(connection,struct.unpack('!I',header)[0]))
                    assert payload=={'operation':'test'}
                    os.close(received)
                    body=b'{"ok":true}';connection.sendall(struct.pack('!I',len(body))+body)
            except Exception as error:errors.append(error)
        thread=threading.Thread(target=worker);thread.start()
        assert exchange(address,{'operation':'test'},lease.fileno())=={'ok':True}
        thread.join();assert not errors
        # Closing the transferred descriptor did not invalidate the caller's lease.
        with lock.open() as other:
            with pytest.raises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)


def test_scheduler_does_not_accept_external_process_as_resident(monkeypatch):
    from arenalab_workbench.gpu_scheduler import resident_only
    from types import SimpleNamespace
    import subprocess
    def run(command,**kw):
        return SimpleNamespace(stdout='GPU-ours\n' if '--query-gpu=uuid' in command else 'GPU-ours, 21\nGPU-ours, 99\n')
    monkeypatch.setattr(subprocess,'run',run)
    assert not resident_only(0,{'pid':21,'state':'idle'})
    assert not resident_only(0,{'pid':21,'state':'running'})


def test_scheduler_records_lease_lifecycle_on_an_empty_gpu(tmp_path,monkeypatch):
    from arenalab_workbench import gpu_scheduler as scheduler
    monkeypatch.setattr(scheduler,'ROOT',tmp_path)
    monkeypatch.setattr(scheduler,'memory',lambda:{0:(48000,1)})
    with scheduler.lease('test',env={'ARENA_GPU_POOL':'0'}) as env:
        assert env['CUDA_VISIBLE_DEVICES']=='0'
        assert os.fstat(int(env['ARENA_GPU_LOCK_FD'])).st_size==0
        record=json.loads(next((tmp_path/'events').glob('*.json')).read_text())
        assert record['state']=='running' and record['label']=='test'
    assert json.loads(next((tmp_path/'events').glob('*.json')).read_text())['state']=='completed'
