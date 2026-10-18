"""Recording boundary and clock tests; no model weights or GPU required."""
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'experiments/manipulation4d'))
from dataset_io import storage
from dataset_io.ingest import unpack, validate_normalized


@pytest.fixture
def package(tmp_path):
    root = tmp_path / 'package'
    root.mkdir()
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=gray:s=64x48:r=30', '-frames:v', '3', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(root/'original.mp4')], check=True)
    camera = np.repeat(np.eye(4)[None], 3, 0)
    camera[:, 2, 3] = 1.6
    np.savez(root/'observations.npz', timestamps_s=np.arange(3)/30, wrist_position=np.zeros((3, 2, 3)), camera_to_world=camera, ee_rotation=np.broadcast_to(np.eye(3), (3, 2, 3, 3)), intrinsic=np.array([[50,0,32],[0,50,24],[0,0,1]]))
    manifest = dict(schema='arenalabs.recording.v1', fps=30, units='m', world_up='z', end_effector='none', floor_z_m=0, image_size=[64,48], video='original.mp4')
    (root/'recording.json').write_text(json.dumps(manifest))
    return root


def test_valid_recording_and_reject_wrong_clock(package):
    manifest, arrays, video = validate_normalized(package)
    assert arrays['wrist_position'].shape == (3,2,3) and video.is_file()
    with np.load(package/'observations.npz') as source:
        data = dict(source)
    data['timestamps_s'] = [0,.02,.07]
    np.savez(package/'observations.npz', **data)
    with pytest.raises(ValueError, match='30 Hz'):
        validate_normalized(package)


def test_left_handed_camera_is_rejected(package):
    with np.load(package/'observations.npz') as source:
        data = dict(source)
    data['camera_to_world'][:,0,0] = -1
    np.savez(package/'observations.npz', **data)
    with pytest.raises(ValueError, match='right-handed'):
        validate_normalized(package)


@pytest.mark.parametrize('name', ['../escape', '/etc/passwd', 'C:/secret', 'folder/../../escape', '..\\escape'])
def test_archive_path_escape_is_rejected(tmp_path, name):
    archive = tmp_path/'input.zip'
    with zipfile.ZipFile(archive, 'w') as output:
        output.writestr(name, 'payload')
    with pytest.raises(ValueError):
        unpack(archive, tmp_path/'extracted')
    assert not (tmp_path/'escape').exists()


def test_workspace_isolation_and_upload(package, tmp_path, monkeypatch):
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    from fastapi.staticfiles import StaticFiles
    from dataset_io.routes import attach_dataset_routes, protect_uploaded_assets, require_scene
    import threading
    monkeypatch.setattr(storage, 'ROOT', tmp_path/'runtime')
    monkeypatch.setattr(storage, 'STORE', tmp_path/'store')
    storage.write(storage.STORE/'manipulation4d/catalog.json', [])
    app = FastAPI()
    attach_dataset_routes(app, threading.Lock())
    protect_uploaded_assets(app)
    app.mount('/assets', StaticFiles(directory=storage.STORE))
    @app.get('/check/{scene_id}')
    def check(scene_id:str, request:Request):
        require_scene(scene_id,request)
        return {'ok':True}
    own, other = TestClient(app), TestClient(app)
    assert own.get('/library').status_code == 200
    assert other.get('/library').json() == []
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as zipped:
        for file in package.iterdir():
            zipped.write(file, file.name)
    result = own.post('/uploads', files={'file':('recording.zip',archive.getvalue(),'application/zip')})
    assert result.status_code == 200, result.text
    row = result.json()
    assert len(own.get('/library').json()) == 1 and other.get('/library').json() == []
    assert own.get('/check/'+row['id']).status_code == 200
    assert other.get('/check/'+row['id']).status_code == 404
    assert own.get(row['video']['url']).status_code == 200
    assert other.get(row['video']['url']).status_code == 404
    assert own.get(row['video']['url']).headers['cache-control'] == 'private, no-store'
