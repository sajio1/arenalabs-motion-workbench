"""Canonical RGB-D directories are losslessly downloadable through artifact APIs."""
import hashlib
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_platform_api import fixtures, install_output_preflight, post, spec


def cloud_fixture(tmp_path, chunks=2):
    client, _, _, store, app = fixtures(tmp_path)
    install_output_preflight(app)
    folder = store / 'geometry'
    folder.mkdir()
    payloads = {'index.json': json.dumps(dict(schema='rgbd-geometry-v1', rgb='rgb.mp4',
        cameras='cameras.parquet', masks='masks.rle.jsonl', depth='depth.zarr/depth_mm',
        source_human_layer='depth.zarr/human_depth_mm')).encode(),
        'rgb.mp4': b'rgb', 'cameras.parquet': b'camera', 'masks.rle.jsonl': b'masks',
        'depth.zarr/.zgroup': b'{}', 'depth.zarr/depth_mm/.zarray': b'{}',
        'depth.zarr/human_depth_mm/.zarray': b'{}'}
    payloads.update({f'depth.zarr/depth_mm/{i}.0.0':str(i).encode() for i in range(chunks)})
    payloads['depth.zarr/human_depth_mm/0.0.0'] = b'human'
    for relative, value in payloads.items():
        target = folder / relative; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(value)
    (folder / 'unreferenced-private-model.xml').write_bytes(b'not a geometry dependency')
    historical = folder / 'depth.zarr/depth_mm/.artifact-revisions/previous'
    historical.mkdir(parents=True); (historical / 'chunk').write_bytes(b'old')
    (store / 'scene/scene.json').write_text(json.dumps(dict(id='clip', robots=[],
        cloud=dict(format='depth-rgb-v1', geometry_url='/assets/geometry'))))
    job = post(client, spec(recording_id='cached', output_modalities=['point_cloud'])).json()['id']
    return client, app, folder, job, payloads


def test_all_canonical_files_have_reconstructable_bundle_paths_and_immutable_downloads(tmp_path):
    client, app, folder, job, payloads = cloud_fixture(tmp_path)
    response = client.get('/jobs/'+job+'/artifacts')
    assert response.status_code == 200
    cloud = [item for item in response.json()['items'] if item['kind'] == 'point_cloud']
    assert {item['bundle_relative_path'] for item in cloud} == set(payloads)
    assert len({item['bundle_id'] for item in cloud}) == 1
    assert all(item['bundle_format'] == 'rgbd-geometry-v1' for item in cloud)
    rebuilt = tmp_path / 'downloaded'
    for item in cloud:
        relative = item['bundle_relative_path']
        content = client.get(item['download_url'].removeprefix('/api/v1'))
        assert content.status_code == 200 and content.content == payloads[relative]
        assert item['size'] == len(content.content)
        assert item['sha256'] == hashlib.sha256(content.content).hexdigest()
        target = rebuilt / relative; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(content.content)
    assert {p.relative_to(rebuilt).as_posix() for p in rebuilt.rglob('*') if p.is_file()} == set(payloads)
    original = next(item for item in cloud if item['bundle_relative_path'].endswith('/0.0.0'))
    url = original['download_url'].removeprefix('/api/v1')
    before = client.get(url)
    outsider = TestClient(app); outsider.cookies.set('arena_workspace','unrelated')
    assert outsider.get(url).status_code == 404
    assert client.get(url, headers={'If-None-Match':before.headers['etag']}).status_code == 304
    assert client.get(url, headers={'Range':'bytes=0-0'}).status_code == 206
    (folder / original['bundle_relative_path']).write_bytes(b'new-content')
    after = client.get('/jobs/'+job+'/artifacts').json()['items']
    changed = next(item for item in after if item.get('bundle_relative_path') == original['bundle_relative_path'])
    assert changed['id'] != original['id']
    assert client.get(url).content == before.content
    # Re-indexing never includes snapshots created by prior downloads/indexing.
    assert {item['bundle_relative_path'] for item in after if item['kind']=='point_cloud'} == set(payloads)


@pytest.mark.parametrize('failure', ['missing', 'index_escape', 'chunk_escape', 'linked_directory'])
def test_invalid_geometry_dependencies_fail_instead_of_publishing_partial_bundles(tmp_path, failure):
    client, _, folder, job, _ = cloud_fixture(tmp_path)
    if failure == 'missing': (folder / 'rgb.mp4').unlink()
    elif failure == 'index_escape':
        metadata = json.loads((folder / 'index.json').read_text()); metadata['rgb'] = '../outside.mp4'
        (folder / 'index.json').write_text(json.dumps(metadata))
    elif failure == 'chunk_escape':
        outside = tmp_path / 'private'; outside.write_bytes(b'private')
        (folder / 'depth.zarr/depth_mm/escaped').symlink_to(outside)
    else:
        (folder / 'depth.zarr/linked').symlink_to(tmp_path, target_is_directory=True)
    response = client.get('/jobs/'+job+'/artifacts')
    assert response.status_code == 409
    assert 'incomplete' in response.json()['detail']


def test_thousand_chunk_bundle_has_one_entry_per_source_file(tmp_path):
    client, _, _, job, payloads = cloud_fixture(tmp_path, chunks=1000)
    started = time.monotonic()
    response = client.get('/jobs/'+job+'/artifacts')
    elapsed = time.monotonic() - started
    assert response.status_code == 200
    cloud = [item for item in response.json()['items'] if item['kind'] == 'point_cloud']
    assert len(cloud) == len(payloads)
    print(json.dumps(dict(benchmark='1000 chunks / complete HTTP artifact indexing', files=len(cloud), elapsed_s=elapsed)))
