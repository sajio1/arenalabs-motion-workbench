"""A registry migration cannot reuse or publish work for its previous source."""
import json
from types import SimpleNamespace

import pytest

from arenalab_workbench import source_identity
from tests.workbench.test_output_queue import queue


@pytest.mark.parametrize('change', ['motion', 'objects', 'observations', 'clock', 'spec', 'video'])
def test_source_revision_binds_input_bytes_and_metadata_without_decoding(tmp_path, monkeypatch, change):
    files = {}
    for name in ('motion', 'objects', 'observations', 'video'):
        files[name] = tmp_path/name
        files[name].write_bytes(name.encode())
    spec = dict(format='robot_joint_state', family='robot', robot_id='local-only',
                path=str(files['motion']), objects_path=str(files['objects']), evidence='simulation_fk')
    row = dict(source=str(files['motion']), observations_source=str(files['observations']),
               frames=299, fps=30, video=dict(path=str(files['video']), start_seconds=0))
    monkeypatch.setattr(source_identity, 'registered_source', lambda *a, **k: (row, spec))
    # No robot model is required to bind a queue to immutable source bytes.
    from arenalab_workbench import robot_motion_source
    monkeypatch.setattr(robot_motion_source, 'source_robot', lambda *a: pytest.fail('Queue decoded a model'))
    before = source_identity.source_revision('clip')
    assert len(before) == 64 and source_identity.source_revision('clip') == before
    if change == 'clock': row['frames'] = 301
    elif change == 'spec': spec['robot_id'] = 'different-local-model'
    else: files[change].write_bytes(b'replaced-' + change.encode())
    assert source_identity.source_revision('clip') != before


def test_source_revision_rejects_stale_declared_hash(tmp_path, monkeypatch):
    path = tmp_path/'motion.npz'; path.write_bytes(b'source')
    spec = dict(path=str(path), sha256='outdated')
    monkeypatch.setattr(source_identity, 'registered_source', lambda *a, **k: (dict(source=str(path)), spec))
    with pytest.raises(ValueError, match='source hash changed'):
        source_identity.source_revision('clip')


def test_missing_optional_video_does_not_block_motion_only_revision(tmp_path, monkeypatch):
    path = tmp_path/'motion.npz'; path.write_bytes(b'source')
    video = tmp_path/'optional.mp4'
    row = dict(source=str(path), video=dict(path=str(video)))
    monkeypatch.setattr(source_identity, 'registered_source', lambda *a, **k: (row, None))
    before = source_identity.source_revision('clip')
    video.write_bytes(b'camera-now-available')
    assert source_identity.source_revision('clip') != before


def submit(client):
    response = client.post('/generate', json=dict(scene_id='clip', robot_id='robot', output_modalities=['trajectory']))
    assert response.status_code == 200, response.text
    return response.json()


def test_active_job_deduplication_requires_current_source_revision(queue):
    client, submissions, api, root = queue
    first = submit(client)
    api.save(root/'queue'/first['id']/'state.json', dict(status='running'))
    assert submit(client)['id'] == first['id']
    client.app.state.test_source_revision['value'] = 'source-after'
    second = submit(client)
    assert second['id'] != first['id'] and len(submissions) == 2
    assert second['source_revision'] == 'source-after'
    assert json.loads((root/'queue'/first['id']/'job.json').read_text())['source_revision'] == 'source-before'


def test_changed_queued_source_fails_before_starting_worker(queue, monkeypatch):
    client, submissions, api, root = queue
    first = submit(client)
    client.app.state.test_source_revision['value'] = 'source-after'
    monkeypatch.setattr(api.subprocess, 'run', lambda *a, **k: pytest.fail('Stale job started a worker'))
    fn, args = submissions[0]; fn(*args)
    state = client.get('/jobs/'+first['id']).json()
    assert state['status'] == 'failed' and state['source_changed']
    assert state['error'].startswith('source_changed:')


@pytest.mark.parametrize('uploaded', [False, True])
@pytest.mark.parametrize('changed_during_worker', [False, True])
def test_default_publication_checks_revision_under_registry_lock(queue, monkeypatch, uploaded, changed_during_worker):
    client, submissions, api, root = queue
    from arenalab_workbench import trajectory_height, gpu_scheduler
    from arenalab_workbench.dataset_io import storage
    monkeypatch.setattr(storage, 'ROOT', root)
    monkeypatch.setattr(api.paths, 'store', root/'store')
    entry = dict(public=dict(id='clip', scene_url='/assets/current-default.json'))
    if uploaded:
        storage.write(root/'user-data/registry.json', {'clip': entry})
    catalog = root/'store/manipulation4d/catalog.json'
    storage.write(catalog, [entry['public']])
    from contextlib import contextmanager
    held = []
    @contextmanager
    def registry_lock(key):
        held.append(key)
        try: yield
        finally: held.pop()
    monkeypatch.setattr(gpu_scheduler, 'resource_lock', registry_lock)
    calls = []
    def complete_height(*args):
        assert held == ['user-recording-registry']
        calls.append(args)
        return False
    monkeypatch.setattr(trajectory_height, 'complete', complete_height)
    first = submit(client)
    def run(*args, **kwargs):
        if changed_during_worker: client.app.state.test_source_revision['value'] = 'source-after'
        api.save(root/'queue'/first['id']/'state.json', dict(status='completed', scene_url='/assets/old-worker-result.json'))
        # The API must not commit a success between child completion and the
        # parent's post-worker source check.
        provisional = client.get('/jobs/'+first['id']).json()
        assert provisional['status'] == 'running'
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(api.subprocess, 'run', run)
    fn, args = submissions[0]; fn(*args)
    state = client.get('/jobs/'+first['id']).json()
    actual = storage.entry('clip')['public'] if uploaded else storage.read(catalog)[0]
    if changed_during_worker:
        assert state['status'] == 'failed' and state['source_changed']
        assert actual['scene_url'] == '/assets/current-default.json' and calls == []
        # Preserve the historical result location as failure evidence.
        assert state['scene_url'] == '/assets/old-worker-result.json'
    else:
        assert state['status'] == 'completed' and len(calls) == 1
        assert actual['scene_url'] == '/assets/old-worker-result.json'


def test_legacy_job_remains_readable_but_is_not_deduplicated(queue):
    client, submissions, api, root = queue
    first = submit(client)
    del client.app.state.generation_jobs[first['id']]['source_revision']
    assert client.get('/jobs/'+first['id']).status_code == 200
    assert submit(client)['id'] != first['id']


@pytest.mark.parametrize('legacy', [False, True])
def test_restart_does_not_leave_unconfirmed_completion_running_forever(queue, legacy):
    client, submissions, api, root = queue
    first = submit(client)
    if legacy:
        old = dict(first); old.pop('source_revision')
        api.save(root/'queue'/first['id']/'job.json', old)
    api.save(root/'queue'/first['id']/'state.json', dict(status='completed', scene_url='/assets/result.json'))
    from fastapi.testclient import TestClient
    restarted = TestClient(api.create_api(root/'viewer.json'))
    state = restarted.get('/jobs/'+first['id']).json()
    assert state['status'] == ('completed' if legacy else 'interrupted')
    assert state['scene_url'] == '/assets/result.json'
