"""Output selection survives the real queue and subprocess boundary."""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def queue(tmp_path, monkeypatch):
    from arenalab_workbench import workbench_api as api
    from arenalab_workbench import robot_registry, trajectory_height, scene_geometry
    from arenalab_workbench import source_requirements, height_registry, source_identity, gpu_scheduler
    from arenalab_workbench.dataset_io import routes
    from arenalab_workbench.output_modalities import source_capabilities
    submissions = []
    class Pool:
        def __init__(self, **kwargs): pass
        def submit(self, fn, *args): submissions.append((fn, args))
    monkeypatch.setattr(api.concurrent.futures, 'ThreadPoolExecutor', Pool)
    monkeypatch.setattr(api, 'ROOT', tmp_path)
    monkeypatch.setattr(gpu_scheduler, 'ROOT', tmp_path/'scheduler')
    revision = {'value': 'source-before'}
    monkeypatch.setattr(source_identity, 'source_revision', lambda *args, **kwargs: revision['value'])
    monkeypatch.setattr(api, 'attach_hand_routes', lambda *args: None)
    monkeypatch.setattr(trajectory_height, 'attach_height_routes', lambda *args: None)
    monkeypatch.setattr(scene_geometry, 'attach_routes', lambda *args: None)
    monkeypatch.setattr(routes, 'attach_dataset_routes', lambda *args: None)
    monkeypatch.setattr(routes, 'require_scene', lambda *args: None)
    monkeypatch.setattr(robot_registry, 'roster', lambda *args: {'robot': {'id': 'robot'}})
    monkeypatch.setattr(api, 'supported', lambda r: True)
    monkeypatch.setattr(height_registry, 'default_id', lambda sid: None)
    monkeypatch.setattr(height_registry, 'public_entry', lambda sid: {'dataset': 'Uploaded'})
    caps = source_capabilities({'end_effector': 'wuji'}, {
        'source': 'observed.hdf5', 'video': {'path': 'observed.mp4'}, 'has_observed_hands': True})
    monkeypatch.setattr(source_requirements, 'input_requirements', lambda *args: {
        'output_capabilities': caps, 'hand_id': 'wuji'})
    monkeypatch.setattr(source_requirements, 'resolve_scale', lambda value, req: value or 'preserve')
    monkeypatch.setattr(api, 'job_view', lambda job, jobs, folder: {
        **job, **json.loads((folder/job['id']/'state.json').read_text())})
    app = api.create_api(tmp_path/'viewer.json')
    app.state.test_source_revision = revision
    return TestClient(app), submissions, api, tmp_path


def test_selected_outputs_deduplicate_canonically_and_reach_worker(queue, monkeypatch):
    client, submissions, api, root = queue
    payload = {'scene_id': 'clip', 'robot_id': 'robot',
               'output_modalities': ['contact', 'point_cloud']}
    first = client.post('/generate', json=payload)
    assert first.status_code == 200, first.text
    payload['output_modalities'].reverse()
    again = client.post('/reconstruct', json=payload)
    assert again.json()['id'] == first.json()['id']
    assert len(submissions) == 1
    trajectory = client.post('/generate', json={**payload, 'output_modalities': ['trajectory']})
    assert trajectory.json()['id'] != first.json()['id']
    assert len(submissions) == 2
    captured = []
    def run(command, **kwargs):
        captured.append(json.loads(kwargs['env']['ARENA_OUTPUT_MODALITIES']))
        state = root/'queue'/first.json()['id']/'state.json'
        state.write_text(json.dumps({'status': 'completed', 'scene_url': '/assets/scene.json'}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(api.subprocess, 'run', run)
    from arenalab_workbench import trajectory_height
    monkeypatch.setattr(trajectory_height, 'complete', lambda *args: True)
    fn, args = submissions[0]
    fn(*args)
    assert captured == [['point_cloud', 'contact']]


@pytest.mark.parametrize('outputs', [[], ['trajectory', 'trajectory'], ['invalid']])
def test_invalid_selection_never_enqueues(queue, outputs):
    client, submissions, _, _ = queue
    result = client.post('/generate', json={
        'scene_id': 'clip', 'robot_id': 'robot', 'output_modalities': outputs})
    assert result.status_code in (409, 422)
    assert submissions == []


def test_unavailable_selection_never_enqueues(queue, monkeypatch):
    client, submissions, _, _ = queue
    from arenalab_workbench import source_requirements
    monkeypatch.setattr(source_requirements, 'resolve_scale', lambda value, req: 'preserve')
    # The app captured input_requirements at initialization; mutate its returned
    # capabilities rather than replacing the callable after construction.
    req = source_requirements.input_requirements('clip')
    req['output_capabilities']['outputs']['contact']['available'] = False
    result = client.post('/generate', json={
        'scene_id': 'clip', 'robot_id': 'robot', 'output_modalities': ['contact']})
    assert result.status_code == 409
    assert submissions == []
