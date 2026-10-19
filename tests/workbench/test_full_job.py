"""Scene completion cannot masquerade as completion of all robot modalities."""
import json
from arenalab_workbench.progressive_job import Progress
from arenalab_workbench.job_status import job_view


def test_progressive_modalities_finish_independently(tmp_path):
    status = tmp_path / 'job' / 'state.json'
    status.parent.mkdir()
    progress = Progress(status, '/assets/scene.json', 'robot')
    progress.publish('motion', 'Motion ready')
    state = json.loads(status.read_text())
    assert state['status'] == 'running'
    assert state['robot_ready'] is True
    progress.publish('scene', 'Scene ready')
    assert json.loads(status.read_text())['status'] == 'running'
    progress.publish('fov', 'FOV ready')
    progress.publish('contact', 'Contact ready')
    progress.complete()
    view = job_view({'id': 'job', 'created_at': 0}, {}, tmp_path)
    assert view['status'] == 'completed'
    assert view['scene_url'] == '/assets/scene.json'
