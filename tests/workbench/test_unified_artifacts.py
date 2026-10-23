import pytest
import json
import time
from concurrent.futures import ThreadPoolExecutor
from arenalab_workbench.unified_scene import scene_key
from arenalab_workbench.motion_backend import run_gpu_motion


def test_scaled_worlds_cannot_overwrite_another_robot():
    assert scene_key('clip','match_robot','a') != scene_key('clip','match_robot','b')
    assert scene_key('clip','preserve','a') == scene_key('clip','preserve','b')
    assert scene_key('clip','match_robot','a') != scene_key('clip','preserve','a')


def test_retry_cannot_replace_original_request(tmp_path):
    original=tmp_path/'request.json'
    original.write_text('{"attempt":"original"}')
    with pytest.raises(FileExistsError,match='fresh'):
        run_gpu_motion({'output_dir':str(tmp_path),'attempt':'different'})
    assert original.read_text() == '{"attempt":"original"}'


def test_parallel_robots_materialize_one_source_version(tmp_path,monkeypatch):
    from arenalab_workbench import motion_backend as backend,motion_sources
    (tmp_path/'raw.npz').write_bytes(b'input')
    (tmp_path/'input_manifest.json').write_text(json.dumps({'motion_source':{'path':'raw.npz','family':'smplh'}}))
    calls=[]
    def prepare(source,output):
        calls.append(output)
        time.sleep(.04)
        output.mkdir()
        (output/'native_source.npz').write_bytes(b'complete')
        (output/'source_provenance.json').write_text(json.dumps({'output_sha256':{'native_source.npz':backend.sha(output/'native_source.npz')}}))
        return output
    monkeypatch.setattr(motion_sources,'prepare_body_source',prepare)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs=list(pool.map(backend.normalized_source,[tmp_path,tmp_path]))
    assert outputs[0]==outputs[1]
    assert len(calls)==1
