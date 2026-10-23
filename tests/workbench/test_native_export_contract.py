"""Resolve a selected native artifact without guessing from the live registry.

Model/FK numeric validation belongs to the adapter tests. These checks isolate
the packaging boundary, legacy fallback and exact immutable-result selection.
"""
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import zipfile

import numpy as np
import pytest

from arenalab_workbench.dataset_io import export as exporter, storage


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    store = tmp_path / 'store'
    public = store / 'selected'
    public.mkdir(parents=True)
    playback = public / 'playback.json'
    write_json(playback, {})
    monkeypatch.setattr(exporter, 'STORE', store)
    monkeypatch.setattr(storage, 'ROOT', tmp_path / 'workbench')
    calls = []
    adapter = ModuleType('arenalab_workbench.dataset_io.native_reference')

    def build(source, model, output, **kwargs):
        calls.append((Path(source), Path(model), Path(output), kwargs))
        write_json(Path(output) / 'manifest.json', {'frames': 3, 'model_included': False})
        # Private implementation details must not pass through the public summary.
        return dict(frames=3, private_path=str(tmp_path / 'private-runtime'))

    adapter.build_native_reference = build
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    return SimpleNamespace(
        root=tmp_path, store=store, public=public, calls=calls,
        robot={'id': 'g1', 'playback_url': '/assets/selected/playback.json?v=revision'},
        motion={'fps': 30, 'frames': 3}, output=tmp_path / 'export/native_reference',
    )


def native_result(bundle, *, symlink=False, request_in_work=False, model_field='xml'):
    native = bundle.root / 'private-results/immutable-version'
    native.mkdir(parents=True)
    source = native / 'motion.npz'
    source.write_bytes(b'Fixture: adapter is stubbed')
    selected = bundle.root / 'private-results/selected'
    selected.mkdir()
    if symlink:
        (selected / 'motion.npz').symlink_to(source)
    else:
        selected = native
    model = bundle.root / 'private-models/customer.xml'
    model.parent.mkdir()
    model.write_text('<mujoco/>')
    work = bundle.root / 'private-results/runner-work'
    request_path = (work if request_in_work else native) / 'request.json'
    request = {'robot_id': 'g1', 'output_dir': str(native), 'row': {model_field: str(model)}}
    write_json(request_path, request)
    validation = {'native_output_dir': str(selected)}
    if request_in_work:
        validation['native_work_dir'] = str(work)
    write_json(bundle.public / 'validation.json', validation)
    return SimpleNamespace(native=native, selected=selected, source=source, model=model,
                           work=work, request_path=request_path, request=request,
                           validation=validation)


@pytest.mark.parametrize('symlink,request_in_work,model_field', [
    (False, False, 'xml'), (False, True, 'xml'), (True, False, 'xml'),
    (False, False, 'model'),
])
def test_native_reference_uses_producing_request_and_private_model(
        bundle, symlink, request_in_work, model_field):
    case = native_result(bundle, symlink=symlink, request_in_work=request_in_work,
                         model_field=model_field)
    result = exporter.native_reference(bundle.robot, bundle.motion, bundle.output)
    assert result == dict(
        available=True, contract='extensions/arenalabs/native_reference/manifest.json',
        mode='articulated_kinematic_reference', model_included=False,
        physics_control_validated=False, frames=3,
    )
    assert len(bundle.calls) == 1
    source, model, output, kwargs = bundle.calls[0]
    assert source.resolve() == case.source
    assert model == case.model and output == bundle.output
    assert kwargs == {'playback': bundle.motion, 'robot_id': 'g1'}
    assert str(bundle.root) not in json.dumps(result)
    assert 'private-runtime' not in json.dumps(result)


def test_native_reference_accepts_nested_producing_robot_row(bundle):
    case = native_result(bundle)
    request = {'robot_id': 'g1', 'output_dir': str(case.native),
               'robot': {'pipeline_row': {'xml': str(case.model)}}}
    write_json(case.request_path, request)
    assert exporter.native_reference(bundle.robot, bundle.motion, bundle.output)['available']
    assert bundle.calls[0][1] == case.model


@pytest.mark.parametrize('validation', [None, {}, {'physical_validation': False}])
def test_legacy_without_authoritative_native_state_retains_visual_export(bundle, validation):
    if validation is not None:
        write_json(bundle.public / 'validation.json', validation)
    result = exporter.native_reference(bundle.robot, bundle.motion, bundle.output)
    assert result['available'] is False
    assert 'visual replay' in result['reason']
    assert not bundle.calls and not bundle.output.exists()
    assert str(bundle.root) not in json.dumps(result)


def test_missing_producing_request_is_explicitly_unavailable(bundle):
    case = native_result(bundle)
    case.request_path.unlink()
    result = exporter.native_reference(bundle.robot, bundle.motion, bundle.output)
    assert result['available'] is False and 'model contract' in result['reason']
    assert not bundle.calls and not bundle.output.exists()
    assert str(bundle.root) not in json.dumps(result)


def test_recorded_but_missing_native_state_fails_without_leaking_path(bundle):
    case = native_result(bundle)
    case.source.unlink()
    with pytest.raises(ValueError, match='native motion is missing') as exc:
        exporter.native_reference(bundle.robot, bundle.motion, bundle.output)
    assert str(bundle.root) not in str(exc.value)
    assert not bundle.calls


def test_wrong_robot_request_cannot_be_used_for_selected_result(bundle):
    case = native_result(bundle)
    case.request['robot_id'] = 'x2'
    write_json(case.request_path, case.request)
    with pytest.raises(ValueError, match='robot identity') as exc:
        exporter.native_reference(bundle.robot, bundle.motion, bundle.output)
    assert str(bundle.root) not in str(exc.value)
    assert not bundle.calls


@pytest.mark.parametrize('row', [{}, {'xml': '/nonexistent/customer-private-model.xml'}])
def test_missing_producing_model_fails_without_registry_fallback(bundle, row):
    case = native_result(bundle)
    case.request['row'] = row
    write_json(case.request_path, case.request)
    with pytest.raises(ValueError, match='Producing robot model is unavailable') as exc:
        exporter.native_reference(bundle.robot, bundle.motion, bundle.output)
    assert str(bundle.root) not in str(exc.value)
    assert 'customer-private-model' not in str(exc.value)
    assert not bundle.calls


def test_same_robot_request_for_another_output_is_rejected(bundle):
    case = native_result(bundle, request_in_work=True)
    case.request['output_dir'] = str(bundle.root / 'private-results/another-run')
    write_json(case.request_path, case.request)
    with pytest.raises(ValueError, match='(?i)(output|producing|native)') as exc:
        exporter.native_reference(bundle.robot, bundle.motion, bundle.output)
    assert str(bundle.root) not in str(exc.value)
    assert not bundle.calls


@pytest.mark.parametrize('misrouted_raw', [False, True])
def test_complete_export_advertises_and_checksums_native_contract_without_model(bundle, monkeypatch, misrouted_raw):
    pytest.importorskip('pyarrow')
    trimesh = pytest.importorskip('trimesh')
    mj = pytest.importorskip('mujoco')
    monkeypatch.delitem(sys.modules, 'arenalab_workbench.dataset_io.native_reference')
    case = native_result(bundle)
    case.model.write_text('<mujoco><worldbody><body name="base"><freejoint name="root"/>'
                         '<geom size=".1"/><body name="arm"><joint name="joint"/>'
                         '<geom size=".1"/></body></body></worldbody></mujoco>')
    model = mj.MjModel.from_xml_path(str(case.model))
    qpos = np.tile(model.qpos0, (3, 1))
    qpos[:, 7] = [0, .1, .2]
    np.savez(case.source, qpos=qpos, fps=30)
    visual = trimesh.Scene()
    for index in range(model.ngeom):
        visual.add_geometry(trimesh.creation.box(), node_name=f'geom_{index}', geom_name=f'geom_{index}')
    (bundle.public / 'model.glb').write_bytes(visual.export(file_type='glb'))
    motion = {
        'fps': 30, 'frames': 3,
        'positions': [[[0, 0, 0]]] * 3, 'quaternions': [[[0, 0, 0, 1]]] * 3,
        'joints': {'names': ['joint'], 'values': [[0.], [.1], [.2]], 'unit': 'rad'},
    }
    from scipy.spatial.transform import Rotation
    basis=np.array([[1,0,0],[0,0,1],[0,-1,0]])
    state=mj.MjData(model); positions=[]; rotations=[]
    for row in qpos:
        state.qpos[:]=row; mj.mj_forward(model,state)
        positions.append(state.geom_xpos.copy()); rotations.append(state.geom_xmat.copy().reshape(-1,3,3))
    positions,rotations=np.asarray(positions),np.asarray(rotations)
    np.savez(case.source.parent/'robot-render.npz',positions=positions,rotations=rotations,native_body_geom_ids=np.arange(model.ngeom))
    motion['positions']=(positions@basis.T).tolist()
    motion['quaternions']=Rotation.from_matrix((basis@rotations).reshape(-1,3,3)).as_quat().reshape(3,model.ngeom,4).tolist()
    write_json(bundle.public / 'playback.json', motion)
    scene = {
        'source_world_to_display':basis.tolist(),
        'id': 'clip', 'title': 'Fixture', 'dataset': 'Synthetic', 'frames': 3,
        'fps': 30, 'video': None, 'output_modalities': ['trajectory'],
        'completed_outputs': ['trajectory'], 'output_status': {'trajectory': 'completed'},
        'generation_status': {'status': 'completed'},
        'robots': [{**bundle.robot, 'model_url': '/assets/selected/model.glb'}],
    }
    scene_path = bundle.store / 'scene.json'
    write_json(scene_path, scene)
    output = bundle.root / 'complete-dataset'
    raw=bundle.root/'optional-raw';raw.mkdir()
    write_json(raw/'report.json',{'private_source':'/mnt/customer/private/motion.npz',
                                'private_windows':'D:\\customer\\private\\robot.urdf'})
    if misrouted_raw:
        np.savez(raw/'motion.npz',qpos=qpos+1,fps=30)
        original=case.source.read_bytes()
        with pytest.raises(ValueError,match='RAW_NATIVE_REFERENCE_MISMATCH'):
            exporter.export_dataset(scene_path, 'g1', output,raw_sources=[('robot/refined',raw)])
        assert case.source.read_bytes()==original
        assert not output.with_suffix('.zip').exists()
        return
    result = exporter.export_dataset(scene_path, 'g1', output,raw_sources=[('diagnostics',raw)])
    with zipfile.ZipFile(result['path']) as package:
        manifest = json.loads(package.read('extensions/arenalabs/manifest.json'))
        contract = manifest['native_reference']['contract']
        checksums = json.loads(package.read('checksums.json'))
        assert manifest['native_reference']['available'] is True
        assert manifest['native_reference']['physics_control_validated'] is False
        assert contract in package.namelist() and contract in checksums
        assert checksums[contract] == exporter.digest(output / contract)
        assert not any(name.endswith('.xml') for name in package.namelist())
        assert str(bundle.root) not in json.dumps(manifest)
        native_raw='extensions/arenalabs/raw/robot/refined/motion.npz'
        assert native_raw in manifest['raw_files']
        assert package.read(native_raw)==case.source.read_bytes()
        report=json.loads(package.read('extensions/arenalabs/raw/diagnostics/report.json'))
        assert report=={'private_source':'motion.npz','private_windows':'robot.urdf'}


def collector_dependencies(bundle, monkeypatch, gripper):
    body = bundle.root / 'body'
    body.mkdir()
    work = bundle.root / 'gripper-source'
    work.mkdir()
    definitions = {
        'arenalab_workbench.gripper_registry': {'sources': lambda: {'clip': {'work': str(work)}} if gripper else {}},
        'arenalab_workbench.gripper_body': {'body_path': lambda path: body},
        'arenalab_workbench.body_prior': {'folder': lambda ident: body},
        'arenalab_workbench.wheeled_body': {'POLICY': 'mobile-fixture'},
    }
    for name, members in definitions.items():
        module = ModuleType(name)
        for key, value in members.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
    return body, work


@pytest.mark.parametrize('gripper', [False, True])
def test_raw_collection_prefers_selected_exact_output_over_legacy_paths(bundle, monkeypatch, gripper):
    body, work = collector_dependencies(bundle, monkeypatch, gripper)
    exact = bundle.root / 'immutable-output/match_robot/g1'
    write_json(bundle.public / 'validation.json', {
        'native_output_dir': str(exact), 'native_work_dir': str(bundle.root / 'different-run-root'),
    })
    unrelated = bundle.store / 'other'
    write_json(unrelated / 'validation.json', {'native_output_dir': str(bundle.root / 'wrong-robot')})
    scene = {'robots': [{'id': 'x2', 'playback_url': '/assets/other/playback.json'}, bundle.robot]}
    values = exporter.collect_raw_sources('clip', scene, 'g1')
    assert [path for category, path in values if category == 'robot/refined'] == [str(exact)]
    assert ('human', str(body)) in values
    assert (('source', str(work)) in values) is gripper


@pytest.mark.parametrize('gripper', [False, True])
def test_raw_collection_preserves_legacy_fallback_and_single_robot_inference(bundle, monkeypatch, gripper):
    body, work = collector_dependencies(bundle, monkeypatch, gripper)
    values = exporter.collect_raw_sources('clip', {'robots': [bundle.robot]})
    expected = work / 'g1' if gripper else body / 'retarget/g1/operation-accuracy/palm/g1'
    assert ('robot/refined', str(expected)) in values


def test_raw_collection_keeps_mobile_artifact_separate_from_body_reference(bundle, monkeypatch):
    body, _ = collector_dependencies(bundle, monkeypatch, False)
    mobile = body / 'mobile/mobile-fixture/g1'
    mobile.mkdir(parents=True)
    values = exporter.collect_raw_sources('clip', {'robots': [bundle.robot]})
    assert ('robot/mobile', str(mobile)) in values
    assert sum(category == 'robot/refined' for category, _ in values) == 1
