"""Downloaded dataset acceptance: real native references, no inference or cloud."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

pa = pytest.importorskip('pyarrow')
pq = pytest.importorskip('pyarrow.parquet')
mj = pytest.importorskip('mujoco')
from arenalab_workbench.dataset_io.native_reference import build_native_reference
from arenalab_workbench.dataset_io import readiness
from arenalab_workbench.dataset_io.native_visual import validate_native_visual
from scipy.spatial.transform import Rotation

BASIS = np.array([[1,0,0],[0,0,1],[0,-1,0]])

XML = '''<mujoco><worldbody><body name="base" pos="0 0 1">
<freejoint name="root"/><geom size=".1"/>
<body name="arm" pos=".2 0 .1"><joint name="hinge" axis="0 1 0"/>
<geom size=".1"/><body name="tool" pos=".1 0 0"><joint name="slide" type="slide"/>
<geom size=".05"/></body></body></body></worldbody></mujoco>'''


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def read(path):
    return json.loads(path.read_text())


def seal(root):
    write(root / 'checksums.json', {
        p.relative_to(root).as_posix(): readiness._sha(p)
        for p in root.rglob('*') if p.is_file() and p.name != 'checksums.json'
    })


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / 'dataset'
    ext = root / 'extensions/arenalabs'
    assets = ext / 'assets'
    assets.mkdir(parents=True)
    model_path = tmp_path / 'private-local-robot.xml'
    model_path.write_text(XML)
    model = mj.MjModel.from_xml_path(str(model_path))
    q = np.tile(model.qpos0, (4, 1))
    q[:, 0] = [0, .01, .02, .04]
    q[:, 7] = [0, .1, .2, .25]
    q[:, 8] = [0, .01, .02, .03]
    source = tmp_path / 'motion.npz'
    np.savez(source, qpos=q, fps=30, timestamps_s=123 + np.arange(4) / 30)
    names = ['external_finger', 'slide', 'hinge']
    values = np.c_[np.linspace(0, .2, 4), q[:, 8], q[:, 7]]
    state = mj.MjData(model)
    positions, rotations = [], []
    for row in q:
        state.qpos[:] = row; mj.mj_forward(model,state)
        positions.append(state.geom_xpos.copy()); rotations.append(state.geom_xmat.copy().reshape(-1,3,3))
    positions, rotations = np.asarray(positions), np.asarray(rotations)
    np.savez(tmp_path/'robot-render.npz', positions=positions, rotations=rotations, native_body_geom_ids=np.arange(model.ngeom))
    playback = dict(fps=30, frames=4, positions=(positions@BASIS.T).tolist(),
                    quaternions=Rotation.from_matrix((BASIS@rotations).reshape(-1,3,3)).as_quat().reshape(4,model.ngeom,4).tolist(),
                    joints=dict(names=names, values=values.tolist(),
                    unit='mixed', units=['rad', 'm', 'rad']))
    write(assets / 'playback.json', playback)
    (assets / 'model.glb').write_bytes(b'visual fixture; mesh replay tested separately')
    native = ext / 'native_reference'
    build_native_reference(source, model_path, native, playback=playback, robot_id='g1')
    raw_motion=ext/'raw/robot/refined/motion.npz'
    raw_motion.parent.mkdir(parents=True)
    shutil.copyfile(source,raw_motion)
    robot = dict(id='g1', hand='fixture', playback_url='extensions/arenalabs/assets/playback.json',
                 model_url='extensions/arenalabs/assets/model.glb')
    scene = dict(id='clip', dataset='Synthetic', frames=4, fps=30, robots=[robot], source_world_to_display=BASIS.tolist(),
                 output_modalities=['trajectory'], completed_outputs=['trajectory'],
                 output_status={'trajectory': 'completed'}, generation_status={'status': 'completed'})
    validate_native_visual(source, model_path, assets/'playback.json', scene, robot, native)
    # Verification must not need the producer's robot model or raw motion file.
    model_path.unlink(); source.unlink(); (tmp_path/'robot-render.npz').unlink()
    write(ext / 'scene.json', scene)
    write(ext/'replay/replay.json',dict(fps=30,frames=4,meshes=model.ngeom,units='m',world_up='Z',
                                      handedness='right',physics_articulation=False))
    (ext/'replay/robot.usda').write_text('#usda 1.0\ndef Xform "World" {}\n')
    manifest = dict(schema='arenalabs-lerobot-v3.0-4-native-reference', scene_id='clip', robot_id='g1',
                    frames=4, fps=30, state_units=['rad', 'm', 'rad'],
                    raw_files=['extensions/arenalabs/raw/robot/refined/motion.npz'],
                    frame_map='extensions/arenalabs/frame_map.parquet',
                    replay=dict(stage='extensions/arenalabs/replay/robot.usda',
                                contract='extensions/arenalabs/replay/replay.json'),
                    readiness=dict(status='ready', scope='complete_kinematic_reference_data',
                                   validator='validate_dataset.py', dynamics_validated=False),
                    native_reference=dict(available=True, frames=4, physics_control_validated=False,
                        visual_validation='extensions/arenalabs/native_reference/visual-validation.json',
                        contract='extensions/arenalabs/native_reference/manifest.json'))
    manifest['modalities'] = readiness._modality_contract(root, scene, manifest, 'g1')
    write(ext / 'manifest.json', manifest)
    write(root / 'meta/info.json', dict(fps=30, total_frames=4, total_episodes=1, robot_type='g1',
          features={name: {'names': names} for name in ('observation.state', 'action')}))
    state = values.astype('f4')
    data = dict(**{'observation.state': state.tolist(),
                   'action': np.concatenate([state[1:], state[-1:]]).tolist()},
                timestamp=(np.arange(4, dtype='f4') / 30).tolist(), frame_index=list(range(4)),
                index=list(range(4)), episode_index=[0] * 4)
    path = root / 'data/chunk-000/file-000.parquet'
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table(data), path)
    pq.write_table(pa.table(dict(frame_index=list(range(4)), timestamp=(np.arange(4) / 30).tolist(),
                                cloud_index=[None]*4,cloud_source_frame=[None]*4,
                                cloud_observed_this_frame=[False]*4)),
                   ext / 'frame_map.parquet')
    shutil.copyfile(readiness.__file__, root / 'validate_dataset.py')
    seal(root)
    return root


def mutate(root, relative, edit):
    path = root / relative
    value = read(path)
    edit(value)
    write(path, value)
    seal(root)


@pytest.mark.parametrize('source_label', ['EgoDex', 'HiPhi', 'GRAIL', 'Nymeria', 'UMI', 'DROID'])
def test_same_reference_contract_for_every_source_label(dataset, source_label):
    mutate(dataset, 'extensions/arenalabs/scene.json', lambda value: value.update(dataset=source_label))
    result = readiness.validate_dataset(dataset)
    assert result['ready'] and result['frames'] == 4 and result['all_frames_checked']
    assert result['body_joint_count'] == 2 and result['named_joint_count'] == 3
    assert result['attachment_components']['names'] == ['external_finger']
    assert result['attachment_components']['channels_retained']
    assert not result['attachment_components']['physics_validated']
    assert not result['attachment_components']['applied_by_body_adapter']
    assert result['modalities']['trajectory']['status'] == 'ready'
    assert all(result['modalities'][name]['status'] == 'not_selected' for name in readiness.MODALITIES[1:])
    assert not result['dynamics_validated'] and not result['controller_validated']


def test_relocated_standalone_validator_requires_no_repo_or_robot_model(dataset, tmp_path):
    relocated = tmp_path / 'customer/unpacked'
    shutil.copytree(dataset, relocated)
    result = subprocess.run([sys.executable, str(relocated / 'validate_dataset.py'), str(relocated)],
                            cwd=tmp_path, capture_output=True, text=True, env={'PATH': '/usr/bin:/bin'})
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads(result.stdout)['ready']
    assert not list(relocated.rglob('__pycache__'))
    assert readiness.validate_dataset(relocated)['ready']


@pytest.mark.parametrize('failure', ['changed', 'missing', 'extra', 'escape', 'symlink'])
def test_checksum_inventory_rejects_corruption_and_escape(dataset, tmp_path, failure):
    artifact = dataset / 'extensions/arenalabs/assets/model.glb'
    if failure == 'changed':
        artifact.write_bytes(b'tampered')
    elif failure == 'missing':
        artifact.unlink()
    elif failure == 'extra':
        (dataset / 'unexpected.txt').write_text('unlisted')
    elif failure == 'escape':
        checksum = read(dataset / 'checksums.json')
        checksum['../outside'] = '0' * 64
        write(dataset / 'checksums.json', checksum)
    else:
        outside = tmp_path / 'private'
        outside.write_text('outside')
        (dataset / 'link').symlink_to(outside)
    with pytest.raises(ValueError):
        readiness.validate_dataset(dataset)


@pytest.mark.parametrize('edit', [
    lambda value: value['native_reference'].update(available=False),
    lambda value: value['native_reference'].update(frames=3),
    lambda value: value['native_reference'].update(physics_control_validated=True),
    lambda value: value.update(robot_id='x2'),
    lambda value: value.pop('modalities'),
    lambda value: value.pop('readiness'),
    lambda value: value['readiness'].update(status='pending'),
    lambda value: value.update(state_units=['rad', 'rad', 'rad']),
])
def test_incomplete_or_inconsistent_manifest_is_not_ready(dataset, edit):
    mutate(dataset, 'extensions/arenalabs/manifest.json', edit)
    with pytest.raises(ValueError):
        readiness.validate_dataset(dataset)


@pytest.mark.parametrize('edit', [
    lambda value: value.update(kinematic_replay_passed=False),
    lambda value: value.update(frames=3),
    lambda value: value.update(checked_link_poses=1),
    lambda value: value.update(position_max_m=.01),
    lambda value: value.update(controller_used=True),
])
def test_partial_or_failed_native_receipt_is_not_ready(dataset, edit):
    mutate(dataset, 'extensions/arenalabs/native_reference/validation.json', edit)
    with pytest.raises(ValueError, match='(?i)(native|kinematic)'):
        readiness.validate_dataset(dataset)


@pytest.mark.parametrize('column', [0, 2])
def test_attachment_and_body_channel_drift_are_both_rejected(dataset, column):
    def edit(value):
        value['joints']['values'][1][column] += .01
    mutate(dataset, 'extensions/arenalabs/assets/playback.json', edit)
    with pytest.raises(ValueError, match='named joint position differs'):
        readiness.validate_dataset(dataset)


@pytest.mark.parametrize('field', ['observation.state', 'action', 'timestamp', 'frame_index'])
def test_lerobot_reference_semantics_and_clock_are_checked(dataset, field):
    path = dataset / 'data/chunk-000/file-000.parquet'
    data = pq.read_table(path).to_pydict()
    if field in ('observation.state', 'action'):
        data[field][1][0] += .1
    else:
        data[field][1] += 1
    pq.write_table(pa.table(data), path)
    seal(dataset)
    with pytest.raises(ValueError, match='LeRobot'):
        readiness.validate_dataset(dataset)


def test_selected_modality_requires_completion_and_portable_file(dataset):
    def selected(value):
        value['output_modalities'].append('contact')
        value['completed_outputs'].append('contact')
        value['output_status']['contact'] = 'completed'
        value['tactile'] = {'url': 'extensions/arenalabs/assets/contact.json'}
    mutate(dataset, 'extensions/arenalabs/scene.json', selected)
    with pytest.raises(ValueError, match='missing'):
        readiness.validate_dataset(dataset)
    write(dataset / 'extensions/arenalabs/assets/contact.json', {'status': 'estimated', 'frames': 4})
    scene = read(dataset / 'extensions/arenalabs/scene.json')
    manifest_path = dataset / 'extensions/arenalabs/manifest.json'
    manifest = read(manifest_path)
    manifest.pop('modalities')
    manifest['modalities'] = readiness._modality_contract(dataset, scene, manifest, 'g1')
    write(manifest_path, manifest)
    seal(dataset)
    assert readiness.validate_dataset(dataset)['modalities']['contact']['status'] == 'ready'
    mutate(dataset, 'extensions/arenalabs/scene.json', lambda value: value['output_status'].update(contact='failed'))
    with pytest.raises(ValueError, match='not completed'):
        readiness.validate_dataset(dataset)


def test_external_media_dependency_is_rejected(dataset):
    mutate(dataset, 'extensions/arenalabs/scene.json',
           lambda value: value.update(video={'url': 'https://example.invalid/original.mp4'}))
    with pytest.raises(ValueError, match='external'):
        readiness.validate_dataset(dataset)


def test_native_structural_corruption_is_checked_after_resealing(dataset):
    mutate(dataset, 'extensions/arenalabs/native_reference/manifest.json',
           lambda value: value['world'].update(up_axis='Y'))
    with pytest.raises(ValueError, match='world conventions'):
        readiness.validate_dataset(dataset)


def test_wrong_raw_native_source_fails_even_with_recomputed_package_hashes(dataset):
    raw=dataset/'extensions/arenalabs/raw/robot/refined/motion.npz'
    np.savez(raw,qpos=np.zeros((4,9)),fps=30)
    seal(dataset)
    with pytest.raises(ValueError,match='Raw native motion SHA256'):
        readiness.validate_dataset(dataset)


def test_raw_native_source_cannot_be_omitted_from_manifest(dataset):
    mutate(dataset,'extensions/arenalabs/manifest.json',lambda value:value.update(raw_files=[]))
    with pytest.raises(ValueError,match='raw motion is not declared'):
        readiness.validate_dataset(dataset)


@pytest.mark.parametrize('name',['native_reference_isaac.py','native_reference_mujoco.py'])
def test_both_simulator_adapters_are_required(dataset,name):
    (dataset/'extensions/arenalabs/native_reference'/name).unlink()
    seal(dataset)
    with pytest.raises(ValueError,match='artifact is missing'):
        readiness.validate_dataset(dataset)


def test_invalid_visual_quaternion_cannot_be_advertised_ready(dataset):
    mutate(dataset,'extensions/arenalabs/assets/playback.json',
           lambda value:value['quaternions'][1][0].__setitem__(3,0))
    with pytest.raises(ValueError,match='mesh poses'):
        readiness.validate_dataset(dataset)


def test_cloud_frame_map_cannot_fabricate_observations(dataset):
    path=dataset/'extensions/arenalabs/frame_map.parquet'
    data=pq.read_table(path).to_pydict()
    data['cloud_index'][1]=0
    pq.write_table(pa.table(data),path)
    seal(dataset)
    with pytest.raises(ValueError,match='observed cloud frames'):
        readiness.validate_dataset(dataset)


@pytest.mark.parametrize('failure', ['missing', 'partial', 'playback_changed', 'basis_changed', 'model_changed'])
def test_native_visual_binding_cannot_be_forged_by_resealing_files(dataset, failure):
    path = 'extensions/arenalabs/native_reference/visual-validation.json'
    if failure == 'missing':
        (dataset/path).unlink(); seal(dataset)
    elif failure == 'partial':
        mutate(dataset,path,lambda value:value.update(checked_body_mesh_poses=1))
    elif failure == 'model_changed':
        mutate(dataset,path,lambda value:value.update(model_source_sha256='0'*64))
    elif failure == 'basis_changed':
        mutate(dataset,'extensions/arenalabs/scene.json',
               lambda value:value.update(source_world_to_display=[[0,-1,0],[0,0,1],[-1,0,0]]))
    else:
        mutate(dataset,'extensions/arenalabs/assets/playback.json',
               lambda value:value['positions'][1][0].__setitem__(0,.2))
    with pytest.raises(ValueError):
        readiness.validate_dataset(dataset)
