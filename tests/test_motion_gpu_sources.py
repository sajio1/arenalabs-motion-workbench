import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('torch')
pytest.importorskip('mujoco')
from arenalab_motion.sources import load_source,ROLE_NAMES,BONE_NAMES
from arenalab_motion.prior import C,encode_source,_write_new
from arenalab_motion.scaling import transform_arrays
from arenalab_motion.runtime import Runtime


def source_fixture(tmp_path):
    names=list(dict.fromkeys(['Hips','Head',*[n for p in BONE_NAMES for n in p]]));J=len(names);T=4
    rest=np.zeros((J,3));rest[1:]=np.array([10.,2.,3.])
    positions=np.arange(J)[None,:,None]*np.array([.1,.02,.03])[None,None,:]+np.zeros((T,1,3))
    rotations=np.broadcast_to(np.eye(3),(T,J,3,3)).copy();roles=[names.index(n) for n in ROLE_NAMES]
    native=dict(names=np.array(names),parents=np.r_[-1,np.arange(J-1)],rest_offsets_cm=rest,
        full_body_position_yup_m=positions,full_body_rotation_yup=rotations,
        timestamps_s=np.arange(T)/30,exact_fps=30.,source_frame_indices=np.arange(T),floor_z_m=0.,
        wrist_position_world_m=(positions@C)[:,roles[2:4]],six_position_zup_m=(positions@C)[:,roles],
        reference_world_R_zup=rotations[:,roles],sole_low_z_m=np.zeros((T,2)),
        palm_forward=np.broadcast_to([1.,0,0],(T,2,3)),palm_normal=np.broadcast_to([0.,0,1.],(T,2,3)))
    path=tmp_path/'source';path.mkdir();np.savez_compressed(path/'native_source.npz',**native)
    digest=hashlib.sha256((path/'native_source.npz').read_bytes()).hexdigest()
    (path/'input_manifest.json').write_text(json.dumps(dict(native_source_sha256=digest,source_body='synthetic complete body')))
    return path,native


def test_complete_source_encodes_every_joint_and_keeps_raw_immutable(tmp_path):
    import torch
    folder,z=source_fixture(tmp_path);source=load_source(folder)
    class Codec:
        device='cpu';_motion_model_identity={'fixture':'fake-only-for-source-contract'}
        def encode(self,pose,spec,dof):
            assert pose.shape[1]==len(z['names'])
            self.pose=pose.clone()
            return torch.zeros((len(pose),132,13),dtype=torch.long)
    codec=Codec();output=tmp_path/'prior';encode_source(source,output,codec)
    saved=np.load(output/'source_prime_original.npz')
    np.testing.assert_array_equal(saved['observed_nodes'],np.arange(len(z['names'])))
    assert not bool(saved['source_decoder_used'])
    original=(output/'source_prime_original.npz').read_bytes()
    encode_source(source,output,codec)
    assert (output/'source_prime_original.npz').read_bytes()==original
    with pytest.raises(FileExistsError):_write_new(output/'source_prime_original.npz',dict(changed=np.zeros(1)))


def test_source_corruption_and_sparse_only_are_rejected(tmp_path):
    path,z=source_fixture(tmp_path);z['floor_z_m']=1.;np.savez_compressed(path/'native_source.npz',**z)
    with pytest.raises(ValueError,match='hash changed'):load_source(path)
    missing=tmp_path/'sparse';missing.mkdir()
    with pytest.raises(FileNotFoundError,match='upstream'):load_source(missing)


def test_similarity_moves_object_and_human_about_same_anchor_without_rotating_time():
    scaling=dict(scale=.5,anchor_world_m=[2.,3.,0.],anchor_source_yup_m=[3.,0.,2.])
    human=dict(vertices_world_m=np.array([[[4.,5.,2.]]]),timestamps_s=np.array([1.]),floor_world_m=np.array(0.))
    objects=dict(positions_world_m=np.array([[[4.,5.,2.]]]),object_0_vertices_local_m=np.array([[2.,0.,0.]]),rotations_world=np.eye(3)[None,None],timestamps_s=np.array([1.]))
    a=transform_arrays('source_mesh.npz',human,scaling);b=transform_arrays('objects.npz',objects,scaling)
    np.testing.assert_array_equal(a['vertices_world_m'],b['positions_world_m'])
    np.testing.assert_array_equal(a['vertices_world_m'],[[[3.,4.,1.]]])
    np.testing.assert_array_equal(b['object_0_vertices_local_m'],[[1.,0.,0.]])
    np.testing.assert_array_equal(b['rotations_world'],objects['rotations_world'])
    np.testing.assert_array_equal(a['timestamps_s'],human['timestamps_s'])


def test_runtime_requires_local_explicit_assets(tmp_path,monkeypatch):
    monkeypatch.delenv('ARENA_MOTION_RUNTIME',raising=False);monkeypatch.delenv('ARENA_GPU_MOTION_RUNTIME',raising=False)
    with pytest.raises(ValueError,match='explicit'):Runtime.read()
    with pytest.raises(FileNotFoundError):Runtime.read(dict(weights=str(tmp_path/'missing'),cache=str(tmp_path/'cache')))
