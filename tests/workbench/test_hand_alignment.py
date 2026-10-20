import numpy as np
from scipy.spatial.transform import Rotation
from arenalab_workbench.hand_alignment import fit_registration,transform_points


def test_global_similarity_is_robust_on_heldout_frames_and_preserves_raw_points():
    rng=np.random.default_rng(18)
    source=rng.normal(size=(1200,3))*.3
    expected=np.eye(4);expected[:3,:3]=1.2*Rotation.from_euler('xyz',[.05,-.08,.12]).as_matrix();expected[:3,3]=[.1,-.15,.07]
    target=transform_points(source,expected)+rng.normal(scale=.001,size=source.shape)
    target[::5]+=rng.normal(scale=.4,size=target[::5].shape)
    raw=source.copy()
    report=fit_registration(source,target,np.ones(len(source)),np.repeat(np.arange(120),10))
    assert report['status']=='registered'
    np.testing.assert_allclose(report['raw_to_reference'],expected,atol=.003)
    assert report['heldout_after_median_mm']<3
    assert report['robot_motion_modified'] is False and report['independent_of_visibility'] is True
    np.testing.assert_array_equal(source,raw)


def test_no_alignment_is_invented_for_stationary_or_missing_anchors():
    source=np.zeros((100,3))
    report=fit_registration(source,source,np.ones(100),np.arange(100))
    assert report['status']=='insufficient_evidence'
    np.testing.assert_array_equal(report['raw_to_reference'],np.eye(4))
    empty=np.empty((0,3))
    assert fit_registration(empty,empty,[],[])['status']=='insufficient_evidence'


def test_good_alignment_is_not_perturbed_by_fitting_noise():
    rng=np.random.default_rng(4);source=rng.normal(size=(600,3))
    target=source+rng.normal(scale=.002,size=source.shape)
    report=fit_registration(source,target,np.ones(600),np.repeat(np.arange(60),10))
    assert report['status']=='no_improvement'
    np.testing.assert_array_equal(report['raw_to_reference'],np.eye(4))


def test_exported_boxes_and_points_share_translation_but_optical_camera_stays_original(tmp_path,monkeypatch):
    import json,sys,runpy
    from pathlib import Path
    from contextlib import nullcontext
    import arenalab_workbench.hand_alignment as alignment
    import arenalab_workbench.gpu_scheduler as scheduler
    work=tmp_path/'work';work.mkdir();(work/'masks').mkdir();dest=tmp_path/'result';dest.mkdir()
    transform=np.eye(4);transform[:3,:3]*=.8;transform[:3,3]=[.1,.2,-.1]
    monkeypatch.setattr(alignment,'register',lambda *args:dict(policy=alignment.POLICY,status='registered',raw_to_reference=transform.tolist()))
    monkeypatch.setattr(scheduler,'resource_lock',lambda *args:nullcontext())
    for name,value in [('manifest.json',dict(source=dict(frames=1),public=dict(id='example',title='example',dataset='EgoDex',video=dict(url='/example.mp4',start_seconds=0)),destination=str(dest),url='/assets/example/',source_sha256='unchanged')),('preparation.json',{}),('semantic.json',dict(objects=['object'])),('prompts.json',[['object-01','object']]),('depth.json',dict(model='test'))]:
        (work/name).write_text(json.dumps(value))
    np.savez(work/'depth.npz',depth=np.ones((1,4,4)),intrinsics=np.eye(3)[None],extrinsics=np.eye(4)[None,:3],images=np.full((1,4,4,3),120,dtype='u1'),ids=[0])
    np.savez(work/'cameras.npz',poses=np.eye(4)[None],intrinsic=np.eye(3))
    np.savez(work/'masks/object-01-masks.npz',masks=np.ones((1,4,4),dtype='u2'),frame_indices=[0])
    monkeypatch.setattr(sys,'argv',['export_scene_cloud.py',str(work)])
    runpy.run_module('arenalab_workbench.export_scene_cloud',run_name='__main__')
    result=json.loads((dest/'scene.json').read_text());box=result['segmented_instances']['frames'][0]['instances'][0]['bounds']
    np.testing.assert_allclose(np.mean(box,axis=0),[1.3,1.4,.7])
    xyz=np.fromfile(dest/'cloud.bin',dtype='<f4',count=12).reshape(-1,3)
    np.testing.assert_allclose(xyz,[[.1,.2,.7],[1.7,.2,.7],[.1,1.8,.7],[1.7,1.8,.7]],atol=1e-6)
    np.testing.assert_allclose(result['source_camera']['poses'][0],np.eye(4))
