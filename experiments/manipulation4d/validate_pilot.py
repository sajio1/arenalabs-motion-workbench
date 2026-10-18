"""Validate source preservation, metric scene coordinates, synchronization and asset bounds."""
from pathlib import Path
import json, hashlib
import cv2, h5py, numpy as np, trimesh

ROOT=Path('/home/ubuntu/arenalab-experiments/manipulation4d');DEST=Path('/opt/arenalab/store/manipulation4d/pilot')
EXP=Path('/home/ubuntu/arenalab-experiments/egodex-threepoint');scene=json.loads((DEST/'scene.json').read_text());report={}
manifest=json.loads((EXP/'input_manifest.json').read_text());source=np.load(EXP/'source_hands.npz');points=source['points']@source['source_to_zup']
with h5py.File(manifest['source']) as f:
    original=[]
    for side in ['left','right']:
        names=[side+'Hand']+[side+'Thumb'+suffix for suffix in ['Knuckle','IntermediateBase','IntermediateTip','Tip']]
        for finger in ['Index','Middle','Ring','Little']:names.extend(side+finger+'Finger'+suffix for suffix in ['Knuckle','IntermediateBase','IntermediateTip','Tip'])
        original.append(np.stack([f['transforms/'+name][:,:3,3] for name in names],1))
    error=float(np.max(np.abs(points-np.stack(original,1))));assert error<1e-7
source_sha=hashlib.sha256(Path(manifest['source']).read_bytes()).hexdigest()
assert source_sha==json.loads((EXP/'validation.json').read_text())['source_preservation']['sha256']
report['source_preservation']=dict(all_42_hand_landmarks_roundtrip_max_m=error,original_hdf5_sha256=source_sha)
blob=(DEST/'cloud.bin').read_bytes();valid=0
for frame in scene['cloud']['frames']:
    xyz=np.frombuffer(blob,dtype='<f4',count=frame['count']*3,offset=frame['positions']);assert np.isfinite(xyz).all()
    assert frame['labels']+frame['count']<=len(blob);valid+=1
report['pointcloud']=dict(frames=valid,fps=15,bytes=len(blob),world='Original EgoDex world, Y up, metres',depth_ground_truth=False)
calibration=scene['geometry'].get('depth_scale_calibration')
if calibration:
    prior=Path('/home/ubuntu/arenalab-experiments/da3/no-hands/depth-sequence.npz')
    assert hashlib.sha256(prior.read_bytes()).hexdigest()==calibration['prior_sha256']
    with np.load(prior) as original,np.load(ROOT/'depth-calibrated.npz') as corrected:
        assert np.array_equal(original['ids'],corrected['ids'])
        assert np.array_equal(original['extrinsics'],corrected['extrinsics'])
        assert np.array_equal(original['intrinsics'],corrected['intrinsics'])
        assert np.allclose(corrected['depth'],original['depth']*calibration['scale'])
    report['depth_calibration']=calibration
report['robots']={}
for robot in scene['robots']:
    playback_path=DEST/robot['playback_url'].split('/pilot/',1)[1]
    d=json.loads(playback_path.read_text());p=np.asarray(d['positions']);q=np.asarray(d['quaternions']);assert len(p)==266 and np.isfinite(p).all() and np.isfinite(q).all()
    assert np.max(np.abs(np.linalg.norm(q,axis=-1)-1))<1e-5
    actual=np.asarray(d['tracking']['actual']);target=np.asarray(d['tracking']['target']);e=np.linalg.norm(actual-target,axis=-1)*1000
    assert np.allclose(e,d['tracking']['error_mm'],atol=1e-5)
    assert len(d['joints']['values'])==266 and np.isfinite(d['joints']['values']).all()
    report['robots'][robot['id']]=dict(frames=len(p),palm_error_mean_mm=float(e.mean()),palm_error_max_mm=float(e.max()),native_joint_channels=len(d['joints']['names']),finite=True)
    if 'fingertip_error_mm' in d['tracking']:
        tips=np.asarray(d['tracking']['fingertip_error_mm']);assert tips.shape==(266,2,5) and np.isfinite(tips).all() and np.min(tips)>=0
        report['robots'][robot['id']]['source_fingertip_deviation_mm']=dict(mean=float(tips.mean()),p95=float(np.percentile(tips,95)),max=float(tips.max()),measures_grasp_success=False)
report['objects']={}
for obj in scene['objects']:
    mesh=trimesh.load(DEST/Path(obj['mesh_url']).name,force='mesh',process=False);assert np.isfinite(mesh.vertices).all()
    frames=json.loads((DEST/Path(obj['track_url']).name).read_text());errors=[]
    for frame in frames:
        if frame['status']!='observed':assert 'matrix' not in frame;continue
        transform=np.asarray(frame['matrix']);assert np.isfinite(transform).all()
        assert np.allclose(transform[3],[0,0,0,1])
        assert np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-6)
        assert np.linalg.det(transform[:3,:3])>.99999;errors.append(frame['surface_rmse_m'])
    surfaces=obj.get('observed_surface');surface_count=0
    if surfaces:
        surface_blob=(DEST/Path(surfaces['url']).name).read_bytes()
        assert len(surfaces['frames'])==133
        for row in surfaces['frames']:
            vertices=np.frombuffer(surface_blob,dtype='<f4',count=row['vertices']*3,offset=row['positions']).reshape(-1,3)
            faces=np.frombuffer(surface_blob,dtype='<u4',count=row['faces']*3,offset=row['indices'])
            assert np.isfinite(vertices).all()
            assert not len(faces) or faces.max()<len(vertices)
            assert row['colors']+row['vertices']*3<=len(surface_blob)
            surface_count+=bool(row['faces'])
    report['objects'][obj['id']]=dict(valid_pose_frames=len(errors),lost_frames=len(frames)-len(errors),total_frames=len(frames),mean_observed_surface_residual_mm=float(np.mean(errors)*1000),rigid_transforms=True,observed_surface_frames=surface_count,real_world_accuracy_measured=False)
report['videos']={}
for name in ['original.mp4']+[Path(url).name for url in scene['patch_videos'].values()]:
    cap=cv2.VideoCapture(str(DEST/name));fps=cap.get(cv2.CAP_PROP_FPS);n=0
    while True:
        ok,image=cap.read()
        if not ok:break
        n+=1
    cap.release();assert n==266 and abs(fps-30)<1e-5,(name,n,fps);report['videos'][name]=dict(decoded_frames=n,fps=fps)
# Persistent map is causal: no point is visible before its second observation.
memory=scene['spatial_memory'];blob=(DEST/Path(memory['url']).name).read_bytes();n=memory['count']
xyz=np.frombuffer(blob,'<f4',n*3,memory['positions']);birth=np.frombuffer(blob,'<u4',n,memory['first_frames']);death=np.frombuffer(blob,'<u4',n,memory['last_frames'])
assert np.isfinite(xyz).all() and np.all(birth<death) and np.min(birth)>0
counts=[int(np.sum((birth<=f)&(death>f))) for f in memory['source_frame_ids']]
assert counts==memory['active_points']
with h5py.File(manifest['source']) as f:
    assert np.allclose(scene['source_camera']['poses'],f['transforms/camera'][:],atol=1e-7)
report['spatial_memory']=dict(total_voxel_versions=n,final_active_points=counts[-1],causal_birth_and_retirement=True,source_camera_unchanged=True,estimated_geometry=True)
arm_report=json.loads((DEST/'arm-view-validation.json').read_text())
assert set(arm_report)==set(scene['patch_videos'])
for row in arm_report.values():assert row['native_mesh_binding_verified'] and row['native_arm_meshes']>0 and row['robot_pixels_min']>0
report['robot_arm_composites']=arm_report
audit=json.loads((DEST/'hand-motion-audit.json').read_text());assert audit['h2_a3_hand_angle_max_difference_rad']==0 and audit['g1_hand_angle_max_difference_rad']==0 and len(audit['finger_motion'])==10
assert all(row['maximum_joint_range_deg']>1 and row['curl_correlation']>.5 for row in audit['finger_motion'])
report['hand_motion']=dict(frames=audit['frames'],frozen=False,verified_robots=audit['verified_robots'],shared_hand_motion=True,curl_correlation_range=[min(r['curl_correlation'] for r in audit['finger_motion']),max(r['curl_correlation'] for r in audit['finger_motion'])],grasp_success_validated=False)
report['semantic']=dict(model=scene['semantic'].get('model'),phases=len(scene['semantic'].get('phases',[])),estimates=True)
report['tactile']=dict(type=scene['tactile']['status'],force_available=False)
report['limitations']=['No dynamics or successful grasp validation','Hidden object surfaces are inferred','Depth and object poses are estimates, not measured geometry','Symmetric-object orientation may be ambiguous','Robot-hand image gaps are synthetically inpainted']
(DEST/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
