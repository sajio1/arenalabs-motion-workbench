"""Portable body-family and no-transfer regression checks; no private weights."""
import hashlib,json,os,tempfile,unittest,shutil
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from arenalab_workbench.emit_smplh import export,canonical_names
from arenalab_workbench.human_reference import SCHEMA,requested_family,family_for_work,load_reference
from greenwich_motion_sdk.motion import MotionClip

class HumanReferenceTests(unittest.TestCase):
 def make_native_fixture(self,root):
  # A translated, rotated articulated fixture checks the world/local basis,
  # root offset and H topology without assuming an identity pose.
  n,j=3,52;parents=np.r_[-1,np.arange(51)]
  rest=np.stack([np.arange(j)*.005,np.arange(j)*.01,np.zeros(j)],1).astype('f4')
  rest+=np.array([.3,.7,-.2],dtype='f4')
  shape=np.zeros((j,3,2),dtype='f4');shape[:,0,0]=np.linspace(0,.01,j)
  beta=np.array([.4,-.2],dtype='f4');shaped=rest+np.einsum('vck,k->vc',shape,beta)
  local=Rotation.from_euler('z',np.linspace(-.1,.1,n*j)[:,None]).as_matrix().reshape(n,j,3,3)
  global_r=np.zeros_like(local);global_r[:,0]=Rotation.from_euler('x',np.array([.2,.25,.3])[:,None]).as_matrix()
  world=np.zeros((n,j,3));world[:,0]=np.array([[2,3,4],[2.1,3.2,4.1],[2.2,3.4,4.2]])
  for k in range(1,j):
   global_r[:,k]=global_r[:,parents[k]]@local[:,k]
   world[:,k]=world[:,parents[k]]+np.einsum('tij,j->ti',global_r[:,parents[k]],shaped[k]-shaped[parents[k]])
  q=Rotation.from_matrix(global_r.reshape(-1,3,3)).as_quat().reshape(n,j,4)[...,[3,0,1,2]]
  lq=Rotation.from_matrix(local.reshape(-1,3,3)).as_quat().reshape(n,j,4)[...,[3,0,1,2]]
  poses=np.concatenate([q,world],-1).astype('f4');vertices=world.astype('f4')+.007
  np.savez(root/'model.npz',v_template=rest,shapedirs=shape,J_regressor=np.eye(j,dtype='f4'),kintree_table=np.stack([parents,np.arange(j)]),weights=np.eye(j,dtype='f4'))
  np.savez(root/'egoallo_smplh.npz',root_pose=poses[:,0],joint_poses=poses[:,1:],local_quats=lq[:,1:].astype('f4'),vertices=vertices,faces=np.array([[0,1,2]],dtype='i4'),betas=np.tile(beta,(n,1)),timestamps_s=np.arange(n)/30)
  np.savez(root/'observations.npz',wrist_position=world[:,[20,21]])
  (root/'input_manifest.json').write_text(json.dumps(dict(dataset='Nymeria',clip='fixture',human_model='smplh')))
  return poses,vertices

 def test_native_export_preserves_geometry_world_and_clock(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);poses,vertices=self.make_native_fixture(root)
   before=hashlib.sha256((root/'egoallo_smplh.npz').read_bytes()).hexdigest()
   report=export(root);human=load_reference(root);clip=MotionClip.load(root/'canonical.npz')
   np.testing.assert_array_equal(human['vertices'],vertices)
   np.testing.assert_array_equal(human['joints'],poses[...,4:])
   np.testing.assert_array_equal(human['timestamps_s'],np.arange(3)/30)
   C=np.array([[0,1,0],[0,0,1],[1,0,0]])
   np.testing.assert_allclose(clip.world_position_cm,poses[...,4:]@C.T*100,atol=1e-5)
   self.assertEqual(report['transfer'],'none');self.assertTrue(report['source_unchanged'])
   self.assertFalse((root/'smplx_motion.npz').exists())
   self.assertEqual(before,hashlib.sha256((root/'egoallo_smplh.npz').read_bytes()).hexdigest())
   with self.assertRaises(FileExistsError):export(root)

 def test_unfixed_shape_is_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);self.make_native_fixture(root)
   with np.load(root/'egoallo_smplh.npz') as d:values={k:d[k] for k in d.files}
   values['betas'][1,0]+=.1;np.savez(root/'egoallo_smplh.npz',**values)
   with self.assertRaises(ValueError):export(root)
   self.assertFalse((root/'canonical.npz').exists())

 def test_existing_x_is_never_reinterpreted_as_h(self):
  with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'ARENA_NYMERIA_HUMAN_MODEL':'smplh'}):
   root=Path(tmp);(root/'smplx_motion.npz').touch()
   with self.assertRaisesRegex(ValueError,'retired'):family_for_work(root)

 def test_family_mismatch_is_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'human_reference.json').write_text(json.dumps(dict(schema=SCHEMA,family='smplh')))
   np.savez(root/'smplh_motion.npz',human_model=np.array('smplx'))
   with self.assertRaises(ValueError):load_reference(root)

 def test_retired_manifest_cannot_be_hidden_by_native_descriptor(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   (root/'input_manifest.json').write_text(json.dumps(dict(dataset='Nymeria',human_model='smplx')))
   (root/'human_reference.json').write_text(json.dumps(dict(schema=SCHEMA,family='smplh')))
   with self.assertRaisesRegex(ValueError,'Retired'):family_for_work(root)

 def test_retired_descriptor_is_rejected_even_with_native_filename(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'human_reference.json').write_text(json.dumps(dict(schema=SCHEMA,family='smplx')))
   np.savez(root/'smplh_motion.npz',human_model=np.array('smplh'))
   with self.assertRaisesRegex(ValueError,'retired'):load_reference(root)

 def test_input_policy_and_semantic_hand_frames(self):
  with patch.dict(os.environ,{'ARENA_NYMERIA_HUMAN_MODEL':'smplh'}):
   for dataset in ['Nymeria','EgoDex','Uploaded','Real robot']:
    self.assertEqual(requested_family(dataset),'smplh')
  with patch.dict(os.environ,{'ARENA_NYMERIA_HUMAN_MODEL':'smplx'}):
   self.assertEqual(requested_family('Nymeria'),'smplh')
  for name in ['Hips','Head','LeftHand','RightHand','LeftHandMiddle1','RightHandMiddle1']:
   self.assertIn(name,canonical_names())


 def completed_window_fixture(self,root):
  poses,vertices=self.make_native_fixture(root)
  weights=root/'egoallo_checkpoint_april13/checkpoints_3000000/model.safetensors'
  weights.parent.mkdir(parents=True);weights.write_bytes(b'test checkpoint identity only')
  window=root/'windowed-v1';window.mkdir()
  shutil.copy2(root/'egoallo_smplh.npz',window/'egoallo_smplh.npz')
  digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
  (window/'manifest.json').write_text(json.dumps(dict(source_sha256=digest(root/'observations.npz'),code_sha256={})))
  (window/'progress.json').write_text(json.dumps(dict(stage='complete',output_sha256=digest(window/'egoallo_smplh.npz'))))
  return vertices

 def test_completed_windows_export_native_without_x_assets(self):
  from arenalab_workbench import body_inference
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);vertices=self.completed_window_fixture(root);calls=[]
   def run_export(command,**kwargs):
    calls.append(Path(command[1]).name)
    self.assertEqual(calls[-1],'emit_smplh.py')
    self.assertEqual(kwargs['env']['ARENA_EGO_WORKDIR'],str(root))
    export(root);return SimpleNamespace(returncode=0)
   with patch.object(body_inference,'gpu_environment',side_effect=lambda env:env),patch.object(body_inference.subprocess,'run',side_effect=run_export),patch.object(body_inference.subprocess,'Popen',side_effect=AssertionError('Completed windows must not rerun inference')):
    self.assertEqual(body_inference.run(root),root)
   self.assertEqual(calls,['emit_smplh.py'])
   np.testing.assert_array_equal(load_reference(root)['vertices'],vertices)
   identity=json.loads((root/'inference_identity.json').read_text())
   self.assertEqual(identity['human_family'],'smplh')
   self.assertNotIn('smplx_sha256',identity)

 def test_partial_native_export_is_preserved_and_rejected(self):
  from arenalab_workbench import body_inference
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);self.completed_window_fixture(root);export(root)
   before=(root/'canonical.npz').read_bytes();(root/'smplh_motion.npz').unlink()
   with patch.object(body_inference,'gpu_environment',side_effect=lambda env:env),patch.object(body_inference.subprocess,'run',side_effect=AssertionError('Immutable partial exports cannot be overwritten')):
    with self.assertRaisesRegex(RuntimeError,'Incomplete native H export'):body_inference.run(root)
   self.assertEqual((root/'canonical.npz').read_bytes(),before)

if __name__=='__main__':unittest.main(verbosity=2)
