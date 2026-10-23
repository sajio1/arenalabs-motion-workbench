"""Capability checks use actual registered input artifacts, not dataset labels."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import h5py
import numpy as np
from arenalab_workbench.source_requirements import input_requirements, observed_hand_stream


def hand_hdf(path, frames=5):
    with h5py.File(path,'w') as f:
        poses=np.repeat(np.eye(4)[None],frames,0)
        f['transforms/camera']=poses;f['camera/intrinsic']=np.eye(3)
        for side in ('left','right'):
            names=[side+'Hand']+[side+'Thumb'+p for p in ('Knuckle','IntermediateBase','IntermediateTip','Tip')]
            names += [side+finger+'Finger'+p for finger in ('Index','Middle','Ring','Little') for p in ('Knuckle','IntermediateBase','IntermediateTip','Tip')]
            for name in names:
                f['transforms/'+name]=poses;f['confidences/'+name]=np.ones(frames)


class SourceOutputRequirementsTests(unittest.TestCase):
    def requirements(self, root, public, row, upload=None, grippers=None):
        config=root/'viewer.json';config.write_text(json.dumps({'clips':[row]}))
        with ExitStack() as stack:
            for target,value in [('height_registry.default_id',None),('height_registry.variant',None),('height_registry.public_entry',public),('dataset_io.storage.entry',upload),('gripper_registry.sources',grippers or {})]:
                stack.enter_context(patch('arenalab_workbench.'+target,return_value=value))
            return input_requirements(public['id'],config_path=config)

    def test_hdf_requires_both_complete_hands_and_current_window(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'hands.hdf5';hand_hdf(path)
            row={'source':str(path),'frames':3,'source_window':{'start_frame':1}}
            self.assertTrue(observed_hand_stream(row))
            self.assertFalse(observed_hand_stream(dict(row,frames=5)))
            with h5py.File(path,'a') as f:
                del f['camera/intrinsic'];f['camera/intrinsic']=np.repeat(np.eye(3)[None],5,0)
            self.assertFalse(observed_hand_stream(row))
            with h5py.File(path,'a') as f:
                del f['camera/intrinsic'];f['camera/intrinsic']=np.eye(3)
                del f['confidences/leftHand'];f['confidences/leftHand']=np.ones(1)
            self.assertFalse(observed_hand_stream(row))
            with h5py.File(path,'a') as f:
                del f['confidences/leftHand'];f['confidences/leftHand']=np.ones(5)
            with h5py.File(path,'a') as f:del f['transforms/rightLittleFingerTip']
            self.assertFalse(observed_hand_stream(row))
            with h5py.File(path,'a') as f:f['transforms/rightLittleFingerTip']=np.zeros((5,3))
            self.assertFalse(observed_hand_stream(row))
            path.unlink();self.assertFalse(observed_hand_stream(row))

    def test_egodex_legacy_default_hand_and_disappearing_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'hands.hdf5';hand_hdf(source)
            video=root/'camera.mp4';video.touch()
            public={'id':'ego','dataset':'EgoDex'} # Existing catalog omits explicit hand.
            row={'id':'ego','source':str(source),'frames':3,'fps':30,'video':{'path':str(video)}}
            caps=self.requirements(root,public,row)['output_capabilities']
            self.assertEqual(caps['available_outputs'],['trajectory','point_cloud','contact','inpaint_fov'])
            with h5py.File(source,'a') as f:del f['transforms/leftHand']
            caps=self.requirements(root,public,row)['output_capabilities']
            self.assertEqual(caps['available_outputs'],['trajectory','point_cloud'])
            video.unlink()
            self.assertEqual(self.requirements(root,public,row)['output_capabilities']['available_outputs'],['trajectory'])

    def test_hdf_missing_camera_does_not_offer_point_cloud(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'hands.hdf5';hand_hdf(source)
            video=root/'camera.mp4';video.touch()
            public={'id':'ego','dataset':'EgoDex','end_effector':'wuji'}
            row={'id':'ego','source':str(source),'frames':3,'fps':30,'video':{'path':str(video)}}
            with h5py.File(source,'a') as f:del f['camera/intrinsic']
            self.assertEqual(self.requirements(root,public,row)['output_capabilities']['available_outputs'],['trajectory'])

    def test_hiphi_and_grail_preview_only_have_trajectory(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'motion.npz';np.savez(source,poses=np.zeros((3,55,3)))
            preview=root/'preview.mp4';preview.touch()
            for dataset in ('HiPhi','Real robot'):
                with self.subTest(dataset=dataset):
                    public={'id':'clip','dataset':dataset,'end_effector':'none','video':{'url':str(preview)}}
                    row={'id':'clip','source':str(source),'frames':3,'fps':30,'video':None}
                    record={'public':public,'source_row':row,'manifest':{}}
                    if dataset=='HiPhi':record['motion_source']={'path':str(source),'family':'smplx','format':'smpl_parameters'}
                    self.assertEqual(self.requirements(root,public,row,upload=record)['output_capabilities']['available_outputs'],['trajectory'])

    def test_supplied_body_plus_video_needs_observed_camera_arrays(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'motion.npz';np.savez(source,poses=np.zeros((3,55,3)))
            video=root/'camera.mp4';video.touch()
            public={'id':'body','dataset':'Uploaded','end_effector':'none','uploaded':True}
            row={'id':'body','source':str(source),'frames':3,'fps':30,'video':{'path':str(video)}}
            upload={'public':public,'source_row':row,'motion_source':{'path':str(source),'family':'smplx'},'manifest':{}}
            self.assertEqual(self.requirements(root,public,row,upload)['output_capabilities']['available_outputs'],['trajectory'])
            observed=root/'observed.npz';np.savez(observed,camera=np.repeat(np.eye(4)[None],3,0),intrinsic=np.eye(3));row['observations_source']=str(observed)
            self.assertEqual(self.requirements(root,public,row,upload)['output_capabilities']['available_outputs'],['trajectory','point_cloud'])
            np.savez(observed,camera=np.zeros((3,3)),intrinsic=np.eye(3))
            self.assertEqual(self.requirements(root,public,row,upload)['output_capabilities']['available_outputs'],['trajectory'])
            observed.unlink()
            self.assertEqual(self.requirements(root,public,row,upload)['output_capabilities']['available_outputs'],['trajectory'])

    def test_gripper_uses_actual_head_video_and_relative_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);work=root/'gripper';work.mkdir();video=work/'head.mp4';video.touch()
            np.savez(work/'source.npz',camera=np.repeat(np.eye(4)[None],3,0))
            (work/'source.json').write_text(json.dumps({'head_video':'head.mp4'}))
            public={'id':'grip','dataset':'Real robot','end_effector':'robotiq-2f-140','video':{'url':'preview.mp4'}}
            row={'id':'grip'};grippers={'grip':{'work':str(work)}}
            caps=self.requirements(root,public,row,grippers=grippers)['output_capabilities']
            self.assertEqual(caps['available_outputs'],['trajectory','point_cloud','inpaint_fov'])
            video.unlink()
            self.assertEqual(self.requirements(root,public,row,grippers=grippers)['output_capabilities']['available_outputs'],['trajectory'])
            (work/'source.json').write_text(json.dumps({'head_video':None}))
            self.assertEqual(self.requirements(root,public,row,grippers=grippers)['output_capabilities']['available_outputs'],['trajectory'])


if __name__=='__main__':unittest.main()
