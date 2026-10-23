import unittest
from arenalab_workbench.output_modalities import source_capabilities, resolve_outputs, plan_outputs


class OutputModalitiesTests(unittest.TestCase):
    def test_motion_only_and_preview_only_do_not_enable_cv(self):
        public={'end_effector':'none','video':{'url':'/assets/sim.mp4'}}
        for row in ({'source':'motion.npz','video':None}, {'source':'observed.npz','video':{'path':'sim.mp4'},'metadata':{'preview_video_only':True}}):
            caps=source_capabilities(public,row)
            self.assertEqual(caps['available_outputs'],['trajectory'])
            self.assertEqual(resolve_outputs(None,caps)['effective_outputs'],['trajectory'])
            with self.assertRaisesRegex(ValueError,'OUTPUTS_UNAVAILABLE'):resolve_outputs(['point_cloud'],caps)

    def test_real_camera_is_not_enough_for_contact_or_nohand_fov(self):
        caps=source_capabilities({'end_effector':'none'},{'source':'motion.npz','video':{'path':'camera.mp4'}})
        self.assertEqual(caps['available_outputs'],['trajectory','point_cloud'])
        caps=source_capabilities({'end_effector':'wuji'},{'source':'motion.npz','video':{'path':'camera.mp4'}})
        self.assertEqual(caps['available_outputs'],['trajectory','point_cloud'])

    def test_wuji_requires_actual_landmark_attestation(self):
        caps=source_capabilities({'end_effector':'wuji'},{'source':'hands.hdf5','video':{'path':'camera.mp4'},'hand_landmark_count':21})
        self.assertEqual(caps['available_outputs'],['trajectory','point_cloud','contact','inpaint_fov'])
        self.assertTrue(resolve_outputs(None,caps)['defaulted'])
        self.assertEqual(resolve_outputs(['contact','trajectory'],caps)['effective_outputs'],['trajectory','contact'])

    def test_gripper_fov_available_contact_unavailable(self):
        caps=source_capabilities({'end_effector':'robotiq-2f-140'},{'work':'/test/gripper','video':{'path':'head.mp4'}},gripper=True)
        self.assertEqual(caps['available_outputs'],['trajectory','point_cloud','inpaint_fov'])
        self.assertIn('gripper contact',caps['outputs']['contact']['reason'])

    def test_explicit_missing_camera_calibration_disables_cv(self):
        caps=source_capabilities({'end_effector':'wuji'},{'source':'hands.hdf5','video':{'path':'camera.mp4'},'has_observed_hands':True,'camera_input_registered':False})
        self.assertEqual(caps['available_outputs'],['trajectory'])

    def test_invalid_explicit_outputs_fail_before_dispatch(self):
        caps=source_capabilities({'end_effector':'none'},{'source':'motion.npz'})
        for requested in ([],['trajectory','trajectory'],['unknown'],'trajectory',True,[None],{'trajectory'}):
            with self.subTest(requested=requested),self.assertRaises(ValueError):resolve_outputs(requested,caps)
        with self.assertRaisesRegex(ValueError,'OUTPUTS_UNAVAILABLE'):resolve_outputs(None,source_capabilities({},{}))

    def test_dependency_graph_does_not_compute_unrequested_expensive_stages(self):
        p=plan_outputs(['point_cloud'])
        self.assertTrue(p['scene']);self.assertFalse(p['body']);self.assertFalse(p['retarget']);self.assertFalse(p['render'])
        p=plan_outputs(['contact'])
        self.assertTrue(p['contact']);self.assertFalse(p['scene']);self.assertFalse(p['body']);self.assertFalse(p['retarget']);self.assertFalse(p['proximity'])
        self.assertTrue(plan_outputs(['contact'],point_cloud_available=True)['proximity'])
        p=plan_outputs(['inpaint_fov'])
        self.assertTrue(all(p[k] for k in ['body','retarget','scene','render']))
        self.assertEqual(p['effective_outputs'],['inpaint_fov']);self.assertFalse(p['contact'])

    def test_match_robot_cloud_only_needs_body_but_no_robot_retarget(self):
        p=plan_outputs(['point_cloud'],trajectory_scale='match_robot')
        self.assertTrue(p['body']);self.assertTrue(p['source_body_for_scene_scale']);self.assertTrue(p['scene'])
        self.assertFalse(p['retarget']);self.assertFalse(p['render'])
        self.assertFalse(plan_outputs(['contact'],trajectory_scale='match_robot')['body'])


if __name__=='__main__':unittest.main()
