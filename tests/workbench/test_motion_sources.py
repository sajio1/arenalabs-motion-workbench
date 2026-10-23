"""Source adapters preserve measured/fitted body geometry without private assets."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from arenalab_workbench import motion_sources as m


def fixture(root, family='smplx'):
    count = m.FAMILY_JOINTS[family]
    rest = np.zeros((count, 3))
    rest[:22] = [[.2,1.,.1],[.1,.9,.1],[.3,.9,.1],[.2,1.1,.1],[.1,.5,.1],[.3,.5,.1],
        [.2,1.3,.1],[.1,.08,.1],[.3,.08,.1],[.2,1.5,.1],[.1,.03,.2],[.3,.03,.2],
        [.2,1.6,.1],[0.,1.5,.1],[.4,1.5,.1],[.2,1.8,.1],[-.1,1.5,.1],[.5,1.5,.1],
        [-.4,1.5,.1],[.8,1.5,.1],[-.7,1.5,.1],[1.1,1.5,.1]]
    parents = np.r_[m.PARENTS, np.zeros(count-22, dtype=int)]
    if family in {'smplh','smplx'}:
        delta = 3 if family == 'smplx' else 0
        if delta:
            rest[22:25] = rest[15] + [[0,-.05,.02],[-.02,0,.02],[.02,0,.02]]
            parents[22:25] = 15
        for start, wrist, sign in [(22+delta,20,-1),(37+delta,21,1)]:
            for finger in range(5):
                for joint in range(3):
                    idx = start + finger*3 + joint
                    rest[idx] = rest[wrist] + [sign*(.08+.03*joint), 0., .04-.02*finger]
                    parents[idx] = wrist if joint == 0 else idx-1
    else:
        rest[22:] = rest[20:22] + [[-.1,0,0],[.1,0,0]]; parents[22:] = [20,21]
    vertices = np.repeat(rest, 3, axis=0) + np.tile([[-.003,0,0],[.003,0,0],[0,0,.003]],(count,1))
    weights = np.repeat(np.eye(count),3,axis=0)
    regressor = weights.T/3
    # Keep posedirs nonzero so the implementation cannot accidentally omit it.
    posedirs = np.zeros((len(vertices),3,(count-1)*9)); posedirs[:,2,0] = .003
    np.savez(root/'model.npz',v_template=vertices,shapedirs=np.zeros((len(vertices),3,2)),
             J_regressor=regressor,kintree_table=np.stack([parents,np.arange(count)]),
             weights=weights,posedirs=posedirs,f=np.arange(len(vertices)).reshape(-1,3))
    poses = np.zeros((9,count,3)); poses[:,0,2] = np.linspace(.1,.3,9)
    poses[:,16,1] = .2
    trans = np.tile([1.,.1,-.3],(9,1)); trans[:,0] += np.arange(9)*.01
    np.savez(root/'motion.npz',poses=poses,trans=trans,betas=np.zeros(2),model_type=np.array(family),
             frame_time=np.array(.011111),fitted_truth_label=np.array('synthetic'))
    spec = dict(format='smpl_parameters', family=family,path=str(root/'motion.npz'),
                world_up='y',units='m',evidence='synthetic',floor_z_m=0.,stride=3)
    if family == 'smpl':spec['wrist_axes_local']=np.tile(np.eye(3),(2,1,1)).tolist()
    return spec, rest


class MotionSourcesTests(unittest.TestCase):
    def test_route_is_evidence_based_and_invalid_body_never_falls_back(self):
        self.assertEqual(m.source_route({})['generated_human_model'],'smplh')
        self.assertEqual(m.source_route({'motion_source':{'format':'smpl_parameters','family':'smplx'}})['route'],'supplied-body')
        with self.assertRaises(ValueError):m.source_route({'motion_source':{'format':'bad'}})

    def test_full_x_preserves_trans_clock_geometry_and_no_egoallo(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);spec,_=fixture(root);before=m.sha256(root/'motion.npz')
            with patch('arenalab_workbench.body_inference.run',side_effect=AssertionError('must bypass reconstruction')):
                out=m.prepare_supplied_source(spec,root/'out',model_path=root/'model.npz')
            d=dict(np.load(out/'native_source.npz'));mesh=dict(np.load(out/'source_mesh.npz'))
            raw=dict(np.load(root/'motion.npz'));model=dict(np.load(root/'model.npz'))
            rest=model['J_regressor']@model['v_template']
            take=np.array([0,3,6]);np.testing.assert_array_equal(d['timestamps_s'],take*.011111)
            np.testing.assert_array_equal(d['source_frame_indices'],take)
            np.testing.assert_allclose(d['full_body_position_yup_m'][:,0],raw['trans'][take]+rest[0])
            np.testing.assert_allclose(d['six_position_zup_m'],d['full_body_position_yup_m'][:,m.ROLES]@m.C)
            np.testing.assert_allclose(d['reference_world_R_zup'],m.C.T@d['full_body_rotation_yup'][:,m.ROLES]@m.C)
            # Root-weighted vertices have a direct independent closed form.
            root_r=Rotation.from_rotvec(raw['poses'][take,0]).as_matrix()
            expected=(np.einsum('tij,vj->tvi',root_r,model['v_template'][:3]-rest[0])+raw['trans'][take,None]+rest[0])@m.C
            np.testing.assert_allclose(mesh['vertices_world_m'][:,:3],expected,atol=1e-7)
            report=json.loads((out/'source_provenance.json').read_text())
            self.assertEqual(report['human_model'],'smplx');self.assertFalse(report['source_completion'])
            self.assertEqual(report['body_encoder_joints'],22);self.assertEqual(before,m.sha256(root/'motion.npz'))
            self.assertFalse((out/'smplh_motion.npz').exists())
            with self.assertRaises(FileExistsError):m.prepare_supplied_source(spec,out,model_path=root/'model.npz')

    def test_native_h_decoded_keeps_world_local_frame_semantics(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);spec,_=fixture(root,'smplh');raw=m._npz(root/'motion.npz')
            _,_,take=m._clock(raw,spec)
            joints,r,vertices,_,_,parents,_,faces,beta,_=m._decode(raw,spec,take,root/'model.npz')
            # Native H export stores world Z-up, model local Y-up.
            np.savez(root/'decoded.npz',joints=joints@m.C,global_rotation=m.C.T@r,vertices=vertices@m.C,
                     faces=faces,parents=parents,betas=beta,human_model=np.array('smplh'),timestamps_s=np.arange(3)*.033333)
            decoded=dict(spec,format='smpl_decoded',path=str(root/'decoded.npz'),world_up='z',stride=1)
            out=m.prepare_supplied_source(decoded,root/'out',model_path=root/'model.npz')
            with np.load(out/'native_source.npz') as d:
                np.testing.assert_allclose(d['full_body_position_yup_m'],joints[:,:22],atol=1e-8)
                np.testing.assert_allclose(d['full_body_rotation_yup'],r[:,:22],atol=1e-8)

    def test_motion_only_job_keeps_recording_identity_without_observations(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);spec,_=fixture(root)
            (root/'input_manifest.json').write_text(json.dumps(dict(
                clip='supplied-clip',dataset='HiPhi',video=None,end_effector='none',motion_source=spec)))
            out=m.prepare_body_source(root,root/'normalized',model_paths={'smplx':root/'model.npz'})
            manifest=json.loads((out/'input_manifest.json').read_text())
            self.assertEqual(manifest['clip'],'supplied-clip')
            self.assertEqual(manifest['dataset'],'HiPhi')
            self.assertEqual(manifest['end_effector'],'none')
            self.assertFalse(manifest['observed_source_sha256'])

    def test_does_not_drop_nonzero_extra_poses_and_rejects_family_lie(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);spec,_=fixture(root);raw=m._npz(root/'motion.npz')
            raw['poses'][:,26,2]=.5;np.savez(root/'motion.npz',**raw)
            m.prepare_supplied_source(spec,root/'out',model_path=root/'model.npz')
            with self.assertRaisesRegex(ValueError,'family'):
                m.prepare_supplied_source(dict(spec,family='smplh'),root/'bad',model_path=root/'model.npz')

    def test_unsupported_rate_or_missing_mcp_never_fabricates(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);spec,_=fixture(root,'smpl')
            with self.assertRaisesRegex(ValueError,'30 Hz'):
                m.prepare_supplied_source(dict(spec,stride=1),root/'bad',model_path=root/'model.npz')
            spec.pop('wrist_axes_local')
            with self.assertRaisesRegex(ValueError,'MCP'):
                m.prepare_supplied_source(spec,root/'bad',model_path=root/'model.npz')

    def test_upload_cannot_select_model_paths_or_escape_package(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);spec,_=fixture(root);spec['path']='motion.npz'
            manifest={'motion_source':spec}
            self.assertEqual(m.validate_body_attachment(root,manifest)['family'],'smplx')
            with self.assertRaisesRegex(ValueError,'configuration'):
                m.validate_body_attachment(root,{'motion_source':dict(spec,model_path='/etc/passwd')})
            with self.assertRaisesRegex(ValueError,'inside'):
                m.validate_body_attachment(root,{'motion_source':dict(spec,path='../motion.npz')})

    def test_three_point_hands_only_must_not_invent_head(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            np.savez(root/'observations.npz',wrist_position=np.ones((3,2,3)),timestamps_s=np.arange(3)/30)
            with self.assertRaises(KeyError):m.validate_three_point_observations(root)

    def test_original_observations_are_preserved_separately_from_body(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);spec,_=fixture(root);raw=m._npz(root/'motion.npz')
            times,_,take=m._clock(raw,spec)
            work=root/'work';work.mkdir();(work/'operation-accuracy').mkdir()
            wrist=np.zeros((len(times),2,3));wrist[...,0]=8.
            forward=np.broadcast_to([1.,0.,0.],wrist.shape);normal=np.broadcast_to([0.,0.,1.],wrist.shape)
            np.savez(work/'observations.npz',wrist_position=wrist,palm_forward=forward,palm_normal=normal,timestamps_s=times)
            np.savez(work/'operation-accuracy/source-targets.npz',points=wrist[:,:,None])
            np.savez(work/'source_hands.npz',points=np.ones((len(times),2,21,3)))
            (work/'input_manifest.json').write_text(json.dumps({'clip':'original','dataset':'fixture','end_effector':'wuji'}))
            out=m.prepare_supplied_source(spec,root/'out',model_path=root/'model.npz',observed_work=work)
            for name in ['observations.npz','source_hands.npz','operation-accuracy/source-targets.npz']:
                self.assertEqual((work/name).read_bytes(),(out/name).read_bytes())
            with np.load(out/'native_source.npz') as d:
                np.testing.assert_array_equal(d['wrist_position_world_m'],wrist)
                np.testing.assert_array_equal(d['six_position_zup_m'][:,2:4],wrist)
                self.assertFalse(np.allclose(d['source_body_position_world_m'][:,[20,21]],wrist))

    def test_motion_only_upload_needs_no_fabricated_video_or_camera(self):
        from arenalab_workbench.dataset_io.ingest import validate_normalized,prepare_body
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);spec,_=fixture(root);spec['path']='motion.npz'
            (root/'recording.json').write_text(json.dumps(dict(schema='arenalabs.recording.v1',fps=30,units='m',world_up='z',end_effector='none',motion_source=spec)))
            manifest,arrays,video=validate_normalized(root)
            self.assertIsNone(video);self.assertNotIn('camera',arrays)
            np.savez(root/'normalized.npz',**arrays)
            record=dict(public={'id':'test','end_effector':'none'},source_row={'source':str(root/'normalized.npz'),'frames':3},manifest=manifest)
            work=root/'work';prepare_body(record,work)
            self.assertFalse((work/'observations.npz').exists())
            self.assertEqual(m.source_route(json.loads((work/'input_manifest.json').read_text()))['route'],'supplied-body')

    def test_motion_only_registry_import_skips_video_and_reconstruction(self):
        from contextlib import nullcontext
        import zipfile
        from arenalab_workbench.dataset_io import routes,storage
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);package=root/'package';package.mkdir();spec,_=fixture(package);spec['path']='motion.npz'
            (package/'recording.json').write_text(json.dumps(dict(schema='arenalabs.recording.v1',fps=30,units='m',world_up='z',end_effector='none',motion_source=spec)))
            archive=root/'motion.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.write(package/'motion.npz','motion.npz');z.write(package/'recording.json','recording.json')
            with patch.object(storage,'ROOT',root/'work'),patch.object(storage,'STORE',root/'store'),patch('arenalab_workbench.gpu_scheduler.resource_lock',return_value=nullcontext()),patch.object(routes.subprocess,'check_output',side_effect=AssertionError('No video tools for motion-only upload')):
                public=routes.import_recording(archive,'source-only',None)
                record=storage.uploads()['source-only']
            self.assertIsNone(public['video']);self.assertTrue(public['has_supplied_body'])
            self.assertEqual(record['body_adapter'],'supplied-body')
            self.assertEqual(record['motion_source']['family'],'smplx')

    def test_describe_does_not_animate_mesh_or_run_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);spec,_=fixture(root)
            with patch.object(m,'_decode',side_effect=AssertionError('no LBS')):
                result=m.describe_source(spec,model_paths={'smplx':root/'model.npz'})
            self.assertTrue(result['morphology_available']);self.assertEqual(len(result['body_bone_lengths_m']),22)

    def test_supplied_gripper_body_bypasses_egoallo_and_preserves_tcp_clock(self):
        from arenalab_workbench import gripper_body
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);spec,_=fixture(root);raw=m._npz(root/'motion.npz')
            raw['timestamps_s']=50.+np.arange(9)*.011111;np.savez(root/'motion.npz',**raw)
            times,_,_=m._clock(raw,spec);work=root/'gripper';work.mkdir()
            tcp=np.full((3,2,3),.7);camera=np.repeat(np.eye(4)[None],3,0)
            tool=np.broadcast_to(np.eye(3),(3,2,3,3))
            np.savez(work/'source.npz',camera=camera,tcp_zup=tcp,tool_R_zup=tool,floor_z=np.array(0.),timestamps_s=times)
            (work/'source.json').write_text(json.dumps(dict(motion_source=spec,dataset='Uploaded',target_reference='Observed TCP')))
            with patch.dict('os.environ',{'ARENA_SMPLX_MODEL':str(root/'model.npz')}),patch('arenalab_workbench.body_inference.run',side_effect=AssertionError('Supplied body must not invoke EgoAllo')):
                body=gripper_body.run(work)
                self.assertEqual(gripper_body.run(work),body)
            receipt=json.loads((body/'prior-ready.json').read_text())
            self.assertEqual(receipt['human_model'],'smplx');self.assertEqual(receipt['body_route'],'supplied_body')
            self.assertFalse((body/'model.npz').exists());self.assertFalse((body/'smplh_motion.npz').exists())
            with np.load(Path(receipt['normalized_source'])/'observations.npz') as observed:
                np.testing.assert_array_equal(observed['wrist_position'],tcp)
                np.testing.assert_array_equal(observed['timestamps_s'],times)
            with np.load(Path(receipt['normalized_source'])/'native_source.npz') as native:
                np.testing.assert_array_equal(native['wrist_position_world_m'],tcp)
            changed=dict(spec,evidence='fitted');meta=json.loads((work/'source.json').read_text());meta['motion_source']=changed;(work/'source.json').write_text(json.dumps(meta))
            self.assertNotEqual(gripper_body.body_path(work),body)

    def test_registered_observed_head_adapter_does_not_fabricate_a_camera(self):
        from arenalab_workbench.dataset_io.ingest import prepare_body
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);n=4;wrist=np.full((n,2,3),.3)
            np.savez(root/'observed.npz',wrist_position=wrist,palm_position=wrist,
                     palm_forward=np.broadcast_to([1.,0.,0.],wrist.shape),palm_normal=np.broadcast_to([0.,0.,1.],wrist.shape),
                     head_position=np.ones((n+1,3)),head_rotation=np.broadcast_to(np.eye(3),(n+1,3,3)),timestamps_s=5.+np.arange(n)/30,floor_z=np.array(0.))
            record=dict(body_adapter='head-wrists-v1',public=dict(id='sim',dataset='Real robot',end_effector='none'),source_row=dict(source=str(root/'observed.npz'),frames=n,video=None),manifest=dict(head_reference='Fixed source robot head marker',source_resampling='Existing 25 to 30Hz source adapter'))
            prepare_body(record,root/'work')
            self.assertEqual((root/'observed.npz').read_bytes(),(root/'work/observations.npz').read_bytes())
            manifest=json.loads((root/'work/input_manifest.json').read_text())
            self.assertEqual(manifest['human_model'],'smplh');self.assertEqual(manifest['cpf_policy'],'Fixed source robot head marker')
            self.assertNotIn('motion_source',manifest)
            with np.load(root/'work/operation-accuracy/source-targets.npz') as d:np.testing.assert_array_equal(d['points'],wrist[:,:,None])

if __name__=='__main__':unittest.main()
