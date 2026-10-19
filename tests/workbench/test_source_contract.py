"""CPU contract checks for sparse source fidelity and native mobile FK."""
import json,unittest,os,tempfile
from pathlib import Path
import numpy as np,mujoco,torch
from scipy.spatial.transform import Rotation
from arenalab_workbench.scene_job import resolve_clip,save
from arenalab_workbench.nymeria_body import prepare
from greenwich_pipelines.pipelines.gento.source import FK

@unittest.skipUnless(os.environ.get("ARENA_PRIVATE_INTEGRATION") == "1", "Requires registered private source and robot fixtures; set ARENA_PRIVATE_INTEGRATION=1 on the compute host")
class ContractTests(unittest.TestCase):
    def test_nymeria_sparse_source(self):
        clip='record_9d3419b9e8a9873ec1_f0016934'
        public,row=resolve_clip(clip)
        self.workspace=tempfile.TemporaryDirectory();self.addCleanup(self.workspace.cleanup);work=Path(self.workspace.name)
        prepare(public,row,work,save)
        obs=np.load(work/'observations.npz');targets=np.load(work/'operation-accuracy/source-targets.npz')['points']
        np.testing.assert_array_equal(obs['wrist_position'],targets[:,:,0])
        self.assertEqual(len(obs['head_position']),len(targets)+1)
        self.assertTrue(all(np.isfinite(obs[k]).all() for k in obs.files))
        self.assertFalse((work/'source_hands.npz').exists())
        self.assertFalse(any(k in obs.files for k in ['pelvis','feet','body_joints']))
    def test_mobile_mixed_joint_fk(self):
        model=mujoco.MjModel.from_xml_path('/opt/arenalab/assets/robots/gento/skye/model.xml')
        random=np.random.default_rng(7)
        x=random.uniform(model.jnt_range[1:,0],model.jnt_range[1:,1],size=(4,model.nq-7))
        xy=random.normal(size=(4,2));yaw=random.normal(size=4);z=.14
        fk=FK(model,'cpu',[0,0,0],dtype=torch.float64)
        p,r=fk(torch.tensor(x));p=p.numpy();r=r.numpy();rot=Rotation.from_euler('z',yaw[:,None]).as_matrix()
        pp=np.einsum('tij,tkj->tki',rot,p)+np.c_[xy,np.full(4,z-model.qpos0[2])][:,None]
        rr=rot[:,None]@r
        for t in range(4):
            d=mujoco.MjData(model);d.qpos[:2]=xy[t];d.qpos[2]=z;d.qpos[3:7]=Rotation.from_matrix(rot[t]).as_quat()[[3,0,1,2]];d.qpos[7:]=x[t];mujoco.mj_forward(model,d)
            np.testing.assert_allclose(pp[t,1:],d.xpos[1:],atol=1e-8)
            np.testing.assert_allclose(rr[t,1:],d.xmat[1:].reshape(-1,3,3),atol=1e-8)

if __name__=='__main__':unittest.main()
