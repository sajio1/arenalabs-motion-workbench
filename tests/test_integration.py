import json
from pathlib import Path
import numpy as np
import pytest
from greenwich_pipelines.config import Job,robot_rows,ROOT
from greenwich_pipelines.artifacts import inspect_native

def job_file(tmp_path,**updates):
    row=dict(schema='greenwich.job.v1',id='test',pipeline='nymeria',source='input.npz',robots='robots.json',runtime='runtime.json',output='out',fps=30)
    row.update(updates);p=tmp_path/'job.json';p.write_text(json.dumps(row));return p

def test_relative_paths_belong_to_job(tmp_path,monkeypatch):
    p=job_file(tmp_path);monkeypatch.chdir(tmp_path.parent)
    j=Job.load(p);assert j.source==tmp_path/'input.npz' and j.output==tmp_path/'out'

@pytest.mark.parametrize('change',[{'fps':60},{'id':'../bad'},{'window_frames':2},{'policy':{}},{'acceptance':'missing_colon'}])
def test_invalid_job_contract(tmp_path,change):
    with pytest.raises(ValueError):Job.load(job_file(tmp_path,**change))

def test_robot_metadata_cannot_carry_solver_policy(tmp_path):
    p=tmp_path/'robots.json';p.write_text(json.dumps([{'body':'new_robot','xml':'model.xml','solver':{'iterations':3}}]))
    with pytest.raises(ValueError):robot_rows(p)

def sparse_file(tmp_path,n=10):
    p=tmp_path/'sparse.npz';pos=np.zeros((n,4,3));R=np.tile(np.eye(3),(n,4,1,1))
    np.savez(p,position_yup_cm=pos,semantic_rotation_yup=R,foot_contacts=np.ones((n,4)),fps=30.)
    return p

def test_sparse_contract_and_window_tail(tmp_path):
    from greenwich_pipelines.pipelines.nymeria.input import read_source,windows
    source=read_source(sparse_file(tmp_path,11));segments=list(windows(source,5))
    assert [(a,b) for a,b,_ in segments]==[(0,5),(5,10),(10,11)]
    assert segments[-1][2] is None
    assert source['position_yup_cm'].shape==(11,4,3)

def test_corrupt_rotation_rejected(tmp_path):
    from greenwich_pipelines.pipelines.nymeria.input import read_source
    p=sparse_file(tmp_path)
    with np.load(p) as z:d=dict(z)
    d['semantic_rotation_yup'][0,0,0,0]=-1
    np.savez(p,**d)
    with pytest.raises(ValueError):read_source(p)

def test_nonfinite_final_artifact_rejected(tmp_path):
    p=tmp_path/'result.npz';np.savez(p,q=np.zeros((3,2,3)),root_t=np.full((3,3),np.nan),rot6d=np.zeros((3,2,6)))
    with pytest.raises(ValueError):inspect_native(p,'nymeria')

def test_bundled_gento_metadata_and_meshes_resolve(tmp_path):
    import mujoco
    from greenwich_pipelines.pipelines.gento.runner import prepare_metadata
    for row in robot_rows(ROOT/'configs/robots.gento.json'):
        meta=prepare_metadata(row,tmp_path)
        assert Path(meta['source']).exists()
        model=mujoco.MjModel.from_xml_path(meta['model'])
        assert model.nq>7 and len(meta['end_effector_frames'])==2

def test_mixed_joint_fk_matches_saved_mujoco_coordinates():
    import mujoco,torch
    from greenwich_pipelines.pipelines.gento.source import FK
    from scipy.spatial.transform import Rotation
    m=mujoco.MjModel.from_xml_path(str(ROOT/'assets/robots/gento/skye/model.xml'))
    data=mujoco.MjData(m);mujoco.mj_forward(m,data)
    anchor=np.r_[m.qpos0[:2],0.]
    fk=FK(m,'cpu',anchor,dtype=torch.float64)
    x=torch.tensor(np.tile(m.qpos0[7:],(3,1)),dtype=torch.float64)
    pp,rr=fk(x)
    np.testing.assert_allclose(pp.numpy(),np.broadcast_to(data.xpos,pp.shape),atol=1e-9)
    np.testing.assert_allclose(rr.numpy(),np.broadcast_to(data.xmat.reshape(-1,3,3),rr.shape),atol=1e-9)

def test_worker_reports_no_default_quality_certificate(tmp_path,monkeypatch):
    from greenwich_pipelines import worker
    from types import SimpleNamespace
    p=job_file(tmp_path)
    (tmp_path/'input.npz').write_bytes(b'input')
    (tmp_path/'runtime.json').write_text('{}')
    (tmp_path/'robots.json').write_text(json.dumps([{'body':'robot','xml':'model.xml'}]))
    monkeypatch.setattr(worker,'configure',lambda _:None)
    monkeypatch.setattr(worker.importlib,'import_module',lambda _:SimpleNamespace(run=lambda *a:[]))
    result=worker.run(Job.load(p))
    assert result['accepted'] is None and result['quality_status']=='not_evaluated'
    with pytest.raises(FileExistsError):worker.run(Job.load(p))

def test_worker_keeps_failure_for_diagnosis(tmp_path,monkeypatch):
    from greenwich_pipelines import worker
    from types import SimpleNamespace
    p=job_file(tmp_path)
    (tmp_path/'input.npz').write_bytes(b'input')
    (tmp_path/'runtime.json').write_text('{}')
    (tmp_path/'robots.json').write_text(json.dumps([{'body':'robot','xml':'model.xml'}]))
    monkeypatch.setattr(worker,'configure',lambda _:None)
    def fail(*args):raise RuntimeError('failed sample')
    monkeypatch.setattr(worker.importlib,'import_module',lambda _:SimpleNamespace(run=fail))
    with pytest.raises(RuntimeError):worker.run(Job.load(p))
    status=json.loads((tmp_path/'out/status.json').read_text())
    assert status['state']=='failed' and status['message']=='failed sample'

def test_acceptance_cannot_rewrite_saved_motion(tmp_path,monkeypatch):
    from greenwich_pipelines import worker
    from greenwich_pipelines.artifacts import artifact
    from types import SimpleNamespace
    p=job_file(tmp_path,acceptance='custom:evaluate')
    (tmp_path/'input.npz').write_bytes(b'input')
    (tmp_path/'runtime.json').write_text('{}')
    (tmp_path/'robots.json').write_text(json.dumps([{'body':'robot','xml':'model.xml'}]))
    monkeypatch.setattr(worker,'configure',lambda _:None)
    def run(j,*_):
        path=j.output/'result.npz';path.write_bytes(b'original')
        return [artifact(j.output,path,'motion')]
    def evaluate(path):
        (path.parent/'result.npz').write_bytes(b'changed')
        return {'accepted':True}
    monkeypatch.setattr(worker.importlib,'import_module',lambda name:SimpleNamespace(evaluate=evaluate) if name=='custom' else SimpleNamespace(run=run))
    with pytest.raises(RuntimeError,match='changed numerical artifacts'):worker.run(Job.load(p))
