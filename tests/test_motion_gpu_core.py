"""Independent CPU checks of the packaged CUDA-compatible numerical operators."""
import numpy as np
import pytest

torch=pytest.importorskip('torch')
mj=pytest.importorskip('mujoco')
from arenalab_motion.kinematics import KinTree
from arenalab_motion.block_qp import BlockCR
from arenalab_motion.stable_ipm import StableWindowIPM


def test_native_fk_and_point_jacobian_match_mujoco():
    model=mj.MjModel.from_xml_string('''<mujoco><worldbody><body name="root" pos="0 0 1"><freejoint/><geom type="box" size=".1 .1 .1" mass="1"/>
      <body name="arm" pos=".2 0 .3" quat=".9238795 0 .3826834 0"><joint type="hinge" axis="0 1 0" pos=".02 0 0"/><geom type="box" size=".1 .02 .02" mass="1"/>
      <body name="hand" pos=".3 0 0"><joint type="hinge" axis="1 0 0"/><geom type="box" size=".05 .02 .02" mass="1"/></body></body></body></worldbody></mujoco>''')
    qs=np.tile(model.qpos0,(4,1));qs[:,7:]=np.array([[0,0],[.4,-.2],[-.7,.5],[1.2,-1.]])
    qs[:,:3]+=np.array([[0,0,0],[.1,.2,-.1],[-.3,.1,.2],[.2,-.1,.3]])
    tree=KinTree(model,'cpu',torch.float64);K=tree.fk(torch.as_tensor(qs));body=model.body('hand').id
    point=K['xpos'][:,body,None]+(K['xmat'][:,body,None]@torch.tensor([.02,.01,.03],dtype=torch.float64)[None,:,None])[...,0]
    Jp,Jr=tree.jac(K,point,torch.tensor([body]),rot=True)
    data=mj.MjData(model)
    for i,q in enumerate(qs):
        data.qpos[:]=q;mj.mj_forward(model,data)
        np.testing.assert_allclose(K['xpos'][i],data.xpos,atol=1e-12)
        np.testing.assert_allclose(K['xmat'][i],data.xmat.reshape(-1,3,3),atol=1e-12)
        jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
        mj.mj_jac(model,data,jp,jr,point[i,0].numpy(),body)
        np.testing.assert_allclose(Jp[i,0],jp,atol=1e-12)
        np.testing.assert_allclose(Jr[i,0],jr,atol=1e-12)


@pytest.mark.parametrize('n',[1,3,8,13])
def test_block_cr_matches_independent_dense_solve(n):
    generator=torch.Generator().manual_seed(18)
    B,d=2,3
    U=torch.randn(B,n-1,d,d,generator=generator,dtype=torch.float64)*.03
    A=torch.eye(d,dtype=torch.float64).expand(B,n,d,d).clone()*2
    rhs=torch.randn(B,n,d,generator=generator,dtype=torch.float64)
    dense=torch.zeros(B,n*d,n*d,dtype=torch.float64)
    for j in range(n):
        dense[:,j*d:(j+1)*d,j*d:(j+1)*d]=A[:,j]
        if j<n-1:
            dense[:,j*d:(j+1)*d,(j+1)*d:(j+2)*d]=U[:,j]
            dense[:,(j+1)*d:(j+2)*d,j*d:(j+1)*d]=U[:,j].transpose(-1,-2)
    actual=BlockCR(A,U).solve(rhs)
    expected=torch.linalg.solve(dense,rhs.flatten(1)[...,None])[...,0].reshape_as(rhs)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)


def test_ipm_fixed_variables_and_constant_row_preserve_feasible_qp():
    dtype=torch.float64;B,T,d=1,3,2
    H=torch.eye(d,dtype=dtype).expand(B,T,d,d).clone()
    g=torch.tensor([-.5,.4],dtype=dtype).expand(B,T,d).clone()
    C=torch.zeros(B,T,1,d,dtype=dtype);lower=torch.full((B,T,1),-torch.inf,dtype=dtype);upper=torch.ones_like(lower)
    lo=torch.tensor([-1.,0.],dtype=dtype).expand(B,T,d).clone();hi=torch.tensor([1.,0.],dtype=dtype).expand(B,T,d).clone()
    qp=StableWindowIPM(H,g,C,lower,upper,lo,hi,torch.zeros(d,dtype=dtype))
    for _ in range(24):qp.step()
    assert not bool(qp.factor_failed.any())
    expected=torch.tensor([.5,0.],dtype=dtype).expand(B,T,d)
    torch.testing.assert_close(qp.result(),expected,atol=2e-5,rtol=0)
    assert float(qp.residuals()[0].max())<2e-5
    bad=StableWindowIPM(H,g,C,torch.ones_like(lower),upper,lo,hi,torch.zeros(d,dtype=dtype))
    assert bool(bad.factor_failed.all()),'An infeasible true zero row must not be normalized into an artificial constraint'
