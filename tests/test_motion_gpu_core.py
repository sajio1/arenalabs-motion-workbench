"""Independent CPU checks of the packaged CUDA-compatible numerical operators."""
import numpy as np
import pytest

torch=pytest.importorskip('torch')
mj=pytest.importorskip('mujoco')
from arenalab_motion.kinematics import KinTree
from arenalab_motion.block_qp import BlockCR,pair_blocks
from arenalab_motion.stable_ipm import StableWindowIPM,equilibrate_newton


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
@pytest.mark.parametrize('factorization',['cholesky','pivoted_lu'])
def test_block_cr_matches_independent_dense_solve(n,factorization):
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
    actual=BlockCR(A,U,factorization=factorization).solve(rhs)
    expected=torch.linalg.solve(dense,rhs.flatten(1)[...,None])[...,0].reshape_as(rhs)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)


def test_pivoted_factor_solves_invertible_block_when_cholesky_rejects():
    # Near-singular symmetric Schur blocks can lose positive definiteness in
    # floating point. LU must solve the supplied coefficients, not add a ridge.
    block=torch.tensor([[1.,1.+1e-12],[1.+1e-12,1.]],dtype=torch.float64)
    A=block.reshape(1,1,2,2);U=torch.empty(1,0,2,2,dtype=torch.float64)
    chol=BlockCR(A,U)
    assert bool((chol.info[-1]!=0).any())
    expected=torch.tensor([[[.25,-.5]]],dtype=torch.float64)
    rhs=(A@expected[...,None])[...,0]
    pivoted=BlockCR(A,U,factorization='pivoted_lu')
    assert not bool((pivoted.info[-1]!=0).any())
    actual=pivoted.solve(rhs)
    dense=torch.linalg.solve(block,rhs[0,0])
    torch.testing.assert_close(actual[0,0],dense,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close((A@actual[...,None])[...,0],rhs,atol=1e-15,rtol=1e-15)


def test_ipm_fixed_variables_and_constant_row_preserve_feasible_qp():
    dtype=torch.float64;B,T,d=1,3,2
    H=torch.eye(d,dtype=dtype).expand(B,T,d,d).clone()
    g=torch.tensor([-.5,.4],dtype=dtype).expand(B,T,d).clone()
    C=torch.zeros(B,T,1,d,dtype=dtype);lower=torch.full((B,T,1),-torch.inf,dtype=dtype);upper=torch.ones_like(lower)
    lo=torch.tensor([-1.,0.],dtype=dtype).expand(B,T,d).clone();hi=torch.tensor([1.,0.],dtype=dtype).expand(B,T,d).clone()
    qp=StableWindowIPM(H,g,C,lower,upper,lo,hi,torch.zeros(d,dtype=dtype))
    assert qp.factorization=='pivoted_lu'
    for _ in range(24):qp.step()
    assert not bool(qp.factor_failed.any())
    expected=torch.tensor([.5,0.],dtype=dtype).expand(B,T,d)
    torch.testing.assert_close(qp.result(),expected,atol=2e-5,rtol=0)
    assert float(qp.residuals()[0].max())<2e-5
    bad=StableWindowIPM(H,g,C,torch.ones_like(lower),upper,lo,hi,torch.zeros(d,dtype=dtype))
    assert bool(bad.factor_failed.all()),'An infeasible true zero row must not be normalized into an artificial constraint'


@pytest.mark.parametrize('frames',[3,4,7,12])
@pytest.mark.parametrize('factorization',['cholesky','pivoted_lu'])
def test_newton_equilibration_preserves_full_coupled_linear_system(frames,factorization):
    """Known SPD system under extreme unit changes; no problem regularization."""
    dtype=torch.float64;B,d=2,3
    base=torch.tensor([[2.,.2,-.1],[.2,1.5,.15],[-.1,.15,1.8]],dtype=dtype).expand(B,frames,d,d).clone()
    off1=torch.full((B,frames-1,d),-.12,dtype=dtype)
    off2=torch.full((B,frames-2,d),.04,dtype=dtype)
    units=torch.logspace(-8,8,frames*d,dtype=dtype).reshape(1,frames,d).expand(B,-1,-1).clone()
    units[1]=units[1].flip(0)
    diagonal=base*units[...,None]*units[...,None,:]
    first=off1*units[:,:-1]*units[:,1:]
    second=off2*units[:,:-2]*units[:,2:]
    balanced,b1,b2,scale=equilibrate_newton(diagonal,first,second)
    # Build the independent dense reference in well-conditioned original units.
    reference=torch.zeros(B,frames*d,frames*d,dtype=dtype)
    for t in range(frames):
        reference[:,t*d:(t+1)*d,t*d:(t+1)*d]=base[:,t]
        for delta,off in [(1,off1),(2,off2)]:
            if t+delta<frames:
                block=torch.diag_embed(off[:,t])
                reference[:,t*d:(t+1)*d,(t+delta)*d:(t+delta+1)*d]=block
                reference[:,(t+delta)*d:(t+delta+1)*d,t*d:(t+1)*d]=block
    generator=torch.Generator().manual_seed(43)
    rhs_base=torch.randn(B,frames,d,generator=generator,dtype=dtype)
    rhs=rhs_base*units
    aa,upper=pair_blocks(balanced,b1,b2);factor=BlockCR(aa,upper,factorization=factorization)
    assert all(not bool((info!=0).any()) for info in factor.info)
    transformed_rhs=rhs*scale
    if frames%2:transformed_rhs=torch.cat([transformed_rhs,transformed_rhs.new_zeros(B,1,d)],1)
    actual=factor.solve(transformed_rhs.reshape(B,(frames+1)//2,2*d)).reshape(B,-1,d)[:,:frames]*scale
    expected=torch.linalg.solve(reference,rhs_base.flatten(1)[...,None])[...,0].reshape(B,frames,d)
    # Verify recovered physical variables, including both temporal off-diagonals.
    torch.testing.assert_close(actual*units,expected,atol=2e-14,rtol=2e-14)
    # Congruence is reversible and has not altered any coefficient.
    torch.testing.assert_close(balanced/scale[...,None]/scale[...,None,:],diagonal,atol=0,rtol=5e-16)
    torch.testing.assert_close(b1/scale[:,:-1]/scale[:,1:],first,atol=0,rtol=5e-16)
    torch.testing.assert_close(b2/scale[:,:-2]/scale[:,2:],second,atol=0,rtol=5e-16)


@pytest.mark.parametrize('frames',[3,4,9])
@pytest.mark.parametrize('factorization',['cholesky','pivoted_lu'])
def test_equilibrated_ipm_keeps_original_acceptance_and_frozen_solution(frames,factorization):
    dtype=torch.float64;B,d=2,3
    H=torch.eye(d,dtype=dtype).expand(B,frames,d,d).clone()
    g=torch.tensor([-2.,.3,-.2],dtype=dtype).expand(B,frames,d).clone()
    C=torch.zeros(B,frames,1,d,dtype=dtype);C[...,0,1]=1.
    lower=torch.full((B,frames,1),-torch.inf,dtype=dtype)
    upper=torch.full_like(lower,.15)
    lo=torch.full((B,frames,d),-.5,dtype=dtype);hi=-lo
    # An exact fixed coordinate exercises elimination as well as active bounds.
    lo[1,:,2]=hi[1,:,2]=0.
    qp=StableWindowIPM(H,g,C,lower,upper,lo,hi,torch.tensor([3.,.5,.2],dtype=dtype),factorization=factorization)
    for _ in range(96):qp.step()
    primal,stationarity=qp.residuals();mu=(qp.s*qp.z).sum((1,2))/qp.count
    assert torch.isfinite(qp.result()).all() and not qp.factor_failed.any()
    assert (primal<2e-5).all() and (stationarity<1e-6).all() and (mu<1e-8).all()
    expected=torch.tensor([.5,-.3,.2],dtype=dtype).expand(B,frames,d).clone();expected[1,:,2]=0.
    torch.testing.assert_close(qp.result(),expected,atol=2e-5,rtol=0)
    before=qp.result().clone()
    for _ in range(16):qp.step()
    torch.testing.assert_close(qp.result(),before,atol=0,rtol=0)
    assert not qp.factor_failed.any()
