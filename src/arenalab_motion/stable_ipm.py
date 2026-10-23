"""Mehrotra predictor-corrector QP using the same temporal block factorization.

Two RHS solves share each iteration's factor. Fixed box variables eliminated
algebraically, not relaxed. No convergence/feasibility implied by step count.
"""
import torch
from .block_qp import BlockCR,pair_blocks,temporal_coeff,temporal_mv


def equilibrate_newton(diagonal,off1,off2):
    """Exact diagonal congruence for the current barrier Newton system.

    Objective equilibration is fixed for a QP, while z/s can change by many
    orders of magnitude during IPM. Rebalance the *linear solve*, including
    temporal couplings; recover its original variable after solving. No ridge,
    constraint, objective or convergence tolerance is changed here.
    """
    scale=diagonal.diagonal(dim1=-2,dim2=-1).rsqrt()
    balanced=diagonal*scale[...,None]*scale[...,None,:]
    first=off1*scale[:,:-1]*scale[:,1:]
    second=off2*scale[:,:-2]*scale[:,2:]
    return balanced,first,second,scale


class StableWindowIPM:
    def __init__(self,H,g,C,l,u,lo,hi,w,factorization='pivoted_lu'):
        self.H,self.g,self.originalC=H,g,C;self.l,self.u,self.lo,self.hi=l,u,lo,hi
        self.factorization=factorization
        B,T,d=g.shape;self.shape=B,T,d;self.w2=w*w;self.a,self.b,self.c=temporal_coeff(T,g.device,g.dtype)
        self.fixed=torch.isfinite(lo)&torch.isfinite(hi)&(lo==hi)
        self.fix=torch.where(self.fixed,lo,torch.zeros_like(lo));free=(~self.fixed).to(g.dtype)
        self.scale=(H.diagonal(dim1=-2,dim2=-1)+self.a[None,:,None]*self.w2).clamp_min(1e-8).rsqrt()*free
        self.gs=(g+(H@self.fix[...,None])[...,0]+temporal_mv(self.fix,self.w2))*self.scale
        self.Hs=H*self.scale[...,None]*self.scale[...,None,:]+torch.diag_embed(self.a[None,:,None]*self.w2*self.scale**2+self.fixed.to(g.dtype))
        self.off1=self.b[None,:,None]*self.w2*self.scale[:,:-1]*self.scale[:,1:]
        self.off2=self.c[None,:,None]*self.w2*self.scale[:,:-2]*self.scale[:,2:]
        cs=C*self.scale[:,:,None,:];offset=(C@self.fix[...,None])[...,0]
        raw_norm=cs.norm(dim=-1)
        constant=raw_norm==0
        # Fixed-variable elimination can leave a constant linear row. A true
        # zero row is either redundant or infeasible; it must not enter the
        # barrier with an artificial 1e12 bound from normalization. Do not use
        # a tolerance that would silently discard small nonzero coefficients.
        self.constant_infeasible=(constant & ((offset<l)|(offset>u))).any((1,2))
        norm=raw_norm.clamp_min(1e-12);cs=cs/norm[...,None]
        eye=torch.eye(d,device=g.device,dtype=g.dtype).expand(B,T,d,d)
        finite=torch.cat([torch.isfinite(u)&~constant,torch.isfinite(l)&~constant,torch.isfinite(hi)&~self.fixed,torch.isfinite(lo)&~self.fixed],-1)
        ss=self.scale.clamp_min(1e-30)
        hh=torch.cat([(u-offset)/norm,(offset-l)/norm,(hi-self.fix)/ss,(self.fix-lo)/ss],-1)
        self.G=torch.cat([cs,-cs,eye,-eye],-2)*finite[...,None]
        self.h=torch.where(finite,hh,torch.ones_like(hh));self.live=finite;self.livef=finite.to(g.dtype)
        self.count=self.livef.sum((1,2)).clamp_min(1)
        self.y=torch.zeros_like(g);self.s=torch.maximum(self.h,torch.ones_like(self.h));self.z=self.livef.clone()
        self.eye=eye
        # Reuse the existing rejection flag for presolve infeasibility. It
        # stays on device and survives reset/CUDA graph replay.
        self.factor_failed=self.constant_infeasible.clone()

    def reset(self):
        self.y.zero_();self.s.copy_(torch.maximum(self.h,torch.ones_like(self.h)));self.z.copy_(self.livef);self.factor_failed.copy_(self.constant_infeasible)

    def scaled_Hmv(self,x):
        v=(self.Hs@x[...,None])[...,0]
        v[:,:-1]+=self.off1*x[:,1:];v[:,1:]+=self.off1*x[:,:-1]
        v[:,:-2]+=self.off2*x[:,2:];v[:,2:]+=self.off2*x[:,:-2]
        return v

    @staticmethod
    def alpha(v,dv):
        ratio=torch.where(dv<0,-v/dv.clamp_max(-1e-300),torch.full_like(v,float('inf')))
        return ratio.amin((1,2)).clamp(max=1)

    def step(self):
        rd=self.scaled_Hmv(self.y)+self.gs+(self.G.transpose(-1,-2)@self.z[...,None])[...,0]
        rp=((self.G@self.y[...,None])[...,0]+self.s-self.h)*self.livef
        mu=(self.s*self.z).sum((1,2))/self.count
        # Freeze at the EXACT outer acceptance tolerances, including the
        # ORIGINAL unscaled primal inequalities. An eight-step CUDA replay
        # must not destroy an already acceptable solution by over-iteration.
        primal, stationarity = self.residuals()
        done = (primal < 2e-5) & (stationarity < 1e-6) & (mu < 1e-8)
        done = done & torch.isfinite(primal) & torch.isfinite(stationarity) & torch.isfinite(mu) & ~self.factor_failed
        ratio=self.z/self.s
        D=self.Hs+self.G.transpose(-1,-2)@(self.G*ratio[...,None])+1e-10*self.eye
        # A converged lane may have tiny slacks and a badly conditioned barrier
        # system. Mask BEFORE factorization: multiplying a NaN step by zero
        # afterwards would still corrupt the solution during graph replay.
        D=torch.where(done[:,None,None,None],self.eye,D)
        off1=torch.where(done[:,None,None],torch.zeros_like(self.off1),self.off1)
        off2=torch.where(done[:,None,None],torch.zeros_like(self.off2),self.off2)
        D,off1,off2,newton_scale=equilibrate_newton(D,off1,off2)
        diag,upper=pair_blocks(D,off1,off2);factor=BlockCR(diag,upper,factorization=self.factorization)
        for info in factor.info:self.factor_failed.logical_or_((info!=0).any(-1))
        B,T,d=self.shape
        def direction(rc):
            rhs=-rd+(self.G.transpose(-1,-2)@((rc-self.z*rp)/self.s)[...,None])[...,0]
            rhs=torch.where(done[:,None,None],torch.zeros_like(rhs),rhs)
            rhs=rhs*newton_scale
            padded=torch.cat([rhs,rhs.new_zeros(B,1,d)],1) if T%2 else rhs
            dy=factor.solve(padded.reshape(B,(T+1)//2,2*d)).reshape(B,-1,d)[:,:T]*newton_scale
            ds=(-rp-(self.G@dy[...,None])[...,0])*self.livef
            dz=((-rc-self.z*ds)/self.s)*self.livef
            return dy,ds,dz
        dy,ds,dz=direction(self.s*self.z)
        ap=self.alpha(self.s,ds);ad=self.alpha(self.z,dz)
        mua=((self.s+ap[:,None,None]*ds)*(self.z+ad[:,None,None]*dz)).sum((1,2))/self.count
        sigma=(mua/mu.clamp_min(1e-30)).clamp(0,1)**3
        rc=self.s*self.z+ds*dz-sigma[:,None,None]*mu[:,None,None]*self.livef
        dy,ds,dz=direction(rc)
        ap=(.995*self.alpha(self.s,ds))[:,None,None];ad=(.995*self.alpha(self.z,dz))[:,None,None]
        ap=ap*(~done)[:,None,None];ad=ad*(~done)[:,None,None]
        self.y.add_(ap*dy);self.s.copy_(torch.where(self.live,(self.s+ap*ds).clamp_min(1e-14),torch.ones_like(self.s)))
        self.z.copy_(torch.where(self.live,(self.z+ad*dz).clamp_min(1e-14),torch.zeros_like(self.z)))

    def result(self):return self.fix+self.scale*self.y

    def residuals(self):
        x=self.result();cx=(self.originalC@x[...,None])[...,0]
        p=torch.maximum((self.l-cx).clamp_min(0).amax((1,2)),(cx-self.u).clamp_min(0).amax((1,2)))
        p=torch.maximum(p,torch.maximum((self.lo-x).clamp_min(0),(x-self.hi).clamp_min(0)).amax((1,2)))
        # Scaled stationarity: complementarity is separately available for QA.
        rd=(self.scaled_Hmv(self.y)+self.gs+(self.G.transpose(-1,-2)@self.z[...,None])[...,0]).abs().amax((1,2))
        return p,rd
