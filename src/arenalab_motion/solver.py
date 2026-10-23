"""Resident SQP with an explicit, cost-tested nonlinear floor restoration.

Deep kneeling exposed global line-search stalls in the baseline: one frame's
curved mesh contact rejected every finite candidate for the complete clip.
Each candidate is raised only as needed within its root-z trust radius, then
the ORIGINAL full objective, temporal cost, hand budgets and limits are tested.
This is an optimizer step, not an export-time trajectory repair.
"""
import torch
from .resident_solver import FullMotion as BaselineFullMotion
from .block_qp import temporal_mv
from .stable_ipm import StableWindowIPM


class FullMotion(BaselineFullMotion):
    def build_qp(self):
        o=self.ops;q=self.q
        self.assembly=o.linearize(q)
        H=self.assembly['H']+.01*o.eye
        g=self.assembly['g']+temporal_mv(q[:,o.cols][None],self.w**2)[0]
        lo=torch.maximum(o.lo-q[:,o.cols],-o.trust)
        hi=torch.minimum(o.hi-q[:,o.cols],o.trust)
        self.qp=StableWindowIPM(H[None],g[None],self.assembly['C'][None],self.assembly['l'][None],self.assembly['u'][None],lo[None],hi[None],self.w)

    def line_search(self):
        o=self.ops;T=self.T;A=len(self.alphas)
        candidates=self.q[None].expand(A,-1,-1).clone()
        candidates[:,:,o.cols]=(candidates[:,:,o.cols]+self.alphas[:,None,None]*self.delta[None]).clamp(o.lo,o.hi)
        # Native free-root z translates the entire physical mesh rigidly.
        # The hull minimum therefore changes exactly by the same displacement.
        geometry=o.geometry(candidates.reshape(A*T,-1),False)
        floor=o.floor_minimum(geometry).reshape(A,T)
        required=(.001-floor).clamp_min(0)
        room=(o.trust[2]-(candidates[:,:,2]-self.q[None,:,2])).clamp_min(0)
        lift=torch.minimum(required,room)
        candidates[:,:,2]+=lift
        cost,valid,hand,floor=o.residual_only(candidates.reshape(A*T,-1),*self.line_inputs)
        x=candidates[:,:,o.cols];dd=(x[:,:-2]-2*x[:,1:-1]+x[:,2:])*self.w
        cost=cost.reshape(A,T).sum(-1)+dd.square().sum((1,2))
        before=self.assembly['residual'].square().sum()+(((self.q[:-2,o.cols]-2*self.q[1:-1,o.cols]+self.q[2:,o.cols])*self.w)**2).sum()
        allowed=valid.reshape(A,T).all(-1)&torch.isfinite(cost)&(cost<=before+1e-7)
        idx=allowed.to(torch.int64).argmax();accept=allowed.any()
        pick=lambda v:torch.index_select(v,0,idx.reshape(1))[0]
        nextq=pick(candidates);step=(nextq[:,o.cols]-self.q[:,o.cols]).abs().amax()
        self.q.copy_(torch.where(accept,nextq,self.q))
        self.line_stats=torch.stack([accept.to(self.q.dtype),pick(self.alphas)*accept,before,
            torch.where(accept,pick(cost),before),step*accept,
            torch.where(accept,pick(hand.reshape(A,T,2)).amax(),self.assembly['hand_error'].amax()),
            torch.where(accept,pick(floor.reshape(A,T)).amin(),self.assembly['floor_min'].amin())])

    def solve(self,*args,**kwargs):
        result,record=super().solve(*args,**kwargs)
        record['qp_freeze_policy']='Same original-space primal, scaled stationarity and complementarity thresholds as outer acceptance; no tolerance relaxation'
        record['line_feasibility_projection']='Full native-hull root-z restoration to 1 mm within the same trust radius; original objective and all budgets rechecked before acceptance'
        return result,record
