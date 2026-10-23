"""Universal full-clip GPU SQP on a local native-robot prior, exact nonlinear acceptance.

No WBC. Static setup is CPU-side; optimization tensors stay on the device.
Only small iteration diagnostics are read back. CUDA Graph capture, setup,
warm solve and deliberate resource cooldown are reported separately.
"""
import os,time,json,argparse,hashlib
from pathlib import Path
import numpy as np
import torch
from .block_qp import temporal_mv,capture_steps
from .stable_ipm import StableWindowIPM as WindowIPM
HERE=Path(__file__).resolve().parent

class FullMotion:
    def __init__(self,ops,chunk=8,initial=None):
        self.ops=ops;self.initial=ops.base.clone() if initial is None else initial.clone()
        self.q=self.initial.clone();self.T=len(self.q);self.d=ops.d
        self.w=self.q.new_tensor([45.]*3+[6.]*(self.d-3))
        self.alphas=self.q.new_tensor([1.,.5,.25,.125,.0625,.03125,.015625])
        A=len(self.alphas)
        self.line_inputs=[v.repeat(A,*([1]*(v.ndim-1))) for v in
                          [ops.base,ops.refs,ops.bones,ops.spos,ops.snorm,ops.orientation,ops.budget]]
        self.delta=torch.zeros_like(self.q[:,ops.cols])
        self.assemble_graph=None;self.line_graph=None
        start=time.perf_counter()
        self.build_qp() # initialize libraries before graph capture
        torch.cuda.synchronize()
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):self.build_qp()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.assemble_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.assemble_graph):self.build_qp()
        self.iter_graph=capture_steps(self.qp,chunk)
        # Warm residual kernels without changing q; delta is zero.
        with torch.cuda.stream(stream):self.line_search()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.q.copy_(self.initial);self.line_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.line_graph):self.line_search()
        self.q.copy_(self.initial);torch.cuda.synchronize()
        self.capture_s=time.perf_counter()-start;self.chunk=chunk

    def build_qp(self):
        o=self.ops;q=self.q
        self.assembly=o.linearize(q)
        H=self.assembly['H']+.01*o.eye
        g=self.assembly['g']+temporal_mv(q[:,o.cols][None],self.w**2)[0]
        lo=torch.maximum(o.lo-q[:,o.cols],-o.trust)
        hi=torch.minimum(o.hi-q[:,o.cols],o.trust)
        self.qp=WindowIPM(H[None],g[None],self.assembly['C'][None],self.assembly['l'][None],self.assembly['u'][None],lo[None],hi[None],self.w)

    def line_search(self):
        o=self.ops;T=self.T;A=len(self.alphas)
        candidates=self.q[None].expand(A,-1,-1).clone()
        candidates[:,:,o.cols]=(candidates[:,:,o.cols]+self.alphas[:,None,None]*self.delta[None]).clamp(o.lo,o.hi)
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

    def solve(self,iters=20,max_qp=64,cooldown=.5):
        self.q.copy_(self.initial);torch.cuda.synchronize();start=time.perf_counter();logs=[];cool=0.
        for it in range(iters):
            s=time.perf_counter();self.assemble_graph.replay();torch.cuda.synchronize();assembly=time.perf_counter()-s
            s=time.perf_counter();qpstats=None
            for n in range(self.chunk,max_qp+1,self.chunk):
                self.iter_graph.replay()
                rp,rd=self.qp.residuals()
                mu=(self.qp.s*self.qp.z).sum((1,2))/self.qp.count
                vals=torch.stack([rp[0],rd[0],mu[0],self.qp.factor_failed[0].to(rp.dtype)]).cpu().numpy()
                qpstats=dict(iterations=n,factorization=self.qp.factorization,primal_max=float(vals[0]),scaled_stationarity=float(vals[1]),complementarity=float(vals[2]),factor_failed=bool(vals[3]))
                if np.isfinite(vals).all() and vals[0]<2e-5 and vals[1]<1e-6 and vals[2]<1e-8 and not vals[3]:break
            qp_s=time.perf_counter()-s
            qpokay=np.isfinite(vals).all() and vals[0]<2e-5 and vals[1]<1e-6 and vals[2]<1e-8 and not vals[3]
            if not qpokay:
                logs.append(dict(iter=it,assembly_s=assembly,qp_s=qp_s,qp=qpstats,status='QP_FAILED'));break
            s=time.perf_counter();self.delta.copy_(self.qp.result()[0]);self.line_graph.replay();stat=self.line_stats.cpu().numpy();line_s=time.perf_counter()-s
            log=dict(iter=it,assembly_s=assembly,qp_s=qp_s,line_search_s=line_s,qp=qpstats,accepted=bool(stat[0]),alpha=float(stat[1]),previous_cost=float(stat[2]),cost=float(stat[3]),step_max=float(stat[4]),max_hand_cm=float(stat[5]*100),floor_min_m=float(stat[6]));logs.append(log)
            print('GPU_ITER',it,json.dumps(log),flush=True)
            if not stat[0] or stat[4]<2e-4:break
            delay=(time.perf_counter()-s+assembly+qp_s)*cooldown
            if delay:time.sleep(delay);cool+=delay
        torch.cuda.synchronize();final_start=time.perf_counter()
        valid,he,floor=self.ops.verify(self.q)
        final=torch.stack([valid.all().to(self.q.dtype),(he-self.ops.budget).amax(),floor.amin()]).cpu().numpy()
        output=self.q.detach().cpu().numpy()
        final_validation_transfer_s=time.perf_counter()-final_start
        wall=time.perf_counter()-start
        status='QP_FAILED' if logs[-1].get('status')=='QP_FAILED' else ('VALID_STOPPED' if final[0] else 'NONLINEAR_FAILED')
        return output,dict(status=status,frames=self.T,iterations=logs,wall_s=wall,resource_cooldown_s=cool,active_s=wall-cool,final_validation_transfer_s=final_validation_transfer_s,valid=bool(final[0]),hand_budget_max_violation_m=float(final[1]),floor_min_m=float(final[2]),max_iterations=iters,linear_solver=self.qp.factorization,newton_equilibration='exact_diagonal_congruence',solve_backend='GPU FP64 IPM + block cyclic reduction with pivoted LU; no CPU QP fallback')
