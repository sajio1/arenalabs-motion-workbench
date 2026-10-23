"""GPU-resident time-coupled QP, no dense (T*d)^2 matrix.

Prototype: block cyclic reduction of SPD systems + box/linear ADMM.
All numeric iterations stay on device. CUDA Graph capture is external.
No convergence claim is implied by the configured iteration count.
"""
import torch

def chol_solve(rhs, L):
    # Explicit triangular solves avoid MAGMA potrs allocation during graph capture.
    y=torch.linalg.solve_triangular(L,rhs,upper=False)
    return torch.linalg.solve_triangular(L.transpose(-1,-2),y,upper=True)

def lu_factor(A):
    factor,pivots,info=torch.linalg.lu_factor_ex(A)
    return (factor,pivots),info

def lu_solve(rhs,factor):
    return torch.linalg.lu_solve(factor[0],factor[1],rhs)

class BlockCR:
    """Batched SPD block-tridiagonal factor/solve, logarithmic time levels.

    A [batch,n,d,d], U [batch,n-1,d,d]. Factor each odd diagonal in
    parallel, eliminate into even neighbours, recursively solve Schur system.
    Numeric factors are reusable across every ADMM RHS of this SQP step.
    """
    def __init__(self, A, U, factorization='cholesky'):
        if factorization not in {'cholesky','pivoted_lu'}:
            raise ValueError('Unknown block factorization')
        factor=torch.linalg.cholesky_ex if factorization=='cholesky' else lu_factor
        self.solve_block=chol_solve if factorization=='cholesky' else lu_solve
        self.levels=[]; self.info=[]
        while A.shape[1]>1:
            n=A.shape[1];odd=A[:,1::2];ne=A[:,::2].shape[1]
            L,info=factor(odd);self.info.append(info)
            left=U[:,::2]
            right=torch.zeros_like(left)
            right[:,:U[:,1::2].shape[1]]=U[:,1::2]
            invleft=self.solve_block(left.transpose(-1,-2),L)
            invright=self.solve_block(right,L)
            S=A[:,::2].clone();no=odd.shape[1]
            S[:,:no]-=left@invleft
            S[:,1:]-=(right.transpose(-1,-2)@invright)[:,:ne-1]
            upper=-(left@invright)[:,:ne-1]
            self.levels.append((n,L,left,right))
            A=(S+S.transpose(-1,-2))*.5;U=upper
        self.last,info=factor(A);self.info.append(info)

    def solve(self,b):
        stack=[]
        for n,L,left,right in self.levels:
            odd=b[:,1::2];y=self.solve_block(odd[...,None],L)[...,0]
            even=b[:,::2].clone();no=odd.shape[1]
            even[:,:no]-=(left@y[...,None])[...,0]
            even[:,1:]-=(right.transpose(-1,-2)@y[...,None])[:,:even.shape[1]-1,:,0]
            stack.append(odd);b=even
        x=self.solve_block(b[...,None],self.last)[...,0]
        for (n,L,left,right),odd in zip(reversed(self.levels),reversed(stack)):
            no=odd.shape[1]
            rhs=odd-(left.transpose(-1,-2)@x[:,:no,:,None])[...,0]
            rhs[:,:x.shape[1]-1]-=(right[:,:x.shape[1]-1]@x[:,1:,:,None])[...,0]
            xo=self.solve_block(rhs[...,None],L)[...,0]
            full=x.new_empty(x.shape[0],n,x.shape[-1]);full[:,::2]=x;full[:,1::2]=xo;x=full
        return x

def temporal_coeff(T,device,dtype):
    """D2.T D2 diagonals, built once per shape bucket."""
    a=torch.zeros(T,device=device,dtype=dtype);b=torch.zeros(T-1,device=device,dtype=dtype)
    c=torch.ones(T-2,device=device,dtype=dtype)
    a[:-2]+=1;a[1:-1]+=4;a[2:]+=1;b[:-1]-=2;b[1:]-=2
    return a,b,c

def temporal_mv(x,w2):
    dd=(x[:,:-2]-2*x[:,1:-1]+x[:,2:])*w2
    out=torch.zeros_like(x);out[:,:-2]+=dd;out[:,1:-1]-=2*dd;out[:,2:]+=dd
    return out

def pair_blocks(diag,off1,off2):
    """Pentadiagonal d blocks -> tridiagonal 2d supernodes. Odd T uses an isolated identity block, preserving the original objective exactly."""
    B,T,d,_=diag.shape
    if T%2:
        diag=torch.cat([diag,torch.eye(d,device=diag.device,dtype=diag.dtype).expand(B,1,d,d)],1)
        off1=torch.cat([off1,off1.new_zeros(B,1,d)],1)
        off2=torch.cat([off2,off2.new_zeros(B,1,d)],1)
        T+=1
    N=T//2;A=diag.new_zeros(B,N,2*d,2*d);U=diag.new_zeros(B,N-1,2*d,2*d)
    A[:,:,:d,:d]=diag[:,::2];A[:,:,d:,d:]=diag[:,1::2]
    A[:,:,:d,d:]=torch.diag_embed(off1[:,::2]);A[:,:,d:,:d]=A[:,:,:d,d:].transpose(-1,-2)
    U[:,:,:d,:d]=torch.diag_embed(off2[:,::2]);U[:,:,d:,d:]=torch.diag_embed(off2[:,1::2])
    U[:,:,d:,:d]=torch.diag_embed(off1[:,1::2])
    return A,U

class WindowQP:
    """min .5 dx' (H + D2'W²D2) dx + g'dx, l <= C dx <= u, box.

    H [B,T,d,d], g [B,T,d], C [B,T,m,d]. Variable equilibration
    and row normalization preserve the QP. Fixed rho permits factor reuse.
    Final original-unit residuals must be checked before accepting a step.
    """
    def __init__(self,H,g,C,l,u,lo,hi,w,rho=.5):
        self.H,self.g,self.originalC=H,g,C
        self.l,self.u,self.lo,self.hi=l,u,lo,hi
        self.w2=w*w;self.rho=rho;self.sigma=1e-8
        B,T,d=g.shape;self.shape=(B,T,d)
        a,b,c=temporal_coeff(T,g.device,g.dtype)
        scale=(H.diagonal(dim1=-2,dim2=-1)+a[None,:,None]*self.w2).clamp_min(1e-8).rsqrt()
        self.scale=scale
        cs=C*scale[:,:,None,:];norm=cs.norm(dim=-1).clamp_min(1e-12)
        self.C=cs/norm[...,None];self.lower=l/norm;self.upper=u/norm
        self.lb=lo/scale;self.ub=hi/scale;self.gs=g*scale
        D=H*scale[...,None]*scale[...,None,:]
        D=D+torch.diag_embed(a[None,:,None]*self.w2*scale*scale+rho+self.sigma)
        D=D+rho*(self.C.transpose(-1,-2)@self.C)
        off1=b[None,:,None]*self.w2*scale[:,:-1]*scale[:,1:]
        off2=c[None,:,None]*self.w2*scale[:,:-2]*scale[:,2:]
        aa,uu=pair_blocks(D,off1,off2);self.factor=BlockCR(aa,uu)
        self.x=torch.zeros_like(g);self.zb=self.x.clamp(self.lb,self.ub);self.ubox=torch.zeros_like(g)
        self.z=torch.zeros_like(l).clamp(self.lower,self.upper);self.dual=torch.zeros_like(l)

    def reset(self):
        self.x.zero_();self.zb.copy_(self.x.clamp(self.lb,self.ub));self.ubox.zero_()
        self.z.copy_(torch.zeros_like(self.z).clamp(self.lower,self.upper));self.dual.zero_()

    def step(self):
        rhs=-self.gs+self.rho*((self.C.transpose(-1,-2)@(self.z-self.dual)[...,None])[...,0]+self.zb-self.ubox)+self.sigma*self.x
        B,T,d=self.shape
        nxt=self.factor.solve(rhs.reshape(B,T//2,2*d)).reshape(B,T,d)
        ax=(self.C@nxt[...,None])[...,0]
        # Over-relaxation affects constraint projections, not the stationarity point.
        axbar=1.6*ax-.6*self.z;xb=1.6*nxt-.6*self.zb
        znew=(axbar+self.dual).clamp(self.lower,self.upper);zbnew=(xb+self.ubox).clamp(self.lb,self.ub)
        self.dual.add_(axbar-znew);self.ubox.add_(xb-zbnew)
        self.z.copy_(znew);self.zb.copy_(zbnew);self.x.copy_(nxt)

    def result(self):return self.x*self.scale

    def residuals(self):
        dx=self.result();cx=(self.originalC@dx[...,None])[...,0]
        primal=torch.maximum((self.l-cx).clamp_min(0).amax((1,2)),(cx-self.u).clamp_min(0).amax((1,2)))
        primal=torch.maximum(primal,torch.maximum((self.lo-dx).clamp_min(0),(dx-self.hi).clamp_min(0)).amax((1,2)))
        grad=(self.H@dx[...,None])[...,0]+temporal_mv(dx,self.w2)+self.g
        aty=self.rho*((self.C.transpose(-1,-2)@self.dual[...,None])[...,0]+self.ubox)/self.scale
        dual=(grad+aty).abs().amax((1,2))
        return primal,dual

def capture_steps(qp,steps=16):
    """Capture warmed, fixed-address state updates. No host tensor reads."""
    stream=torch.cuda.Stream(device=qp.g.device);stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):qp.step()
    torch.cuda.current_stream().wait_stream(stream);qp.reset();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(steps):qp.step()
    qp.reset()
    return graph
