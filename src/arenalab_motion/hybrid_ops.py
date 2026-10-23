"""Resident skeleton/surface/task linearization from ONE shared FK/Jacobian.

Private robot and source data are imported once from the existing local oracle.
Prototype keeps per-frame Jacobians for parity; production can stream Gram tiles.
No NumPy/MuJoCo or device-to-host operation occurs inside linearize().
"""
import torch
import numpy as np
from .kinematics import KinTree,skew

class HybridOps:
    def __init__(self,oracle,base,targets,orientation,budget,device='cuda',dtype=torch.float64,task_offsets=None):
        self.device=device;self.dtype=dtype;self.m=oracle.b.m
        f=lambda x:torch.as_tensor(np.asarray(x),device=device,dtype=dtype)
        ix=lambda x:torch.as_tensor(np.asarray(x),device=device,dtype=torch.long)
        self.f=f;self.kt=KinTree(self.m,device,dtype);self.cols=ix(oracle.cols);self.vcols=ix(oracle.vcols)
        self.base=f(base);self.refs,self.bones,self.spos,self.snorm=map(f,targets)
        self.task_offsets=f(np.zeros((len(base),6,3)) if task_offsets is None else task_offsets)
        self.orientation=f(orientation);self.budget=f(budget)
        self.bids=ix(oracle.bids);self.allb=ix(np.arange(self.m.nbody));self.d=len(oracle.cols)
        self.ba=ix([v[0] for v in oracle.bones]);self.bc=ix([v[1] for v in oracle.bones])
        self.gids=ix(oracle.geom);self.slocal=f(oracle.local);self.normal=f(oracle.normal);self.anchor=ix(oracle.anchor)
        self.directions=f(oracle.directions);self.lo=f(oracle.lo);self.hi=f(oracle.hi);self.trust=f(oracle.trust)
        self.floor_gid=ix(oracle.surface.ids);self.floor_local=f(oracle.surface.vertices)
        groups=oracle.floor_groups;K=max(map(len,groups));table=np.zeros((len(groups),K),dtype=int);valid=np.zeros_like(table,dtype=bool)
        for j,ids in enumerate(groups):table[j,:len(ids)]=ids;valid[j,:len(ids)]=True
        self.floor_table=ix(table);self.floor_valid=torch.as_tensor(valid,device=device)
        self.floor_groups=[ix(group) for group in groups]
        self.slopes=f([(x,y) for x in [-.5,-.25,0,.25,.5] for y in [-.5,-.25,0,.25,.5]])
        self.prior=f(np.r_[np.zeros(3),np.full(self.d-3,.6)])
        self.eye=torch.eye(self.d,device=device,dtype=dtype)
        self.weight=f([12.,0.,150.,150.,60.,60.]);self.task_indices=ix([0,2,3,4,5])

    def geometry(self,q,jacobian=True):
        K=self.kt.fk(q);gp,gR=self.kt.geom_pose(K,torch.arange(self.m.ngeom,device=self.device))
        result=dict(K=K,gp=gp,gR=gR)
        if jacobian:
            jp,jr=self.kt.jac(K,K['xpos'],self.allb,rot=True)
            result.update(jp=jp[...,self.vcols],jr=jr[...,self.vcols])
        return result

    def floor_points(self,geo):
        return (geo['gR'][:,self.floor_gid]@self.floor_local[None,...,None])[...,0]+geo['gp'][:,self.floor_gid]

    def floor_subset(self,geo,ids):
        gids=self.floor_gid[ids]
        return (geo['gR'][:,gids]@self.floor_local[ids][None,...,None])[...,0]+geo['gp'][:,gids]

    def floor_minimum(self,geo):
        # Full exact mesh hull, bounded temporary storage even for a seven-way
        # line search. Conservative curved-primitive bounds are documented by
        # the local geometry builder; no decimated mesh is used for acceptance.
        result=geo['gp'].new_full((len(geo['gp']),),float('inf'))
        for start in range(0,len(self.floor_gid),2048):
            ids=torch.arange(start,min(start+2048,len(self.floor_gid)),device=self.device)
            result=torch.minimum(result,self.floor_subset(geo,ids)[...,2].amin(-1))
        return result

    def support_points(self,geo):
        F=len(geo['gp']);selected=[];masks=[]
        # Compute the same 25 support directions over all vertices, in tiles.
        # Avoid a [frames,groups,max_vertices,25] padded intermediate (>10 GB).
        for group in self.floor_groups:
            best=geo['gp'].new_full((F,len(self.slopes)),float('inf'))
            indices=torch.zeros(F,len(self.slopes),device=self.device,dtype=torch.long)
            for start in range(0,len(group),1024):
                ids=group[start:start+1024];p=self.floor_subset(geo,ids)
                score=p[...,2,None]+p[...,:2]@self.slopes.T
                value,index=score.masked_fill(p[...,2,None]>=.06,float('inf')).min(-2)
                improve=value<best
                indices=torch.where(improve,ids[index],indices)
                best=torch.minimum(best,value)
            index,order=indices.sort(-1);valid=torch.isfinite(torch.gather(best,-1,order))
            valid[:,1:] &= index[:,1:]!=index[:,:-1]
            selected.append(index);masks.append(valid)
        return torch.cat(selected,-1),torch.cat(masks,-1)

    def task_geometry(self,geo,jacobian=False):
        K=geo['K'];P,R=K['xpos'],K['xmat']
        local=self.task_offsets.repeat(len(P)//len(self.task_offsets),1,1)
        offset=(R[:,self.bids]@local[...,None])[...,0]
        point=P[:,self.bids]+offset
        if not jacobian:return point
        J=geo['jp'][:,self.bids]-skew(offset)@geo['jr'][:,self.bids]
        return point,J

    def linearize(self,q,base=None,refs=None,bones=None,spos=None,snorm=None,orientation=None,budget=None):
        base=self.base if base is None else base;refs=self.refs if refs is None else refs
        bones=self.bones if bones is None else bones;spos=self.spos if spos is None else spos
        snorm=self.snorm if snorm is None else snorm;orientation=self.orientation if orientation is None else orientation;budget=self.budget if budget is None else budget
        geo=self.geometry(q);K=geo['K'];P,R=K['xpos'],K['xmat'];jp,jr=geo['jp'],geo['jr'];F=len(q)
        taskP,taskJ=self.task_geometry(geo,True);err=taskP-refs;ids=self.task_indices
        residual=[(err[:,ids]*self.weight[ids,None]).reshape(F,-1)]
        jac=[(taskJ[:,ids]*self.weight[ids,None,None]).reshape(F,-1,self.d)]
        residual.append((q[:,self.cols]-base[:,self.cols])*self.prior);jac.append((self.eye*self.prior).expand(F,-1,-1))
        v=P[:,self.bc]-P[:,self.ba];length=v.norm(dim=-1,keepdim=True).clamp_min(1e-12);u=v/length
        J=(torch.eye(3,device=q.device,dtype=q.dtype)-u[...,None]*u[...,None,:])/length[...,None]@(jp[:,self.bc]-jp[:,self.ba])
        residual.append(((u-bones)*2).reshape(F,-1));jac.append((J*2).reshape(F,-1,self.d))
        wr=R[:,self.bids[2:4]];wJ=jr[:,self.bids[2:4]]
        jrot=skew(wJ.transpose(-1,-2))@wr[:,:,None]
        residual.append(((wr-orientation)*10).reshape(F,-1));jac.append((jrot.flatten(-2).transpose(-1,-2)*10).reshape(F,-1,self.d))
        sR=geo['gR'][:,self.gids];sp=(sR@self.slocal[None,...,None])[...,0]+geo['gp'][:,self.gids];normal=(sR@self.normal[None,...,None])[...,0]
        sb=self.kt.geom_body[self.gids]
        sj=jp[:,sb]-skew(sp-P[:,sb])@jr[:,sb]-jp[:,self.anchor]
        nj=-skew(normal)@jr[:,sb]
        residual += [((sp-P[:,self.anchor]-spos)*8).reshape(F,-1),((normal-snorm)*.35).reshape(F,-1)]
        jac += [(sj*8).reshape(F,-1,self.d),(nj*.35).reshape(F,-1,self.d)]
        # Original 26-plane outer approximation plus inward radial tangent.
        he=err[:,2:4];hj=taskJ[:,2:4]
        c=(self.directions[None,None]@hj).reshape(F,-1,self.d)
        upper=(budget[...,None]-torch.einsum('nc,fhc->fhn',self.directions,he)).reshape(F,-1)
        norm=he.norm(dim=-1,keepdim=True);rad=he/norm.clamp_min(1e-12)
        tangent=(rad[...,None,:]@hj).squeeze(-2)
        tu=budget-norm[...,0]-torch.minimum(budget*.1,torch.full_like(budget,.0003))
        tangent=torch.where(norm>1e-10,tangent,torch.zeros_like(tangent));tu=torch.where(norm[...,0]>1e-10,tu,torch.full_like(tu,float('inf')))
        C=[c,tangent];lower=[torch.full_like(upper,-float('inf')),torch.full_like(tu,-float('inf'))];uppers=[upper,tu]
        # Fixed-capacity support selection: duplicate indices masked, no .unique sync.
        selected,mask=self.support_points(geo)
        gids=self.floor_gid[selected];batch=torch.arange(F,device=q.device)[:,None]
        b=self.kt.geom_body[gids]
        p=(geo['gR'][batch,gids]@self.floor_local[selected][...,None])[...,0]+geo['gp'][batch,gids]
        off=p-P[batch,b];fJ=jp[batch,b][:,:,2]+off[:,:,1,None]*jr[batch,b][:,:,0]-off[:,:,0,None]*jr[batch,b][:,:,1]
        C.append(torch.where(mask[...,None],fJ,torch.zeros_like(fJ)));lower.append(torch.where(mask,.001-p[:,:,2],torch.full_like(p[:,:,2],-float('inf'))));uppers.append(torch.full_like(p[:,:,2],float('inf')))
        rv=torch.cat(residual,1);J=torch.cat(jac,1);C=torch.cat(C,1);lo=torch.cat(lower,1);hi=torch.cat(uppers,1)
        return dict(residual=rv,J=J,H=J.transpose(-1,-2)@J,g=(J.transpose(-1,-2)@rv[...,None])[...,0],C=C,l=lo,u=hi,geo=geo,floor_min=self.floor_minimum(geo),hand_error=he.norm(dim=-1))

    def verify(self,q,budget=None):
        geo=self.geometry(q,False);e=(self.task_geometry(geo)[:,2:4]-self.refs[:,2:4]).norm(dim=-1)
        floor=self.floor_minimum(geo);budget=self.budget if budget is None else budget
        limit=torch.maximum(self.lo-q[:,self.cols],q[:,self.cols]-self.hi).amax(-1)
        return (e<=budget+2e-6).all(-1)&(floor>=-.0001)&(limit<=1e-8),e,floor

    def residual_only(self,q,base,refs,bones,spos,snorm,orientation,budget):
        """Exact nonlinear objective and full visual-hull acceptance, no Jacobian.

        Inputs may concatenate multiple step-length candidates along the frame
        axis. They share the same robot geometry and remain on the GPU.
        """
        geo=self.geometry(q,False);P,R=geo['K']['xpos'],geo['K']['xmat'];F=len(q)
        err=self.task_geometry(geo)-refs;ids=self.task_indices
        res=[(err[:,ids]*self.weight[ids,None]).reshape(F,-1),
             (q[:,self.cols]-base[:,self.cols])*self.prior]
        v=P[:,self.bc]-P[:,self.ba];u=v/v.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        res.append(((u-bones)*2).reshape(F,-1))
        res.append(((R[:,self.bids[2:4]]-orientation)*10).reshape(F,-1))
        sr=geo['gR'][:,self.gids]
        sp=(sr@self.slocal[None,...,None])[...,0]+geo['gp'][:,self.gids]
        normal=(sr@self.normal[None,...,None])[...,0]
        res.extend([((sp-P[:,self.anchor]-spos)*8).reshape(F,-1),
                    ((normal-snorm)*.35).reshape(F,-1)])
        cost=torch.cat(res,1).square().sum(-1)
        hand=err[:,2:4].norm(dim=-1)
        floor=self.floor_minimum(geo)
        limit=torch.maximum(self.lo-q[:,self.cols],q[:,self.cols]-self.hi).amax(-1)
        valid=(hand<=budget+2e-6).all(-1)&(floor>=-.0001)&(limit<=1e-8)
        return cost,valid,hand,floor
