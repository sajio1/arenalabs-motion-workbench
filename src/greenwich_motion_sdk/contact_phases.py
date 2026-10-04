"""Conservative geometric support candidates, not measured contact forces."""
import numpy as np


def near_ground_confidence(clearance_cm, release_cm):
    """Vertical-only proximity cue, independent of stationary support.

    A rolling/sliding source foot can be near the floor without being planted.
    Fade smoothly to zero before flight; never impose horizontal anchoring.
    """
    c=np.asarray(clearance_cm,float)
    if not np.isfinite(c).all() or not np.isfinite(release_cm) or release_cm<=0:
        raise ValueError('Finite clearance and a positive release distance required')
    u=np.clip(c/release_cm,0,1)
    return (1-u)**2*(1+2*u)


def smooth_residual(values, sigma_frames):
    """Differentiable, symmetric low-pass correction; no world-position target.

    Runtime is linear in frame count for fixed FPS/window. Replicated boundaries
    preserve constant offsets. This never snaps a point to a contact surface.
    """
    import torch
    import torch.nn.functional as F
    if not np.isfinite(sigma_frames) or sigma_frames<0:
        raise ValueError('Smoothing sigma must be finite and nonnegative')
    if not sigma_frames:return values
    radius=max(1,int(np.ceil(3*sigma_frames)))
    x=torch.arange(-radius,radius+1,device=values.device,dtype=values.dtype)
    kernel=torch.exp(-.5*(x/sigma_frames).square());kernel=kernel/kernel.sum()
    channels=values.shape[1]
    data=F.pad(values.T[None],(radius,radius),mode='replicate')
    return F.conv1d(data,kernel[None,None].expand(channels,1,-1),groups=channels)[0].T


def released_support(foot_positions_cm, neutral_heights_cm, fps,
                     height_tolerance_cm=1.5, speed_tolerance_cm_s=15.):
    """Require the SAME landmark to be low and slow; release rolling soles.

    Inputs are ankle/toe pairs in left,right order, with source bind offsets.
    A single point can indicate contact but cannot establish a flat sole. Only
    both low/slow landmarks permit whole-sole flattening or center slip costs.
    Motion above/away from support receives no downward attraction here.
    """
    p=np.asarray(foot_positions_cm,float); offsets=np.asarray(neutral_heights_cm,float)
    if p.ndim!=3 or p.shape[1:]!=(4,3) or offsets.shape!=(4,) or not np.isfinite([fps,height_tolerance_cm,speed_tolerance_cm_s]).all() or min(fps,height_tolerance_cm,speed_tolerance_cm_s)<=0:
        raise ValueError('Expected [T,4,3] ankle/toe pairs, four offsets and positive fps')
    if len(p)<3 or not np.isfinite(p).all() or not np.isfinite(offsets).all():
        raise ValueError('Finite source positions and at least three frames required')
    speed=np.linalg.norm(np.diff(p,axis=0),axis=-1)*fps
    # At both sides of a frame, including vertical lift-off. No unconditional
    # first-frame contact and no centered difference cancelling reversals.
    speed=np.maximum(np.vstack([speed[:1],speed]),np.vstack([speed,speed[-1:]]))
    low_slow=((p[:,:,1]-offsets)<=height_tolerance_cm)&(speed<=speed_tolerance_cm_s)
    pair=low_slow.reshape(len(p),2,2)
    return pair.any(2),pair.all(2)


def material_patch_slip(clouds, heights, edges, band_cm=.5, allowance_cm=0.):
    """Differentiable XZ displacement of persistent lowest material points.

    Contact selection is detached and source-gated. This is a geometric slip
    residual, not a friction model; unsupported/unknown frames incur zero cost.
    """
    import torch
    if not np.isfinite(allowance_cm) or allowance_cm<0:
        raise ValueError('Slip allowance must be finite and nonnegative')
    terms=[]
    for side,cloud in enumerate(clouds):
        low=(cloud[:,:,1]-heights[:,side,None]).detach()<band_cm
        active=(low[1:]&low[:-1]).to(cloud.dtype)*edges[:,side,None]
        delta=cloud[1:,:,[0,2]]-cloud[:-1,:,[0,2]]
        cost=delta.square().sum(-1)
        if allowance_cm:
            cost=torch.relu(torch.linalg.vector_norm(delta,dim=-1)-allowance_cm).square()
        terms.append((cost*active).sum()/active.sum().clamp_min(1))
    return torch.stack(terms).mean()


def source_phase_patch_speed(vertices_cm, support, fps, band_cm=.5):
    """Diagnostic material speed without censoring hovering/penetrating feet.

    Source support is independent of target height/speed. This measures drift
    during intended support, not confirmed load-bearing frictional contact.
    Unknown edges stay NaN, never zero. Rolling can retain a stationary toe.
    """
    v=np.asarray(vertices_cm,float);support=np.asarray(support,bool)
    if v.ndim!=3 or v.shape[2]!=3 or support.shape!=(len(v),) or len(v)<2 or fps<=0 or band_cm<=0 or not np.isfinite(v).all():
        raise ValueError('Expected finite [T,V,3] vertices, [T] support and positive rate/band')
    low=v[:,:,1]-v[:,:,1].min(1)[:,None]<band_cm
    active=low[1:]&low[:-1]&support[1:,None]&support[:-1,None]
    speed=np.linalg.norm(np.diff(v[:,:,[0,2]],axis=0),axis=-1)*fps
    out=np.full(len(v),np.nan)
    for i in np.flatnonzero(active.any(1)):
        out[i+1]=np.median(speed[i,active[i]])
    return out


def material_patch_drift(clouds, heights, support, band_cm=.5):
    """Position variance within each continuous material-point contact episode.

    Each native point has its own free episode centroid, not a prescribed foot
    location. Releasing that point ends its episode, allowing roll and new steps.
    Detached contact membership prevents differentiating through phase selection.
    Segmented scatter reductions are linear in frames times sampled vertices.
    """
    import torch
    terms=[]
    for side,cloud in enumerate(clouds):
        active=((cloud[:,:,1]-heights[:,side,None]).detach()<band_cm)&support[:,side,None].bool()
        starts=active&~torch.cat([torch.zeros_like(active[:1]),active[:-1]],0)
        # Disjoint ids for every material vertex and every support episode.
        T,V=active.shape
        ids=starts.to(torch.int64).cumsum(0)+torch.arange(V,device=cloud.device)[None]*(T+1)
        ids=ids.reshape(-1)
        mask=active.reshape(-1).to(cloud.dtype)
        xy=cloud[:,:,[0,2]].reshape(-1,2)
        counts=torch.zeros(V*(T+1),device=cloud.device,dtype=cloud.dtype).scatter_add(0,ids,mask)
        sums=torch.zeros((V*(T+1),2),device=cloud.device,dtype=cloud.dtype).scatter_add(0,ids[:,None].expand(-1,2),xy*mask[:,None])
        center=sums[ids]/counts[ids,None].clamp_min(1)
        terms.append(((xy-center).square().sum(-1)*mask).sum()/mask.sum().clamp_min(1))
    return torch.stack(terms).mean()
