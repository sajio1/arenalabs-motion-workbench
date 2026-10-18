"""Source-only ground hypothesis and soft stance confidence, shared by robots.

The saved SMPL-X and observed task poses are never changed. Ground is inferred
from low/slow reconstructed soles, not a measured floor or a source registration.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

POLICY = 'source-sole-plane-v1'

def support_reference(human, model):
    neutral = model['v_template'] + np.einsum('vck,k->vc', model['shapedirs'][..., :len(human['betas'])], human['betas'])
    soles=[]
    for joints in ([7,10], [8,11]):
        region=model['weights'][:,joints].sum(1)>.7
        mask=region & (neutral[:,1] <= neutral[region,1].min()+.012)
        if mask.sum()<8: raise ValueError('SMPL-X sole region unavailable')
        soles.append(human['vertices'][:,mask])
    low=np.stack([np.percentile(p[:,:,2],5,axis=1) for p in soles],1)
    centres=np.stack([p.mean(1) for p in soles],1)
    speed=np.linalg.norm(np.gradient(centres,axis=0)*30,axis=-1)
    base=np.percentile(low,15,axis=0)
    confidence=np.exp(-(np.maximum(low-base-.025,0)/.045)**2)*np.exp(-(speed/.35)**2)
    confidence=gaussian_filter1d(confidence,1.5,axis=0)
    stable=low[confidence>.7]
    if not len(stable): raise ValueError('No stable source support for estimating a ground plane')
    floor=float(np.percentile(stable,10))
    clearance=np.maximum(gaussian_filter1d(low-base,1.5,axis=0)-.025,0)
    return floor, confidence, clearance, dict(policy=POLICY,ground_height_m=floor,
        ground_status='estimated from reconstructed SMPL-X soles; not measured',
        stable_samples=int(len(stable)),source_sole_height_range_m=[float(low.min()),float(low.max())],
        source_trajectory_edited=False,shared_across_robots=True)
