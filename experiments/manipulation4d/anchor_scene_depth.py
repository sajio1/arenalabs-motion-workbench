"""Sparse metric hand anchors refine scene depth; source targets remain unchanged.

One scale/shift sequence is shared by every target robot. Missing observations
are constrained by temporal regularization, not invented skeleton trajectories.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

ROOT=Path('/home/ubuntu/arenalab-experiments/manipulation4d')
PRIOR=Path('/home/ubuntu/arenalab-experiments/da3/no-hands/depth-sequence.npz')
rows=json.loads((ROOT/'cup-lids/depth-anchors.json').read_text())['rows']
with np.load(PRIOR) as saved:arrays={k:saved[k] for k in saved.files}
T=len(arrays['ids']);frame=np.asarray([r['frame']//2 for r in rows]);x=np.asarray([r['predicted'] for r in rows]);y=np.asarray([r['source'] for r in rows]);joint=np.asarray([r['joint'] for r in rows])
temporal_test=frame%5==0;spatial_test=joint%2==1;train=~(temporal_test|spatial_test)
global_scale=json.loads((ROOT/'depth-calibration.json').read_text())['scale']
def fit(keep):
    f=frame[keep];a=x[keep];b=y[keep];n=len(a)
    def residual(v):
        scale=np.exp(v[:T]);shift=v[T:]
        return np.r_[(scale[f]*a+shift[f]-b)/.01,(v[:T]-np.log(global_scale))/.3,shift/.1,np.diff(v[:T],n=2)/.03,np.diff(shift,n=2)/.01]
    pattern=lil_matrix((n+2*T+2*(T-2),2*T),dtype=int)
    for k,t in enumerate(f):pattern[k,t]=1;pattern[k,T+t]=1
    for t in range(T):pattern[n+t,t]=1;pattern[n+T+t,T+t]=1
    for k in range(T-2):pattern[n+2*T+k,k:k+3]=1;pattern[n+3*T-2+k,T+k:T+k+3]=1
    result=least_squares(residual,np.r_[np.full(T,np.log(global_scale)),np.zeros(T)],jac_sparsity=pattern.tocsr(),loss='soft_l1',max_nfev=150,ftol=1e-7,xtol=1e-7)
    return np.exp(result.x[:T]),result.x[T:],dict(cost=float(result.cost),evaluations=int(result.nfev),status=int(result.status))
def metrics(error):return dict(count=len(error),median_mm=float(np.median(abs(error))*1000),p95_mm=float(np.percentile(abs(error),95)*1000))
scale,shift,solver=fit(train);estimate=scale[frame]*x+shift[frame]
report=dict(method='Robust per-frame positive depth scale plus shift, shared temporal second-difference regularization and common weak global prior',source_trajectory_edited=False,ground_truth=False,prior_sha256=hashlib.sha256(PRIOR.read_bytes()).hexdigest(),solver=solver,split='Train even finger landmarks except every fifth frame; hold out odd landmarks and every fifth entire frame',heldout_time=dict(before=metrics(global_scale*x[temporal_test]-y[temporal_test]),after=metrics(estimate[temporal_test]-y[temporal_test])),heldout_landmarks=dict(before=metrics(global_scale*x[spatial_test&~temporal_test]-y[spatial_test&~temporal_test]),after=metrics(estimate[spatial_test&~temporal_test]-y[spatial_test&~temporal_test])),limitation='Source skeleton joints are inside the visible skin. Monocular geometry and sparse-anchor interpolation remain estimates, not contact truth.')
before=report['heldout_time']['before'];after=report['heldout_time']['after']
report['exported']=after['median_mm']<before['median_mm'] and after['p95_mm']<before['p95_mm']
if report['exported']:
    scale,shift,report['final_solver']=fit(np.ones(len(x),bool));new=arrays['depth']*scale[:,None,None]+shift[:,None,None]
    if not np.isfinite(new).all() or np.min(new)<=.01:raise ValueError('Scene depth fit produced invalid extrapolated geometry')
    arrays['depth']=new.astype('float32');np.savez_compressed(ROOT/'depth-hand-anchored.npz',**arrays)
    report.update(scales=scale.tolist(),shifts_m=shift.tolist(),source_frame_ids=arrays['ids'].tolist())
(ROOT/'depth-hand-anchored.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['scales','shifts_m','source_frame_ids']},indent=2))
