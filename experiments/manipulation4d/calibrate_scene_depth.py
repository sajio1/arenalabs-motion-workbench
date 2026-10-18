"""Estimate one scene-depth scale from visible source skeleton anchors.

Source poses, camera extrinsics and the original monocular output stay immutable.
This is an estimated scene calibration, not ground-truth depth or hand contact.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.optimize import least_squares

ROOT=Path('/home/ubuntu/arenalab-experiments/manipulation4d')
PRIOR=Path('/home/ubuntu/arenalab-experiments/da3/no-hands/depth-sequence.npz')
rows=json.loads((ROOT/'cup-lids/depth-anchors.json').read_text())['rows']
x=np.asarray([r['predicted'] for r in rows]);y=np.asarray([r['source'] for r in rows]);test=np.asarray([r['frame']//2%5==0 for r in rows])
def fit(keep):
    result=least_squares(lambda value:(np.exp(value[0])*x[keep]-y[keep])/.01,[0.],loss='soft_l1',f_scale=1)
    return float(np.exp(result.x[0]))
def metrics(error):return dict(median_mm=float(np.median(abs(error))*1000),p95_mm=float(np.percentile(abs(error),95)*1000))
scale=fit(~test);before=metrics(x[test]-y[test]);after=metrics(scale*x[test]-y[test])
report=dict(method='One common positive camera-depth scale, robust source hand anchors; source skeleton is unchanged',fit_scale=scale,train_anchors=int((~test).sum()),heldout_anchors=int(test.sum()),heldout_split='Every fifth sampled source frame held out',before=before,after=after,prior_sha256=hashlib.sha256(PRIOR.read_bytes()).hexdigest(),source_trajectory_edited=False,ground_truth=False,limitation='Video surface and skeleton joint centers differ; this estimates scene scale and does not establish metric object accuracy')
# Do not export a new scene estimate that makes the held-out check worse.
report['exported']=after['median_mm']<before['median_mm'] and after['p95_mm']<before['p95_mm']
if report['exported']:
    report['scale']=fit(np.ones(len(x),bool))
    with np.load(PRIOR) as data:arrays={k:data[k] for k in data.files}
    arrays['depth']=arrays['depth']*report['scale']
    np.savez_compressed(ROOT/'depth-calibrated.npz',**arrays)
(ROOT/'depth-calibration.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
