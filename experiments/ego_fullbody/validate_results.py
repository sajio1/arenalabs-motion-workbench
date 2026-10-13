"""Report final kinematic metrics without relabeling them as physical acceptance."""
from pathlib import Path
import json,numpy as np
root=Path(__import__("os").environ.get("ARENA_EGO_WORKDIR", str(Path(__file__).resolve().parent)))
obs=np.load(root/'observations.npz');h=np.load(root/'egoallo_smplh.npz');x=np.load(root/'smplx_motion.npz');truth=np.load(root/'heldout_xsens.npz')['positions']
def stats(a):return dict(mean=float(np.mean(a)),p95=float(np.percentile(a,95)),max=float(np.max(a)))
report={'human':json.loads((root/'human_validation.json').read_text()),'heldout_comparison':{},'robots':{}}
for name,hi,xi in [('pelvis',0,0),('left_ankle',7,21),('right_ankle',8,17)]:
 report['heldout_comparison'][name+'_position_error_cm']=stats(np.linalg.norm(x['joints'][:,hi]-truth[:,xi],axis=-1)*100)
report['human']['minimum_vertex_height_m']=float(x['vertices'][...,2].min())
report['human']['pelvis_vertical_accel_m_s2']=stats(np.abs(np.diff(x['joints'][:,0,2],n=2)*900))
for robot in ['h2','a3']:
 path=next((root/(robot+'-result')).rglob('motion.npz'));d=np.load(path)
 final=np.asarray(d['world_position_cm']);roles=d['source_role_indices'];src=d['source_positions_cm']
 keys=d['constraint_joint_indices'];err=np.linalg.norm(final[:,keys]-src[:,roles[2:4]],axis=-1)
 report['robots'][robot]=dict(frames=len(final),finite=bool(np.isfinite(final).all()),wrist_world_position_error_to_smplx_cm=stats(err),minimum_support_height_cm=float(d['support_surface_height_cm'].min()),pelvis_vertical_accel_m_s2=stats(np.abs(np.diff(d['root_t'][:,1],n=2)*9)),physical_validation=False)
(root/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
