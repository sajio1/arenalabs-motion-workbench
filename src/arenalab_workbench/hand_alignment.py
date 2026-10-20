"""Register estimated geometry to observed source hands; never alter robot motion.

One right-handed similarity transform applies to the entire clip, including
retained human points and object boxes. Visibility never changes the solve.
Raw depth and observed trajectories remain immutable. Residuals are diagnostics,
not verified hand-object contact or a claim of complete scene accuracy.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

POLICY = 'observed-hand-global-similarity-v1'


def transform_points(points, transform):
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def similarity(source, target, weights, scale=True):
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    a, b = weights @ source, weights @ target
    x, y = source-a, target-b
    u, singular, vt = np.linalg.svd((y*weights[:, None]).T @ x)
    sign = np.ones(3);sign[-1] = np.linalg.det(u@vt)
    rotation = (u*sign)@vt
    factor = float(singular@sign / np.sum(weights*np.sum(x*x, axis=1))) if scale else 1.
    result = np.eye(4);result[:3, :3] = factor*rotation;result[:3, 3] = b-factor*rotation@a
    return result


def fit_registration(source, target, weights, frame_ids):
    """Robust correspondences, temporally held-out frames, fixed clip transform."""
    source, target, weights = map(np.asarray, (source, target, weights))
    base = dict(policy=POLICY, raw_to_reference=np.eye(4).tolist(), robot_motion_modified=False,
                independent_of_visibility=True, correspondence_count=len(source))
    unique = np.unique(frame_ids)
    if len(source)<40 or len(unique)<8 or np.linalg.matrix_rank(source-source.mean(0),tol=.005)<2 or np.linalg.matrix_rank(target-target.mean(0),tol=.005)<2:
        return {**base, 'status':'insufficient_evidence'}
    held = np.isin(frame_ids, unique[::5]);train = ~held
    x, y, w = source[train], target[train], weights[train]
    before = np.linalg.norm(source[held]-target[held],axis=1)
    candidates=[];rng=np.random.default_rng(0)
    for with_scale in [False, True]:
        seeds=[similarity(x,y,w,with_scale)]
        for _ in range(96):
            chosen=rng.choice(len(x),3,replace=False)
            if np.linalg.matrix_rank(x[chosen]-x[chosen].mean(0),tol=.002)<2:continue
            seeds.append(similarity(x[chosen],y[chosen],w[chosen],with_scale))
        matrix=min(seeds,key=lambda m:np.median(np.linalg.norm(transform_points(x,m)-y,axis=1)))
        for _ in range(12):
            errors=np.linalg.norm(transform_points(x,matrix)-y,axis=1)
            robust=np.minimum(1.,.025/np.maximum(errors,1e-9))**2
            matrix=similarity(x,y,w*robust,with_scale)
        factor=float(np.cbrt(np.linalg.det(matrix[:3,:3])))
        if not .5<=factor<=2:continue
        errors=np.linalg.norm(transform_points(source[held],matrix)-target[held],axis=1)
        candidates.append((float(np.median(errors)),with_scale,matrix,errors,factor))
    if not candidates:return {**base,'status':'unresolved','reason':'No supported global scale'}
    # Prefer rigid registration unless scale materially improves held-out frames.
    best=min(candidates,key=lambda row:row[0])
    rigid=next((row for row in candidates if not row[1]),None)
    if rigid is not None and rigid[0]<=best[0]*1.05:best=rigid
    median,with_scale,matrix,after,factor=best
    applied=median<np.median(before)*.9
    return {**base,'status':'registered' if applied else 'no_improvement',
            'raw_to_reference':matrix.tolist() if applied else np.eye(4).tolist(),
            'scale':factor if applied else 1.,'candidate_transform':matrix.tolist(),
            'fit':'similarity' if with_scale else 'rigid','training_samples':int(train.sum()),
            'heldout_samples':int(held.sum()),'heldout_frames':unique[::5].tolist(),
            'heldout_before_median_mm':float(np.median(before)*1000),
            'heldout_after_median_mm':float(np.median(after if applied else before)*1000),
            'heldout_after_p90_mm':float(np.percentile(after if applied else before,90)*1000),
            'scope':'Observed hand/arm alignment; residual local shape/depth errors may remain'}


def observed_anchors(row):
    """Read observed data in the same untouched world as the optical camera."""
    source=Path(row['source']);n=int(row['frames']);start=int(row.get('source_window',{}).get('start_frame',0))
    if source.suffix.lower() in {'.h5','.hdf5'}:
        import h5py
        with h5py.File(source) as data:
            if 'transforms' not in data or 'confidences' not in data:return None
            names=[name for name in data['transforms'] if name.startswith(('left','right')) and any(word in name for word in ['Finger','Thumb','Hand','Forearm','Arm'])]
            names=[name for name in names if name in data['confidences']]
            if not names:return None
            points=np.stack([data['transforms/'+name][start:start+n,:3,3] for name in names],1)
            confidence=np.stack([data['confidences/'+name][start:start+n] for name in names],1).reshape(n,-1)
        return points,confidence,'Observed hand/arm transforms in optical source world'
    if source.suffix.lower()=='.npz':
        with np.load(source) as data:
            if 'alignment_points_world' in data and 'alignment_confidence' in data:
                return data['alignment_points_world'][start:start+n],data['alignment_confidence'][start:start+n],'Declared observed world-space alignment points'
            if 'segment_tXYZ' in data and 'timestamps_us' in data:
                from greenwich_pipelines.pipelines.nymeria import xsens_clock
                times,_=xsens_clock.recording_clock(data['timestamps_us']);query=times[0]+np.arange(start,start+n)*1e6/30
                if query[-1]>times[-1]:raise ValueError('Alignment window exceeds observed source clock')
                indices=xsens_clock.nearest_source_rows(times,query)
                points=data['segment_tXYZ'][indices].reshape(n,23,3)[:,[13,14,9,10]]
                return points,np.ones(points.shape[:2]),'Observed Xsens elbows/wrists; no generated fingers'
    return None


def register(work,row,depth,reference_masks):
    work=Path(work)
    if not reference_masks.any():return dict(policy=POLICY,status='unavailable',reason='No source hand/arm segmentation',raw_to_reference=np.eye(4).tolist())
    if not any((work/name).exists() for name in ['cameras.npz','camera-poses.npz']):return dict(policy=POLICY,status='unavailable',reason='Observed source camera world is missing',raw_to_reference=np.eye(4).tolist())
    anchors=observed_anchors(row)
    if anchors is None:return dict(policy=POLICY,status='unavailable',reason='No observed hand/arm 3D correspondences; point-cloud display remains available',raw_to_reference=np.eye(4).tolist())
    points,confidence,provenance=anchors
    src=[];dst=[];weights=[];frames=[]
    chosen=np.unique(np.linspace(0,len(depth['ids'])-1,min(96,len(depth['ids']))).astype(int))
    for index in chosen:
        frame=int(depth['ids'][index]);e=depth['extrinsics'][index];k=depth['intrinsics'][index];z=depth['depth'][index]
        world=points[frame].reshape(-1,3);cam=world@e[:,:3].T+e[:,3];uv=cam@k.T
        uv=uv[:,:2]/np.maximum(uv[:,2:],1e-9)
        for joint in np.flatnonzero((confidence[frame].ravel()>=.6)&(cam[:,2]>.05)):
            if not np.isfinite(uv[joint]).all():continue
            x,y=np.rint(uv[joint]).astype(int)
            if not(1<=x<z.shape[1]-1 and 1<=y<z.shape[0]-1) or not reference_masks[index,y,x]:continue
            patch=z[y-1:y+2,x-1:x+2];valid=np.isfinite(patch)&(patch>.05)
            if 'valid' in depth:valid &= depth['valid'][index][y-1:y+2,x-1:x+2]
            values=patch[valid]
            if len(values)<3 or np.ptp(values)>.08:continue
            predicted=float(np.median(values));ray=np.linalg.inv(k)@np.r_[uv[joint],1.]
            point=(ray*predicted-e[:,3])@e[:,:3]
            src.append(point);dst.append(world[joint]);weights.append(float(confidence[frame].ravel()[joint]));frames.append(frame)
    report=fit_registration(np.asarray(src).reshape(-1,3),np.asarray(dst).reshape(-1,3),np.asarray(weights),np.asarray(frames))
    report.update(anchor_provenance=provenance,source_sha256=hashlib.sha256(Path(row['source']).read_bytes()).hexdigest(),
                  depth_measured=False,source_trajectory_edited=False,correspondence_method='Projected observed 3D landmarks inside source hand/arm masks; robust raw-depth correspondence')
    np.savez_compressed(work/'hand-alignment-correspondences.npz',source=np.asarray(src).reshape(-1,3),target=np.asarray(dst).reshape(-1,3),weights=weights,source_frames=frames)
    (work/'hand-alignment.json').write_text(json.dumps(report,indent=2))
    return report
