"""Estimate one camera correction from explicit image correspondences.

Diagnostic output only: does not change source poses, inference or live assets.
Annotations contain frame, side (0/1), pixel [u,v], and split (fit/validation).
The pixel must identify the source EE marker, not an arbitrary visible hand point.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def run(source_path, annotation_path, output_path):
    raw = Path(source_path).read_bytes()
    source = np.load(source_path)
    measurements = json.loads(Path(annotation_path).read_text())
    rows = measurements['observations']
    k = source['intrinsic']
    points = np.array([(source['tcp_zup'][a['frame'], a['side']]
                       - source['camera'][a['frame'], :3, 3])
                      @ source['camera'][a['frame'], :3, :3] for a in rows])
    pixels = np.array([a['pixel'] for a in rows], dtype=float)
    x, y = ((pixels-k[:2, 2])/np.diag(k)[:2]).T
    k1, k2, p1, p2, k3 = measurements['inverse_brown_coefficients']
    r2 = x*x + y*y
    radial = 1+k1*r2+k2*r2*r2+k3*r2*r2*r2
    ideal = np.stack([x*radial+2*p1*x*y+p2*(r2+2*x*x),
                      y*radial+2*p2*x*y+p1*(r2+2*y*y)], axis=1)
    ideal = ideal*np.diag(k)[:2]+k[:2, 2]
    fit = np.array([a['split'] == 'fit' for a in rows])
    if fit.sum() < 4 or (~fit).sum() < 1:
        raise ValueError('Need at least four fit observations and independent validation')

    def project(v):
        q = points@Rotation.from_rotvec(v[:3]).as_matrix().T+v[3:]
        return (q@k.T)[:, :2]/q[:, 2:]

    result = least_squares(lambda v: (project(v)[fit]-ideal[fit]).ravel(),
                           np.zeros(6), loss='soft_l1', f_scale=3)
    correction = np.eye(4)
    correction[:3, :3] = Rotation.from_rotvec(result.x[:3]).as_matrix()
    correction[:3, 3] = result.x[3:]
    before = np.linalg.norm(project(np.zeros(6))-ideal, axis=1)
    after = np.linalg.norm(project(result.x)-ideal, axis=1)
    report = dict(estimated=True,diagnostic_only=True,
                  source_sha256=hashlib.sha256(raw).hexdigest(),
                  annotation_sha256=hashlib.sha256(Path(annotation_path).read_bytes()).hexdigest(),
                  annotation_provenance=measurements['provenance'],
                  corrected_camera_from_recorded_camera=correction.tolist(),
                  camera_pose_application='world_from_recorded_camera @ inverse(correction)',
                  before_pixels=before.tolist(),after_pixels=after.tolist(),
                  fit_mean_pixels=float(after[fit].mean()),
                  validation_mean_pixels=float(after[~fit].mean()),
                  validation_max_pixels=float(after[~fit].max()),
                  source_trajectory_edited=False)
    assert Path(source_path).read_bytes() == raw
    Path(output_path).write_text(json.dumps(report,indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source'); parser.add_argument('annotations'); parser.add_argument('output')
    args = parser.parse_args()
    print(json.dumps(run(args.source,args.annotations,args.output),indent=2))
