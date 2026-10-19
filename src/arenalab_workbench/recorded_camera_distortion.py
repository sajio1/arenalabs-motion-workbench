"""Map a pinhole render into the recorded RealSense color image domain.

Inverse Brown coefficients map distorted pixels to ideal rays. This is the
destination-to-source mapping required by cv2.remap, not initUndistortRectifyMap.
No task trajectories or camera extrinsics are fitted or modified here.
"""
import numpy as np


def inverse_brown_render_map(intrinsic, coefficients, width, height):
    k = np.asarray(intrinsic, dtype=float)
    k1, k2, p1, p2, k3 = np.asarray(coefficients, dtype=float)
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    x = (u - k[0, 2]) / k[0, 0]
    y = (v - k[1, 2]) / k[1, 1]
    r2 = x*x + y*y
    radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
    xu = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
    yu = y*radial + 2*p2*x*y + p1*(r2 + 2*y*y)
    return ((xu*k[0, 0] + k[0, 2]).astype(np.float32),
            (yu*k[1, 1] + k[1, 2]).astype(np.float32))


def recorded_render_map(source_metadata, intrinsic, width, height):
    """Return None for sources without an explicit recorded calibration."""
    import json
    from pathlib import Path
    if source_metadata.get('dataset') != 'Real robot':
        return None
    info = json.loads((Path(source_metadata['source'])/'meta/info.json').read_text())
    color = info.get('recording_metadata',{}).get('camera_calibration',{}).get('observation.images.head_cam',{}).get('streams',{}).get('color',{})
    distortion = color.get('distortion')
    if distortion is None:return None
    if distortion['model'].lower().replace('_', ' ') != 'inverse brown conrady':
        raise ValueError('Unsupported recorded color distortion: '+distortion['model'])
    return inverse_brown_render_map(intrinsic, distortion['coeffs'], width, height)
