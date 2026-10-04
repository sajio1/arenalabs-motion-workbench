"""Semantic palm frames, independent of trajectory time and robot rest pose.

Columns: distal (toward middle knuckle), transverse, outward palmar normal.
World basis changes act on the LEFT; tool-frame calibrations act on the RIGHT.
"""
import numpy as np


def unit(v):
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(n < 1e-8) or not np.isfinite(v).all():
        raise ValueError('Degenerate/non-finite palm landmarks or axes')
    return v/n


def palm_frame(distal, normal):
    d = unit(distal)
    n = unit(np.asarray(normal) - d*np.sum(d*normal, axis=-1, keepdims=True))
    return np.stack([d, np.cross(n, d), n], axis=-1)


def egodex_palm_calibration(wrist_rotation, wrist, middle, index, little, side):
    """Compatibility import for existing ego callers; behavior unchanged."""
    from .input_adapters.ego import egodex_palm_calibration as calibrate
    return calibrate(wrist_rotation, wrist, middle, index, little, side)


def robot_palm_local(descriptor, role):
    """Return semantic palm axes in Greenwich wrist-link coordinates.

    Descriptor vectors are in the vendor MJCF wrist-link frame (Z-up model).
    Missing axes fail explicitly: guessing from a trajectory is not calibration.
    """
    if role not in descriptor:
        raise ValueError(f'Missing robot palm-frame calibration for {role}')
    axes = descriptor[role]
    vendor = palm_frame(axes['distal'], axes['normal'])
    yup_to_vendor = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    return yup_to_vendor.T @ vendor


def map_world_palms(episode, descriptor, alignment):
    return {role: alignment @ track.rotation @ robot_palm_local(descriptor, role).T
            for role, track in episode.tracks.items() if role.endswith('_wrist')}


def humi_site_episode_to_world_palms(episode):
    """Compatibility import; HuMI calibration lives in its source adapter."""
    from .input_adapters.humi import humi_site_episode_to_world_palms as adapt
    return adapt(episode)
