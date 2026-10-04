"""Derive geometric foot observations from immutable GEM-X SOMA vertices.

This is a source observation, not a force/contact ground truth or a change to
the predicted body parameters.  The same fixed rules apply to every clip.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from gem.utils.soma_utils.soma_layer import SomaLayer


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prediction', type=Path, required=True)
    p.add_argument('--soma-assets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fps', type=float, default=30.)
    a = p.parse_args()
    if not np.isfinite(a.fps) or a.fps <= 0:
        raise ValueError('Invalid fps')
    raw = torch.load(a.prediction, map_location='cpu', weights_only=False)['body_params_global']
    body = {k: v.float() if torch.is_tensor(v) else torch.tensor(v, dtype=torch.float32) for k, v in raw.items()}
    count = len(body['global_orient'])
    model = SomaLayer(data_root=str(a.soma_assets), low_lod=True, device='cpu', identity_model_type='mhr', mode='warp')
    rig = np.load(a.soma_assets / 'SOMA_neutral.npz', allow_pickle=False)
    names = [str(n) for n in rig['joint_names'][1:]]
    with torch.no_grad():
        neutral = model.static_forward(torch.zeros(1, 77, 3), body['identity_coeffs'][:1],
                                       body['scale_params'][:1], torch.zeros(1, 3))
        posed = []
        for start in range(0, count, 16):
            stop = min(start + 16, count)
            chunk = {k: v[start:stop] for k, v in body.items()}
            posed.append(model(**chunk)['vertices'].numpy())
    vertex = np.concatenate(posed, axis=0)
    neutral_vertex = neutral['vertices'][0].numpy()
    joint = neutral['joints'][0].numpy()
    observations = np.zeros((count, 4), bool)
    heights = np.empty((count, 4), np.float32)
    material_height = []
    material_speed = []
    regions = []
    for side_index, side in enumerate(('Left', 'Right')):
        ankle = joint[names.index(side+'Foot')]
        toe = joint[names.index(side+'ToeBase')]
        other = joint[names.index(('Right' if side == 'Left' else 'Left')+'Foot')]
        xy = neutral_vertex[:, [0, 2]]
        own = np.minimum(np.linalg.norm(xy-ankle[[0,2]],axis=1), np.linalg.norm(xy-toe[[0,2]],axis=1))
        opposite = np.linalg.norm(xy-other[[0,2]],axis=1)
        foot = np.flatnonzero((neutral_vertex[:,1] < ankle[1] + .015) & (own < .18) & (own < opposite))
        axis = toe[[0,2]] - ankle[[0,2]]
        axis /= np.linalg.norm(axis)
        distance = (xy[foot] - ankle[[0,2]]) @ axis
        cut = np.median(distance)
        for patch_index, patch in enumerate((foot[distance < cut], foot[distance >= cut])):
            if len(patch) < 20:
                raise ValueError('Source mesh foot patch is too small')
            points = vertex[:,patch]
            height = points[:,:,1] * 100
            speed_edge = np.linalg.norm(np.diff(points,axis=0),axis=2) * (100*a.fps)
            # A material point must be low and slow on both adjacent edges.
            speed = np.maximum(np.vstack([speed_edge[:1],speed_edge]),
                               np.vstack([speed_edge,speed_edge[-1:]]))
            col = side_index*2 + patch_index
            heights[:,col] = height.min(1)
            material_height.append(height)
            material_speed.append(speed)
            regions.append(len(patch))
    for col in range(4):
        observations[:,col] = ((material_height[col] <= 3.) &
                                (material_speed[col] <= 15.)).any(1)
    source_contact = observations.reshape(count,2,2).any(2)
    flat_support = observations.reshape(count,2,2).all(2)
    clearance = heights.reshape(count,2,2).min(2)
    if not np.isfinite(clearance).all():
        raise ValueError('Nonfinite mesh heights')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, schema=np.array('genmo.soma.source_foot_surface.v1'),
                        prediction_sha256=np.array(sha(a.prediction)), fps=np.array(a.fps),
                        source_contact_proxy=source_contact, flat_support=flat_support,
                        source_clearance_cm=clearance, heel_toe_low_slow=observations,
                        heel_toe_min_height_cm=heights,
                        patch_vertex_count=np.asarray(regions))
    print(json.dumps({'source_frames':count,'contact_frames':source_contact.sum(0).tolist(),
                      'flat_frames':flat_support.sum(0).tolist(),
                      'source_clearance_min_max_cm':[float(clearance.min()),float(clearance.max())],
                      'patch_vertices':regions,'output':str(a.output),'sha256':sha(a.output)}))

if __name__ == '__main__':
    main()
