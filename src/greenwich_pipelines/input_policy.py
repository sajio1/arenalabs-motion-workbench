"""Shared six-landmark visibility for human-to-robot spatial encoding."""
import torch
from alphamotion.engine.descriptor import geom_batch

POLICY = 'human-six-pelvis-head-hands-feet-v1'
ROLE_SLOTS = (0, 1, 2, 3, 4, 5)


def observed_indices(role_indices):
    if len(role_indices) != 6:
        raise ValueError('Expected root/head/L-hand/R-hand/L-foot/R-foot metadata')
    indices = [int(role_indices[i]) for i in ROLE_SLOTS]
    if len(set(indices)) != 6 or min(indices) < 0 or indices[0] != 0:
        raise ValueError('Six distinct anatomical input joints are required')
    return indices


@torch.no_grad()
def encode_observed(codec, pose9, spec, dof, role_indices, contact=None):
    """Frames are independent batch items; hidden joints remain skeleton metadata."""
    indices = observed_indices(role_indices)
    if max(indices) >= pose9.shape[1]:
        raise ValueError('Observed joint is outside source skeleton')
    mask = torch.ones(pose9.shape[:2], dtype=torch.bool, device=pose9.device)
    mask[:, indices] = False
    gb = geom_batch(codec.geom(spec, dof), len(pose9))
    fp, fr = codec.model.encode_features(pose9, gb, joint_mask=mask, con=contact)
    ip, ir = codec.model.ints_from_features(fp, fr)
    return torch.cat([ip[0], ir[0]], 1).long()


def receipt(names, role_indices):
    return dict(policy=POLICY, observed_joints=[names[i] for i in observed_indices(role_indices)],
                source_joint_count=len(names), temporal_model=False,
                refinement='Original complete body and operation refinement')
