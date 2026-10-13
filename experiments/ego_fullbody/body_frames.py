"""Anatomical body frames from topology and neutral geometry, no robot names."""
import numpy as np
import torch

def unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)

def frame(left, up):
    y = unit(up)
    x = unit(left - (left * y).sum(-1, keepdim=True) * y)
    z = unit(torch.cross(x, y, dim=-1))
    return torch.stack((x, y, z), -1)

def angle(a, b):
    delta = a @ b.transpose(-1, -2)
    skew = torch.stack((delta[..., 2, 1] - delta[..., 1, 2],
                        delta[..., 0, 2] - delta[..., 2, 0],
                        delta[..., 1, 0] - delta[..., 0, 1]), -1) * .5
    return torch.atan2(skew.norm(dim=-1), .5 * (delta.diagonal(dim1=-2, dim2=-1).sum(-1) - 1))

class BodyFrames:
    def __init__(self, body, fk, wrists, ankles):
        parents = np.asarray(body.spec.parents)
        def ancestry(j):
            out = []
            while j >= 0:
                out.append(j)
                j = int(parents[j])
            return out
        def common(pair):
            return next(j for j in ancestry(pair[0]) if j in ancestry(pair[1]))
        self.chest = common(wrists)
        self.pelvis = common(ankles)
        with torch.no_grad():
            q = fk.lower.new_zeros((1, len(fk.active_indices)))
            R, p = fk(q, fk.identity[None])
            def anchor(endpoint, parent):
                chain = list(reversed(ancestry(endpoint)[:ancestry(endpoint).index(parent)]))
                return next(j for j in chain if (p[0, j] - p[0, parent]).norm() > .025)
            self.shoulders = [anchor(j, self.chest) for j in wrists]
            self.hips = [anchor(j, self.pelvis) for j in ankles]
            up = p[:, self.shoulders].mean(1) - p[:, self.hips].mean(1)
            chest = frame(p[:, self.shoulders[0]] - p[:, self.shoulders[1]], up)
            pelvis = frame(p[:, self.hips[0]] - p[:, self.hips[1]], fk.lower.new_tensor([[0., 1., 0.]]))
            self.chest_local = R[:, self.chest].transpose(-1, -2) @ chest
            self.pelvis_local = R[:, self.pelvis].transpose(-1, -2) @ pelvis
        names = body.spec.joint_names
        self.metadata = dict(chest=names[self.chest], pelvis=names[self.pelvis],
                             shoulders=[names[j] for j in self.shoulders], hips=[names[j] for j in self.hips],
                             chest_local=self.chest_local.cpu().numpy().tolist(), pelvis_local=self.pelvis_local.cpu().numpy().tolist())

    def __call__(self, rotations):
        chest = rotations[:, self.chest] @ self.chest_local
        pelvis = rotations[:, self.pelvis] @ self.pelvis_local
        return chest, pelvis.transpose(-1, -2) @ chest, pelvis
