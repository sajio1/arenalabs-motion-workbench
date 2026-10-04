"""Export GEM-X SOMA motion through the generic MotionClip source interface.

The reconstruction is read without modification.  SOMA's rig and predicted
identity define the fixed bind skeleton; the same conversion is used for every
video.  No robot or action dependent correction is applied here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from greenwich_motion_sdk.motion import MotionClip


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--soma-assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    pred = torch.load(args.prediction, map_location="cpu", weights_only=False)
    body = pred["body_params_global"]
    def array(name: str) -> np.ndarray:
        value = body[name]
        return np.asarray(value.detach().cpu() if torch.is_tensor(value) else value, dtype=np.float64)

    global_orient = array("global_orient")
    body_pose = array("body_pose")
    transl = array("transl")
    identity = array("identity_coeffs")
    scale = array("scale_params")
    frames = len(global_orient)
    if global_orient.shape != (frames, 3) or body_pose.shape != (frames, 228) or transl.shape != (frames, 3):
        raise ValueError("Unexpected GEM-X SOMA parameter shapes")
    if identity.shape[0] != frames or scale.shape[0] != frames:
        raise ValueError("Identity/scale frames do not match motion")

    rig = np.load(args.soma_assets / "SOMA_neutral.npz", allow_pickle=False)
    names = [str(x) for x in rig["joint_names"][1:]]
    parent_ids = np.asarray(rig["joint_parent_ids"], dtype=np.int64)
    parents = parent_ids[1:] - 1
    if len(names) != 77 or parents[0] != -1:
        raise ValueError("Unexpected SOMA rig hierarchy")
    orient = np.asarray(rig["t_pose_world"][:, :3, :3], dtype=np.float64)

    from gem.utils.soma_utils.soma_layer import SomaLayer

    soma = SomaLayer(data_root=str(args.soma_assets), low_lod=True, device="cpu", identity_model_type="mhr", mode="warp")
    with torch.no_grad():
        rest = soma.get_skeleton(torch.as_tensor(identity[:1], dtype=torch.float32),
                                 torch.as_tensor(scale[:1], dtype=torch.float32))[0].numpy().astype(np.float64)
    if rest.shape != (77, 3):
        raise ValueError("Unexpected shaped SOMA skeleton")

    local_offsets = np.zeros((77, 3), dtype=np.float64)
    for j in range(1, 77):
        old_j, old_p = j + 1, int(parent_ids[j + 1])
        local_offsets[j] = orient[old_p].T @ (rest[j] - rest[old_p - 1])

    aa = np.concatenate((global_orient[:, None], body_pose.reshape(frames, 76, 3)), axis=1)
    posed = Rotation.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(frames, 77, 3, 3)
    global_rot = np.empty_like(posed)
    world = np.empty((frames, 77, 3), dtype=np.float64)
    world[:, 0] = transl
    for j in range(77):
        old_j, old_p = j + 1, int(parent_ids[j + 1])
        local = orient[old_p].T @ posed[:, j] @ orient[old_j]
        if j == 0:
            global_rot[:, j] = local
        else:
            p = int(parents[j])
            global_rot[:, j] = global_rot[:, p] @ local
            world[:, j] = world[:, p] + np.einsum("tij,j->ti", global_rot[:, p], local_offsets[j])

    clip = MotionClip(
        names, parents, local_offsets * 100.0, global_rot, world * 100.0,
        np.arange(frames, dtype=np.float64) / args.fps,
        {"source": str(args.prediction), "source_sha256": sha256(args.prediction),
         "format": "gem-x-soma77", "fps": args.fps,
         "soma_rig_sha256": sha256(args.soma_assets / "SOMA_neutral.npz"),
         "identity_reference_frame": 0,
         "units": "cm", "up": "Y"},
    ).validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clip.save(args.output)
    print(json.dumps({"frames": frames, "joints": len(names), "output": str(args.output),
                      "sha256": sha256(args.output),
                      "root_y_cm": [float(world[:, 0, 1].min() * 100), float(world[:, 0, 1].max() * 100)]}))


if __name__ == "__main__":
    main()
