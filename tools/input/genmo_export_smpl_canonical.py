"""Export immutable GEM-SMPL predictions through the generic MotionClip API.

All clips use the same SMPL-X skeleton and coordinate conversion.  This file
does not infer contacts, edit poses, or alter the model's world trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from gem.utils.body_model.smplx_lite import SmplxLite
from greenwich_motion_sdk.motion import MotionClip
from smplx.joint_names import JOINT_NAMES


ALIASES = {
    "pelvis": "Hips", "left_hip": "LeftLeg", "right_hip": "RightLeg",
    "left_knee": "LeftShin", "right_knee": "RightShin",
    "left_ankle": "LeftFoot", "right_ankle": "RightFoot",
    "left_foot": "LeftToeBase", "right_foot": "RightToeBase",
    "left_shoulder": "LeftArm", "right_shoulder": "RightArm",
    "left_elbow": "LeftForeArm", "right_elbow": "RightForeArm",
    "left_wrist": "LeftHand", "right_wrist": "RightHand",
}
def anatomical_name(name: str) -> str:
    if name in ALIASES:
        return ALIASES[name]
    side, _, distal = name.partition("_")
    if side in ("left", "right") and distal.startswith(("index", "middle", "pinky", "ring", "thumb")):
        return side.title() + "Hand" + distal.title()
    return "".join(part.title() for part in name.split("_"))


NAMES = [anatomical_name(name) for name in JOINT_NAMES[:55]]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--body-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    if not np.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("Invalid FPS")

    prediction = torch.load(args.prediction, map_location="cpu", weights_only=False)
    params = prediction["body_params_global"]
    arr = lambda key: np.asarray(params[key].detach().cpu(), dtype=np.float64)
    global_orient, body_pose, transl, betas = map(arr, ("global_orient", "body_pose", "transl", "betas"))
    count = len(global_orient)
    if global_orient.shape != (count, 3) or body_pose.shape != (count, 63) or transl.shape != (count, 3) or betas.shape != (count, 10):
        raise ValueError("Unexpected GEM-SMPL parameter shapes")
    if not all(np.isfinite(x).all() for x in (global_orient, body_pose, transl, betas)):
        raise ValueError("Nonfinite GEM-SMPL parameters")

    model = SmplxLite(model_path=str(args.body_model))
    with torch.no_grad():
        rest = model.get_skeleton(torch.as_tensor(betas[:1], dtype=torch.float32))[0].numpy().astype(np.float64)
        neutral_vertices = model(
            torch.zeros(1, 63), torch.as_tensor(betas[:1], dtype=torch.float32),
            torch.zeros(1, 3), torch.zeros(1, 3),
        )[0].numpy().astype(np.float64)
    neutral_floor = float(neutral_vertices[:, 1].min())
    foot_landmark_heights_cm = ((rest[[7, 10, 8, 11], 1] - neutral_floor) * 100.0).tolist()
    parents = model.parents.numpy().astype(np.int64)
    if parents[0] != -1 or len(NAMES) != len(parents) or len(set(NAMES)) != 55:
        raise ValueError("Unexpected SMPL-X hierarchy")
    offsets = np.zeros((55, 3), np.float64)
    for joint in range(1, 55):
        offsets[joint] = rest[joint] - rest[parents[joint]]

    default_pose = model.other_default_pose.detach().numpy().astype(np.float64).reshape(33, 3)
    aa = np.concatenate((global_orient[:, None], body_pose.reshape(count, 21, 3),
                         np.broadcast_to(default_pose, (count, 33, 3))), axis=1)
    local_rotation = Rotation.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(count, 55, 3, 3)
    global_rotation = np.empty_like(local_rotation)
    world = np.empty((count, 55, 3), np.float64)
    for joint in range(55):
        parent = int(parents[joint])
        if parent < 0:
            global_rotation[:, joint] = local_rotation[:, joint]
            world[:, joint] = transl + rest[0]
        else:
            global_rotation[:, joint] = global_rotation[:, parent] @ local_rotation[:, joint]
            world[:, joint] = world[:, parent] + np.einsum("tij,j->ti", global_rotation[:, parent], offsets[joint])
    clip = MotionClip(
        NAMES, parents, offsets * 100.0, global_rotation, world * 100.0,
        np.arange(count, dtype=np.float64) / args.fps,
        {"source": str(args.prediction), "source_sha256": digest(args.prediction),
         "body_model_sha256": digest(args.body_model), "format": "gem-smplx55-skeleton-from-smpl22",
         "fps": args.fps, "units": "cm", "up": "Y", "identity_reference_frame": 0,
         "foot_landmark_heights_cm": foot_landmark_heights_cm,
         "neutral_mesh_floor_cm": neutral_floor * 100.0,
         "hand_scope": "SMPL-X static default hand pose; video model predicts wrists but not articulated fingers"},
    ).validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clip.save(args.output)
    print(json.dumps({"frames": count, "joints": 55, "output": str(args.output),
                      "sha256": digest(args.output), "foot_landmark_heights_cm": foot_landmark_heights_cm,
                      "root_y_cm": [float(world[:, 0, 1].min() * 100), float(world[:, 0, 1].max() * 100)]}))


if __name__ == "__main__":
    main()
