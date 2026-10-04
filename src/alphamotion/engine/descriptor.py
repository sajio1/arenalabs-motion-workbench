"""Embodiment descriptors: what the codec needs to know about a body.

(spec, dof[J,19], rest[J,3,3], qnames, xml) — built from a bundled cache entry
or live from any MJCF/URDF. Port of eval_newbody's target surface.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..paths import assets_root
from .nets.rotations import SkeletonSpec
from .spatial import static_cond

PART_KW = {
    "leg": ("hip", "knee", "ankle", "thigh", "shin", "calf", "foot", "toe", "leg"),
    "arm": ("shoulder", "elbow", "wrist", "hand", "arm", "clav", "scap"),
    "torso": ("waist", "torso", "spine", "pelvis", "chest", "trunk", "neck",
              "head", "base"),
}


def build_from_mjcf(xml: str | Path, name: str):
    """MJCF (or URDF converted upstream) -> full descriptor tuple."""
    import mujoco

    from ..embodiment import dof_features as PD
    from ..embodiment import mjcf_build as PG
    model = mujoco.MjModel.from_xml_path(str(xml))
    keep, frame = PG.build_merge(model)
    spec = PG.build_spec(model, name, keep)
    D, snames, *_rest_parts, Rst = PD.mjcf_dof(
        name, str(xml), {"parents": spec.parents,
                         "joint_names": spec.joint_names})
    qn = [[x for x in s.split(",") if x] for s in snames]
    return spec, D.astype(np.float32), Rst.astype(np.float32), qn, str(xml)


def build_from_cache(cache: str | Path, name: str):
    """Bundled/packaged spec+dof npz pair -> descriptor (no meshes needed)."""
    cache = Path(cache)
    d = np.load(cache / f"{name}_spec.npz", allow_pickle=True)
    spec = SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"],
                        [str(x) for x in d["joint_names"]])
    z = np.load(cache / f"{name}_dof.npz", allow_pickle=True)
    return spec, z["dof"].astype(np.float32), z["rest"].astype(np.float32), \
        None, None


def bundled_cache(version: int = 1) -> Path:
    """version 1: the v0.1 descriptor set (greenwich / C1 codec, Atlas, library).
    version 2: the LAFAN-v3 skeleton descriptors the v2 codecs (bumblebee /
    megatron / prime) were trained against. Rest offsets differ by up to ~90 cm
    for some bodies, so a codec must always be fed ITS OWN descriptor set."""
    return assets_root() / ("embodiments" if version == 1 else "embodiments_v2")


def target_geom(spec: SkeletonSpec, dof, device, lut=None, sem_encoder="qwen3"):
    """Static conditioning + joint-name text semantics + DOF descriptor.
    sem_encoder must match the codec's training encoder (qwen3 for greenwich,
    siglip for the v2 models)."""
    from ..embodiment.semantics_map import build_name_embeddings, embed_matrix
    st = static_cond(spec, device)
    lut = lut or build_name_embeddings(list(spec.joint_names), device=device,
                                       encoder=sem_encoder)
    missing = [n for n in spec.joint_names if n not in lut]
    if missing:
        lut = build_name_embeddings(
            list(lut.keys()) + list(spec.joint_names), device=device,
            encoder=sem_encoder)
    sem = embed_matrix(spec.joint_names, lut).to(device)
    return {"st": st, "sem": sem, "dof": torch.as_tensor(dof, device=device)}


def geom_batch(g: dict, B: int) -> dict:
    st = g["st"]
    return dict(rest_off=st["rest_off"][None].expand(B, -1, -1),
                bonelen=st["bonelen"][None].expand(B, -1),
                depth=st["depth"][None].expand(B, -1),
                hop=st["hop"][None].expand(B, -1, -1),
                sem=g["sem"][None].expand(B, -1, -1),
                dof=g["dof"][None].expand(B, -1, -1))
