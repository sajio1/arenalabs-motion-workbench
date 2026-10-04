"""Weight registry: our artifacts from HF, third-party from their origins.

Our HF repo carries ONLY AlphaMotion's own weights and curated motion assets.
The complete core download is roughly 1 GB because native playback requires
the lossless packed dual-stream code library in addition to model weights.
Big third-party
models (GENMO perception, T5) stay at their original sources and download on
first use — cleaner licensing, small repo.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .paths import weights_dir

# Windows: HF cache symlinks need admin rights; disable before any hub import.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

# The spatial codecs. greenwich = v0.1 (C1); bumblebee / megatron / prime = the
# v0.2 Greenwich-v2 size ladder (S / M / L). All four live in the same HF repo.
MODELS = ("greenwich", "bumblebee", "megatron", "prime")

ARTIFACTS = {
    # artifact -> (subdir in the HF repo, files)
    "greenwich": ("greenwich", ("model.safetensors", "config.json")),
    "bumblebee": ("bumblebee", ("model.safetensors", "config.json")),
    "megatron": ("megatron", ("model.safetensors", "config.json")),
    "prime": ("prime", ("model.safetensors", "config.json")),
    "equator_a": ("equator_a", ("model.safetensors", "config.json")),
    "equator_b": ("equator_b", ("model.safetensors", "config.json")),
    "atlas": ("atlas", ("atlas.npz", "atlas_meta.json")),
    # library.npz contains editor tokens and boundary codes. Native playback
    # additionally requires the packed frame codes and root trajectories; a
    # release missing any of these files is not a usable release.
    "library": ("library", ("library.npz", "library_meta.json",
                             "library_codes.npy", "library_root.npy",
                             "library_root_meta.json")),
    "embodiments": ("embodiments", ("name_embeddings.pt",)),
}


def hf_repo() -> str:
    from .config import CONFIG
    return CONFIG.hf_repo


def resolve(name: str, download: bool = True) -> Path:
    """Local dir for an artifact, downloading from HF on first use."""
    sub, files = ARTIFACTS[name]
    local = weights_dir() / sub
    if all((local / f).exists() for f in files):
        return local
    if not download:
        raise FileNotFoundError(f"artifact '{name}' not downloaded "
                                f"(expected under {local})")
    from huggingface_hub import hf_hub_download
    local.mkdir(parents=True, exist_ok=True)
    for f in files:
        hf_hub_download(repo_id=hf_repo(), filename=f"{sub}/{f}",
                        local_dir=weights_dir())
    return local


# What `alphamotion download` fetches by default: the v0.1 service stack plus
# the recommended v2 codec. Pass models=... to pick others.
CORE = ("greenwich", "megatron", "equator_a", "equator_b", "atlas", "library",
        "embodiments")


def download_all(models: tuple[str, ...] | None = None) -> dict[str, str]:
    names = list(CORE) if models is None else \
        [n for n in CORE if n not in MODELS] + list(models)
    out = {}
    for name in names:
        out[name] = str(resolve(name))
    return out


def status() -> dict[str, bool]:
    return {name: all((weights_dir() / sub / f).exists() for f in files)
            for name, (sub, files) in ARTIFACTS.items()}
