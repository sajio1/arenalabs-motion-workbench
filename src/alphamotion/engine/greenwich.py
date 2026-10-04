"""Greenwich — the cross-embodiment spatial codec (release surface).

One latent code space shared by every body. encode() a motion observed on any
registered embodiment; decode() it onto any other.

v0.2 ships four checkpoints behind the same class:

| model       | params | codes            | notes |
|-------------|--------|------------------|-------|
| greenwich   | 18.1M  | [T,256,20] (8-lvl) | v0.1 C1 codec; the Atlas, library and Equator A/B are built on ITS codes |
| bumblebee   | 3.7M   | [T,132,13] (32-lvl) | Greenwich-v2 recipe, size S (d128, 3+3 layers) |
| megatron    | 18.0M  | [T,132,13] (32-lvl) | Greenwich-v2 recipe, size M (d256, 4+4 layers) — recommended |
| prime       | 57.7M  | [T,132,13] (32-lvl) | Greenwich-v2 recipe, size L (d384, 6+6 layers) |

The v2 models (bumblebee / megatron / prime) share one code space with each
other but NOT with greenwich (C1): their codes are 4 position slots + 128
rotation slots over a 13-dim 32-level FSQ grid, they were trained on the
LAFAN v3 + AMASS (timebase-fixed) corpora with SigLIP joint-name semantics and
the LAFAN-v3 skeleton descriptors (assets/embodiments_v2), and their position
head's root slot carries the root height above ground in metres. Use them
through encode / decode / decode_full / translate; the timeline editor,
Atlas and Equator stay on greenwich until they are re-indexed on v2 codes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .descriptor import geom_batch, target_geom
from .nets.codec import DualTranslator
from .nets.rotations import SkeletonSpec
from .spatial import build_global, fk_pos

SEM_DIM = {"siglip": 768, "qwen3": 1024}

# Locked architecture contracts. Startup fails on any mismatch.
RELEASE_CONTRACT = {
    "n_latent": 128, "fsq_stages": 1,
    "fsq_levels": "8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8",
    "d_model": 256, "n_heads": 8, "enc_layers": 4, "dec_layers": 4,
    "sem_encoder": "qwen3", "use_dof": True, "rot_head": "free",
}
_V2_COMMON = {
    "n_latent": 128, "n_latent_p": 4, "fsq_stages": 1,
    "fsq_levels": "32,32,32,32,32,32,32,32,32,32,32,32,32",
    "sem_encoder": "siglip", "use_dof": True, "rot_head": "free",
    "contact_chan": True, "root_head": False, "single_codebook": False,
    "vis_residual": False, "rot_only": False,
}
MODEL_CONTRACTS = {
    "greenwich": RELEASE_CONTRACT,
    "bumblebee": {**_V2_COMMON, "d_model": 128, "n_heads": 4, "enc_layers": 3, "dec_layers": 3},
    "megatron": {**_V2_COMMON, "d_model": 256, "n_heads": 8, "enc_layers": 4, "dec_layers": 4},
    "prime": {**_V2_COMMON, "d_model": 384, "n_heads": 12, "enc_layers": 6, "dec_layers": 6},
}
# Which bundled embodiment descriptor set each model was trained against.
MODEL_EMBODIMENTS = {"greenwich": 1, "bumblebee": 2, "megatron": 2, "prime": 2}
DEFAULT_MODEL = "greenwich"


def _validate_release_config(cfg: dict, model: str = "greenwich") -> None:
    contract = MODEL_CONTRACTS[model]
    bad = {key: (cfg.get(key), expected)
           for key, expected in contract.items()
           if cfg.get(key, False if isinstance(expected, bool) else
                      (0 if key == "n_latent_p" else None)) != expected}
    if bad:
        raise ValueError(f"Greenwich checkpoint does not match the locked "
                         f"AlphaMotion release contract for '{model}': {bad}")


def build_model(cfg: dict, device: str) -> DualTranslator:
    """Construct the codec from a run config (dual_common.build_model port)."""
    from .nets.blocks import parse_fsq_levels
    model = DualTranslator(
        d_model=cfg.get("d_model", 256),
        n_heads=cfg.get("n_heads", 8),
        enc_layers=cfg.get("enc_layers", 4),
        dec_layers=cfg.get("dec_layers", 4),
        n_latent=cfg.get("n_latent", 32),
        fsq_stages=cfg.get("fsq_stages", 2),
        fsq_levels=parse_fsq_levels(cfg.get("fsq_levels", "8,8,8,5,5,5")),
        single_codebook=cfg.get("single_codebook", False),
        n_judges=cfg.get("n_judges", 0),
        use_dof=cfg.get("use_dof", False),
        rot_head=cfg.get("rot_head", "free"),
        sem_dim=SEM_DIM[cfg.get("sem_encoder", "siglip")],
        vis_residual=cfg.get("vis_residual", False),
        rot_only=cfg.get("rot_only", False),
        contact_chan=cfg.get("contact_chan", False),
        root_head=cfg.get("root_head", False),
        n_latent_p=cfg.get("n_latent_p", 0),
    ).to(device)
    return model


class Greenwich:
    def __init__(self, model: DualTranslator, cfg: dict, device: str,
                 name: str = "greenwich"):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.name = name
        self.sem_encoder = cfg.get("sem_encoder", "siglip")
        self.embodiment_version = MODEL_EMBODIMENTS.get(name, 1)
        self._geoms: dict[str, dict] = {}

    # --------------------------------------------------------------- shape --
    @property
    def n_slots(self) -> tuple[int, int]:
        """(position slots, rotation slots) of one frame's code tensor."""
        m = self.model
        return (getattr(m, "n_latent_p", m.n_latent), m.n_latent)

    @property
    def code_shape(self) -> tuple[int, int]:
        """[slots, fsq dims] of one frame's integer code tensor."""
        p, r = self.n_slots
        return (p + r, self.model.fsqr.f)

    # ------------------------------------------------------------- loading --
    @classmethod
    def load(cls, run_dir: str | Path | None = None,
             device: str | None = None, model: str | None = None) -> "Greenwich":
        """model: 'greenwich' (v0.1 default) | 'bumblebee' | 'megatron' | 'prime'.
        Defaults to ALPHAMOTION_MODEL / CONFIG.greenwich_model. run_dir overrides
        the weight location but the config is still checked against `model`."""
        from ..config import CONFIG
        from ..weights import resolve
        device = device or CONFIG.device
        name = model or CONFIG.greenwich_model
        if name not in MODEL_CONTRACTS:
            raise ValueError(f"unknown Greenwich model '{name}'; "
                             f"choose one of {sorted(MODEL_CONTRACTS)}")
        run = Path(run_dir) if run_dir else resolve(name)
        cfg = json.load(open(run / "config.json"))
        _validate_release_config(cfg, name)
        net = build_model(cfg, device)
        sd = _load_state(run)
        net.load_state_dict(sd, strict=True)
        net.eval()
        return cls(net, cfg, device, name)

    # ---------------------------------------------------------- embodiment --
    def embodiment(self, name: str):
        """(spec, dof, rest) of a bundled body from the descriptor set THIS model
        was trained on (v1 for greenwich, v2 = LAFAN-v3 skeletons for the others)."""
        from .descriptor import build_from_cache, bundled_cache
        spec, dof, rest, _, _ = build_from_cache(
            bundled_cache(self.embodiment_version), name)
        return spec, dof, rest

    # ---------------------------------------------------------- descriptors --
    def geom(self, spec: SkeletonSpec, dof) -> dict:
        key = spec.name
        if key not in self._geoms:
            self._geoms[key] = target_geom(spec, dof, self.device,
                                           sem_encoder=self.sem_encoder)
        return self._geoms[key]

    def pose9(self, local_or_global6d: torch.Tensor, spec: SkeletonSpec,
              is_global: bool = False, reach: float | None = None,
              frame_adapter=None):
        """[T,J,6] rotations -> the [T,J,9] observation the encoder eats.

        Cached corpora store LOCAL rotations (compose first — feeding local
        rot6d straight to FK keeps bone lengths plausible while displacing
        joints ~12 cm, the classic silent bug); adapters that already composed
        global rotations pass is_global=True.
        """
        if is_global:
            gr6 = local_or_global6d.float()
            gp = torch.from_numpy(fk_pos(gr6.numpy(), spec)).float()
            r = reach or float(gp.norm(dim=-1).max())
        else:
            gr6, gp, r = build_global(local_or_global6d, spec, "cpu")
            gr6, gp = gr6.cpu(), gp.cpu()
            r = reach or r
        pose = torch.cat([gr6, gp / max(r, 1e-6)], -1)
        if frame_adapter is not None:
            from ..embodiment.frames import load_frame_adapter
            adapter = load_frame_adapter(frame_adapter) \
                if isinstance(frame_adapter, str) else frame_adapter
            pose = adapter.pose9(pose)
        return pose.to(self.device), r

    # ------------------------------------------------------------- codes ----
    @torch.no_grad()
    def encode(self, pose9: torch.Tensor, spec: SkeletonSpec, dof,
               contact: torch.Tensor | None = None):
        """[T,J,9] -> integer codes [T, P+R, f]: P position slots then R rotation
        slots (greenwich: 128+128 x 20; v2 models: 4+128 x 13).
        contact [T,J] in [0,1] (v2 models only): per-joint soft contact label;
        None = no contact information (bit-identical to a model without it)."""
        g = self.geom(spec, dof)
        gb = geom_batch(g, pose9.shape[0])
        if contact is not None:
            fp, fr = self.model.encode_features(pose9, gb, con=contact.to(pose9.device))
        else:
            fp, fr = self.model.encode_features(pose9, gb)
        ipS, irS = self.model.ints_from_features(fp, fr)
        return torch.cat([ipS[0], irS[0]], 1).long()

    @torch.no_grad()
    def decode(self, codes: torch.Tensor, spec: SkeletonSpec, dof):
        """[T,P+R,f] int codes -> global rot6d [T,J,6] on the target body."""
        return self.decode_full(codes, spec, dof)[0]

    @torch.no_grad()
    def decode_full(self, codes: torch.Tensor, spec: SkeletonSpec, dof,
                    contact: bool = False):
        """Both decoder heads: (rot6d [T,J,6], pos [T,J,3] root-relative,
        reach-normalised). The position head carries placement information the
        rotation stream alone cannot express on a mismatched skeleton (owner
        call 0814: pose the mesh from position + rot together).

        v2 models: pos[:, root, 0] is the root height above ground in metres
        (the only per-frame world quantity the codec resolves); the rest of the
        root slot is unsupervised. contact=True additionally returns per-joint
        contact logits [T,J] (v2 models only)."""
        g = self.geom(spec, dof)
        P, R = self.n_slots
        assert codes.shape[1] == P + R and codes.shape[2] == self.model.fsqr.f, \
            f"codes {tuple(codes.shape)} do not fit model '{self.name}' " \
            f"(expects [T,{P + R},{self.model.fsqr.f}])"
        zp = self.model.fsqp.from_ints(codes[None, :, :P])
        zr = self.model.fsqr.from_ints(codes[None, :, P:])
        gb = geom_batch(g, codes.shape[0])
        if contact:
            return self.model.decode(zp, zr, gb, ret_con=True)
        return self.model.decode(zp, zr, gb)

    @torch.no_grad()
    def translate(self, pose9: torch.Tensor, src_spec, src_dof,
                  tgt_spec, tgt_dof):
        """Source observation -> target-body global rot6d, one hop."""
        g_s = self.geom(src_spec, src_dof)
        zp, zr = self.model.encode(pose9, geom_batch(g_s, pose9.shape[0]))
        g_t = self.geom(tgt_spec, tgt_dof)
        rot, _ = self.model.decode(zp, zr, geom_batch(g_t, pose9.shape[0]))
        return rot


def _load_state(run: Path) -> dict:
    """model.safetensors preferred; model.pt (pure state_dict) accepted."""
    st = run / "model.safetensors"
    if st.exists():
        from safetensors.torch import load_file
        return load_file(st)
    return torch.load(run / "model.pt", map_location="cpu")
