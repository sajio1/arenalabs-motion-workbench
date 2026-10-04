"""Explicit coordinate-frame adapters for motion sources.

Greenwich consumes global joint rotations and root-relative positions in one
model-canonical frame.  Skeleton topology does not encode axis conventions, so
data sources must declare their frame conversion instead of relying on a
renderer or target robot to repair it implicitly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch

from ..engine.nets.rotations import matrix_to_rot6d, rot6d_to_matrix
from ..paths import assets_root


@dataclass(frozen=True)
class PoseFrameAdapter:
    """Rigid source-frame to AlphaMotion-canonical rotation.

    The transform is applied on the left to global joint rotations and to
    root-relative position vectors.  World-root trajectories are separate
    data; callers may rotate them explicitly with :meth:`vectors` when their
    source is in the same frame.
    """

    name: str
    matrix: np.ndarray
    metadata: dict

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError(f"{self.name}: frame matrix must be finite [3,3]")
        if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5):
            raise ValueError(f"{self.name}: frame matrix is not orthonormal")
        if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-5):
            raise ValueError(f"{self.name}: frame matrix must be a rotation")
        object.__setattr__(self, "matrix", matrix)

    def _torch_matrix(self, value: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.matrix, dtype=value.dtype,
                               device=value.device)

    def pose9(self, pose: torch.Tensor) -> torch.Tensor:
        """Transform ``[..., J, 9]`` global-rot6d + relative-position poses."""
        if pose.shape[-1] != 9:
            raise ValueError(f"pose9 last dimension must be 9, got {pose.shape}")
        frame = self._torch_matrix(pose)
        rotations = frame @ rot6d_to_matrix(pose[..., :6])
        positions = torch.einsum("ij,...j->...i", frame, pose[..., 6:9])
        return torch.cat((matrix_to_rot6d(rotations), positions), dim=-1)

    def rot6d(self, rotations: torch.Tensor) -> torch.Tensor:
        """Transform global rotations represented as ``[..., J, 6]``."""
        if rotations.shape[-1] != 6:
            raise ValueError(
                f"rot6d last dimension must be 6, got {rotations.shape}")
        frame = self._torch_matrix(rotations)
        return matrix_to_rot6d(frame @ rot6d_to_matrix(rotations))

    def vectors(self, vectors: torch.Tensor | np.ndarray):
        """Rotate position/displacement vectors while preserving input type."""
        if isinstance(vectors, torch.Tensor):
            frame = self._torch_matrix(vectors)
            return torch.einsum("ij,...j->...i", frame, vectors)
        value = np.asarray(vectors)
        if value.shape[-1] != 3:
            raise ValueError(f"vector last dimension must be 3, got {value.shape}")
        return np.einsum("ij,...j->...i", self.matrix, value)

    def inverse(self) -> "PoseFrameAdapter":
        return PoseFrameAdapter(
            name=f"{self.name}:inverse",
            matrix=self.matrix.T,
            metadata={**self.metadata, "inverse_of": self.name},
        )


def _registry() -> dict:
    path = assets_root() / "embodiments" / "frame_adapters.json"
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported frame-adapter schema in {path}")
    return payload["adapters"]


def frame_adapter_names() -> list[str]:
    return sorted(_registry())


def load_frame_adapter(name: str) -> PoseFrameAdapter:
    try:
        entry = _registry()[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown frame adapter '{name}'; available={frame_adapter_names()}"
        ) from exc
    return PoseFrameAdapter(name=name, matrix=np.asarray(entry["matrix"]),
                            metadata={key: value for key, value in entry.items()
                                      if key != "matrix"})
