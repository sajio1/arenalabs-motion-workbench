"""Public, reversible storage primitives; no learned decoder or point-cloud archive."""
from pathlib import Path
import hashlib
import json

import numpy as np


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def rle_encode(mask):
    """COCO-compatible uncompressed counts, column-major, starting with zeros."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("RLE requires an H x W binary mask")
    flat = mask.ravel(order="F")
    boundaries = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.r_[0, boundaries, len(flat)]).tolist()
    if len(flat) and flat[0]:
        counts.insert(0, 0)
    return {"size": list(mask.shape), "counts": counts}


def rle_decode(rle):
    shape = tuple(rle["size"])
    counts = np.asarray(rle["counts"], dtype=np.int64)
    if (counts < 0).any() or counts.sum() != np.prod(shape):
        raise ValueError("Invalid RLE run lengths")
    return np.repeat(np.arange(len(counts)) % 2, counts).astype(bool).reshape(shape, order="F")


def depth_mm(depth):
    """0=invalid; uint16 millimetres; out-of-range pixels are invalid, never wrap."""
    depth = np.asarray(depth)
    result = np.zeros(depth.shape, dtype="uint16")
    valid = np.isfinite(depth) & (depth >= .0005) & (depth <= 65.535)
    result[valid] = np.rint(depth[valid] * 1000).astype("uint16")
    return result


def sparse_cloud(depth, intrinsic, camera_pose, count, rgb=None, seed=0, labels=None):
    """Sample observed depth pixels, no duplication or invented points."""
    if count not in {1024, 4096, 8192}:
        raise ValueError("Supported point counts: 1024, 4096, 8192")
    depth = np.asarray(depth)
    y, x = np.where(np.isfinite(depth) & (depth > 0))
    selected = np.random.default_rng(seed).choice(len(y), min(count, len(y)), replace=False)
    y, x = y[selected], x[selected]
    rays = np.stack([x, y, np.ones_like(x)], axis=-1) @ np.linalg.inv(intrinsic).T
    points = rays * depth[y, x, None]
    points = points @ camera_pose[:3, :3].T + camera_pose[:3, 3]
    result = {"xyz": points.astype("float32")}
    if rgb is not None:
        result["rgb"] = np.asarray(rgb)[y, x].astype("uint8")
    if labels is not None:
        result["instance_ids"] = np.asarray(labels)[y, x].astype("uint16")
    return result


def write_arrays(path, arrays, frame_ids=None, fps=30):
    """One Parquet row per timestamp, with explicit tensor shape/dtype metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    values = {name: np.asarray(value) for name, value in arrays.items()}
    lengths = {len(value) for value in values.values()}
    if len(lengths) != 1:
        raise ValueError("Array lengths disagree")
    n = lengths.pop()
    ids = np.arange(n) if frame_ids is None else np.asarray(frame_ids)
    if ids.shape != (n,) or (np.diff(ids) <= 0).any():
        raise ValueError("Frame ids must be unique and increasing")
    table = pa.table({"source_frame": ids, "timestamp_s": ids / fps, **{
        name: value.reshape(n, -1).tolist() for name, value in values.items()}})
    metadata = {name: {"shape": list(value.shape[1:]), "dtype": str(value.dtype)} for name, value in values.items()}
    table = table.replace_schema_metadata({b"tensor_schema": json.dumps(metadata).encode(), b"fps": str(fps).encode()})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def read_arrays(path):
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    metadata = json.loads(table.schema.metadata[b"tensor_schema"])
    return {name: np.asarray(table[name].to_pylist(), dtype=spec["dtype"]).reshape((len(table), *spec["shape"]))
            for name, spec in metadata.items()}


def storage_budget(points=4096, fps=30, bytes_per_point=18):
    """XYZ float32 + RGB uint8 + label uint16 + validity uint8, decimal GB/hour."""
    return points * fps * 3600 * bytes_per_point / 1e9
