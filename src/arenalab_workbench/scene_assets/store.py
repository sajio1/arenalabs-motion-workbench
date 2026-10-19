"""Disk-backed scratch arrays; sequence length never determines host/GPU memory.

Legacy NPZ members are extracted once to NPY memmaps. New reconstructions use
frame-chunked Zarr. Neither representation is part of permanent delivery assets.
"""
from pathlib import Path
import json
import shutil
import tempfile
import zipfile

import numpy as np

from .formats import sha256
from .geometry import backproject


class NpzArrays:
    def __init__(self, path, scratch=None):
        self.path = Path(path)
        self.root = Path(scratch) if scratch else self.path.parent / (self.path.stem + "-arrays")
        self.root.mkdir(parents=True, exist_ok=True)
        fingerprint = dict(source=str(self.path.resolve()), sha256=sha256(self.path))
        receipt = self.root / "source.json"
        if receipt.exists() and json.loads(receipt.read_text()).get("sha256") != fingerprint["sha256"]:
            raise ValueError("Scratch array source changed; choose a fresh directory")
        if not receipt.exists():
            with tempfile.NamedTemporaryFile(mode="w",dir=self.root,delete=False,suffix=".json.tmp") as stream:
                json.dump(fingerprint,stream)
                temporary=Path(stream.name)
            temporary.replace(receipt)
        with zipfile.ZipFile(self.path) as archive:
            self.names = {Path(n).stem: n for n in archive.namelist() if n.endswith(".npy")}
        self.loaded = {}

    def __contains__(self, name):
        return name in self.names

    def __getitem__(self, name):
        if name not in self.loaded:
            target = self.root / f"{name}.npy"
            if not target.exists():
                with tempfile.NamedTemporaryFile(dir=self.root,delete=False,suffix=".npy.tmp") as out:
                    temporary=Path(out.name)
                    with zipfile.ZipFile(self.path) as archive, archive.open(self.names[name]) as source:
                        shutil.copyfileobj(source, out, length=4*1024*1024)
                # Concurrent workers may extract the same immutable member.
                # Never expose partial arrays or replace an open Windows memmap.
                if target.exists():
                    temporary.unlink()
                else:
                    try:
                        temporary.replace(target)
                    except PermissionError:
                        if not target.exists():
                            raise
                        temporary.unlink()
            self.loaded[name] = np.load(target, mmap_mode="r", allow_pickle=False)
        return self.loaded[name]


class GeometryStore:
    def __init__(self, path):
        self.path = Path(path)
        if self.path.is_dir():
            import zarr
            self.arrays = zarr.open_group(str(self.path), mode="r")
        else:
            self.arrays = NpzArrays(self.path)
        self.metadata = json.loads(self.path.with_suffix(".json").read_text()) if self.path.with_suffix(".json").exists() else {}

    def __len__(self):
        return len(self.arrays["ids"])

    def __getitem__(self, name):
        return self.arrays[name]

    def __contains__(self, name):
        return name in self.arrays

    def frame(self, index):
        result = {name: np.asarray(self.arrays[name][index]) for name in
                  ["ids", "depth", "valid", "k", "poses", "images", "labels"]}
        result["points"] = backproject(result["depth"], result["k"], result["poses"])
        return result


def create_geometry(path, count, height, width):
    import zarr
    from numcodecs import Blosc
    root = zarr.open_group(str(path), mode="w-")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    for name, tail, dtype in [("ids", (), "i8"), ("depth", (height,width), "f4"),
        ("valid", (height,width), "bool"), ("labels", (height,width), "u2"),
        ("images", (height,width,3), "u1"), ("poses", (4,4), "f8"),
        ("k", (3,3), "f8"), ("preprocessing_affine", (3,3), "f8")]:
        root.create_dataset(name, shape=(count,*tail), chunks=(1,*tail), dtype=dtype, compressor=compressor)
    root.attrs.update(scratch_only=True, point_coordinates="computed on demand from Z-depth, K and source-world optical poses")
    return root
