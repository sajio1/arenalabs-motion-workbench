"""Bounded CPU cache of prompt-independent SAM vision-trunk outputs.

Scoped to one immutable video session/model/autocast configuration. Detection,
language-conditioned features, tracker memories and masks are never cached.
"""
import copy
from dataclasses import dataclass

import torch


def map_features(value, transform, memo=None):
    """Copy supported vision containers, preserving shared tensor references."""
    memo = {} if memo is None else memo
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if isinstance(value, torch.Tensor):
        out = transform(value)
    elif isinstance(value, dict):
        out = {k: map_features(v, transform, memo) for k, v in value.items()}
    elif isinstance(value, list):
        out = [map_features(v, transform, memo) for v in value]
    elif isinstance(value, tuple):
        out = tuple(map_features(v, transform, memo) for v in value)
    elif hasattr(value, 'tensors') and hasattr(value, 'mask'):
        # SAM NestedTensor is not registered as a torch pytree container.
        out = copy.copy(value)
        out.tensors = map_features(value.tensors, transform, memo)
        out.mask = map_features(value.mask, transform, memo)
    elif value is None or isinstance(value, (int, float, bool, str, torch.Size)):
        out = value
    else:
        raise TypeError(f'Unsupported SAM vision feature container: {type(value).__name__}')
    memo[identity] = out
    return out


def feature_bytes(value, batch_size=None):
    size = 0
    def count(tensor):
        nonlocal size
        elements=tensor.numel()
        if batch_size and tensor.ndim and tensor.shape[0]==batch_size:
            elements//=batch_size
        size += elements * tensor.element_size()
        return tensor
    map_features(value, count)
    return size


@dataclass
class CachedTensor:
    value: torch.Tensor
    batched: bool


def combine_frames(values, device, memo=None):
    """Reassemble frame-independent raw features; retain tensor aliases."""
    memo = {} if memo is None else memo
    key = tuple(id(v) for v in values)
    if key in memo:
        return memo[key]
    first = values[0]
    if isinstance(first, CachedTensor):
        value = torch.cat([v.value for v in values], 0) if first.batched else first.value
        out = value.to(device, copy=True)
    elif isinstance(first, dict):
        out = {k: combine_frames([v[k] for v in values], device, memo) for k in first}
    elif isinstance(first, list):
        out = [combine_frames([v[i] for v in values], device, memo) for i in range(len(first))]
    elif isinstance(first, tuple):
        out = tuple(combine_frames([v[i] for v in values], device, memo) for i in range(len(first)))
    elif hasattr(first, 'tensors') and hasattr(first, 'mask'):
        out = copy.copy(first)
        out.tensors = combine_frames([v.tensors for v in values], device, memo)
        out.mask = combine_frames([v.mask for v in values], device, memo)
    else:
        out = first
    memo[key] = out
    return out


class ImageFeatureCache:
    def __init__(self, detector, max_bytes=8 * 1024**3):
        self.detector = detector
        self.original_get = detector._get_img_feats
        self.trunk = detector.backbone.vision_backbone.trunk
        self.original_image = self.trunk.forward
        self.max_bytes = max_bytes
        self.entries = {}
        self.inputs = None
        self.pending = None
        self.bytes = self.hits = self.misses = self.skipped_capacity = 0
        self.reused_frames = 0

    def get_feats(self, backbone_out, img_ids):
        if 'backbone_fpn' in backbone_out:
            return self.original_get(backbone_out, img_ids)
        inputs = backbone_out['img_batch_all_stages']
        if self.inputs is not inputs:
            self.entries.clear()
            self.bytes = 0
            self.inputs = inputs
        previous = self.pending
        self.pending = tuple(torch.unique(img_ids).tolist())
        try:
            return self.original_get(backbone_out, img_ids)
        finally:
            self.pending = previous

    def forward_image(self, samples, **kwargs):
        if self.pending is None or len(self.pending) != len(samples):
            return self.original_image(samples, **kwargs)
        # Keep batch shape in the identity to preserve the original numerical
        # path, while still sharing overlapping equal-sized prefetch batches.
        mode = (tuple(samples.shape), tuple(samples.stride()), tuple(sorted(kwargs.items())), str(samples.device),
               samples.dtype, torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda'))
        keys = [(frame, *mode) for frame in self.pending]
        if all(key in self.entries for key in keys):
            self.hits += 1
            self.reused_frames += len(keys)
            return combine_frames([self.entries[key] for key in keys], samples.device)
        self.misses += 1
        output = self.original_image(samples, **kwargs)
        size=feature_bytes(output,batch_size=len(samples))
        # Transfer each tensor once for the whole batch, then split on CPU.
        # Per-frame CUDA -> CPU copies would synchronize hundreds of times.
        host = None
        for index, key in enumerate(keys):
            if key in self.entries:
                continue
            if self.bytes + size > self.max_bytes:
                self.skipped_capacity += 1
                continue
            if host is None:
                host = map_features(output, lambda t: t.detach().to('cpu', copy=True))
            def pack(tensor):
                batched = bool(tensor.ndim and tensor.shape[0] == len(samples))
                part = tensor[index:index+1] if batched else tensor
                return CachedTensor(part.clone(), batched)
            frame = map_features(host, pack)
            # Keep admitted frames; sequential scans would thrash an LRU.
            self.entries[key] = frame
            self.bytes += size
        return output

    def __enter__(self):
        self.detector._get_img_feats = self.get_feats
        self.trunk.forward = self.forward_image
        return self

    def __exit__(self, *exc):
        self.detector._get_img_feats = self.original_get
        self.trunk.forward = self.original_image
        self.entries.clear()
        self.inputs = None

    def report(self):
        return dict(target='shared_vision_trunk',hits=self.hits, misses=self.misses, reused_frames=self.reused_frames, cpu_bytes=self.bytes,
                    capacity_bytes=self.max_bytes, skipped_capacity=self.skipped_capacity)
