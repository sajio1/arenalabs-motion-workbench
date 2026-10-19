"""Bounded, process-local reuse of text-only SAM features.

No detection, tracker state, masks or source frames are shared across prompts.
The caller keeps model weights, device and autocast mode fixed for its lifetime.
"""
from collections import OrderedDict
from contextlib import contextmanager

import torch
from torch.utils._pytree import tree_map


@contextmanager
def video_session(predictor, frames):
    response = predictor.handle_request({'type': 'start_session', 'resource_path': str(frames)})
    session = response['session_id']
    try:
        yield session
    finally:
        predictor.handle_request({'type': 'close_session', 'session_id': session})


class TextFeatureCache:
    def __init__(self, backbone, capacity=16):
        self.backbone = backbone
        self.original = backbone.forward_text
        self.capacity = capacity
        self.entries = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def clone(value):
        # Consumers get their own tensors, just as with a fresh forward pass.
        return tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, value)

    def forward(self, captions, *args, **kwargs):
        # Only the released video path's explicit (captions, device=...) call.
        # Unknown call options retain the original implementation.
        if args or set(kwargs) - {'device'}:
            return self.original(captions, *args, **kwargs)
        key = (tuple(captions), str(kwargs.get('device')),
               torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda'))
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.clone(self.entries[key])
        self.misses += 1
        output = self.original(captions, **kwargs)
        self.entries[key] = self.clone(output)
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return output

    def __enter__(self):
        self.backbone.forward_text = self.forward
        return self

    def __exit__(self, *exc):
        self.backbone.forward_text = self.original
        self.entries.clear()

    def report(self):
        return dict(hits=self.hits, misses=self.misses, capacity=self.capacity)
