"""Raw vision reuse must not reuse text-conditioned or temporal state."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from arenalab_workbench.sam_features import ImageFeatureCache, feature_bytes


class Nested:
    def __init__(self, tensors, mask=None):
        self.tensors, self.mask = tensors, mask


class Detector:
    def __init__(self):
        self.calls = 0
        trunk=SimpleNamespace(forward=self.forward_trunk)
        self.backbone = SimpleNamespace(forward_image=self.forward_image,
                                        vision_backbone=SimpleNamespace(trunk=trunk))

    def forward_image(self, samples, **kwargs):
        return self.backbone.vision_backbone.trunk.forward(samples)

    def forward_trunk(self, samples):
        self.calls += 1
        value = samples + 1
        return {'backbone_fpn': [Nested(value)], 'vision_features': value}

    def _get_img_feats(self, backbone_out, img_ids):
        if 'backbone_fpn' not in backbone_out:
            result = dict(backbone_out)
            result.update(self.backbone.forward_image(backbone_out['img_batch_all_stages'][img_ids]))
            return self._get_img_feats(result, img_ids)
        return backbone_out, backbone_out['vision_features']


def test_vision_cache_preserves_values_and_current_language():
    detector = Detector()
    original = detector._get_img_feats
    frames = torch.arange(12).reshape(3, 4).float()
    with ImageFeatureCache(detector, max_bytes=16) as cache:
        def get(index, text='cup', inputs=frames):
            return detector._get_img_feats({'img_batch_all_stages': inputs, 'text': text}, torch.tensor([index]))
        first, feature = get(0)
        feature[:] = 99  # Must not corrupt the CPU cache.
        second, feature = get(0, 'lid')
        assert second['text'] == 'lid'
        assert torch.equal(feature, frames[:1] + 1)
        assert second['backbone_fpn'][0].tensors is feature
        assert feature_bytes(second['backbone_fpn']) == 16
        feature[:] = 98
        assert torch.equal(get(0)[1], frames[:1] + 1)
        get(1)
        get(1)  # Capacity exceeded: recompute rather than evict admitted frame 0.
        assert detector.calls == 3
        assert cache.report()['hits'] == 2
        assert cache.report()['skipped_capacity'] == 2
        replacement = frames + 100
        assert torch.equal(get(0, inputs=replacement)[1], replacement[:1] + 1)
        assert detector.calls == 4  # A different video cannot hit the old cache.
    assert detector._get_img_feats == original
    assert not cache.entries


def test_overlapping_batches_share_frames_without_sharing_text():
    detector = Detector()
    frames = torch.arange(8).reshape(4, 2).float()
    def get(ids, text):
        return detector._get_img_feats({'img_batch_all_stages': frames, 'text': text}, torch.tensor(ids))
    with ImageFeatureCache(detector) as cache:
        get([0, 1], 'cup')
        get([1, 2], 'lid')
        result, value = get([0, 2], 'hand')
        assert result['text'] == 'hand'
        assert torch.equal(value, frames[[0, 2]] + 1)
        assert detector.calls == 2
        assert cache.report()['reused_frames'] == 2
        # A different batch shape retains its original numerical path.
        assert torch.equal(get([0], 'cup')[1], frames[:1] + 1)
        assert detector.calls == 3


def test_zero_budget_runs_the_original_trunk_without_retaining_features():
    detector = Detector()
    frames = torch.ones(1, 4)
    with ImageFeatureCache(detector, max_bytes=0) as cache:
        for _ in range(2):
            _, output = detector._get_img_feats({'img_batch_all_stages': frames}, torch.tensor([0]))
            assert torch.equal(output, frames + 1)
        assert detector.calls == 2
        assert not cache.entries and cache.report()['cpu_bytes'] == 0
