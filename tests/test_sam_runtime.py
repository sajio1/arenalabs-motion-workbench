"""Reuse immutable inputs while isolating text tensors and tracker state."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'experiments/manipulation4d'))
from sam_runtime import TextFeatureCache, video_session


class Backbone:
    def __init__(self):
        self.calls = 0

    def forward_text(self, captions, **kwargs):
        self.calls += 1
        return {'features': torch.tensor([len(s) for s in captions]),
                'nested': [torch.tensor([3])], 'label': 'original'}


def test_text_cache_is_bounded_and_outputs_cannot_mutate_it():
    backbone = Backbone()
    original = backbone.forward_text
    with TextFeatureCache(backbone, capacity=1) as cache:
        a = backbone.forward_text(['cup'], device='cuda')
        a['features'][0] = 999
        b = backbone.forward_text(['cup'], device='cuda')
        assert b['features'].tolist() == [3]
        b['nested'][0][0] = 999
        assert backbone.forward_text(['cup'], device='cuda')['nested'][0].tolist() == [3]
        backbone.forward_text(['lid'], device='cuda')
        backbone.forward_text(['cup'], device='cuda')  # Evicted, must recompute.
        assert backbone.calls == 3
        assert cache.report() == dict(hits=2, misses=3, capacity=1)
        backbone.forward_text(['cup'], device='cuda', another_option=True)
        assert backbone.calls == 4  # Unsupported options bypass caching.
    assert backbone.forward_text == original
    assert not cache.entries


def test_video_session_closes_after_failure():
    calls = []
    class Predictor:
        def handle_request(self, request):
            calls.append(request)
            return {'session_id': 's'}
    with pytest.raises(RuntimeError, match='test failure'):
        with video_session(Predictor(), '/frames') as session:
            assert session == 's'
            raise RuntimeError('test failure')
    assert [c['type'] for c in calls] == ['start_session', 'close_session']
