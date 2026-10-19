"""Cached frames must retain their exact identity and original frame order."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

source = Path(__file__).parents[1] / 'src/greenwich_pipelines/frame_cache.py'
module_spec = importlib.util.spec_from_file_location('tested_frame_cache', source)
cache = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(cache)


class Codec:
    def decode_full(self, codes, spec, dof, contact=True):
        x = codes.float()
        return x[..., None].expand(-1, -1, 6), x[..., None].expand(-1, -1, 3), x


@pytest.mark.parametrize('mode', ['none', 'all', 'partial'])
def test_reuse_checks_identity_and_preserves_order(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(cache, 'target_key', lambda *args: 'target')
    codec = Codec()
    codes = torch.arange(12).reshape(6, 2)
    ids = np.arange(10, 16)
    times = np.arange(6) / 30
    baseline = codec.decode_full(codes, None, None)
    expected = [v.clone() for v in baseline]
    selected = []
    if mode != 'none':
        # A shuffled cache exercises destination order, not just concatenation.
        order = np.array([4, 0, 5, 2, 1, 3])
        cached_codes = codes.numpy()[order].copy()
        cached_times = times[order].copy()
        if mode == 'partial':
            cached_codes[0, 0] += 1  # Wrong integer code: decode frame 4.
            cached_times[1] += .01  # Wrong timestamp: decode frame 0.
        selected = [int(i) for i in order if mode == 'all' or i not in (4, 0)]
        folder = tmp_path / 'chunk-000'
        folder.mkdir()
        np.savez_compressed(folder / 'robot.npz', target_key='target',
                            frame_ids=ids[order], timestamps_s=cached_times,
                            codes=cached_codes,
                            **{k: v.numpy()[order] + 100 for k, v in zip(
                                ['raw_rot6d', 'decoder_position', 'model_contact'], baseline)})
        for v in expected:
            v[selected] += 100
        # A later file with a different target must never replace valid frames.
        folder = tmp_path / 'chunk-999'
        folder.mkdir()
        np.savez_compressed(folder / 'robot.npz', target_key='other')
    result = cache.decode_cached(codec, codes, None, None, ids, times, tmp_path, chunk=2)
    assert result[-1]['reused_frames'] == len(selected)
    assert result[-1]['decoded_frames'] == len(ids) - len(selected)
    assert all(torch.equal(a, b) for a, b in zip(expected, result[:3]))
