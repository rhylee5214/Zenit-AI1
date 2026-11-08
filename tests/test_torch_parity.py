"""Runs on machines with PyTorch installed (skipped otherwise).

Checks the PyTorch model reproduces the NumPy reference exactly from the
same weights, and the KV cache path matches the full forward pass."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from zenit.model.config import ZenitConfig  # noqa: E402
from zenit.model.numpy_model import NumpyZenit  # noqa: E402
from zenit.model.torch_model import KVCache, ZenitModel, load_zenit, save_zenit  # noqa: E402


def cfg():
    return ZenitConfig(vocab_size=60, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                       ffn_hidden=48, max_seq_len=16)


def test_logits_match_numpy(tmp_path):
    np_model = NumpyZenit(cfg(), seed=4)
    np_model.save(tmp_path / "m")
    t_model = load_zenit(tmp_path / "m")
    idx = np.array([[3, 7, 1, 9, 22, 5, 8]])
    ref, ref_loss, _ = np_model.forward(idx, idx)
    with torch.no_grad():
        out, loss = t_model(torch.from_numpy(idx), torch.from_numpy(idx))
    assert np.allclose(ref, out.numpy(), atol=1e-4)
    assert abs(ref_loss - loss.item()) < 1e-4


def test_kv_cache_matches_full():
    model = ZenitModel(cfg()).eval()
    idx = torch.randint(0, 60, (1, 12))
    with torch.no_grad():
        full, _ = model(idx)
        cache = KVCache(model.cfg, 1, "cpu", torch.float32)
        a, _ = model(idx[:, :5], cache=cache)
        b, _ = model(idx[:, 5:9], cache=cache)
        c = [model(idx[:, i:i + 1], cache=cache)[0] for i in range(9, 12)]
    step = torch.cat([a, b] + c, dim=1)
    assert torch.allclose(full, step, atol=1e-5)


def test_backward_runs_and_roundtrip(tmp_path):
    model = ZenitModel(cfg())
    idx = torch.randint(0, 60, (2, 10))
    _, loss = model(idx, idx)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
    save_zenit(model, tmp_path / "t")
    again = load_zenit(tmp_path / "t")
    with torch.no_grad():
        assert torch.allclose(model.eval()(idx)[0], again(idx)[0], atol=1e-6)
