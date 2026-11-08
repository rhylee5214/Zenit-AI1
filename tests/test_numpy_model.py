import numpy as np
import pytest

from zenit.model.config import ZenitConfig
from zenit.model.numpy_model import NumpyZenit


def tiny_cfg(**kw):
    base = dict(vocab_size=50, d_model=16, n_layers=2, n_heads=4, n_kv_heads=2,
                ffn_hidden=24, max_seq_len=12)
    base.update(kw)
    return ZenitConfig(**base)


@pytest.mark.parametrize("tie", [True, False])
def test_gradients_match_finite_differences(tie):
    cfg = tiny_cfg(tie_embeddings=tie)
    model = NumpyZenit(cfg, dtype=np.float64, seed=1)
    # bigger init so the check is not dominated by tiny values
    for k, v in model.params.items():
        if v.ndim == 2:
            model.params[k] = v * 5
    rng = np.random.default_rng(0)
    idx = rng.integers(0, cfg.vocab_size, size=(2, 7))
    tgt = rng.integers(0, cfg.vocab_size, size=(2, 7))
    tgt[0, :2] = -1   # ignored positions must not leak gradient
    _, _, cache = model.forward(idx, tgt)
    grads = model.backward(cache)

    eps = 1e-6
    for name, param in model.params.items():
        flat = param.reshape(-1)
        g = grads[name].reshape(-1)
        for j in rng.choice(flat.size, size=min(6, flat.size), replace=False):
            old = flat[j]
            flat[j] = old + eps
            lp = model.forward(idx, tgt)[1]
            flat[j] = old - eps
            lm = model.forward(idx, tgt)[1]
            flat[j] = old
            num = (lp - lm) / (2 * eps)
            assert abs(num - g[j]) <= 1e-5 + 1e-4 * abs(num), (name, j, num, g[j])


def test_kv_cache_matches_full_forward():
    cfg = tiny_cfg()
    model = NumpyZenit(cfg, dtype=np.float64, seed=3)
    rng = np.random.default_rng(1)
    idx = rng.integers(0, cfg.vocab_size, size=(1, 10))
    full, _, _ = model.forward(idx)
    cache = model.new_cache()
    out_a = model.forward_cached(idx[:, :6], cache)
    out_b = model.forward_cached(idx[:, 6:8], cache)
    out_c = model.forward_cached(idx[:, 8:], cache)
    stepwise = np.concatenate([out_a, out_b, out_c], axis=1)
    assert np.allclose(full, stepwise, atol=1e-9)


def test_parameter_count_formula():
    cfg = tiny_cfg()
    assert NumpyZenit(cfg).num_parameters() == cfg.num_parameters()


def test_save_and_load_roundtrip(tmp_path):
    cfg = tiny_cfg()
    model = NumpyZenit(cfg, seed=5)
    model.save(tmp_path / "m")
    again = NumpyZenit.load(tmp_path / "m")
    idx = np.array([[1, 2, 3]])
    assert np.allclose(model.forward(idx)[0], again.forward(idx)[0])
