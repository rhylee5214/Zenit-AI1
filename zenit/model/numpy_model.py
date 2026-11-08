"""ZENIT transformer in pure NumPy, with a hand-written backward pass.

Why this exists:
  * It trains real ZENIT weights on machines without PyTorch or a GPU.
  * It is the reference implementation. The PyTorch model must produce
    the same logits from the same weights (see tests/test_torch_parity.py).
  * Every operation is visible, which makes the architecture easy to study.

Architecture (decoder-only, pre-norm, Llama-style):
    tokens -> embedding
    N x [ x += Attention(RMSNorm(x)) ; x += SwiGLU(RMSNorm(x)) ]
    RMSNorm -> logits (embedding matrix reused as the output head)

Attention uses rotary position embeddings (RoPE) and grouped-query
attention (GQA). Weight matrices use the PyTorch nn.Linear layout
[out_features, in_features], so y = x @ W.T.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import ZenitConfig


# ---------------------------------------------------------------- primitives
def rms_norm(x, w, eps):
    r = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * r * w, r


def rms_norm_backward(g, x, r, w):
    gw = g * w
    dw = np.sum(g * x * r, axis=tuple(range(g.ndim - 1)))
    dx = r * (gw - x * (r * r) * np.mean(gw * x, axis=-1, keepdims=True))
    return dx, dw


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))   # numerically stable


def silu(x):
    return x * sigmoid(x)


def silu_grad(x):
    s = sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def rotate_half_t(y):
    h = y.shape[-1] // 2
    return np.concatenate([y[..., h:], -y[..., :h]], axis=-1)


def rope_tables(head_dim, max_len, theta, dtype):
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    t = np.arange(max_len, dtype=np.float64)
    freqs = np.outer(t, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(dtype), np.sin(emb).astype(dtype)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# --------------------------------------------------------------------- model
class NumpyZenit:
    def __init__(self, cfg: ZenitConfig, params: dict | None = None,
                 dtype=np.float32, seed: int = 0) -> None:
        self.cfg = cfg
        self.dtype = dtype
        self.params = params if params is not None else self._init_params(seed)
        self.cos, self.sin = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta, dtype)

    # ------------------------------------------------------------------ init
    def _init_params(self, seed: int) -> dict:
        c = self.cfg
        rng = np.random.default_rng(seed)
        std = c.init_std
        resid_std = std / np.sqrt(2 * c.n_layers)   # scaled init for residual outputs
        hd, kv = c.head_dim, c.n_kv_heads * c.head_dim

        def normal(shape, s):
            return rng.normal(0.0, s, size=shape).astype(self.dtype)

        p = {"tok_emb.weight": normal((c.vocab_size, c.d_model), std)}
        for i in range(c.n_layers):
            pre = f"layers.{i}."
            p[pre + "attn_norm.weight"] = np.ones(c.d_model, self.dtype)
            p[pre + "attn.wq.weight"] = normal((c.n_heads * hd, c.d_model), std)
            p[pre + "attn.wk.weight"] = normal((kv, c.d_model), std)
            p[pre + "attn.wv.weight"] = normal((kv, c.d_model), std)
            p[pre + "attn.wo.weight"] = normal((c.d_model, c.n_heads * hd), resid_std)
            p[pre + "mlp_norm.weight"] = np.ones(c.d_model, self.dtype)
            p[pre + "mlp.w1.weight"] = normal((c.ffn_hidden, c.d_model), std)
            p[pre + "mlp.w3.weight"] = normal((c.ffn_hidden, c.d_model), std)
            p[pre + "mlp.w2.weight"] = normal((c.d_model, c.ffn_hidden), resid_std)
        p["norm.weight"] = np.ones(c.d_model, self.dtype)
        if not c.tie_embeddings:
            p["lm_head.weight"] = normal((c.vocab_size, c.d_model), std)
        return p

    def num_parameters(self) -> int:
        return int(sum(v.size for v in self.params.values()))

    @property
    def head_weight(self):
        return self.params["tok_emb.weight" if self.cfg.tie_embeddings else "lm_head.weight"]

    # --------------------------------------------------------------- forward
    def forward(self, idx: np.ndarray, targets: np.ndarray | None = None):
        """idx [B,T] int. targets [B,T] int with -1 = ignore.
        Returns (logits, loss, cache). cache feeds backward()."""
        c, P = self.cfg, self.params
        B, T = idx.shape
        if T > c.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {c.max_seq_len}")
        H, KVH, hd = c.n_heads, c.n_kv_heads, c.head_dim
        groups = H // KVH
        scale = 1.0 / np.sqrt(hd)
        cos, sin = self.cos[:T], self.sin[:T]
        mask = np.triu(np.full((T, T), -1e9, dtype=self.dtype), k=1)

        x = P["tok_emb.weight"][idx]
        layers = []
        for i in range(c.n_layers):
            pre = f"layers.{i}."
            st = {"x_in": x}
            n1, r1 = rms_norm(x, P[pre + "attn_norm.weight"], c.norm_eps)
            st["r1"], st["n1"] = r1, n1
            q = (n1 @ P[pre + "attn.wq.weight"].T).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
            k = (n1 @ P[pre + "attn.wk.weight"].T).reshape(B, T, KVH, hd).transpose(0, 2, 1, 3)
            v = (n1 @ P[pre + "attn.wv.weight"].T).reshape(B, T, KVH, hd).transpose(0, 2, 1, 3)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            k_rep = np.repeat(k, groups, axis=1)
            v_rep = np.repeat(v, groups, axis=1)
            att = (q @ k_rep.transpose(0, 1, 3, 2)) * scale + mask
            p_att = softmax(att)
            o = p_att @ v_rep
            o2 = o.transpose(0, 2, 1, 3).reshape(B, T, H * hd)
            x = x + o2 @ P[pre + "attn.wo.weight"].T
            st.update(q=q, k_rep=k_rep, v_rep=v_rep, p=p_att, o2=o2, x_mid=x)

            n2, r2 = rms_norm(x, P[pre + "mlp_norm.weight"], c.norm_eps)
            a = n2 @ P[pre + "mlp.w1.weight"].T
            b = n2 @ P[pre + "mlp.w3.weight"].T
            sa = silu(a)
            g = sa * b
            x = x + g @ P[pre + "mlp.w2.weight"].T
            st.update(r2=r2, n2=n2, a=a, b=b, sa=sa, g=g)
            layers.append(st)

        nf, rf = rms_norm(x, P["norm.weight"], c.norm_eps)
        logits = nf @ self.head_weight.T
        cache = {"idx": idx, "layers": layers, "x_final": x, "rf": rf, "nf": nf}

        loss = None
        if targets is not None:
            flat = logits.reshape(-1, c.vocab_size).astype(np.float64)
            tgt = targets.reshape(-1)
            valid = tgt >= 0
            n_valid = max(int(valid.sum()), 1)
            probs = softmax(flat)
            safe_t = np.where(valid, tgt, 0)
            picked = probs[np.arange(len(tgt)), safe_t]
            loss = float(-np.sum(np.log(picked + 1e-12) * valid) / n_valid)
            dlogits = probs
            dlogits[np.arange(len(tgt)), safe_t] -= 1.0
            dlogits *= (valid / n_valid)[:, None]
            cache["dlogits"] = dlogits.reshape(B, T, c.vocab_size).astype(self.dtype)
        return logits, loss, cache

    # -------------------------------------------------------------- backward
    def backward(self, cache) -> dict:
        c, P = self.cfg, self.params
        idx = cache["idx"]
        B, T = idx.shape
        H, KVH, hd = c.n_heads, c.n_kv_heads, c.head_dim
        groups = H // KVH
        scale = 1.0 / np.sqrt(hd)
        cos, sin = self.cos[:T], self.sin[:T]
        D, V = c.d_model, c.vocab_size
        grads = {k: np.zeros_like(v) for k, v in P.items()}

        dlogits = cache["dlogits"]
        head_name = "tok_emb.weight" if c.tie_embeddings else "lm_head.weight"
        grads[head_name] += dlogits.reshape(-1, V).T @ cache["nf"].reshape(-1, D)
        dnf = dlogits @ self.head_weight
        dx, grads["norm.weight"] = rms_norm_backward(dnf, cache["x_final"], cache["rf"], P["norm.weight"])

        for i in reversed(range(c.n_layers)):
            pre = f"layers.{i}."
            st = cache["layers"][i]
            F = c.ffn_hidden
            # ---- SwiGLU block
            W1, W2, W3 = P[pre + "mlp.w1.weight"], P[pre + "mlp.w2.weight"], P[pre + "mlp.w3.weight"]
            grads[pre + "mlp.w2.weight"] += dx.reshape(-1, D).T @ st["g"].reshape(-1, F)
            dg = dx @ W2
            dsa = dg * st["b"]
            db = dg * st["sa"]
            da = dsa * silu_grad(st["a"])
            n2 = st["n2"].reshape(-1, D)
            grads[pre + "mlp.w1.weight"] += da.reshape(-1, F).T @ n2
            grads[pre + "mlp.w3.weight"] += db.reshape(-1, F).T @ n2
            dn2 = da @ W1 + db @ W3
            dxm, dw = rms_norm_backward(dn2, st["x_mid"], st["r2"], P[pre + "mlp_norm.weight"])
            grads[pre + "mlp_norm.weight"] += dw
            dx = dx + dxm
            # ---- attention block
            Wo = P[pre + "attn.wo.weight"]
            grads[pre + "attn.wo.weight"] += dx.reshape(-1, D).T @ st["o2"].reshape(-1, H * hd)
            do = (dx @ Wo).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
            p_att = st["p"]
            dp = do @ st["v_rep"].transpose(0, 1, 3, 2)
            dv_rep = p_att.transpose(0, 1, 3, 2) @ do
            datt = p_att * (dp - np.sum(dp * p_att, axis=-1, keepdims=True)) * scale
            dq = datt @ st["k_rep"]
            dk_rep = datt.transpose(0, 1, 3, 2) @ st["q"]
            dk = dk_rep.reshape(B, KVH, groups, T, hd).sum(axis=2)
            dv = dv_rep.reshape(B, KVH, groups, T, hd).sum(axis=2)
            dq = dq * cos + rotate_half_t(dq * sin)
            dk = dk * cos + rotate_half_t(dk * sin)
            dq = dq.transpose(0, 2, 1, 3).reshape(B, T, H * hd)
            dk = dk.transpose(0, 2, 1, 3).reshape(B, T, KVH * hd)
            dv = dv.transpose(0, 2, 1, 3).reshape(B, T, KVH * hd)
            n1 = st["n1"].reshape(-1, D)
            grads[pre + "attn.wq.weight"] += dq.reshape(-1, H * hd).T @ n1
            grads[pre + "attn.wk.weight"] += dk.reshape(-1, KVH * hd).T @ n1
            grads[pre + "attn.wv.weight"] += dv.reshape(-1, KVH * hd).T @ n1
            dn1 = (dq @ P[pre + "attn.wq.weight"] + dk @ P[pre + "attn.wk.weight"]
                   + dv @ P[pre + "attn.wv.weight"])
            dxi, dw = rms_norm_backward(dn1, st["x_in"], st["r1"], P[pre + "attn_norm.weight"])
            grads[pre + "attn_norm.weight"] += dw
            dx = dx + dxi

        np.add.at(grads["tok_emb.weight"], idx.reshape(-1), dx.reshape(-1, D))
        return grads

    # ------------------------------------------------- cached inference path
    def new_cache(self, batch: int = 1) -> dict:
        c = self.cfg
        shape = (batch, c.n_kv_heads, 0, c.head_dim)
        return {"pos": 0, "k": [np.zeros(shape, self.dtype) for _ in range(c.n_layers)],
                "v": [np.zeros(shape, self.dtype) for _ in range(c.n_layers)]}

    def forward_cached(self, idx: np.ndarray, cache: dict) -> np.ndarray:
        """Incremental forward with a KV cache. Returns logits for the new
        positions [B, t_new, V]. Mutates cache."""
        c, P = self.cfg, self.params
        B, tn = idx.shape
        start = cache["pos"]
        end = start + tn
        if end > c.max_seq_len:
            raise ValueError("context window exceeded")
        H, KVH, hd = c.n_heads, c.n_kv_heads, c.head_dim
        groups = H // KVH
        scale = 1.0 / np.sqrt(hd)
        cos, sin = self.cos[start:end], self.sin[start:end]
        q_pos = np.arange(start, end)[:, None]
        k_pos = np.arange(end)[None, :]
        mask = np.where(k_pos > q_pos, -1e9, 0.0).astype(self.dtype)

        x = P["tok_emb.weight"][idx]
        for i in range(c.n_layers):
            pre = f"layers.{i}."
            n1, _ = rms_norm(x, P[pre + "attn_norm.weight"], c.norm_eps)
            q = (n1 @ P[pre + "attn.wq.weight"].T).reshape(B, tn, H, hd).transpose(0, 2, 1, 3)
            k = (n1 @ P[pre + "attn.wk.weight"].T).reshape(B, tn, KVH, hd).transpose(0, 2, 1, 3)
            v = (n1 @ P[pre + "attn.wv.weight"].T).reshape(B, tn, KVH, hd).transpose(0, 2, 1, 3)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            cache["k"][i] = np.concatenate([cache["k"][i], k], axis=2)
            cache["v"][i] = np.concatenate([cache["v"][i], v], axis=2)
            k_rep = np.repeat(cache["k"][i], groups, axis=1)
            v_rep = np.repeat(cache["v"][i], groups, axis=1)
            att = softmax((q @ k_rep.transpose(0, 1, 3, 2)) * scale + mask)
            o = (att @ v_rep).transpose(0, 2, 1, 3).reshape(B, tn, H * hd)
            x = x + o @ P[pre + "attn.wo.weight"].T
            n2, _ = rms_norm(x, P[pre + "mlp_norm.weight"], c.norm_eps)
            g = silu(n2 @ P[pre + "mlp.w1.weight"].T) * (n2 @ P[pre + "mlp.w3.weight"].T)
            x = x + g @ P[pre + "mlp.w2.weight"].T
        cache["pos"] = end
        nf, _ = rms_norm(x, P["norm.weight"], c.norm_eps)
        return nf @ self.head_weight.T

    # ----------------------------------------------------------- persistence
    def save(self, directory: str | Path, extra_meta: dict | None = None) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "weights.npz", **self.params)
        self.cfg.save(d / "config.json")
        meta = {"format": "zenit-npz", "parameters": self.num_parameters()}
        meta.update(extra_meta or {})
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, dtype=np.float32) -> "NumpyZenit":
        d = Path(directory)
        cfg = ZenitConfig.load(d / "config.json")
        with np.load(d / "weights.npz") as data:
            params = {k: data[k].astype(dtype) for k in data.files}
        return cls(cfg, params=params, dtype=dtype)
