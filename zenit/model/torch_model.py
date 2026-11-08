"""ZENIT transformer in PyTorch (GPU training and fast inference).

Same architecture, parameter names and layouts as numpy_model.py, so
checkpoints move freely between the two. tests/test_torch_parity.py checks
the logits match.

Extras over the NumPy reference:
  * fused scaled-dot-product attention (FlashAttention kernels on CUDA)
  * preallocated KV cache for fast generation
  * gradient checkpointing and mixed precision for bigger models
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ZenitConfig


def rope_tables(head_dim: int, max_len: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    t = torch.arange(max_len, dtype=torch.float64)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().float(), emb.sin().float()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos.to(x.dtype) + rotate_half(x) * sin.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


class KVCache:
    def __init__(self, cfg: ZenitConfig, batch: int, device, dtype, max_len: int | None = None) -> None:
        n = max_len or cfg.max_seq_len
        shape = (batch, cfg.n_kv_heads, n, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layers)]
        self.pos = 0
        self.max_len = n

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        t = k.shape[2]
        if self.pos + t > self.max_len:
            raise ValueError("context window exceeded")
        self.k[layer][:, :, self.pos:self.pos + t] = k
        self.v[layer][:, :, self.pos:self.pos + t] = v
        return self.k[layer][:, :, :self.pos + t], self.v[layer][:, :, :self.pos + t]


class Attention(nn.Module):
    def __init__(self, cfg: ZenitConfig) -> None:
        super().__init__()
        self.n_heads, self.n_kv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.groups = cfg.n_heads // cfg.n_kv_heads
        self.dropout = cfg.dropout
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, cache: KVCache | None = None, layer: int = 0):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        mask, causal = None, True
        if cache is not None:
            start = cache.pos
            k, v = cache.update(layer, k, v)
            if T == 1:
                causal = False
            elif start > 0:
                q_pos = torch.arange(start, start + T, device=x.device)[:, None]
                k_pos = torch.arange(start + T, device=x.device)[None, :]
                mask, causal = (k_pos <= q_pos), False
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=causal,
            dropout_p=self.dropout if self.training else 0.0)
        return self.wo(out.transpose(1, 2).reshape(B, T, self.n_heads * self.hd))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ZenitConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w2 = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ZenitConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None, layer=0):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache, layer)
        return x + self.mlp(self.mlp_norm(x))


class ZenitModel(nn.Module):
    def __init__(self, cfg: ZenitConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = None if cfg.tie_embeddings else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        cos, sin = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.grad_checkpointing = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = self.cfg.init_std
        resid = std / math.sqrt(2 * self.cfg.n_layers)
        for name, p in self.named_parameters():
            if p.ndim < 2:
                nn.init.ones_(p)
            elif name.endswith(("attn.wo.weight", "mlp.w2.weight")):
                nn.init.normal_(p, 0.0, resid)
            else:
                nn.init.normal_(p, 0.0, std)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                cache: KVCache | None = None):
        B, T = idx.shape
        start = cache.pos if cache is not None else 0
        if start + T > self.cfg.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        cos = self.rope_cos[start:start + T]
        sin = self.rope_sin[start:start + T]
        x = self.tok_emb(idx)
        for i, layer in enumerate(self.layers):
            if self.grad_checkpointing and self.training and cache is None:
                x = checkpoint(layer, x, cos, sin, None, i, use_reentrant=False)
            else:
                x = layer(x, cos, sin, cache, i)
        if cache is not None:
            cache.pos += T
        x = self.norm(x)
        logits = F.linear(x, self.tok_emb.weight) if self.lm_head is None else self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1)
        return logits, loss

    def optimizer(self, lr: float, weight_decay: float, betas=(0.9, 0.95)):
        decay = [p for p in self.parameters() if p.requires_grad and p.ndim >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.ndim < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        fused = next(self.parameters()).is_cuda
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)


# ------------------------------------------------------------------ loading
def save_zenit(model: ZenitModel, directory: str | Path, extra_meta: dict | None = None) -> Path:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    try:
        from safetensors.torch import save_file
        save_file(sd, str(d / "model.safetensors"))
        fmt = "safetensors"
    except ImportError:
        torch.save(sd, d / "model.pt")
        fmt = "torch-pt"
    model.cfg.save(d / "config.json")
    meta = {"format": f"zenit-{fmt}", "parameters": model.num_parameters(), **(extra_meta or {})}
    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return d


def load_state(directory: str | Path) -> dict:
    d = Path(directory)
    if (d / "model.safetensors").exists():
        from safetensors.torch import load_file
        return load_file(str(d / "model.safetensors"))
    if (d / "model.pt").exists():
        return torch.load(d / "model.pt", map_location="cpu", weights_only=True)
    if (d / "weights.npz").exists():
        import numpy as np
        with np.load(d / "weights.npz") as data:
            return {k: torch.from_numpy(data[k].astype("float32")) for k in data.files}
    raise FileNotFoundError(f"no ZENIT weights in {d}")


def load_zenit(directory: str | Path, device="cpu", dtype=torch.float32) -> ZenitModel:
    cfg = ZenitConfig.load(Path(directory) / "config.json")
    model = ZenitModel(cfg)
    model.load_state_dict(load_state(directory), strict=True)
    return model.to(device=device, dtype=dtype).eval()
