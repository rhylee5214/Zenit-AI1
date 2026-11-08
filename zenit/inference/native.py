"""Inference for native ZENIT checkpoints (NumPy or PyTorch backend)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterator

import numpy as np

from ..tokenizer.bpe import IncrementalDecoder, ZenitTokenizer
from ..tokenizer.chat_template import encode_prompt
from .base import Engine, GenerationEvent, GenerationParams
from .sampling import sample_next


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class NativeEngine(Engine):
    def __init__(self, model_dir: str | Path, model_id: str | None = None,
                 backend: str = "auto", device: str = "auto") -> None:
        self.dir = Path(model_dir)
        self.id = model_id or self.dir.name
        self.backend_pref = backend
        self.device_pref = device
        self.loaded = False
        self.lock = threading.Lock()

    def load(self) -> None:
        if self.loaded:
            return
        self.tok = ZenitTokenizer.load(self.dir / "tokenizer.json")
        card_path = self.dir / "model_card.json"
        self.card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else {}
        has_np = (self.dir / "weights.npz").exists()
        if self.backend_pref == "numpy" or (self.backend_pref == "auto" and not _torch_available()):
            use_torch = False
        else:
            use_torch = True
        if use_torch and not (has_np or self._has_torch_weights()):
            raise FileNotFoundError(f"no weights in {self.dir}")
        if use_torch:
            import torch
            from ..model.torch_model import load_zenit
            dev = self.device_pref
            if dev == "auto":
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if dev == "cuda" else torch.float32
            self.model = load_zenit(self.dir, device=dev, dtype=dtype)
            self.backend, self.device = "torch", dev
            self.cfg = self.model.cfg
        else:
            from ..model.numpy_model import NumpyZenit
            self.model = NumpyZenit.load(self.dir)
            self.backend, self.device = "numpy", "cpu"
            self.cfg = self.model.cfg
        self.context_window = self.cfg.max_seq_len
        self.short_system_prompt = self.cfg.max_seq_len <= 512
        self.loaded = True

    def _has_torch_weights(self) -> bool:
        return (self.dir / "model.safetensors").exists() or (self.dir / "model.pt").exists()

    def count_tokens(self, text: str) -> int:
        self.load()
        return len(self.tok.encode(text, allow_special=False))

    def info(self) -> dict:
        self.load()
        return {"id": self.id, "engine": "native", "backend": self.backend, "device": self.device,
                "context_window": self.context_window, "parameters": self.card.get("parameters"),
                "version": self.card.get("version"), "stage": self.card.get("stage"),
                "trained_from_scratch": self.card.get("trained_from_scratch"),
                "architecture": self.card.get("architecture"), "model_card": self.card}

    # ------------------------------------------------------------- decoding
    def _logits_fn(self):
        if self.backend == "numpy":
            cache = self.model.new_cache()

            def step(ids: list[int]) -> np.ndarray:
                return self.model.forward_cached(np.array([ids]), cache)[0, -1]
            return step
        import torch
        from ..model.torch_model import KVCache
        cache = KVCache(self.cfg, 1, self.device, next(self.model.parameters()).dtype)

        def step(ids: list[int]) -> np.ndarray:
            with torch.no_grad():
                x = torch.tensor([ids], device=self.device)
                return self.model(x, cache=cache)[0][0, -1].float().cpu().numpy()
        return step

    def stream(self, messages: list[dict], params: GenerationParams,
               cancel: threading.Event | None = None,
               tools: list[dict] | None = None) -> Iterator[GenerationEvent]:
        # Native ZENIT models were not trained for tool calls (supports_tools
        # stays False); `tools` is accepted for interface parity and ignored.
        self.load()
        ids, sys_len = encode_prompt(self.tok, messages)
        window = self.cfg.max_seq_len
        reserve = max(8, min(params.max_tokens, window // 2))
        limit = window - reserve
        if len(ids) > limit:              # keep the system block, drop the oldest turns
            keep_sys = sys_len if sys_len < limit // 2 else 1
            ids = ids[:keep_sys] + ids[-(limit - keep_sys):]
        prompt_tokens = len(ids)
        max_new = min(params.max_tokens, window - len(ids))
        stop_ids = {self.tok.end_id, self.tok.eos_id, self.tok.special_id("<|user|>"),
                    self.tok.special_id("<|system|>")}
        rng = np.random.default_rng(params.seed)
        dec = IncrementalDecoder(self.tok)
        hold = max((len(s) for s in params.stop), default=0)
        text_out, pending = "", ""
        generated: list[int] = []
        finish = "length"

        with self.lock:                   # one generation per loaded model at a time
            step = self._logits_fn()
            logits = step(ids)
            for _ in range(max_new):
                if cancel is not None and cancel.is_set():
                    finish = "cancelled"
                    break
                nxt = sample_next(logits, ids + generated, params.temperature, params.top_k,
                                  params.top_p, params.repetition_penalty, rng)
                if nxt in stop_ids:
                    finish = "stop"
                    break
                generated.append(nxt)
                piece = dec.push(nxt)
                if piece:
                    pending += piece
                    hit = next((s for s in params.stop if s in text_out + pending), None)
                    if hit:
                        full = text_out + pending
                        cut = full.index(hit)
                        if cut > len(text_out):
                            yield GenerationEvent("token", full[len(text_out):cut])
                        finish = "stop"
                        pending = ""
                        break
                    if len(pending) > hold:
                        emit = pending[:len(pending) - hold] if hold else pending
                        pending = pending[len(emit):]
                        text_out += emit
                        yield GenerationEvent("token", emit)
                if len(ids) + len(generated) >= window:
                    finish = "length"
                    break
                logits = step([nxt])
        tail = pending + (dec.flush() if finish != "stop" else "")
        if tail:
            yield GenerationEvent("token", tail)
        yield GenerationEvent("done", finish_reason=finish, prompt_tokens=prompt_tokens,
                              completion_tokens=len(generated))
