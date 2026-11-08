"""Inference for ZENIT models built on an open-weight foundation model.

Runs fully locally through Hugging Face transformers. Loads either a full
checkpoint from disk (a merged ZENIT fine-tune or an unmodified open-weight
model such as Qwen3-4B-Instruct-2507) or base weights plus a ZENIT LoRA
adapter. No network call happens at generation time once the weights sit on
disk (set HF_HUB_OFFLINE=1 to enforce it).

Tool calling: when the model's chat template supports tools (Qwen3 does),
stream(..., tools=[...]) renders the tool list into the prompt and the model
answers with <tool_call>{json}</tool_call> blocks, which zenit.tools.agent
parses. Tool results go back as {"role": "tool"} messages.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterator

from .base import Engine, GenerationEvent, GenerationParams


def dtype_kwarg() -> str:
    """transformers 4.56+ takes dtype= (5.x deprecates torch_dtype=); older
    releases only know torch_dtype=."""
    import transformers
    try:
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    except ValueError:
        return "dtype"
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


class HFEngine(Engine):
    def __init__(self, model_id: str, path: str, adapter: str | None = None,
                 load_in_4bit: bool = False, device_map: str = "auto",
                 chat_template_kwargs: dict | None = None, context_window: int = 8192,
                 card: dict | None = None, supports_tools: bool = False,
                 generation: dict | None = None) -> None:
        self.id = model_id
        self.path = path
        self.adapter = adapter
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.context_window = context_window
        self.supports_tools = bool(supports_tools)
        self.generation = dict(generation or {})
        self.loaded = False
        self.lock = threading.Lock()
        self._load_lock = threading.Lock()
        # Card from serve.json; a model_card.json next to the weights wins.
        self.card: dict = dict(card or {})

    def load(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if not self.loaded:
                self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        kwargs = {dtype_kwarg(): torch.bfloat16, "device_map": self.device_map}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16)
        self.tok = AutoTokenizer.from_pretrained(self.adapter or self.path)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, **kwargs)
        if self.adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.adapter)
        self.model.eval()
        for candidate in (self.adapter, self.path):
            if candidate and (Path(candidate) / "model_card.json").exists():
                self.card = json.loads((Path(candidate) / "model_card.json").read_text(encoding="utf-8"))
                break
        self.loaded = True

    def count_tokens(self, text: str) -> int:
        self.load()
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def info(self) -> dict:
        return {"id": self.id, "engine": "hf", "path": self.path, "adapter": self.adapter,
                "quantized_4bit": self.load_in_4bit, "context_window": self.context_window,
                "version": self.card.get("version"), "base_model": self.card.get("base_model"),
                "supports_tools": self.supports_tools, "generation": self.generation,
                "model_card": self.card}

    @staticmethod
    def _fold_roles(messages: list[dict]) -> list[dict]:
        """HF chat templates know system/user/assistant/tool. Memory and
        retrieved context get folded into the system turn under clear,
        separate headings. user, assistant (with any tool_calls) and tool
        messages keep their original order."""
        system = [m["content"] for m in messages if m["role"] == "system"]
        memory = [m["content"] for m in messages if m["role"] == "memory"]
        context = [m["content"] for m in messages if m["role"] == "context"]
        text = "\n\n".join(system)
        if memory:
            text += "\n\n<memory>\n" + "\n".join(memory) + "\n</memory>"
        if context:
            text += "\n\n<context>\n" + "\n\n".join(context) + "\n</context>"
        out = [{"role": "system", "content": text}] if text else []
        for m in messages:
            if m["role"] in ("user", "tool"):
                out.append({"role": m["role"], "content": m["content"]})
            elif m["role"] == "assistant":
                msg = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    msg["tool_calls"] = m["tool_calls"]
                out.append(msg)
        return out

    def stream(self, messages: list[dict], params: GenerationParams,
               cancel: threading.Event | None = None,
               tools: list[dict] | None = None) -> Iterator[GenerationEvent]:
        self.load()
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        class Cancel(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):
                return bool(cancel is not None and cancel.is_set())

        folded = self._fold_roles(messages)
        template_kwargs = dict(self.chat_template_kwargs)
        if tools:
            template_kwargs["tools"] = tools
        inputs = self.tok.apply_chat_template(folded, add_generation_prompt=True, return_tensors="pt",
                                              return_dict=True, **template_kwargs)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[1])
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=params.max_tokens,
                          do_sample=params.temperature > 1e-5, temperature=max(params.temperature, 1e-5),
                          top_p=params.top_p, top_k=params.top_k,
                          repetition_penalty=params.repetition_penalty,
                          stopping_criteria=StoppingCriteriaList([Cancel()]))
        if params.seed is not None:
            torch.manual_seed(params.seed)
        out_holder: dict = {}

        def run():
            try:
                with torch.no_grad():
                    out_holder["ids"] = self.model.generate(**gen_kwargs)
            except Exception as err:          # surface it instead of leaving the streamer waiting
                out_holder["error"] = err
                streamer.end()

        with self.lock:
            th = threading.Thread(target=run, daemon=True)
            th.start()
            text = ""
            for piece in streamer:
                if piece:
                    text += piece
                    hit = next((s for s in params.stop if s in text), None)
                    if hit:
                        if cancel is not None:
                            cancel.set()
                        yield GenerationEvent("token", piece[: max(0, len(piece) - (len(text) - text.index(hit)))])
                        break
                    yield GenerationEvent("token", piece)
            th.join()
        if "error" in out_holder:
            raise out_holder["error"]
        completion = int(out_holder["ids"].shape[1]) - prompt_tokens if "ids" in out_holder else 0
        finish = "cancelled" if cancel is not None and cancel.is_set() else (
            "length" if completion >= params.max_tokens else "stop")
        yield GenerationEvent("done", finish_reason=finish, prompt_tokens=prompt_tokens,
                              completion_tokens=completion)
