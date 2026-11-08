"""Write a BF16 GGUF of a Qwen3 dense checkpoint (Qwen3ForCausalLM) without
llama.cpp's Python tooling, then quantize it with llama-quantize.exe.

    python scripts/make_qwen3_gguf.py models/qwen3-4b-instruct-2507 \
        models/qwen3-4b-instruct-2507-gguf/qwen3-4b-instruct-2507-Q4_K_M.gguf

It mirrors what convert_hf_to_gguf.py (llama.cpp b12110) writes for Qwen3:
the same metadata keys, the GPT-2 style BPE vocab with pre-tokenizer
"qwen2", token types (added "special" or <|...|> tokens are CONTROL, other
added tokens such as <tool_call> are USER_DEFINED so llama-server prints them
as text), merges, special token ids and the chat template. 2-D weights stay
BF16, 1-D norms become F32. The intermediate BF16 file is deleted after
quantization unless --keep-bf16 is given.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path

ALIGN = 32
# GGUF value types
T_UINT32, T_INT32, T_FLOAT32, T_BOOL, T_STRING, T_ARRAY = 4, 5, 6, 7, 8, 9
# ggml tensor types
GGML_F32, GGML_BF16 = 0, 30
# llama token types
NORMAL, UNUSED, CONTROL, USER_DEFINED = 1, 5, 3, 4
MOSTLY_BF16 = 32
COPY_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "generation_config.json", "LICENSE")


def _str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, vtype: int, value) -> bytes:
    out = _str(key) + struct.pack("<I", vtype)
    if vtype == T_STRING:
        return out + _str(value)
    if vtype == T_UINT32:
        return out + struct.pack("<I", value)
    if vtype == T_INT32:
        return out + struct.pack("<i", value)
    if vtype == T_FLOAT32:
        return out + struct.pack("<f", value)
    if vtype == T_BOOL:
        return out + struct.pack("<?", value)
    raise ValueError(vtype)


def _kv_array(key: str, etype: int, values: list) -> bytes:
    parts = [_str(key), struct.pack("<IIQ", T_ARRAY, etype, len(values))]
    if etype == T_STRING:
        parts.extend(_str(v) for v in values)
    elif etype == T_INT32:
        parts.append(struct.pack(f"<{len(values)}i", *values))
    else:
        raise ValueError(etype)
    return b"".join(parts)


def looks_special(tok: str) -> bool:
    return (tok in ("<pad>", "<mask>", "<2mass>", "[@BOS@]")
            or (tok.startswith("<|") and tok.endswith("|>"))
            or (tok.startswith("<｜") and tok.endswith("｜>"))
            or (tok.startswith("<unused") and tok.endswith(">")))


def build_vocab(src: Path, vocab_size: int):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(src))
    vocab = tok.get_vocab()
    assert max(vocab.values()) < vocab_size
    reverse = {i: t for t, i in vocab.items()}
    added = tok.get_added_vocab()
    decoder = tok.added_tokens_decoder
    tokens, types = [], []
    for i in range(vocab_size):
        if i not in reverse:
            tokens.append(f"[PAD{i}]")
            types.append(UNUSED)
            continue
        t = reverse[i]
        if t in added:
            if not decoder[i].normalized:
                t = tok.decode(tok.encode(t, add_special_tokens=False))
            if decoder[i].special or looks_special(t):
                types.append(CONTROL)
            else:
                t = t.replace("▁", " ")
                types.append(USER_DEFINED)
        else:
            types.append(NORMAL)
        tokens.append(t)
    tj = json.loads((src / "tokenizer.json").read_text(encoding="utf-8"))
    merges = [m if isinstance(m, str) else " ".join(m) for m in tj["model"]["merges"]]
    return tok, tokens, types, merges, tj


def write_bf16_gguf(src: Path, out: Path) -> None:
    import torch
    from safetensors import safe_open

    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    assert cfg["architectures"][0] == "Qwen3ForCausalLM", cfg["architectures"]
    tcfg = json.loads((src / "tokenizer_config.json").read_text(encoding="utf-8"))
    tok, tokens, types, merges, tj = build_vocab(src, cfg["vocab_size"])

    def tid(content):
        return tok.convert_tokens_to_ids(content) if content else None

    arch = "qwen3"
    kvs = [
        _kv("general.architecture", T_STRING, arch),
        _kv("general.type", T_STRING, "model"),
        _kv("general.name", T_STRING, src.name),
        _kv(f"{arch}.block_count", T_UINT32, cfg["num_hidden_layers"]),
        _kv(f"{arch}.context_length", T_UINT32, cfg["max_position_embeddings"]),
        _kv(f"{arch}.embedding_length", T_UINT32, cfg["hidden_size"]),
        _kv(f"{arch}.feed_forward_length", T_UINT32, cfg["intermediate_size"]),
        _kv(f"{arch}.attention.head_count", T_UINT32, cfg["num_attention_heads"]),
        _kv(f"{arch}.attention.head_count_kv", T_UINT32, cfg["num_key_value_heads"]),
        _kv(f"{arch}.rope.freq_base", T_FLOAT32, float(cfg["rope_theta"])),
        _kv(f"{arch}.attention.layer_norm_rms_epsilon", T_FLOAT32, float(cfg["rms_norm_eps"])),
        _kv(f"{arch}.attention.key_length", T_UINT32, cfg["head_dim"]),
        _kv(f"{arch}.attention.value_length", T_UINT32, cfg["head_dim"]),
        _kv("general.file_type", T_UINT32, MOSTLY_BF16),
        _kv("general.quantization_version", T_UINT32, 2),
        _kv("tokenizer.ggml.model", T_STRING, "gpt2"),
        _kv("tokenizer.ggml.pre", T_STRING, "qwen2"),
        _kv_array("tokenizer.ggml.tokens", T_STRING, tokens),
        _kv_array("tokenizer.ggml.token_type", T_INT32, types),
        _kv_array("tokenizer.ggml.merges", T_STRING, merges),
    ]
    eos = tid(tcfg.get("eos_token")) if tcfg.get("eos_token") else cfg.get("eos_token_id")
    pad = tid(tcfg.get("pad_token")) if tcfg.get("pad_token") else None
    bos = tid(tcfg.get("bos_token")) if tcfg.get("bos_token") else cfg.get("bos_token_id")
    for key, val in (("eos", eos), ("padding", pad), ("bos", bos)):
        if val is not None:
            kvs.append(_kv(f"tokenizer.ggml.{key}_token_id", T_UINT32, int(val)))
    kvs.append(_kv("tokenizer.ggml.add_bos_token", T_BOOL, bool(tcfg.get("add_bos_token", False))))
    kvs.append(_kv("tokenizer.ggml.add_eos_token", T_BOOL, bool(tcfg.get("add_eos_token", False))))
    if (tj.get("normalizer") or {}).get("type") == "NFC":
        kvs.append(_kv("tokenizer.ggml.normalizer.nfc", T_BOOL, True))
    if isinstance(tcfg.get("clean_up_tokenization_spaces"), bool):
        kvs.append(_kv("tokenizer.ggml.clean_spaces", T_BOOL, tcfg["clean_up_tokenization_spaces"]))
    template = tcfg.get("chat_template")
    if template is None and (src / "chat_template.jinja").exists():
        template = (src / "chat_template.jinja").read_text(encoding="utf-8")
    if template:
        kvs.append(_kv("tokenizer.chat_template", T_STRING, template))

    # tensor name mapping
    def map_name(name: str) -> str | None:
        if name == "model.embed_tokens.weight":
            return "token_embd.weight"
        if name == "model.norm.weight":
            return "output_norm.weight"
        if name == "lm_head.weight":
            return None if cfg.get("tie_word_embeddings") else "output.weight"
        parts = name.split(".")
        if parts[:2] != ["model", "layers"]:
            raise KeyError(name)
        bid, rest = parts[2], ".".join(parts[3:])
        table = {
            "input_layernorm.weight": "attn_norm.weight",
            "self_attn.q_proj.weight": "attn_q.weight",
            "self_attn.k_proj.weight": "attn_k.weight",
            "self_attn.v_proj.weight": "attn_v.weight",
            "self_attn.o_proj.weight": "attn_output.weight",
            "self_attn.q_norm.weight": "attn_q_norm.weight",
            "self_attn.k_norm.weight": "attn_k_norm.weight",
            "post_attention_layernorm.weight": "ffn_norm.weight",
            "mlp.gate_proj.weight": "ffn_gate.weight",
            "mlp.up_proj.weight": "ffn_up.weight",
            "mlp.down_proj.weight": "ffn_down.weight",
        }
        return f"blk.{bid}.{table[rest]}"

    shards = sorted(src.glob("*.safetensors"))
    entries = []        # (gguf_name, shard, hf_name, shape, ggml_type, nbytes)
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for hf_name in f.keys():
                gname = map_name(hf_name)
                if gname is None:
                    continue
                shape = list(f.get_slice(hf_name).get_shape())
                is_1d = len(shape) <= 1 or gname.endswith("_norm.weight")
                gtype = GGML_F32 if is_1d else GGML_BF16
                n = 1
                for d in shape:
                    n *= d
                entries.append((gname, shard, hf_name, shape, gtype, n * (4 if gtype == GGML_F32 else 2)))
    entries.sort(key=lambda e: (e[0].startswith("blk."), int(e[0].split(".")[1]) if e[0].startswith("blk.") else 0, e[0]))

    infos, offset = [], 0
    for gname, _s, _h, shape, gtype, nbytes in entries:
        dims = list(reversed(shape))
        infos.append(_str(gname) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
                     + struct.pack("<IQ", gtype, offset))
        offset += (nbytes + ALIGN - 1) // ALIGN * ALIGN

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        fh.write(b"GGUF" + struct.pack("<IQQ", 3, len(entries), len(kvs)))
        for kv in kvs:
            fh.write(kv)
        for info in infos:
            fh.write(info)
        fh.write(b"\0" * ((-fh.tell()) % ALIGN))
        open_files = {}
        try:
            for gname, shard, hf_name, shape, gtype, nbytes in entries:
                if shard not in open_files:
                    open_files[shard] = safe_open(str(shard), framework="pt").__enter__()
                t = open_files[shard].get_tensor(hf_name)
                if gtype == GGML_F32:
                    data = t.to(torch.float32).contiguous().numpy().tobytes()
                else:
                    data = t.to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes()
                assert len(data) == nbytes, (gname, len(data), nbytes)
                fh.write(data)
                fh.write(b"\0" * ((-len(data)) % ALIGN))
        finally:
            for f in open_files.values():
                f.__exit__(None, None, None)
    print(f"wrote {out} ({len(entries)} tensors, {len(kvs)} metadata keys)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path, help="final quantized .gguf path")
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--quantize-exe", type=Path, default=Path("tools/llama.cpp/llama-quantize.exe"))
    ap.add_argument("--keep-bf16", action="store_true")
    a = ap.parse_args()
    bf16 = a.out.with_name(a.out.stem + ".bf16.tmp.gguf")
    write_bf16_gguf(a.src, bf16)
    rc = subprocess.call([str(a.quantize_exe), str(bf16), str(a.out), a.quant])
    if rc != 0:
        print(f"llama-quantize failed with exit code {rc}; kept {bf16}")
        return rc
    if not a.keep_bf16:
        bf16.unlink()
    import shutil
    for name in COPY_FILES:
        if (a.src / name).exists():
            shutil.copy2(a.src / name, a.out.parent / name)
    print(f"done: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
