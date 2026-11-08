"""Merge a ZENIT LoRA adapter into its base model to produce standalone
ZENIT weights (safetensors), servable without PEFT and convertible to GGUF.

    python -m zenit.finetune.merge --adapter models/zenit-2.0-sft/adapter --out models/zenit-2.0
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from ..train.registry import Registry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models-dir", default="models")
    a = ap.parse_args(argv)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = Path(a.adapter)
    card = json.loads((adapter / "model_card.json").read_text(encoding="utf-8"))
    print(f"[zenit] loading {card['base_model']} in bf16 (needs about 2 bytes per parameter of RAM) ...")
    base = AutoModelForCausalLM.from_pretrained(card["base_model"], torch_dtype=torch.bfloat16, device_map="cpu")
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    out = Path(a.out)
    merged.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(str(adapter)).save_pretrained(out)
    card.update(id=out.name, merged=True, weights_format="hf-safetensors")
    (out / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    shutil.copy(adapter / "NOTICE.txt", out / "NOTICE.txt")
    Registry(a.models_dir).register({"id": out.name, "name": "ZENIT", "version": card["version"],
                                     "stage": card.get("stage", "sft") + "-merged", "engine": "hf", "path": str(out),
                                     "base_model": card["base_model"], "run_id": card.get("run_id")})
    print(f"[zenit] merged weights written to {out}")
    print("[zenit] optional GGUF export for llama.cpp / Ollama:\n"
          f"  python llama.cpp/convert_hf_to_gguf.py {out} --outfile {out}/zenit.gguf --outtype q8_0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
