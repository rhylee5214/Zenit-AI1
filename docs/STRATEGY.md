# ZENIT model development strategy

## 1. Hardware inspected

| Machine | What I found |
|---|---|
| Build workspace (where ZENIT 0.1 trained) | 2 CPU cores, 8.4 GB RAM, about 32 GB free disk, no GPU, no PyTorch |
| Your PC | Windows x64, device name "rhylee". The folder-access request was declined and no remote shell exists, so your GPU, RAM and disk stayed unreadable from here. A folder named `ansel` in your home directory shares its name with an NVIDIA screenshot tool, a hint of an NVIDIA card. The hardware check below confirms it. |

Run this first on your PC. It reports GPU, VRAM, RAM and disk, then picks a plan:

```
python scripts/check_hardware.py
```

| Your GPU memory | Fine-tuning base for ZENIT 2.0+ | Method | Native ZENIT size |
|---|---|---|---|
| 22 GB or more | Qwen3-8B-Base | LoRA bf16 | base or medium |
| 14 to 22 GB | Qwen3-8B-Base | QLoRA 4-bit | base |
| 10 to 14 GB | Qwen3-4B-Base | QLoRA 4-bit | small, or base with batch 8 |
| 6 to 10 GB | Qwen3-4B-Base | QLoRA 4-bit, sequence length 1024 | small |
| under 6 GB | Qwen3-1.7B-Base | QLoRA 4-bit | nano or small |
| no NVIDIA GPU | rent a cloud GPU for fine-tuning | | nano on CPU |

## 2. Why ZENIT 5.5 does not come from training from scratch

Training cost grows with parameters times tokens. A rough rule: 6 floating-point operations per parameter per training token.

* A 7B model trained on 140B tokens (20 tokens per parameter, the compute-optimal ratio from the Chinchilla paper) needs about 5.9 × 10²¹ operations. One GPU sustaining 50 TFLOPS needs about 3.7 years.
* ZENIT's base preset (100.7M parameters) on 2.5B tokens needs about 1.5 × 10¹⁸ operations, about 8 hours at 50 TFLOPS.

Competitive assistants sit far above the second number and far beyond a single PC. So ZENIT runs two tracks.

## 3. Two tracks

**Track A: native ZENIT, its own architecture from scratch.** Every weight comes from ZENIT training. It proves the architecture, tokeniser and training stack, and it grows as budget allows.

* 0.1-nano: done. Trained in this build on CPU.
* 1.0: the small (31.6M) or base (100.7M) preset on a licensed open corpus, on your GPU. `configs/zenit-1.0-small.json` holds the plan. The estimator puts it near 629 PFLOPs, about 3.5 hours at 50 TFLOPS sustained or 9 hours at 20.

**Track B: ZENIT fine-tuned from an open-weight foundation.** This track reaches assistant-level quality on a PC.

* Base: Qwen3 Base models. Apache-2.0 license, sizes from 0.6B to 8B, a standard transformer layout, mature LoRA and 4-bit support. ZENIT starts from Base rather than Instruct models, so the instruction following, identity and honesty behaviour all come from ZENIT's own training.
* The newer Qwen3.5 small series (0.8B, 2B, 4B, 9B, each with a Base variant) uses a hybrid Gated DeltaNet and attention design with image input. Switch `base_model` once your transformers and PEFT versions load it. LoRA targets `all-linear`, so no other code changes.
* Pipeline: SFT with LoRA or QLoRA, then DPO on preference pairs, then merge into standalone weights.

## 4. What makes a fine-tuned ZENIT independent

* The weights sit on your disk. The merged model is a complete set of weight files.
* Generation runs on ZENIT's own inference engine and API on your hardware. No request goes to OpenAI, Anthropic or any other AI provider, in training or in serving.
* The behaviour comes from ZENIT data: persona SFT, honesty preferences and your chosen open datasets, all recorded in manifests.
* Every checkpoint carries a model card with its base model, data fingerprint, training run and evals.

What it does not mean: the base model's original training stays part of the weights. Apache-2.0 allows modification, commercial use and redistribution, with the license and a notice kept alongside. ZENIT writes the `NOTICE.txt` automatically, and the persona data teaches ZENIT to state its base model truthfully when asked.

## 5. Version milestones

Each version counts as reached once a checkpoint passes its exit check on the fixed eval suite. The targets below are goals, not results.

| Version | Track | Work | Exit check |
|---|---|---|---|
| 0.1 | A | Tiny native model, pipeline proof | Trains, serves, streams, evaluates end to end (met) |
| 1.0 | A | Native model on a licensed open corpus, GPU | Validation loss below the 0.1 run on held-out text from the new corpus |
| 2.0 | B | SFT on open instruction data plus ZENIT persona | Instruction following 70%+, identity answers correct |
| 3.0 | B | Multi-turn data, memory and document-aware prompts | Conversation 80%+, citations present when context is supplied |
| 4.0 | B | Code and math SFT, DPO | Coding 50%+ with code execution, math 70%+ |
| 5.0 | B | Larger base, curated mix, wider evals (lm-evaluation-harness) | Beats 4.0 on every category |
| 5.5 | B | Refinement pass: accuracy, hallucination, memory, instructions | Hallucination rate under 15%, no regressions against 5.0 |

## 6. Resources per stage

These are planning figures. The trainers measure your GPU and print a fresh estimate before starting, and they refuse jobs over 2 hours without `--yes`.

| Stage | GPU memory | Disk | Compute |
|---|---|---|---|
| 0.1-nano (CPU) | none | under 50 MB | 15 to 20 minutes on 2 cores |
| 1.0-small | about 10 GB at batch 16 | about 2 GB plus corpus | about 629 PFLOPs |
| 2.0 SFT, 4B QLoRA | 6 to 10 GB | about 8 GB base plus 0.2 GB adapter | 20,000 examples of about 600 tokens: about 1.9 × 10¹⁷ operations |
| 4.0 DPO, 4B QLoRA | 8 to 12 GB | as above | depends on pair count |
| Merge 4B | 16 GB system RAM | about 8 GB output | minutes on CPU |

## 7. Next steps on your PC

1. `python scripts/check_hardware.py`
2. Install CUDA PyTorch from pytorch.org, then `pip install -e .[finetune]`.
3. `python -m pytest` (the PyTorch parity tests run once torch exists).
4. Read the dataset cards in `data/manifests/zenit-2.0-sft.json`, then set `permission` to `true` for the ones you accept.
5. `python -m zenit.finetune.sft --config configs/zenit-2.0-sft.json --dry-run`, then without `--dry-run`.
6. Enable the model in `configs/serve.json`, restart the server, and run `python -m zenit.evals.run --model zenit-2.0-sft --allow-code-exec`.

## Sources

* Qwen3 base models, Apache-2.0, sizes and LoRA feasibility: [Qwen3-8B-Base](https://dev.co/ai/llms/qwen3-8b-base), [Qwen3-4B-Base](https://dev.co/ai/llms/qwen3-4b-base), [Qwen3-0.6B-Base](https://dev.co/ai/llms/qwen3-0-6b-base)
* Qwen3.5 small series: [GIGAZINE](https://gigazine.net/gsc_news/en/20260303-qwen-3-5-small), [model list](https://besthub.dev/articles/qwen3-5-small-models-unveiled-from-0-8b-to-9b-with-full-capabilities-e409056d2b3c)
* Open-weight licensing overview: [Onyx open LLM guide](https://onyx.app/insights/best-open-source-llms-2026)
