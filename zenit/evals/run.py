"""Runs the ZENIT evaluation suite against a model.

    python -m zenit.evals.run --model zenit-0.1-nano
    python -m zenit.evals.run --model zenit-2.0-sft --allow-code-exec
    python -m zenit.evals.run --model zenit-2.0-sft --api http://127.0.0.1:8055 --key zk-...

Generation is greedy (temperature 0) so reruns give the same answers. Every
result file records the suite hash, so only runs on identical test sets get
compared. Nothing gets estimated: each score comes from scoring real outputs.

--allow-code-exec runs model-written Python in a subprocess with a timeout.
That code comes from your own model, but treat it as untrusted: run evals in
a VM or container if that worries you.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from ..inference.base import GenerationParams
from ..persona.identity import load_persona
from .scorers import score

TASK_DIR = Path("evals/tasks")
CATEGORIES = ["general_knowledge", "math", "coding", "reasoning", "instruction_following", "conversation", "hallucination"]


def load_tasks(names: list[str]) -> tuple[dict, str]:
    tasks, h = {}, hashlib.sha256()
    for name in CATEGORIES:
        f = TASK_DIR / f"{name}.jsonl"
        if not f.exists():
            continue
        data = f.read_bytes()
        h.update(name.encode() + b"\0" + data)
        if name in names:
            tasks[name] = [json.loads(l) for l in data.decode().splitlines() if l.strip()]
    return tasks, h.hexdigest()[:12]


class LocalRunner:
    def __init__(self, model_id: str) -> None:
        from ..inference.registry import ModelPool
        self.engine = ModelPool().get(model_id)
        self.persona = load_persona()

    def system(self) -> str:
        return self.persona.system_for(None, getattr(self.engine, "card", {}), short=self.engine.short_system_prompt)

    def generate(self, messages: list[dict], max_tokens: int):
        params = GenerationParams(max_tokens=min(max_tokens, self.engine.context_window // 2),
                                  temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0)
        msgs = [{"role": "system", "content": self.system()}] + messages
        t0 = time.perf_counter()
        first, parts, done = None, [], None
        for ev in self.engine.stream(msgs, params):
            if ev.type == "token":
                if first is None:
                    first = time.perf_counter() - t0
                parts.append(ev.text)
            else:
                done = ev
        total = time.perf_counter() - t0
        return "".join(parts), first, total, done.completion_tokens if done else 0


class ApiRunner:
    def __init__(self, url: str, key: str, model_id: str) -> None:
        self.url, self.key, self.model = url.rstrip("/"), key, model_id

    def generate(self, messages: list[dict], max_tokens: int):
        body = json.dumps({"model": self.model, "messages": messages, "stream": True, "max_tokens": max_tokens,
                           "temperature": 0.0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0}).encode()
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=body, method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        t0 = time.perf_counter()
        first, parts, tokens = None, [], 0
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                obj = json.loads(line[6:])
                ch = (obj.get("choices") or [{}])[0]
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if first is None:
                        first = time.perf_counter() - t0
                    parts.append(piece)
                if obj.get("usage"):
                    tokens = obj["usage"]["completion_tokens"]
        return "".join(parts), first, time.perf_counter() - t0, tokens


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--limit", type=int, default=0, help="items per category (0 = all)")
    ap.add_argument("--allow-code-exec", action="store_true")
    ap.add_argument("--api", default="")
    ap.add_argument("--key", default="")
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--out-dir", default="evals/results")
    a = ap.parse_args(argv)

    names = CATEGORIES if a.tasks == "all" else a.tasks.split(",")
    tasks, suite_hash = load_tasks(names)
    runner = ApiRunner(a.api, a.key, a.model) if a.api else LocalRunner(a.model)
    items_out, summary, speed = [], {}, {"first_token_s": [], "tokens_per_sec": []}
    for cat, items in tasks.items():
        if a.limit:
            items = items[: a.limit]
        passed = scored = 0
        for it in items:
            answer, first, total, ntok = runner.generate(it["messages"], it.get("max_tokens", 96))
            ok, detail = score(it, answer, a.allow_code_exec)
            if first is not None:
                speed["first_token_s"].append(first)
            if ntok and total > 0:
                speed["tokens_per_sec"].append(ntok / total)
            if ok is not None:
                scored += 1
                passed += int(ok)
            items_out.append({"category": cat, "id": it["id"], "passed": ok, "detail": detail,
                              "answer": answer[:2000], "first_token_s": first, "seconds": round(total, 3)})
            mark = "skip" if ok is None else ("PASS" if ok else "fail")
            print(f"[{mark}] {cat:22s} {it['id']:18s} {answer[:70]!r}")
        summary[cat] = round(passed / scored, 4) if scored else "skipped"
    if isinstance(summary.get("hallucination"), float):
        summary["hallucination_rate"] = round(1 - summary["hallucination"], 4)
    ft, tps = speed["first_token_s"], speed["tokens_per_sec"]
    summary["first_token_ms"] = round(1000 * sorted(ft)[len(ft) // 2], 1) if ft else None
    summary["tokens_per_sec"] = round(sorted(tps)[len(tps) // 2], 1) if tps else None

    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result = {"model": a.model, "created": created, "suite_hash": suite_hash, "mode": "api" if a.api else "local",
              "code_exec": a.allow_code_exec, "summary": summary, "items": items_out}
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.model}__{created.replace(':', '').replace('-', '')[:15]}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nsummary (share of items passed; speed is the median):")
    for k, v in summary.items():
        print(f"  {k:22s} {v}")
    print(f"saved {path}")
    if not a.no_register:
        try:
            from ..train.registry import Registry
            Registry().add_eval(a.model, {"file": path.name, "suite_hash": suite_hash, "created": created, "summary": summary})
        except KeyError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
