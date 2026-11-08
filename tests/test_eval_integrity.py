"""Guards against benchmark contamination: no eval prompt may appear in the
training corpora, and long word n-grams may not overlap."""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def words(s):
    return re.findall(r"\w+", s.lower())


def training_user_texts():
    texts = []
    for f in list((ROOT / "data" / "raw" / "zenit_seed").glob("*.jsonl")) + [ROOT / "data" / "sft" / "zenit_persona.jsonl"]:
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                for m in json.loads(line)["messages"]:
                    texts.append(m["content"])
    return texts


def eval_prompts():
    out = []
    for f in (ROOT / "evals" / "tasks").glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                it = json.loads(line)
                out.append((it["id"], " ".join(m["content"] for m in it["messages"] if m["role"] == "user")))
    return out


@pytest.mark.skipif(not (ROOT / "evals" / "tasks").exists(), reason="eval sets not generated")
def test_no_exact_or_ngram_overlap():
    train = training_user_texts()
    if not train:
        pytest.skip("training corpus not generated")
    exact = {" ".join(words(t)) for t in train}
    grams = set()
    for t in train:
        w = words(t)
        grams.update(tuple(w[i:i + 10]) for i in range(len(w) - 9))
    for iid, prompt in eval_prompts():
        w = words(prompt)
        assert " ".join(w) not in exact, f"{iid} appears verbatim in training data"
        overlap = [tuple(w[i:i + 10]) for i in range(len(w) - 9) if tuple(w[i:i + 10]) in grams]
        assert not overlap, f"{iid} shares a 10-word span with training data"
