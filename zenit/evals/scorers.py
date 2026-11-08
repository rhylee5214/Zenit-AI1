"""Scoring functions. Each returns (passed: bool, detail: str)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

ABSTAIN = [
    "i don't know", "i do not know", "don't have reliable", "no reliable information", "not aware of",
    "i'm not aware", "i am not aware", "doesn't exist", "does not exist", "may not exist", "might not exist",
    "no record", "i can't find", "cannot find", "i'm not sure", "i am not sure", "no information",
    "don't recognise", "don't recognize", "not familiar", "i have no", "fictional", "can't verify",
    "cannot verify", "unable to find", "there is no", "there's no", "no such",
]
_NUM = re.compile(r"-?\d+(?:[.,]\d+)*")


def norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).lower().strip()


def contains_any(answer: str, answers: list[str]) -> tuple[bool, str]:
    a = norm(answer)
    for ok in answers:
        pattern = r"(?<![\w])" + re.escape(norm(ok)) + r"(?![\w])"
        if re.search(pattern, a):
            return True, f"found {ok!r}"
    return False, f"none of {answers}"


def number(answer: str, value: float, tol: float = 1e-6) -> tuple[bool, str]:
    nums = _NUM.findall(answer.replace("−", "-"))
    if not nums:
        return False, "no number"
    try:
        got = float(nums[-1].replace(",", ""))
    except ValueError:
        return False, f"unparsable {nums[-1]!r}"
    return abs(got - value) <= tol, f"last number {got:g}, expected {value:g}"


def extract_code(answer: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", answer, re.S)
    if blocks:
        return "\n\n".join(blocks)
    m = re.search(r"(def \w+\(.*)", answer, re.S)
    return m.group(1) if m else answer


def run_code(answer: str, tests: str, allow_exec: bool, timeout: float = 6.0) -> tuple[bool | None, str]:
    if not allow_exec:
        return None, "skipped (pass --allow-code-exec to run model code)"
    code = extract_code(answer)
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "candidate.py"
        f.write_text(code + "\n\n" + tests + "\nprint('ZENIT_TESTS_PASSED')\n", encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, "-I", str(f)], capture_output=True, text=True,
                               timeout=timeout, cwd=d)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    ok = r.returncode == 0 and "ZENIT_TESTS_PASSED" in r.stdout
    err = (r.stderr.strip().splitlines() or [""])[-1]
    return ok, "tests passed" if ok else f"failed: {err[:160]}"


def constraints(answer: str, checks: list[dict]) -> tuple[bool, str]:
    text = answer.strip()
    words = re.findall(r"\b\w+\b", text)
    for c in checks:
        (k, v), = c.items()
        if k == "exact":
            ok = re.sub(r"[^\w]", "", text.lower()) == v.lower()
        elif k == "contains":
            ok = v.lower() in text.lower()
        elif k == "contains_cs":
            ok = v in text
        elif k == "not_contains_ci":
            ok = v.lower() not in text.lower()
        elif k == "lowercase":
            ok = text == text.lower() and any(ch.isalpha() for ch in text)
        elif k == "max_words":
            ok = len(words) <= v
        elif k == "min_words":
            ok = len(words) >= v
        elif k == "starts_with":
            ok = text.startswith(v)
        elif k == "bullets":
            lines = [l for l in text.splitlines() if l.strip()]
            bullets = [l for l in lines if re.match(r"^\s*([-*•]|\d+[.)])\s+", l)]
            ok = len(bullets) == v and len(bullets) == len(lines)
        elif k == "regex":
            ok = re.search(v, text) is not None
        elif k == "json_keys":
            try:
                body = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
                obj = json.loads(body)
                ok = isinstance(obj, dict) and all(key in obj for key in v)
            except json.JSONDecodeError:
                ok = False
        else:
            return False, f"unknown check {k}"
        if not ok:
            return False, f"failed {k}={v!r}"
    return True, "all checks passed"


def abstain(answer: str, also_accept: list[str] | None = None) -> tuple[bool, str]:
    a = norm(answer).replace("’", "'")
    for p in ABSTAIN + [norm(x) for x in (also_accept or [])]:
        if p in a:
            return True, f"declined or corrected ({p!r})"
    return False, "answered as if the entity or premise were real"


def score(item: dict, answer: str, allow_exec: bool) -> tuple[bool | None, str]:
    s = item["scorer"]
    t = s["type"]
    if t == "contains_any":
        return contains_any(answer, s["answers"])
    if t == "number":
        return number(answer, s["value"])
    if t == "regex":
        return (re.search(s["pattern"], answer.strip()) is not None), f"pattern {s['pattern']}"
    if t == "code":
        return run_code(answer, s["tests"], allow_exec)
    if t == "constraints":
        return constraints(answer, s["checks"])
    if t == "abstain":
        return abstain(answer, s.get("also_accept"))
    raise ValueError(f"unknown scorer {t}")
