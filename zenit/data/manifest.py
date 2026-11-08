"""Dataset manifests: every training source must declare where it came from
and under what license you have permission to use it.

A manifest is a JSON file:

{
  "name": "zenit-seed",
  "sources": [
    {"name": "seed-chat", "path": "data/raw/zenit_seed/chat.jsonl",
     "kind": "chat_jsonl", "license": "CC0-1.0",
     "origin": "written for the ZENIT project", "permission": true,
     "weight": 1.0}
  ]
}

kind is one of: text (plain .txt/.md, one document per file or per blank
line block), jsonl_text ({"text": ...} per line), chat_jsonl
({"messages": [...]} per line), pref_jsonl ({"prompt","chosen","rejected"}).

Sources without a license or without permission: true are refused. Nothing
gets used silently.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

VALID_KINDS = {"text", "jsonl_text", "chat_jsonl", "pref_jsonl"}


class ManifestError(ValueError):
    pass


@dataclass
class DataSource:
    name: str
    path: str
    kind: str
    license: str
    origin: str
    permission: bool
    weight: float = 1.0
    notes: str = ""
    filters: dict = field(default_factory=dict)   # CleanOptions overrides, per source
    hf_dataset: str = ""          # optional: Hugging Face dataset id
    hf_config: str = ""           # optional: dataset configuration name
    hf_split: str = "train"


@dataclass
class Manifest:
    name: str
    sources: list[DataSource] = field(default_factory=list)
    path: str = ""

    def fingerprint(self) -> str:
        blob = json.dumps([s.__dict__ for s in self.sources], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_manifest(path: str | Path, root: str | Path | None = None) -> Manifest:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    root = Path(root) if root else Path.cwd()
    problems: list[str] = []
    sources = []
    for raw in data.get("sources", []):
        name = raw.get("name", "<unnamed>")
        lic = str(raw.get("license", "")).strip()
        if not lic or lic.lower() in {"unknown", "none", "?"}:
            problems.append(f"{name}: missing license")
        if raw.get("permission") is not True:
            problems.append(f"{name}: permission must be true (you confirm you may train on it)")
        if raw.get("kind") not in VALID_KINDS:
            problems.append(f"{name}: kind must be one of {sorted(VALID_KINDS)}")
        if not raw.get("hf_dataset"):
            fp = root / raw.get("path", "")
            if not fp.exists():
                problems.append(f"{name}: file not found: {fp}")
        sources.append(DataSource(**{k: v for k, v in raw.items() if k in DataSource.__dataclass_fields__}))
    if problems:
        raise ManifestError("manifest rejected:\n  - " + "\n  - ".join(problems))
    return Manifest(name=data.get("name", p.stem), sources=sources, path=str(p))
