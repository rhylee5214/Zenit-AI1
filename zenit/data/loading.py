"""Reads documents from manifest sources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .manifest import DataSource


def _iter_text_file(path: Path) -> Iterator[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    # Documents are separated by a line holding only "<|doc|>", if present.
    if "\n<|doc|>\n" in text:
        yield from (d for d in text.split("\n<|doc|>\n") if d.strip())
    else:
        yield text


def _files(path: Path, patterns=("*.txt", "*.md")) -> list[Path]:
    if path.is_file():
        return [path]
    out: list[Path] = []
    for pat in patterns:
        out.extend(sorted(path.rglob(pat)))
    return out


def iter_records(source: DataSource, root: Path) -> Iterator[dict]:
    """Yields dicts: {"text": str} for text kinds, {"messages": [...]} for chat,
    {"prompt","chosen","rejected"} for preference data."""
    if source.hf_dataset:
        yield from _iter_hf(source)
        return
    path = root / source.path
    if source.kind == "text":
        for f in _files(path):
            for doc in _iter_text_file(f):
                yield {"text": doc}
        return
    for f in _files(path, ("*.jsonl",)):
        with f.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as err:
                    raise ValueError(f"{f}:{line_no}: bad JSON ({err})") from err
                yield rec


def _iter_hf(source: DataSource) -> Iterator[dict]:
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise RuntimeError("pip install datasets to read Hugging Face sources") from err
    ds = load_dataset(source.hf_dataset, source.hf_config or None, split=source.hf_split, streaming=True)
    for row in ds:
        if source.kind == "chat_jsonl" and "messages" in row:
            yield {"messages": row["messages"]}
        elif source.kind == "pref_jsonl":
            yield {k: row[k] for k in ("prompt", "chosen", "rejected") if k in row}
        elif "text" in row:
            yield {"text": row["text"]}


def chat_to_text(messages: list[dict]) -> str:
    """Flattens a chat into the ZENIT template string for pretraining."""
    from ..tokenizer.chat_template import render
    return render(messages, add_generation_prompt=False)
