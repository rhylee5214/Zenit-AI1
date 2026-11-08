"""Document store, vector search and retrieval-augmented context."""

from __future__ import annotations

import hashlib
import threading
import time

import numpy as np

from ..storage import Database, new_id
from .embeddings import from_blob, get_embedder, to_blob
from .ingest import chunk_text, extract_text


class KnowledgeBase:
    def __init__(self, db: Database, embedder=None) -> None:
        self.db = db
        self.embedder = embedder or get_embedder()
        self._cache: dict[str, tuple[int, np.ndarray, list[dict]]] = {}
        self._version: dict[str, int] = {}
        self._lock = threading.Lock()

    def _bump(self, user_id: str) -> None:
        with self._lock:
            self._version[user_id] = self._version.get(user_id, 0) + 1

    # ------------------------------------------------------------ ingestion
    def add_document(self, user_id: str, filename: str, data: bytes, mime: str = "") -> dict:
        pages = extract_text(filename, data)
        chunks = chunk_text(pages)
        if not chunks:
            raise ValueError("no extractable text in file")
        doc_id = new_id("doc_")
        sha = hashlib.sha256(data).hexdigest()
        self.db.execute("INSERT INTO documents (id, user_id, filename, mime, bytes, sha256, n_chunks, created) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (doc_id, user_id, filename, mime, len(data), sha, len(chunks), time.time()))
        vecs = self.embedder.embed([c["text"] for c in chunks])
        for c, v in zip(chunks, vecs):
            self.db.execute("INSERT INTO chunks (document_id, user_id, idx, page, text, embedder, embedding) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (doc_id, user_id, c["idx"], c["page"], c["text"], self.embedder.name, to_blob(v)))
        self._bump(user_id)
        return self.get_document(user_id, doc_id)

    def get_document(self, user_id: str, doc_id: str) -> dict | None:
        return self.db.one("SELECT id, filename, mime, bytes, n_chunks, created FROM documents "
                           "WHERE id=? AND user_id=?", (doc_id, user_id))

    def list_documents(self, user_id: str) -> list[dict]:
        return self.db.query("SELECT id, filename, mime, bytes, n_chunks, created FROM documents "
                             "WHERE user_id=? ORDER BY created DESC", (user_id,))

    def delete_document(self, user_id: str, doc_id: str) -> bool:
        cur = self.db.execute("DELETE FROM documents WHERE id=? AND user_id=?", (doc_id, user_id))
        self.db.execute("DELETE FROM chunks WHERE document_id=? AND user_id=?", (doc_id, user_id))
        self._bump(user_id)
        return cur.rowcount > 0

    # --------------------------------------------------------------- search
    def _matrix(self, user_id: str):
        ver = self._version.get(user_id, 0)
        hit = self._cache.get(user_id)
        if hit and hit[0] == ver:
            return hit[1], hit[2]
        rows = self.db.query("SELECT c.id, c.document_id, c.idx, c.page, c.text, c.embedder, c.embedding, "
                             "d.filename FROM chunks c JOIN documents d ON d.id = c.document_id "
                             "WHERE c.user_id=?", (user_id,))
        stale = [r for r in rows if r["embedder"] != self.embedder.name]
        if stale:                                   # embedder changed: reindex lazily
            vecs = self.embedder.embed([r["text"] for r in stale])
            for r, v in zip(stale, vecs):
                self.db.execute("UPDATE chunks SET embedder=?, embedding=? WHERE id=?",
                                (self.embedder.name, to_blob(v), r["id"]))
                r["embedding"] = to_blob(v)
        meta = [{k: r[k] for k in ("id", "document_id", "idx", "page", "text", "filename")} for r in rows]
        mat = np.stack([from_blob(r["embedding"]) for r in rows]) if rows else np.zeros((0, self.embedder.dim), np.float32)
        self._cache[user_id] = (ver, mat, meta)
        return mat, meta

    def search(self, user_id: str, query: str, k: int = 5, doc_ids: list[str] | None = None,
               min_score: float = 0.15) -> list[dict]:
        mat, meta = self._matrix(user_id)
        if not len(meta):
            return []
        q = self.embedder.embed([query])[0]
        scores = mat @ q
        if doc_ids:
            allow = np.array([m["document_id"] in doc_ids for m in meta])
            scores = np.where(allow, scores, -1.0)
        order = np.argsort(-scores)[: k * 3]
        out, seen = [], set()
        for i in order:
            if scores[i] < min_score or len(out) >= k:
                break
            key = (meta[i]["document_id"], meta[i]["idx"])
            if key in seen:
                continue
            seen.add(key)
            out.append({**meta[i], "score": round(float(scores[i]), 4)})
        return out


def build_context_block(hits: list[dict], token_budget: int, count_tokens) -> tuple[str, list[dict]]:
    """Formats retrieved chunks as numbered sources within a token budget."""
    parts, sources, used = [], [], 0
    header = ("The passages below come from the user's uploaded documents, not from ZENIT's training. "
              "Cite them as [S1], [S2]. If they do not contain the answer, say so.\n")
    used += count_tokens(header)
    for h in hits:
        label = f"S{len(sources) + 1}"
        where = f"{h['filename']}" + (f", page {h['page']}" if h.get("page") else "")
        block = f"[{label}] {where}\n{h['text']}\n"
        cost = count_tokens(block)
        if used + cost > token_budget:
            break
        parts.append(block)
        used += cost
        sources.append({"label": label, "document_id": h["document_id"], "filename": h["filename"],
                        "page": h.get("page"), "chunk": h["idx"], "score": h["score"],
                        "excerpt": h["text"][:300]})
    if not parts:
        return "", []
    return header + "\n".join(parts), sources
