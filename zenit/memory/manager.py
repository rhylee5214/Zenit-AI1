"""Conversation history, long-term memories, summaries and context building.

User control is built in:
  * memory can be switched off per user (setting memory_enabled=false)
  * every memory is listable, editable, deletable and exportable
  * "forget ..." in chat deletes the closest matching memory
  * memories are only extracted from explicit, recognisable statements
"""

from __future__ import annotations

import json
import re
import time

import numpy as np

from ..knowledge.embeddings import from_blob, get_embedder, to_blob
from ..knowledge.store import KnowledgeBase, build_context_block
from ..storage import Database, new_id

# Explicit statements only. Each pattern maps to (kind, template).
_PATTERNS = [
    (re.compile(r"^(?:please\s+)?remember(?: that)?\s+(.{3,300})$", re.I), "fact", "{0}"),
    (re.compile(r"\bmy name is ([A-Z][\w'-]{1,40}(?: [A-Z][\w'-]{1,40})?)", re.I), "profile", "The user's name is {0}."),
    (re.compile(r"\bcall me ([A-Z][\w'-]{1,40})", re.I), "profile", "The user wants to be called {0}."),
    (re.compile(r"\bi prefer (.{3,160}?)(?:[.!]|$)", re.I), "preference", "The user prefers {0}."),
    (re.compile(r"\bi (?:live|am based) in ([A-Z][\w .'-]{1,60}?)(?:[.!,]|$)", re.I), "profile", "The user lives in {0}."),
    (re.compile(r"\bi work (?:as|at) (.{3,100}?)(?:[.!,]|$)", re.I), "profile", "The user works {1} {0}."),
]
_FORGET = re.compile(r"^(?:please\s+)?forget(?: that| about)?\s+(.{3,300})$", re.I)


class MemoryManager:
    def __init__(self, db: Database, kb: KnowledgeBase, embedder=None) -> None:
        self.db = db
        self.kb = kb
        self.embedder = embedder or get_embedder()

    # -------------------------------------------------------- conversations
    def create_conversation(self, user_id: str, title: str = "New chat", model: str = "") -> dict:
        cid = new_id("conv_")
        now = time.time()
        self.db.execute("INSERT INTO conversations (id, user_id, title, model, created, updated) VALUES (?,?,?,?,?,?)",
                        (cid, user_id, title, model, now, now))
        return self.get_conversation(user_id, cid)

    def get_conversation(self, user_id: str, cid: str) -> dict | None:
        return self.db.one("SELECT * FROM conversations WHERE id=? AND user_id=?", (cid, user_id))

    def list_conversations(self, user_id: str, limit: int = 100) -> list[dict]:
        return self.db.query("SELECT id, title, model, created, updated FROM conversations WHERE user_id=? "
                             "ORDER BY updated DESC LIMIT ?", (user_id, limit))

    def rename_conversation(self, user_id: str, cid: str, title: str) -> None:
        self.db.execute("UPDATE conversations SET title=?, updated=? WHERE id=? AND user_id=?",
                        (title[:200], time.time(), cid, user_id))

    def delete_conversation(self, user_id: str, cid: str) -> bool:
        return self.db.execute("DELETE FROM conversations WHERE id=? AND user_id=?", (cid, user_id)).rowcount > 0

    def add_message(self, cid: str, role: str, content: str, model: str = "", meta: dict | None = None,
                    prompt_tokens: int = 0, completion_tokens: int = 0) -> int:
        cur = self.db.execute("INSERT INTO messages (conversation_id, role, content, model, meta, prompt_tokens, "
                              "completion_tokens, created) VALUES (?,?,?,?,?,?,?,?)",
                              (cid, role, content, model, json.dumps(meta or {}), prompt_tokens,
                               completion_tokens, time.time()))
        self.db.execute("UPDATE conversations SET updated=? WHERE id=?", (time.time(), cid))
        return cur.lastrowid

    def messages(self, cid: str) -> list[dict]:
        rows = self.db.query("SELECT id, role, content, model, meta, prompt_tokens, completion_tokens, created "
                             "FROM messages WHERE conversation_id=? ORDER BY id", (cid,))
        for r in rows:
            r["meta"] = json.loads(r["meta"] or "{}")
        return rows

    def pop_last_assistant(self, cid: str) -> None:
        r = self.db.one("SELECT id, role FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT 1", (cid,))
        if r and r["role"] == "assistant":
            self.db.execute("DELETE FROM messages WHERE id=?", (r["id"],))

    # ------------------------------------------------------------- memories
    def enabled(self, user_id: str) -> bool:
        return self.db.get_setting(user_id, "memory_enabled", "true") == "true"

    def add_memory(self, user_id: str, content: str, kind: str = "fact", pinned: bool = False,
                   source_conversation: str | None = None) -> dict:
        content = content.strip()
        vec = self.embedder.embed([content])[0]
        for m in self._all_with_vectors(user_id):   # near-duplicate -> update in place
            if float(m["vec"] @ vec) > 0.9:
                self.update_memory(user_id, m["id"], content=content)
                return self.get_memory(user_id, m["id"])
        mid = new_id("mem_")
        now = time.time()
        self.db.execute("INSERT INTO memories (id, user_id, content, kind, pinned, source_conversation, embedder, "
                        "embedding, created, updated) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (mid, user_id, content, kind, int(pinned), source_conversation, self.embedder.name,
                         to_blob(vec), now, now))
        return self.get_memory(user_id, mid)

    def get_memory(self, user_id: str, mid: str) -> dict | None:
        return self.db.one("SELECT id, content, kind, pinned, source_conversation, created, updated "
                           "FROM memories WHERE id=? AND user_id=?", (mid, user_id))

    def list_memories(self, user_id: str) -> list[dict]:
        return self.db.query("SELECT id, content, kind, pinned, source_conversation, created, updated "
                             "FROM memories WHERE user_id=? ORDER BY pinned DESC, updated DESC", (user_id,))

    def update_memory(self, user_id: str, mid: str, content: str | None = None,
                      pinned: bool | None = None, kind: str | None = None) -> dict | None:
        m = self.get_memory(user_id, mid)
        if not m:
            return None
        if content is not None:
            vec = self.embedder.embed([content])[0]
            self.db.execute("UPDATE memories SET content=?, embedder=?, embedding=?, updated=? WHERE id=?",
                            (content.strip(), self.embedder.name, to_blob(vec), time.time(), mid))
        if pinned is not None:
            self.db.execute("UPDATE memories SET pinned=?, updated=? WHERE id=?", (int(pinned), time.time(), mid))
        if kind is not None:
            self.db.execute("UPDATE memories SET kind=?, updated=? WHERE id=?", (kind, time.time(), mid))
        return self.get_memory(user_id, mid)

    def delete_memory(self, user_id: str, mid: str) -> bool:
        return self.db.execute("DELETE FROM memories WHERE id=? AND user_id=?", (mid, user_id)).rowcount > 0

    def delete_all_memories(self, user_id: str) -> int:
        return self.db.execute("DELETE FROM memories WHERE user_id=?", (user_id,)).rowcount

    def export(self, user_id: str) -> dict:
        convs = self.list_conversations(user_id, limit=10_000)
        return {"exported": time.time(), "memories": self.list_memories(user_id),
                "conversations": [{**c, "messages": self.messages(c["id"])} for c in convs],
                "documents": self.kb.list_documents(user_id)}

    def _all_with_vectors(self, user_id: str) -> list[dict]:
        rows = self.db.query("SELECT id, content, kind, pinned, embedder, embedding FROM memories WHERE user_id=?",
                             (user_id,))
        out = []
        for r in rows:
            if r["embedder"] != self.embedder.name:
                vec = self.embedder.embed([r["content"]])[0]
                self.db.execute("UPDATE memories SET embedder=?, embedding=? WHERE id=?",
                                (self.embedder.name, to_blob(vec), r["id"]))
            else:
                vec = from_blob(r["embedding"])
            out.append({**r, "vec": vec})
        return out

    def relevant_memories(self, user_id: str, query: str, k: int = 6, min_score: float = 0.25) -> list[dict]:
        rows = self._all_with_vectors(user_id)
        if not rows:
            return []
        q = self.embedder.embed([query])[0]
        scored = sorted(((float(r["vec"] @ q), r) for r in rows), key=lambda t: -t[0])
        chosen = [r for r in rows if r["pinned"]]
        for s, r in scored:
            if len(chosen) >= k:
                break
            if s >= min_score and r not in chosen:
                chosen.append(r)
        return [{"id": r["id"], "content": r["content"], "kind": r["kind"]} for r in chosen]

    def process_user_message(self, user_id: str, text: str, cid: str | None = None) -> list[dict]:
        """Handles explicit remember / forget statements. Returns actions taken."""
        if not self.enabled(user_id):
            return []
        actions = []
        stripped = text.strip()
        fm = _FORGET.match(stripped)
        if fm:
            target = self.embedder.embed([fm.group(1)])[0]
            best = max(self._all_with_vectors(user_id), key=lambda r: float(r["vec"] @ target), default=None)
            if best is not None and float(best["vec"] @ target) > 0.25:
                self.delete_memory(user_id, best["id"])
                actions.append({"action": "forgot", "memory": best["content"]})
            return actions
        for pat, kind, template in _PATTERNS:
            for m in pat.finditer(stripped):
                value = m.group(1).strip().rstrip(".")
                verb = "as" if " as " in m.group(0).lower() else "at"
                content = template.format(value, verb)
                mem = self.add_memory(user_id, content, kind, source_conversation=cid)
                actions.append({"action": "remembered", "memory": mem["content"], "id": mem["id"]})
        return actions

    # -------------------------------------------------------------- summary
    @staticmethod
    def extractive_summary(turns: list[dict], max_chars: int = 1200) -> str:
        lines = []
        for t in turns:
            first = re.split(r"(?<=[.!?])\s", t["content"].strip(), maxsplit=1)[0]
            lines.append(f"{t['role']}: {first[:200]}")
        text = "\n".join(lines)
        return text[-max_chars:]

    # ------------------------------------------------------ context builder
    def build_messages(self, user_id: str, engine, persona, profile: str | None,
                       history: list[dict], new_message: str, max_tokens: int,
                       use_memory: bool = True, use_documents: bool = True,
                       document_ids: list[str] | None = None, summary: str | None = None,
                       user_name: str | None = None) -> dict:
        """Assembles the prompt within the model's context window.

        Budget order: system prompt, newest user message, retrieved document
        context, memories, conversation summary, then as many recent turns
        as fit. Returns {"messages", "sources", "memories", "dropped_turns"}."""
        window = engine.context_window
        reserve = min(max_tokens, window // 2)
        budget = window - reserve
        count = engine.count_tokens
        card = getattr(engine, "card", {}) or {}
        system = persona.system_for(profile, card, short=engine.short_system_prompt, user_name=user_name)
        used = count(system) + count(new_message) + 12
        messages_front = [{"role": "system", "content": system}]
        sources, mems = [], []

        if use_documents and budget - used > 200:
            hits = self.kb.search(user_id, new_message, k=6, doc_ids=document_ids)
            if hits:
                block, sources = build_context_block(hits, int((budget - used) * 0.5), count)
                if block:
                    messages_front.append({"role": "context", "content": block})
                    used += count(block) + 4
        if use_memory and self.enabled(user_id) and budget - used > 120:
            mems = self.relevant_memories(user_id, new_message)
            if mems:
                text = "Facts the user asked ZENIT to remember:\n" + "\n".join(f"- {m['content']}" for m in mems)
                if count(text) < (budget - used) * 0.3:
                    messages_front.append({"role": "memory", "content": text})
                    used += count(text) + 4
                else:
                    mems = []

        turns = [{"role": h["role"], "content": h["content"]} for h in history if h["role"] in ("user", "assistant")]
        kept: list[dict] = []
        for t in reversed(turns):
            cost = count(t["content"]) + 4
            if used + cost > budget:
                break
            kept.insert(0, t)
            used += cost
        dropped = turns[: len(turns) - len(kept)]
        if dropped and budget - used > 80:
            text = summary or self.extractive_summary(dropped)
            text = "Summary of earlier conversation:\n" + text
            while text and count(text) > budget - used - 8:
                text = text[len(text) // 4:]
            if text:
                messages_front.insert(1, {"role": "system", "content": text})
        return {"messages": messages_front + kept + [{"role": "user", "content": new_message}],
                "sources": sources, "memories": mems, "dropped_turns": len(dropped)}
