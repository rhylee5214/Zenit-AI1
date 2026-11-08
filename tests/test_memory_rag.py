from zenit.knowledge.ingest import chunk_text, extract_text
from zenit.knowledge.store import KnowledgeBase, build_context_block
from zenit.memory.manager import MemoryManager
from zenit.storage import Database

DOC = """ZENIT Field Guide

The Aurora Station is located on the northern ridge. It stores 340 litres of fresh water.

Maintenance happens every Tuesday. The lead engineer for the station is Dana Whitfield.

The backup generator runs on biodiesel and lasts nine days at full load."""


def setup():
    db = Database(":memory:")
    db.ensure_user("u1")
    db.ensure_user("u2")
    kb = KnowledgeBase(db)
    return db, kb, MemoryManager(db, kb)


def test_chunking_keeps_text_and_pages():
    pages = extract_text("guide.md", DOC.encode())
    chunks = chunk_text(pages, size=120, overlap=20)
    assert len(chunks) >= 3
    joined = " ".join(c["text"] for c in chunks)
    assert "Dana Whitfield" in joined and "biodiesel" in joined


def test_document_search_finds_relevant_passage_and_isolates_users():
    db, kb, _ = setup()
    kb.add_document("u1", "guide.md", DOC.encode())
    hits = kb.search("u1", "who is the lead engineer?")
    assert hits and "Dana Whitfield" in hits[0]["text"]
    assert kb.search("u2", "who is the lead engineer?") == []      # other users see nothing


def test_context_block_numbers_sources_and_respects_budget():
    db, kb, _ = setup()
    kb.add_document("u1", "guide.md", DOC.encode())
    hits = kb.search("u1", "generator fuel", k=3, min_score=0.0)
    block, sources = build_context_block(hits, 10_000, lambda s: len(s) // 4)
    assert "[S1]" in block and sources[0]["label"] == "S1"
    small, few = build_context_block(hits, 40, lambda s: len(s) // 4)
    assert len(few) < len(sources) or small == ""


def test_explicit_memory_extraction_forget_and_disable():
    db, _, mem = setup()
    acts = mem.process_user_message("u1", "Hi, my name is Rhylee and I prefer short answers.")
    contents = [a["memory"] for a in acts]
    assert any("Rhylee" in c for c in contents)
    assert any("short answers" in c for c in contents)
    assert len(mem.list_memories("u1")) == 2
    assert mem.process_user_message("u1", "what's the weather like") == []
    gone = mem.process_user_message("u1", "forget my name")
    assert gone and gone[0]["action"] == "forgot"
    assert all("Rhylee" not in m["content"] for m in mem.list_memories("u1"))
    db.set_setting("u1", "memory_enabled", "false")
    assert mem.process_user_message("u1", "remember that I like tea") == []


def test_memory_dedup_pin_and_relevance():
    db, _, mem = setup()
    a = mem.add_memory("u1", "The user's favourite language is Python.")
    b = mem.add_memory("u1", "The user's favourite language is Python.")
    assert a["id"] == b["id"]
    mem.add_memory("u1", "The user has a dog called Biscuit.", pinned=True)
    rel = mem.relevant_memories("u1", "which programming language should I use?")
    assert any("Python" in r["content"] for r in rel)
    assert any("Biscuit" in r["content"] for r in rel)   # pinned memories always included


def test_conversation_history_and_regenerate_support():
    db, _, mem = setup()
    conv = mem.create_conversation("u1", "test")
    mem.add_message(conv["id"], "user", "hello")
    mem.add_message(conv["id"], "assistant", "hi")
    mem.pop_last_assistant(conv["id"])
    msgs = mem.messages(conv["id"])
    assert [m["role"] for m in msgs] == ["user"]
    assert mem.delete_conversation("u1", conv["id"])
    assert mem.get_conversation("u1", conv["id"]) is None
