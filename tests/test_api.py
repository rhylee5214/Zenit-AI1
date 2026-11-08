"""End-to-end API tests against a real uvicorn server and a tiny ZENIT model."""

import json
import shutil
import socket
import threading
import time
from pathlib import Path

import pytest

requests = pytest.importorskip("requests")
uvicorn = pytest.importorskip("uvicorn")

from zenit.api.app import create_app  # noqa: E402
from zenit.api.auth import create_key  # noqa: E402
from zenit.model.config import ZenitConfig  # noqa: E402
from zenit.model.numpy_model import NumpyZenit  # noqa: E402
from zenit.tokenizer.bpe import ZenitTokenizer  # noqa: E402


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    root = tmp_path_factory.mktemp("zenit")
    (root / "configs").mkdir()
    (root / "web" / "static").mkdir(parents=True)
    (root / "web" / "index.html").write_text("<!DOCTYPE html><title>ZENIT</title>", encoding="utf-8")
    mdir = root / "models" / "tiny"
    tok = ZenitTokenizer.train(["hello world, ZENIT answers questions about documents."] * 30, vocab_size=300)
    cfg = ZenitConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=1, n_heads=2, n_kv_heads=1,
                      max_seq_len=4096, version="test")
    NumpyZenit(cfg, seed=0).save(mdir)
    tok.save(mdir / "tokenizer.json")
    (mdir / "model_card.json").write_text(json.dumps({"version": "test", "parameters": 1, "trained_from_scratch": True}), encoding="utf-8")
    (root / "configs" / "serve.json").write_text(json.dumps(
        {"default_model": "tiny", "models": [{"id": "tiny", "engine": "native", "path": str(mdir)}]}), encoding="utf-8")
    app = create_app(root)
    db = app.state.zenit.db
    admin = create_key(db, "owner", "admin", is_admin=True, rate_per_min=1000)
    user = create_key(db, "guest", "user", rate_per_min=1000)
    slow = create_key(db, "slow", "slow", rate_per_min=3)
    port = free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield {"url": f"http://127.0.0.1:{port}", "admin": admin["key"], "user": user["key"], "slow": slow["key"], "root": root}
    srv.should_exit = True
    th.join(timeout=5)
    shutil.rmtree(root, ignore_errors=True)


def h(key):
    return {"Authorization": f"Bearer {key}"}


def chat(s, key, **body):
    body.setdefault("max_tokens", 12)
    body.setdefault("seed", 1)
    return requests.post(s["url"] + "/v1/chat/completions", json=body, headers=h(key), timeout=60)


def test_pages_are_served(server):
    r = requests.get(server["url"] + "/")
    assert r.status_code == 200 and "ZENIT" in r.text


def test_health_and_auth(server):
    assert requests.get(server["url"] + "/v1/health").json()["status"] == "ok"
    assert requests.get(server["url"] + "/v1/models").status_code == 401
    assert requests.get(server["url"] + "/v1/models", headers=h("zk-wrong")).status_code == 401
    models = requests.get(server["url"] + "/v1/models", headers=h(server["user"])).json()
    assert models["default"] == "tiny"


def test_stateless_chat_completion(server):
    r = chat(server, server["user"], messages=[{"role": "user", "content": "hello"}])
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["object"] == "chat.completion"
    assert isinstance(data["choices"][0]["message"]["content"], str)
    assert data["usage"]["prompt_tokens"] > 0


def test_streaming_conversation_and_regenerate(server):
    r = requests.post(server["url"] + "/v1/chat/completions", headers=h(server["user"]), stream=True, timeout=60,
                      json={"message": "hello there", "stream": True, "max_tokens": 10, "seed": 2})
    assert r.headers["content-type"].startswith("text/event-stream")
    events = [line[6:] for line in r.iter_lines(decode_unicode=True) if line and line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    first = json.loads(events[0])
    cid = first["zenit"]["conversation_id"]
    assert cid
    final = json.loads(events[-2])
    assert final["choices"][0]["finish_reason"] in ("stop", "length")
    conv = requests.get(f"{server['url']}/v1/conversations/{cid}", headers=h(server["user"])).json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant"]
    assert conv["title"] == "hello there"
    r2 = chat(server, server["user"], conversation_id=cid, regenerate=True)
    assert r2.status_code == 200, r2.text
    conv = requests.get(f"{server['url']}/v1/conversations/{cid}", headers=h(server["user"])).json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant"]
    other = requests.get(f"{server['url']}/v1/conversations/{cid}", headers=h(server["admin"]))
    assert other.status_code == 404                     # users cannot read each other's chats


def test_memory_flow_through_chat(server):
    r = chat(server, server["user"], message="my name is Rhylee")
    acts = r.json()["zenit"]["memory_actions"]
    assert acts and "Rhylee" in acts[0]["memory"]
    mems = requests.get(server["url"] + "/v1/memories", headers=h(server["user"])).json()["data"]
    assert any("Rhylee" in m["content"] for m in mems)
    r = chat(server, server["user"], message="what is my name?")
    assert any("Rhylee" in m["content"] for m in r.json()["zenit"]["memories"])
    requests.put(server["url"] + "/v1/settings", json={"memory_enabled": False}, headers=h(server["user"]))
    r = chat(server, server["user"], message="remember that I like tea")
    assert r.json()["zenit"]["memory_actions"] == []
    requests.put(server["url"] + "/v1/settings", json={"memory_enabled": True}, headers=h(server["user"]))
    exp = requests.get(server["url"] + "/v1/export", headers=h(server["user"])).json()
    assert exp["memories"] and exp["conversations"]
    d = requests.delete(server["url"] + "/v1/memories", headers=h(server["user"])).json()
    assert d["deleted"] >= 1


def test_documents_and_rag_sources(server):
    text = ("The Aurora Station stores 340 litres of water. The lead engineer is Dana Whitfield. " * 3).encode()
    up = requests.post(server["url"] + "/v1/documents?filename=station.txt", data=text,
                       headers={**h(server["user"]), "Content-Type": "text/plain"})
    assert up.status_code == 201, up.text
    doc = up.json()
    hits = requests.post(server["url"] + "/v1/documents/search", json={"query": "lead engineer"},
                         headers=h(server["user"])).json()["data"]
    assert hits and "Whitfield" in hits[0]["text"]
    r = chat(server, server["user"], message="who is the lead engineer?", document_ids=[doc["id"]])
    sources = r.json()["zenit"]["sources"]
    assert sources and sources[0]["filename"] == "station.txt"
    bad = requests.post(server["url"] + "/v1/documents?filename=x.exe", data=b"MZ\x00",
                        headers=h(server["user"]))
    assert bad.status_code == 415
    assert requests.delete(f"{server['url']}/v1/documents/{doc['id']}", headers=h(server["user"])).json()["deleted"]


def test_rate_limit_and_admin_permissions(server):
    codes = [chat(server, server["slow"], messages=[{"role": "user", "content": "hi"}], max_tokens=2).status_code
             for _ in range(4)]
    assert codes[:3] == [200, 200, 200] and codes[3] == 429
    assert requests.post(server["url"] + "/v1/keys", json={}, headers=h(server["user"])).status_code == 403
    created = requests.post(server["url"] + "/v1/keys", json={"label": "x"}, headers=h(server["admin"]))
    assert created.status_code == 201 and created.json()["key"].startswith("zk-")


def test_usage_training_and_registry_endpoints(server):
    usage = requests.get(server["url"] + "/v1/usage", headers=h(server["user"])).json()
    assert usage["totals"]["requests"] >= 1
    assert requests.get(server["url"] + "/v1/training/runs", headers=h(server["user"])).json() == {"data": []}
    assert requests.post(server["url"] + "/v1/training/estimate", json={}, headers=h(server["user"])).status_code == 403
    reg = requests.get(server["url"] + "/v1/registry", headers=h(server["user"])).json()
    assert [r["version"] for r in reg["roadmap"]][-1] == "5.5"
    assert requests.post(server["url"] + "/v1/generations/gen_x/cancel", headers=h(server["user"])).json() == {"cancelled": False}


def test_validation_errors(server):
    r = chat(server, server["user"], messages=[{"role": "user", "content": "x"}], temperature=9)
    assert r.status_code == 422
    r = requests.post(server["url"] + "/v1/chat/completions", data="not json", headers=h(server["user"]))
    assert r.status_code == 400
