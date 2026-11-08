"""SQLite storage for the ZENIT platform (users, keys, conversations,
memories, documents, usage). One file, WAL mode, safe for one server
process with many threads. Swap for PostgreSQL later by reimplementing this
module: every SQL statement in the project lives behind these helpers."""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_hash TEXT NOT NULL UNIQUE, prefix TEXT NOT NULL, label TEXT, is_admin INTEGER DEFAULT 0,
    rate_per_min INTEGER DEFAULT 30, daily_token_limit INTEGER DEFAULT 0,
    created REAL NOT NULL, revoked INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS settings (
    user_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT, PRIMARY KEY (user_id, key)
);
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT, model TEXT, summary TEXT, summary_upto INTEGER DEFAULT 0,
    created REAL NOT NULL, updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL, content TEXT NOT NULL, model TEXT, meta TEXT,
    prompt_tokens INTEGER DEFAULT 0, completion_tokens INTEGER DEFAULT 0, created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL, kind TEXT DEFAULT 'fact', pinned INTEGER DEFAULT 0,
    source_conversation TEXT, embedder TEXT, embedding BLOB, created REAL NOT NULL, updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename TEXT NOT NULL, mime TEXT, bytes INTEGER, sha256 TEXT, n_chunks INTEGER DEFAULT 0,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL, idx INTEGER NOT NULL, page INTEGER, text TEXT NOT NULL,
    embedder TEXT, embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_user ON chunks(user_id);
CREATE TABLE IF NOT EXISTS usage (
    key_id TEXT NOT NULL, day TEXT NOT NULL, model TEXT NOT NULL,
    requests INTEGER DEFAULT 0, prompt_tokens INTEGER DEFAULT 0, completion_tokens INTEGER DEFAULT 0,
    PRIMARY KEY (key_id, day, model)
);
CREATE TABLE IF NOT EXISTS google_accounts (
    sub TEXT PRIMARY KEY, user_id TEXT NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    key_id TEXT, email TEXT, name TEXT, given_name TEXT, picture TEXT,
    created REAL NOT NULL, updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created REAL NOT NULL, expires REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
"""


def new_id(prefix: str = "") -> str:
    return prefix + uuid.uuid4().hex[:20]


class Database:
    def __init__(self, path: str | Path = "data/zenit.db") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        with self.conn() as c:
            c.executescript(SCHEMA)

    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA foreign_keys = ON")
            if self.path != ":memory:":
                c.execute("PRAGMA journal_mode = WAL")
            self._local.conn = c
        return c

    def query(self, sql: str, args: tuple = ()) -> list[dict]:
        return [dict(r) for r in self.conn().execute(sql, args).fetchall()]

    def one(self, sql: str, args: tuple = ()) -> dict | None:
        r = self.conn().execute(sql, args).fetchone()
        return dict(r) if r else None

    def execute(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._write_lock:
            c = self.conn()
            cur = c.execute(sql, args)
            c.commit()
            return cur

    # -------------------------------------------------------------- users
    def ensure_user(self, user_id: str, name: str = "user") -> None:
        self.execute("INSERT OR IGNORE INTO users (id, name, created) VALUES (?, ?, ?)",
                     (user_id, name, time.time()))

    def get_setting(self, user_id: str, key: str, default=None):
        r = self.one("SELECT value FROM settings WHERE user_id=? AND key=?", (user_id, key))
        return r["value"] if r else default

    def set_setting(self, user_id: str, key: str, value) -> None:
        self.execute("INSERT INTO settings (user_id, key, value) VALUES (?, ?, ?) "
                     "ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value",
                     (user_id, key, None if value is None else str(value)))
