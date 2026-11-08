"""API keys, rate limiting and usage accounting.

Keys look like zk-<random>. Only a SHA-256 hash is stored, so a leaked
database does not leak usable keys. The plaintext shows once, at creation.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
from datetime import date

from ..storage import Database, new_id


def hash_key(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_key(db: Database, user_name: str = "owner", label: str = "", is_admin: bool = False,
               rate_per_min: int = 30, daily_token_limit: int = 0, user_id: str | None = None) -> dict:
    user_id = user_id or new_id("usr_")
    db.ensure_user(user_id, user_name)
    token = "zk-" + secrets.token_urlsafe(32)
    kid = new_id("key_")
    db.execute("INSERT INTO api_keys (id, user_id, key_hash, prefix, label, is_admin, rate_per_min, "
               "daily_token_limit, created) VALUES (?,?,?,?,?,?,?,?,?)",
               (kid, user_id, hash_key(token), token[:10], label, int(is_admin), rate_per_min,
                daily_token_limit, time.time()))
    return {"id": kid, "user_id": user_id, "key": token, "prefix": token[:10], "label": label,
            "is_admin": is_admin, "rate_per_min": rate_per_min}


def list_keys(db: Database) -> list[dict]:
    return db.query("SELECT k.id, k.prefix, k.label, k.is_admin, k.rate_per_min, k.daily_token_limit, "
                    "k.created, k.revoked, u.name AS user_name, k.user_id FROM api_keys k "
                    "JOIN users u ON u.id = k.user_id ORDER BY k.created")


def revoke_key(db: Database, key_id: str) -> bool:
    return db.execute("UPDATE api_keys SET revoked=1 WHERE id=?", (key_id,)).rowcount > 0


class Authenticator:
    def __init__(self, db: Database) -> None:
        self.db = db
        self._cache: dict[str, tuple[float, dict | None]] = {}

    def check(self, token: str) -> dict | None:
        h = hash_key(token)
        hit = self._cache.get(h)
        if hit and time.time() - hit[0] < 30:
            return hit[1]
        row = self.db.one("SELECT * FROM api_keys WHERE key_hash=? AND revoked=0", (h,))
        self._cache[h] = (time.time(), row)
        return row

    def local_key(self) -> dict:
        """Identity for loopback requests in --local mode."""
        row = self.db.one("SELECT * FROM api_keys WHERE label='local-mode' AND revoked=0")
        if row is None:
            create_key(self.db, "local owner", "local-mode", is_admin=True, rate_per_min=120,
                       user_id="usr_local")
            row = self.db.one("SELECT * FROM api_keys WHERE label='local-mode' AND revoked=0")
        return row


class RateLimiter:
    """Token bucket per key: rate_per_min requests, bursts up to the same."""

    def __init__(self) -> None:
        self.buckets: dict[str, tuple[float, float]] = {}
        self.lock = threading.Lock()

    def allow(self, key_id: str, rate_per_min: int) -> tuple[bool, float]:
        if rate_per_min <= 0:
            return True, 0.0
        now = time.time()
        with self.lock:
            tokens, last = self.buckets.get(key_id, (float(rate_per_min), now))
            tokens = min(float(rate_per_min), tokens + (now - last) * rate_per_min / 60.0)
            if tokens < 1.0:
                self.buckets[key_id] = (tokens, now)
                return False, (1.0 - tokens) * 60.0 / rate_per_min
            self.buckets[key_id] = (tokens - 1.0, now)
            return True, 0.0


def record_usage(db: Database, key_id: str, model: str, prompt_tokens: int, completion_tokens: int) -> None:
    db.execute("INSERT INTO usage (key_id, day, model, requests, prompt_tokens, completion_tokens) "
               "VALUES (?, ?, ?, 1, ?, ?) ON CONFLICT(key_id, day, model) DO UPDATE SET "
               "requests = requests + 1, prompt_tokens = prompt_tokens + excluded.prompt_tokens, "
               "completion_tokens = completion_tokens + excluded.completion_tokens",
               (key_id, date.today().isoformat(), model, prompt_tokens, completion_tokens))


def tokens_today(db: Database, key_id: str) -> int:
    r = db.one("SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0) AS t FROM usage "
               "WHERE key_id=? AND day=?", (key_id, date.today().isoformat()))
    return int(r["t"]) if r else 0


def usage_report(db: Database, key_id: str | None = None, user_id: str | None = None, days: int = 30) -> dict:
    if user_id:
        rows = db.query("SELECT u.day, u.model, SUM(u.requests) AS requests, SUM(u.prompt_tokens) AS prompt_tokens, "
                        "SUM(u.completion_tokens) AS completion_tokens FROM usage u JOIN api_keys k ON k.id=u.key_id "
                        "WHERE k.user_id=? GROUP BY u.day, u.model ORDER BY u.day DESC LIMIT ?", (user_id, days * 10))
    else:
        rows = db.query("SELECT day, model, requests, prompt_tokens, completion_tokens FROM usage WHERE key_id=? "
                        "ORDER BY day DESC LIMIT ?", (key_id, days * 10))
    totals = {"requests": sum(r["requests"] for r in rows),
              "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
              "completion_tokens": sum(r["completion_tokens"] for r in rows)}
    return {"days": rows, "totals": totals}
