"""Start the ZENIT server.

    python -m zenit.api --local            # web UI on http://localhost:8055 (or 127.0.0.1), no key needed from this PC
    python -m zenit.api --host 0.0.0.0     # LAN access, API key required for every request
    python -m zenit.api keys create --name rhylee --admin
    python -m zenit.api keys list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..storage import Database
from .auth import create_key, list_keys, revoke_key


def keys_cli(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m zenit.api keys")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--name", default="owner")
    c.add_argument("--label", default="")
    c.add_argument("--admin", action="store_true")
    c.add_argument("--rate", type=int, default=30, help="requests per minute")
    c.add_argument("--daily-tokens", type=int, default=0, help="0 = unlimited")
    sub.add_parser("list")
    r = sub.add_parser("revoke")
    r.add_argument("key_id")
    ap.add_argument("--db", default="data/zenit.db")
    a = ap.parse_args(argv)
    db = Database(a.db)
    if a.cmd == "create":
        k = create_key(db, a.name, a.label, a.admin, a.rate, a.daily_tokens)
        print(f"API key (shown once, store it safely):\n  {k['key']}\nid={k['id']} admin={k['is_admin']}")
    elif a.cmd == "list":
        for k in list_keys(db):
            print(f"{k['id']}  {k['prefix']}...  user={k['user_name']} admin={bool(k['is_admin'])} "
                  f"rate={k['rate_per_min']}/min revoked={bool(k['revoked'])} label={k['label']}")
    elif a.cmd == "revoke":
        print("revoked" if revoke_key(db, a.key_id) else "not found")
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "keys":
        return keys_cli(sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8055)
    ap.add_argument("--local", action="store_true",
                    help="allow requests from this machine without an API key")
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    if a.local and a.host not in ("127.0.0.1", "localhost", "::1"):
        print("--local only works with a loopback host; refusing to expose keyless access")
        return 2
    import uvicorn
    from .app import create_app
    app = create_app(Path(a.root), local_mode=a.local)
    pool = app.state.zenit.pool
    print(f"[zenit] models: {', '.join(pool.engines) or 'none (train one first)'}; default={pool.default}")
    print(f"[zenit] open http://{a.host}:{a.port}/  (dashboard: /dashboard)")
    if app.state.zenit.google is not None:
        print(f"[zenit] Sign in with Google is on: open http://localhost:{a.port}/ to sign in")
    if pool.default:
        import threading

        def warm():
            try:
                print(f"[zenit] loading {pool.default} in the background...")
                pool.get(pool.default)
                print(f"[zenit] {pool.default} ready")
            except Exception as err:  # serve anyway; the first request will retry
                print(f"[zenit] preload failed: {err}")
        threading.Thread(target=warm, daemon=True).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
