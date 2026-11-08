"""Sign in with Google: ID token verification and browser sessions.

The web app uses Google Identity Services (GIS) to get an ID token (a JWT
signed by Google) and posts it to /v1/auth/google. This module checks it
properly before trusting anything inside it:

  * RS256 only; the signing key is looked up by "kid" in Google's published
    keys (JWKS), cached for an hour and refetched once when a new kid shows up
  * aud must be our OAuth client id, iss must be Google, exp is enforced with
    a small leeway, and the email must be verified by Google

A verified token creates (or finds) a ZENIT user keyed by Google's stable
"sub" id and opens a session. The session token goes to the browser as an
HttpOnly cookie; only its SHA-256 hash is stored, like API keys.

Google sign-in is off until configs/auth.json holds a Web client id:
    {"google_client_id": "1234-abc.apps.googleusercontent.com"}
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import threading
import time
import urllib.request
from pathlib import Path

from ..storage import Database, new_id
from .auth import create_key

GOOGLE_CERTS_URL = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_ISSUERS = ("accounts.google.com", "https://accounts.google.com")
CLIENT_ID_SUFFIX = ".apps.googleusercontent.com"
SESSION_COOKIE = "zenit_session"
SESSION_TTL_S = 30 * 24 * 3600
SESSION_KEY_LABEL = "google-session"
JWKS_TTL_S = 3600
JWKS_MIN_REFETCH_S = 60          # unknown kids cannot make us hammer Google
LEEWAY_S = 60
MAX_TOKEN_CHARS = 8192


class GoogleAuthError(Exception):
    def __init__(self, message: str, status: int = 401) -> None:
        super().__init__(message)
        self.message, self.status = message, status


# ------------------------------------------------------------------ config
def load_auth_config(path: str | Path) -> dict:
    """Reads configs/auth.json. A missing file, null or a malformed client id
    leaves Google sign-in off (with a reason in "error")."""
    out = {"google_client_id": None, "error": None}
    p = Path(path)
    if not p.exists():
        return out
    try:
        data = json.loads(p.read_text(encoding="utf-8") or "{}")
    except (OSError, json.JSONDecodeError) as err:
        out["error"] = f"configs/auth.json is not valid JSON: {err}"
        return out
    cid = data.get("google_client_id") if isinstance(data, dict) else None
    if cid is None or (isinstance(cid, str) and not cid.strip()):
        return out
    cid = str(cid).strip()
    if not cid.endswith(CLIENT_ID_SUFFIX) or len(cid) > 200 or not re.fullmatch(r"[A-Za-z0-9._-]+", cid):
        out["error"] = "google_client_id must look like 1234-abc.apps.googleusercontent.com"
        return out
    out["google_client_id"] = cid
    return out


# -------------------------------------------------------------------- JWKS
def fetch_jwks(url: str = GOOGLE_CERTS_URL, timeout: float = 10.0) -> dict:
    """Downloads Google's public signing keys. Tests replace this function."""
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "ZENIT"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:      # noqa: S310 (fixed https URL)
        return json.loads(resp.read(1_000_000).decode("utf-8"))


def _rsa_keys(jwks: dict) -> dict:
    from jwt.algorithms import RSAAlgorithm
    keys = {}
    for jwk in (jwks or {}).get("keys", []) if isinstance(jwks, dict) else []:
        if not isinstance(jwk, dict) or jwk.get("kty") != "RSA" or not isinstance(jwk.get("kid"), str):
            continue
        if jwk.get("alg") not in (None, "RS256") or jwk.get("use") not in (None, "sig"):
            continue
        try:
            keys[jwk["kid"]] = RSAAlgorithm.from_jwk(json.dumps(jwk))
        except Exception:                               # skip a malformed key, keep the rest
            continue
    return keys


class JWKSCache:
    def __init__(self, url: str = GOOGLE_CERTS_URL, ttl: float = JWKS_TTL_S,
                 min_refetch: float = JWKS_MIN_REFETCH_S) -> None:
        self.url, self.ttl, self.min_refetch = url, ttl, min_refetch
        self.keys: dict = {}
        self.fetched = 0.0
        self.lock = threading.Lock()

    def _refresh(self) -> None:
        try:
            data = fetch_jwks(self.url)          # module global: tests monkeypatch it
        except Exception as err:
            self.fetched = time.time()           # back off; do not retry on every request
            if not self.keys:
                raise GoogleAuthError(f"could not download Google's signing keys: {err}", 503) from err
            return
        keys = _rsa_keys(data)
        self.fetched = time.time()
        if keys:
            self.keys = keys

    def get(self, kid: str):
        with self.lock:
            age = time.time() - self.fetched
            if not self.keys or age > self.ttl:
                self._refresh()
            elif kid not in self.keys and age > self.min_refetch:
                self._refresh()                  # Google rotated its keys
            return self.keys.get(kid)


# ---------------------------------------------------------------- verifier
class GoogleVerifier:
    def __init__(self, client_id: str, jwks: JWKSCache | None = None, leeway: int = LEEWAY_S) -> None:
        self.client_id = client_id
        self.jwks = jwks or JWKSCache()
        self.leeway = leeway

    def verify(self, credential: str) -> dict:
        import jwt
        if not isinstance(credential, str) or not credential or len(credential) > MAX_TOKEN_CHARS \
                or credential.count(".") != 2:
            raise GoogleAuthError("malformed Google credential", 400)
        try:
            header = jwt.get_unverified_header(credential)
        except jwt.PyJWTError:
            raise GoogleAuthError("malformed Google credential", 400)
        if header.get("alg") != "RS256":            # rejects none, HS256 and friends up front
            raise GoogleAuthError("unsupported token algorithm")
        kid = header.get("kid")
        if not isinstance(kid, str) or not kid:
            raise GoogleAuthError("token has no key id")
        key = self.jwks.get(kid)
        if key is None:
            raise GoogleAuthError("token signed with an unknown key")
        try:
            claims = jwt.decode(credential, key, algorithms=["RS256"], audience=self.client_id,
                                issuer=list(GOOGLE_ISSUERS), leeway=self.leeway,
                                options={"require": ["exp", "iat", "iss", "aud", "sub"]})
        except jwt.ExpiredSignatureError:
            raise GoogleAuthError("Google sign-in expired, try again")
        except jwt.InvalidAudienceError:
            raise GoogleAuthError("token was issued for a different app")
        except jwt.InvalidIssuerError:
            raise GoogleAuthError("token was not issued by Google")
        except jwt.InvalidSignatureError:
            raise GoogleAuthError("token signature is invalid")
        except jwt.PyJWTError as err:
            raise GoogleAuthError(f"invalid Google token: {err}")
        sub = claims.get("sub")
        if not isinstance(sub, str) or not sub or len(sub) > 255:
            raise GoogleAuthError("token has no account id")
        if claims.get("email_verified") not in (True, "true"):
            raise GoogleAuthError("Google has not verified this account's email address", 403)
        return claims


# ------------------------------------------------------------ users, sessions
_CTRL = re.compile(r"[\x00-\x1f\x7f]+")


def _clean(value, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    value = re.sub(r"\s+", " ", _CTRL.sub(" ", value)).strip()
    return value[:limit] or None


def _clean_picture(value) -> str | None:
    if not isinstance(value, str) or not value.startswith("https://") or len(value) > 1000:
        return None
    return None if any(c in value for c in "\"'<> \\\n\r\t") else value


def hash_session(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def upsert_google_user(db: Database, claims: dict) -> dict:
    """Finds or creates the ZENIT user for a verified Google account and makes
    sure it has its own key row (for rate limits and usage). Returns the
    google_accounts row."""
    sub = claims["sub"]
    email = _clean(claims.get("email"), 320)
    name = _clean(claims.get("name"), 100) or _clean(claims.get("given_name"), 100) or \
        (email.split("@")[0] if email else "Google user")
    given = _clean(claims.get("given_name"), 60)
    picture = _clean_picture(claims.get("picture"))
    now = time.time()
    row = db.one("SELECT * FROM google_accounts WHERE sub=?", (sub,))
    if row is None:
        uid = new_id("usr_")
        db.ensure_user(uid, name)
        db.execute("INSERT OR IGNORE INTO google_accounts (sub, user_id, email, name, given_name, picture, created, "
                   "updated) VALUES (?,?,?,?,?,?,?,?)", (sub, uid, email, name, given, picture, now, now))
        row = db.one("SELECT * FROM google_accounts WHERE sub=?", (sub,))
    else:
        db.execute("UPDATE google_accounts SET email=?, name=?, given_name=?, picture=?, updated=? WHERE sub=?",
                   (email, name, given, picture, now, sub))
        db.execute("UPDATE users SET name=? WHERE id=?", (name, row["user_id"]))
    uid = row["user_id"]
    key = db.one("SELECT id FROM api_keys WHERE id=? AND revoked=0", (row["key_id"],)) if row["key_id"] else None
    if key is None:
        # The plaintext of this key is thrown away: it is only an anchor for limits and usage.
        k = create_key(db, name, SESSION_KEY_LABEL, is_admin=False, rate_per_min=120, user_id=uid)
        db.execute("UPDATE google_accounts SET key_id=? WHERE sub=?", (k["id"], sub))
    return db.one("SELECT * FROM google_accounts WHERE sub=?", (sub,))


def create_session(db: Database, user_id: str, ttl: float = SESSION_TTL_S) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    db.execute("DELETE FROM sessions WHERE expires <= ?", (now,))
    db.execute("INSERT INTO sessions (token_hash, user_id, created, expires) VALUES (?,?,?,?)",
               (hash_session(token), user_id, now, now + ttl))
    return token


def lookup_session(db: Database, token: str | None) -> dict | None:
    """Returns {"account": google_accounts row, "key": api_keys row} for a
    live session, else None."""
    if not token or len(token) > 200:
        return None
    s = db.one("SELECT user_id FROM sessions WHERE token_hash=? AND expires > ?", (hash_session(token), time.time()))
    if not s:
        return None
    account = db.one("SELECT * FROM google_accounts WHERE user_id=?", (s["user_id"],))
    if not account or not account["key_id"]:
        return None
    key = db.one("SELECT * FROM api_keys WHERE id=? AND revoked=0", (account["key_id"],))
    if not key:
        return None
    return {"account": account, "key": key}


def delete_session(db: Database, token: str | None) -> bool:
    if not token or len(token) > 200:
        return False
    return db.execute("DELETE FROM sessions WHERE token_hash=?", (hash_session(token),)).rowcount > 0
