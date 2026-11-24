from __future__ import annotations

import hashlib
import hmac
import re
import time
from datetime import datetime, timezone

import requests

from .config import DAMI_RAG_SHARED_SECRET, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY

MAX_CLOCK_SKEW_SECONDS = 300
NONCE_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.I)


class RequestAuthError(Exception):
    """Raised when a server-to-server request fails authentication."""


def _claim_nonce(nonce: str, timestamp: int) -> bool:
    """Atomically claim a nonce in Supabase; duplicate primary keys are rejected."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RequestAuthError("request authentication storage is not configured")

    url = f"{SUPABASE_URL}/rest/v1/dami_rag_request_nonces"
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    expires = datetime.fromtimestamp(
        timestamp + MAX_CLOCK_SKEW_SECONDS, tz=timezone.utc
    ).isoformat()

    # Cleanup is best-effort. The unique primary key remains the replay guard.
    try:
        requests.delete(
            url,
            headers=headers,
            params={"expires_at": f"lt.{datetime.now(timezone.utc).isoformat()}"},
            timeout=4,
        )
    except requests.RequestException:
        pass

    try:
        response = requests.post(
            url,
            headers=headers,
            json={"nonce": nonce, "expires_at": expires},
            timeout=6,
        )
    except requests.RequestException as exc:
        raise RequestAuthError("request authentication storage unavailable") from exc

    if response.status_code in (200, 201, 204):
        return True
    if response.status_code == 409:
        return False
    raise RequestAuthError("request authentication storage rejected the request")


def verify_signed_request(
    *, method: str, path: str, raw_body: bytes,
    timestamp: str | None, nonce: str | None, signature: str | None,
) -> None:
    secret = DAMI_RAG_SHARED_SECRET
    if len(secret) < 32:
        raise RequestAuthError("shared secret is not configured")
    if method != "POST" or path != "/chat":
        raise RequestAuthError("invalid request target")
    if not timestamp or not timestamp.isdigit() or len(timestamp) > 12:
        raise RequestAuthError("invalid timestamp")
    if not nonce or not NONCE_RE.fullmatch(nonce):
        raise RequestAuthError("invalid nonce")
    if not signature or not re.fullmatch(r"[0-9a-f]{64}", signature, re.I):
        raise RequestAuthError("invalid signature")

    request_time = int(timestamp)
    now = int(time.time())
    if abs(now - request_time) > MAX_CLOCK_SKEW_SECONDS:
        raise RequestAuthError("request timestamp expired")

    body_hash = hashlib.sha256(raw_body).hexdigest()
    canonical = f"{method}\n{path}\n{timestamp}\n{nonce}\n{body_hash}"
    expected = hmac.new(
        secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected, signature.lower()):
        raise RequestAuthError("signature mismatch")

    if not _claim_nonce(nonce, request_time):
        raise RequestAuthError("request replay detected")
