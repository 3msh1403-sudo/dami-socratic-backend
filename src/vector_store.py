from __future__ import annotations

import requests

from .config import (
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
    SUPABASE_RPC_NAME,
    REQUEST_TIMEOUT,
)


def match_documents(
    query_embedding: list[float],
    *,
    match_count: int = 3,
    filter_metadata: dict | None = None,
    match_threshold: float = 0.0,
) -> list[dict]:
    """
    Call the existing Supabase RPC:
    match_dami_documents(query_embedding, match_count,
                         filter_metadata, match_threshold)

    This module does not create tables, indexes, embeddings, or vector stores.
    It only searches the already-populated Dami database.
    """
    if not query_embedding:
        return []
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Supabase server credentials are not configured.")

    url = f"{SUPABASE_URL}/rest/v1/rpc/{SUPABASE_RPC_NAME}"
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    body = {
        "query_embedding": query_embedding,
        "match_count": max(1, min(int(match_count), 20)),
        "filter_metadata": filter_metadata or {},
        "match_threshold": float(match_threshold),
    }

    response = requests.post(
        url,
        headers=headers,
        json=body,
        timeout=REQUEST_TIMEOUT,
    )
    if not response.ok:
        raise RuntimeError(
            f"Supabase RPC '{SUPABASE_RPC_NAME}' failed "
            f"({response.status_code}): {response.text[:2000]}"
        )

    result = response.json()
    if not isinstance(result, list):
        raise RuntimeError("Supabase RPC returned an unexpected response.")
    return result
