from __future__ import annotations

import requests

from .config import (
    GEMINI_API_KEY,
    GOOGLE_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    REQUEST_TIMEOUT,
)


def embed_query(text: str) -> list[float]:
    """
    Create a query embedding with Google's Gemini Embedding API.

    outputDimensionality=768 is intentional: it must match the existing
    Supabase pgvector column. Do not change the dimension without rebuilding
    the stored vectors with the same embedding model and dimension.
    """
    if not text or not text.strip():
        raise ValueError("Embedding input cannot be empty.")
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GOOGLE_EMBEDDING_MODEL}:embedContent"
    )
    response = requests.post(
        url,
        params={"key": GEMINI_API_KEY},
        json={
            "model": f"models/{GOOGLE_EMBEDDING_MODEL}",
            "content": {"parts": [{"text": text.strip()}]},
            "taskType": "RETRIEVAL_QUERY",
            "outputDimensionality": EMBEDDING_DIMENSION,
        },
        timeout=REQUEST_TIMEOUT,
    )
    if not response.ok:
        raise RuntimeError(
            f"Google Embedding API error ({response.status_code}): "
            f"{response.text[:1500]}"
        )

    payload = response.json()
    vector = payload.get("embedding", {}).get("values")
    if not isinstance(vector, list) or not vector:
        raise RuntimeError("Google Embedding API returned no embedding vector.")
    if len(vector) != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Embedding dimension mismatch: received {len(vector)}, "
            f"expected {EMBEDDING_DIMENSION}."
        )
    return [float(value) for value in vector]
