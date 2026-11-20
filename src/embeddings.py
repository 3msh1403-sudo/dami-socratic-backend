from __future__ import annotations

from .config import (
    GOOGLE_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
)
from .gemini_client import post_gemini_json


def embed_query(text: str) -> list[float]:
    """
    Create a query embedding using the existing Gemini embedding model.
    Keep the 768 dimensions unchanged to match the current Supabase vectors.
    """
    if not text or not text.strip():
        raise ValueError("Embedding input cannot be empty.")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GOOGLE_EMBEDDING_MODEL}:embedContent"
    )
    response = post_gemini_json(
        url,
        {
            "model": f"models/{GOOGLE_EMBEDDING_MODEL}",
            "content": {"parts": [{"text": text.strip()}]},
            "taskType": "RETRIEVAL_QUERY",
            "outputDimensionality": EMBEDDING_DIMENSION,
        },
        purpose="embedding",
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
