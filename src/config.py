from __future__ import annotations

import json
import os
from dotenv import load_dotenv

load_dotenv()


def _json_env(name: str, default: str) -> dict:
    raw = os.getenv(name, default)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{name} must contain valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be a JSON object.")
    return value


# Two independent Google AI Studio projects.
# GEMINI_API_KEY remains as a backward-compatible fallback for older deployments.
LEGACY_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
SCOPE_GEMINI_API_KEY = (
    os.getenv("SCOPE_GEMINI_API_KEY", "").strip() or LEGACY_GEMINI_API_KEY
)
CHAT_GEMINI_API_KEY = (
    os.getenv("CHAT_GEMINI_API_KEY", "").strip() or LEGACY_GEMINI_API_KEY
)
EMBEDDING_GEMINI_API_KEY = os.getenv("EMBEDDING_GEMINI_API_KEY", "").strip()

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "").strip()
SCOPE_MODEL_NAME = os.getenv("SCOPE_MODEL_NAME", "").strip()

GOOGLE_EMBEDDING_MODEL = os.getenv(
    "GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001"
).strip()
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
SUPABASE_RPC_NAME = os.getenv("SUPABASE_RPC_NAME", "match_dami_documents").strip()
DAMI_RAG_SHARED_SECRET = os.getenv("DAMI_RAG_SHARED_SECRET", "").strip()

TEXTBOOK_TOP_K = int(os.getenv("TEXTBOOK_TOP_K", "4"))
QA_TOP_K = int(os.getenv("QA_TOP_K", "3"))
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.0"))

TEXTBOOK_FILTER_METADATA = _json_env("TEXTBOOK_FILTER_METADATA", "{}")
QA_FILTER_METADATA = _json_env("QA_FILTER_METADATA", '{"type":"qa_example"}')
QA_RETRIEVAL_ENABLED = os.getenv("QA_RETRIEVAL_ENABLED", "false").lower() in {
    "1", "true", "yes", "on"
}

MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "4000"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "45"))
CORS_ORIGINS = [
    item.strip()
    for item in os.getenv("CORS_ORIGINS", "*").split(",")
    if item.strip()
]


def gemini_keys_for(purpose: str) -> list[str]:
    """Return ordered, de-duplicated project keys for a Gemini operation."""
    if purpose == "scope":
        candidates = [SCOPE_GEMINI_API_KEY, CHAT_GEMINI_API_KEY]
    elif purpose == "embedding":
        candidates = [EMBEDDING_GEMINI_API_KEY, CHAT_GEMINI_API_KEY, SCOPE_GEMINI_API_KEY]
    elif purpose == "chat":
        candidates = [CHAT_GEMINI_API_KEY, SCOPE_GEMINI_API_KEY]
    else:
        raise ValueError(f"Unknown Gemini purpose: {purpose}")

    result: list[str] = []
    for key in candidates:
        if key and key not in result:
            result.append(key)
    return result


def validate_settings() -> None:
    missing = []
    if not SCOPE_GEMINI_API_KEY:
        missing.append("SCOPE_GEMINI_API_KEY")
    if not CHAT_GEMINI_API_KEY:
        missing.append("CHAT_GEMINI_API_KEY")
    if not GEMINI_MODEL_NAME:
        missing.append("GEMINI_MODEL_NAME")
    if not SCOPE_MODEL_NAME:
        missing.append("SCOPE_MODEL_NAME")
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_SERVICE_ROLE_KEY:
        missing.append("SUPABASE_SERVICE_ROLE_KEY")
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )
    if EMBEDDING_DIMENSION != 768:
        raise RuntimeError(
            "EMBEDDING_DIMENSION must be 768 for the current Dami database schema."
        )

