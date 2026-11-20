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


# Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash").strip()
SCOPE_MODEL_NAME = os.getenv("SCOPE_MODEL_NAME", GEMINI_MODEL_NAME).strip()

# Google Embedding API. The output dimension MUST match the existing
# Supabase pgvector column dimension. Dami's current schema is 768.
GOOGLE_EMBEDDING_MODEL = os.getenv(
    "GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001"
).strip()
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))

# Supabase service-role credentials must remain server-side.
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
SUPABASE_RPC_NAME = os.getenv("SUPABASE_RPC_NAME", "match_dami_documents").strip()

# Retrieval settings
TEXTBOOK_TOP_K = int(os.getenv("TEXTBOOK_TOP_K", "4"))
QA_TOP_K = int(os.getenv("QA_TOP_K", "3"))
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.0"))

# No restrictive textbook metadata filter by default, because the exact
# metadata shape of the live Dami rows has not yet been verified.
TEXTBOOK_FILTER_METADATA = _json_env("TEXTBOOK_FILTER_METADATA", "{}")
QA_FILTER_METADATA = _json_env("QA_FILTER_METADATA", '{"type":"qa_example"}')
QA_RETRIEVAL_ENABLED = os.getenv("QA_RETRIEVAL_ENABLED", "false").lower() in {
    "1", "true", "yes", "on"
}

# Application
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "4000"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "45"))
CORS_ORIGINS = [
    item.strip()
    for item in os.getenv("CORS_ORIGINS", "*").split(",")
    if item.strip()
]


def validate_settings() -> None:
    missing = []
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_SERVICE_ROLE_KEY:
        missing.append("SUPABASE_SERVICE_ROLE_KEY")
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )
    if EMBEDDING_DIMENSION != 768:
        # The currently documented Dami pgvector schema uses 768 dimensions.
        # Change this only if the deployed vector column is confirmed otherwise.
        raise RuntimeError(
            "EMBEDDING_DIMENSION must be 768 for the current Dami database schema."
        )
