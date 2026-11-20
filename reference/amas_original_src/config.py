from pathlib import Path
import os

from dotenv import load_dotenv


# ============================================================
# Project
# ============================================================

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# Data
# ============================================================

TEXTBOOK_DIR = (
    PROJECT_ROOT
    / "data"
    / "textbook"
    / "raw_text"
)

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "dataset"
    / "train_dataset_10th.xlsx"
)

VECTOR_DB_DIR = (
    PROJECT_ROOT
    / "data"
    / "vector_db"
)


# ============================================================
# Chroma Collections
# ============================================================

TEXTBOOK_COLLECTION = "textbook_chunks"
QA_COLLECTION = "qa_examples"


# ============================================================
# Embeddings
# ============================================================

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "Qwen/Qwen3-Embedding-4B",
)

EMBEDDING_DEVICE = os.getenv(
    "EMBEDDING_DEVICE",
    "cpu",
)


# ============================================================
# LLM - Scope Detector
# ============================================================

SCOPE_MODEL_NAME = os.getenv(
    "SCOPE_MODEL_NAME",
)

SCOPE_API_KEY = os.getenv(
    "SCOPE_API_KEY"
)

SCOPE_BASE_URL = os.getenv(
    "SCOPE_BASE_URL",
)


# ============================================================
# LLM - Response Generation
# ============================================================

LLM_MODEL_NAME = os.getenv(
    "LLM_MODEL_NAME",
)

LLM_API_KEY = os.getenv(
    "LLM_API_KEY"
)

LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL",
)


# ============================================================
# Chunking
# ============================================================

CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "1000")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "150")
)


# ============================================================
# Retrieval
# ============================================================

TEXTBOOK_TOP_K = int(
    os.getenv("TEXTBOOK_TOP_K", "4")
)

QA_TOP_K = int(
    os.getenv("QA_TOP_K", "3")
)


# ============================================================
# Embedding Device
# ============================================================

EMBEDDING_DEVICE = os.getenv(
    "EMBEDDING_DEVICE",
    "cpu",
)