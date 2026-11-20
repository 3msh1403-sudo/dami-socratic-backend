from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
)


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Create and cache the local embedding model.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={
            "device": EMBEDDING_DEVICE,
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
        local_dir=r"D:\physics_private_teacher\models\Qwen3-Embedding-4B",
    )