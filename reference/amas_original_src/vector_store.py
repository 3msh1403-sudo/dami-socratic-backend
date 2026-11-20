from pathlib import Path
from typing import Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import (
    QA_COLLECTION,
    TEXTBOOK_COLLECTION,
    VECTOR_DB_DIR,
)
from .embeddings import get_embedding_model


def get_vector_store(
    collection_name: str,
    persist_directory: Path = VECTOR_DB_DIR,
) -> Chroma:
    """
    Open or create a persistent local Chroma collection.

    The embedding function is provided by the project's
    OpenAI-compatible Embedding API.
    """
    persist_directory.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embedding_model(),
        persist_directory=str(persist_directory),
    )


def build_textbook_store(
    chunks: Sequence[Document],
    *,
    reset: bool = False,
) -> Chroma:
    """
    Build/update the textbook vector store.

    Each textbook chunk is stored using its deterministic Document ID,
    e.g. textbook-0, textbook-1, ...
    """

    store = get_vector_store(TEXTBOOK_COLLECTION)

    if reset:
        store.delete_collection()
        store = get_vector_store(TEXTBOOK_COLLECTION)

    documents = list(chunks)

    if not documents:
        raise ValueError("No textbook chunks were provided.")

    ids = [doc.id for doc in documents]

    if any(doc_id is None for doc_id in ids):
        raise ValueError("Every textbook document must have a valid ID.")

    store.add_documents(
        documents=documents,
        ids=ids,
    )

    return store


def build_qa_store(
    examples: Sequence[Document],
    *,
    reset: bool = False,
) -> Chroma:
    """
    Build/update the Q&A example vector store.

    Each example is stored using its deterministic Document ID,
    e.g. qa-0, qa-1, ...
    """

    store = get_vector_store(QA_COLLECTION)

    if reset:
        store.delete_collection()
        store = get_vector_store(QA_COLLECTION)

    documents = list(examples)

    if not documents:
        raise ValueError("No Q&A examples were provided.")

    ids = [doc.id for doc in documents]

    if any(doc_id is None for doc_id in ids):
        raise ValueError("Every Q&A document must have a valid ID.")

    store.add_documents(
        documents=documents,
        ids=ids,
    )

    return store