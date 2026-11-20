from __future__ import annotations

from .config import (
    TEXTBOOK_TOP_K,
    QA_TOP_K,
    MATCH_THRESHOLD,
    TEXTBOOK_FILTER_METADATA,
    QA_FILTER_METADATA,
    QA_RETRIEVAL_ENABLED,
)
from .embeddings import embed_query
from .vector_store import match_documents


def _to_document(row: dict) -> dict:
    """
    Normalize the exact columns returned by match_dami_documents.
    """
    return {
        "id": row.get("id"),
        "document_key": row.get("document_key"),
        "content": str(row.get("content") or ""),
        "source": row.get("source"),
        "page": row.get("page"),
        "chunk_index": row.get("chunk_index"),
        "metadata": row.get("metadata") or {},
        "similarity": row.get("similarity"),
    }


def retrieve_textbook(
    question: str,
    k: int = TEXTBOOK_TOP_K,
    query_embedding: list[float] | None = None,
) -> list[dict]:
    """Retrieve textbook chunks from the existing Dami Supabase database."""
    if not question or not question.strip() or k <= 0:
        return []
    query_vector = query_embedding or embed_query(question.strip())
    rows = match_documents(
        query_vector,
        match_count=k,
        filter_metadata=TEXTBOOK_FILTER_METADATA,
        match_threshold=MATCH_THRESHOLD,
    )
    return [_to_document(row) for row in rows if row.get("content")]


def retrieve_examples(
    question: str,
    k: int = QA_TOP_K,
    query_embedding: list[float] | None = None,
) -> list[dict]:
    """
    Retrieve optional Q&A few-shot examples using the same RPC.
    This is enabled only when matching QA records exist in Supabase metadata.
    """
    if not QA_RETRIEVAL_ENABLED or not question or not question.strip() or k <= 0:
        return []
    query_vector = query_embedding or embed_query(question.strip())
    rows = match_documents(
        query_vector,
        match_count=k,
        filter_metadata=QA_FILTER_METADATA,
        match_threshold=MATCH_THRESHOLD,
    )
    return [_to_document(row) for row in rows if row.get("content")]


def format_textbook_context(documents: list[dict]) -> str:
    """Format retrieved textbook chunks as trusted-source context."""
    if not documents:
        return "هیچ بخش مرتبطی از کتاب درسی بازیابی نشد."

    blocks = []
    for index, doc in enumerate(documents, start=1):
        metadata = doc.get("metadata") or {}
        section_parts = [
            metadata.get("h1"),
            metadata.get("h2"),
            metadata.get("h3"),
            metadata.get("chapter"),
            metadata.get("section"),
        ]
        section = " / ".join(str(value) for value in section_parts if value)
        source = doc.get("source") or metadata.get("source_title") or ""
        page = doc.get("page")
        header = f"[بخش {index}]"
        if section:
            header += f" {section}"
        if source:
            header += f" | منبع: {source}"
        if page is not None:
            header += f" | صفحه: {page}"
        blocks.append(f"{header}\n{doc['content']}")
    return "\n\n".join(blocks)


def format_few_shot_examples(documents: list[dict]) -> str:
    """Format retrieved Q&A examples for style guidance only."""
    if not documents:
        return "هیچ نمونه سؤال و پاسخ مشابهی بازیابی نشد."
    blocks = []
    for index, doc in enumerate(documents, start=1):
        blocks.append(f"[نمونه {index}]\n{doc['content']}")
    return "\n\n".join(blocks)
