from langchain_core.documents import Document

from .config import (
    QA_COLLECTION,
    QA_TOP_K,
    TEXTBOOK_COLLECTION,
    TEXTBOOK_TOP_K,
)
from .vector_store import get_vector_store


def retrieve_textbook(
    question: str,
    k: int = TEXTBOOK_TOP_K,
) -> list[Document]:
    """
    Retrieve the most relevant textbook chunks for a user question.
    """

    if not question or not question.strip():
        return []

    if k <= 0:
        return []

    store = get_vector_store(TEXTBOOK_COLLECTION)

    return store.similarity_search(
        question.strip(),
        k=k,
    )


def retrieve_examples(
    question: str,
    k: int = QA_TOP_K,
) -> list[Document]:
    """
    Retrieve the most relevant Q&A examples for a user question.
    """

    if not question or not question.strip():
        return []

    if k <= 0:
        return []

    store = get_vector_store(QA_COLLECTION)

    return store.similarity_search(
        question.strip(),
        k=k,
    )


def format_textbook_context(
    documents: list[Document],
) -> str:
    """
    Convert retrieved textbook documents into a prompt-ready context.
    """

    if not documents:
        return "هیچ بخش مرتبطی از کتاب درسی بازیابی نشد."

    blocks: list[str] = []

    for index, doc in enumerate(documents, start=1):
        metadata = doc.metadata

        section_parts = [
            metadata.get("h1"),
            metadata.get("h2"),
            metadata.get("h3"),
        ]

        section = " / ".join(
            str(value)
            for value in section_parts
            if value
        )

        header = f"[بخش {index}]"

        if section:
            header += f" {section}"

        blocks.append(
            f"{header}\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(blocks)


def format_few_shot_examples(
    documents: list[Document],
) -> str:
    """
    Convert retrieved Q&A examples into prompt-ready few-shot examples.
    """

    if not documents:
        return "هیچ نمونه سؤال و پاسخ مشابهی بازیابی نشد."

    blocks: list[str] = []

    for index, doc in enumerate(documents, start=1):
        blocks.append(
            f"[نمونه {index}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(blocks)