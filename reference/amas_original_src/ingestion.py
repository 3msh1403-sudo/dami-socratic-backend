from pathlib import Path
from typing import Iterable

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from .config import CHUNK_OVERLAP, CHUNK_SIZE, DATASET_PATH, TEXTBOOK_DIR


def load_markdown_documents(textbook_dir: Path = TEXTBOOK_DIR) -> list[Document]:
    """Load every chapter markdown file as one LangChain Document."""
    documents: list[Document] = []

    for path in sorted(textbook_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "source_path": str(path),
                    "type": "textbook",
                    "chapter": path.stem,
                },
            )
        )

    if not documents:
        raise FileNotFoundError(
            f"No .md files found in {textbook_dir}. "
            "Expected chapter1.md, chapter2.md, chapter3.md, ..."
        )

    return documents


def split_textbook_documents(
    documents: Iterable[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split Markdown semantically first (headers), then by character length.

    The header metadata is preserved so retrieval results retain chapter/section
    information.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ],
        strip_headers=False,
    )

    character_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            "۔ ",
            ". ",
            "، ",
            ", ",
            " ",
            "",
        ],
    )

    chunks: list[Document] = []

    for document in documents:
        header_docs = header_splitter.split_text(document.page_content)

        for section_index, section_doc in enumerate(header_docs):
            metadata = dict(document.metadata)
            metadata.update(section_doc.metadata)
            metadata["section_index"] = section_index

            section_doc.metadata = metadata
            chunks.extend(character_splitter.split_documents([section_doc]))

    # Add deterministic IDs useful for debugging/rebuilding.
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        chunk.id = f"textbook-{index}"

    return chunks


def load_qa_examples(dataset_path: Path = DATASET_PATH) -> list[Document]:
    """
    Load the 98 Q&A examples from Excel.

    The project description says the question column is 'quesion'.
    We also accept 'question' to make the ingestion robust to a later typo fix.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_excel(dataset_path)

    normalized = {str(col).strip().lower(): col for col in df.columns}

    question_col = normalized.get("quesion") or normalized.get("question")
    answer_col = normalized.get("answer")

    if question_col is None or answer_col is None:
        raise ValueError(
            "Excel must contain question/quesion and answer columns. "
            f"Found columns: {list(df.columns)}"
        )

    documents: list[Document] = []

    for index, row in df.iterrows():
        question = str(row[question_col]).strip()
        answer = str(row[answer_col]).strip()

        if not question or question.lower() == "nan":
            continue

        if not answer or answer.lower() == "nan":
            continue

        content = (
            "سؤال نمونه:\n"
            f"{question}\n\n"
            "پاسخ نمونه:\n"
            f"{answer}"
        )

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": dataset_path.name,
                    "type": "qa_example",
                    "qa_index": int(index),
                    "question": question,
                },
                id=f"qa-{int(index)}",
            )
        )

    if not documents:
        raise ValueError("No valid Q&A records were found in the Excel file.")

    return documents
