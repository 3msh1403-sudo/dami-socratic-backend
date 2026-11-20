from __future__ import annotations

import argparse

from src.ingestion import (
    load_markdown_documents,
    split_textbook_documents,
    load_qa_examples,
)
from src.vector_store import (
    get_vector_store,
    build_textbook_store,
    build_qa_store,
)
from src.config import (
    TEXTBOOK_COLLECTION,
    QA_COLLECTION,
)


def collection_exists_and_complete(
    collection_name: str,
    expected_count: int,
) -> bool:
    """
    Check whether a Chroma collection already exists and contains
    the expected number of documents.
    """

    try:
        store = get_vector_store(collection_name)
        count = store._collection.count()

        print(
            f"Collection '{collection_name}': "
            f"{count}/{expected_count} documents"
        )

        return count == expected_count

    except Exception as exc:
        print(
            f"Could not inspect collection "
            f"'{collection_name}': {exc}"
        )
        return False


def build_database(reset: bool = False) -> None:
    print("=" * 60)
    print("VECTOR DATABASE BUILD")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load and prepare textbook
    # ---------------------------------------------------------
    print("\n[1/4] Loading textbook...")

    textbook_documents = load_markdown_documents()

    print(
        f"Loaded {len(textbook_documents)} chapter documents."
    )

    print("\n[2/4] Splitting textbook...")

    textbook_chunks = split_textbook_documents(
        textbook_documents
    )

    print(
        f"Created {len(textbook_chunks)} textbook chunks."
    )

    # ---------------------------------------------------------
    # 2. Load Q&A dataset
    # ---------------------------------------------------------
    print("\n[3/4] Loading Q&A dataset...")

    qa_examples = load_qa_examples()

    print(
        f"Loaded {len(qa_examples)} Q&A examples."
    )

    # ---------------------------------------------------------
    # 3. Check existing vector database
    # ---------------------------------------------------------
    print("\n[4/4] Checking vector database...")

    textbook_complete = collection_exists_and_complete(
        TEXTBOOK_COLLECTION,
        len(textbook_chunks),
    )

    qa_complete = collection_exists_and_complete(
        QA_COLLECTION,
        len(qa_examples),
    )

    # ---------------------------------------------------------
    # Existing DB is complete
    # ---------------------------------------------------------
    if not reset and textbook_complete and qa_complete:
        print("\n" + "=" * 60)
        print("Vector database already exists and is complete.")
        print("No embeddings were generated.")
        print("=" * 60)

        print(
            f"Textbook vectors : {len(textbook_chunks)}"
        )
        print(
            f"Q&A vectors      : {len(qa_examples)}"
        )
        print(
            f"Total vectors    : "
            f"{len(textbook_chunks) + len(qa_examples)}"
        )

        return

    # ---------------------------------------------------------
    # Build / rebuild textbook collection
    # ---------------------------------------------------------
    if reset:
        print("\nReset requested.")
        print("Rebuilding all vector collections...")

        build_textbook_store(
            textbook_chunks,
            reset=True,
        )

        build_qa_store(
            qa_examples,
            reset=True,
        )

    else:
        # -----------------------------------------------------
        # Build only missing/incomplete collections
        # -----------------------------------------------------
        if not textbook_complete:
            print(
                "\nTextbook collection is missing or incomplete."
            )

            build_textbook_store(
                textbook_chunks,
                reset=True,
            )

        else:
            print(
                "\nTextbook collection is already complete."
            )

        if not qa_complete:
            print(
                "\nQ&A collection is missing or incomplete."
            )

            build_qa_store(
                qa_examples,
                reset=True,
            )

        else:
            print(
                "\nQ&A collection is already complete."
            )

    # ---------------------------------------------------------
    # Final verification
    # ---------------------------------------------------------
    print("\nVerifying vector database...")

    final_textbook_count = get_vector_store(
        TEXTBOOK_COLLECTION
    )._collection.count()

    final_qa_count = get_vector_store(
        QA_COLLECTION
    )._collection.count()

    print("\n" + "=" * 60)
    print("VECTOR DATABASE READY")
    print("=" * 60)

    print(
        f"Textbook vectors : "
        f"{final_textbook_count}/{len(textbook_chunks)}"
    )

    print(
        f"Q&A vectors      : "
        f"{final_qa_count}/{len(qa_examples)}"
    )

    print(
        f"Total vectors    : "
        f"{final_textbook_count + final_qa_count}"
    )

    if (
        final_textbook_count != len(textbook_chunks)
        or final_qa_count != len(qa_examples)
    ):
        raise RuntimeError(
            "Vector database verification failed."
        )

    print("\nEverything looks good.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or rebuild the project's Chroma vector database."
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and rebuild all vector collections.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build_database(
        reset=args.reset
    )


if __name__ == "__main__":
    main()