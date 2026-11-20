from __future__ import annotations

from src.config import validate_settings
from src.orchestration import handle_question


def main():
    validate_settings()
    print("DAMI SOCRATIC RAG - INTERACTIVE TEST")
    question = input("\nسؤال را وارد کن:\n> ").strip()
    if not question:
        print("سؤال خالی است.")
        return
    print("\nدر حال پردازش...\n")
    try:
        print(handle_question(question))
    except Exception as exc:
        print(f"خطا: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
