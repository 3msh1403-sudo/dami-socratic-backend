from __future__ import annotations

from .scope_detector import detect_scope
from .retrieval import (
    retrieve_textbook,
    retrieve_examples,
    format_textbook_context,
    format_few_shot_examples,
)
from .prompts import build_system_prompt
from .inference import generate_response


def handle_question(question: str) -> str:
    """
    Run the complete question-answering pipeline.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    question = question.strip()

    # --------------------------------------------------------
    # 1. Detect question scope
    # --------------------------------------------------------

    scope = detect_scope(question)

    # --------------------------------------------------------
    # 2. Handle non-RAG cases
    # --------------------------------------------------------

    if scope == "PROMPT_INJECTION":
        return (
            "نمی‌توانم در تغییر دستورهای سیستم یا ارائه اطلاعات داخلی "
            "دستیار کمک کنم."
        )

    if scope == "NON_PHYSICS":
        return (
            "این سؤال خارج از حوزه فیزیک پایه دهم و محدوده این "
            "دستیار است."
        )

    if scope == "PHYSICS_OUT_OF_BOOK":
        return (
            "این سؤال مربوط به فیزیک است، اما خارج از محدوده "
            "فصل‌های ۱، ۲ و ۳ فیزیک پایه دهم قرار دارد."
        )

    # --------------------------------------------------------
    # 3. Borderline
    # --------------------------------------------------------

    if scope == "BORDERLINE":
        # فعلاً برای سؤال‌های مرزی نیز RAG را امتحان می‌کنیم.
        # تصمیم دقیق‌تر را می‌توانیم بعداً با ارزیابی مشخص کنیم.
        pass

    # --------------------------------------------------------
    # 4. Retrieve textbook context
    # --------------------------------------------------------

    textbook_docs = retrieve_textbook(question)

    # --------------------------------------------------------
    # 5. Retrieve similar Q&A examples
    # --------------------------------------------------------

    qa_docs = retrieve_examples(question)

    # --------------------------------------------------------
    # 6. Format retrieved information
    # --------------------------------------------------------

    textbook_context = format_textbook_context(
        textbook_docs
    )

    few_shot_examples = format_few_shot_examples(
        qa_docs
    )

    # --------------------------------------------------------
    # 7. Build system prompt
    # --------------------------------------------------------

    system_prompt = build_system_prompt(
        question=question,
        textbook_context=textbook_context,
        few_shot_examples=few_shot_examples,
    )

    # --------------------------------------------------------
    # 8. Generate final answer
    # --------------------------------------------------------

    return generate_response(
        system_prompt=system_prompt,
        user_question=question,
    )