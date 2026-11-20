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
from .embeddings import embed_query
from .config import QA_RETRIEVAL_ENABLED
from .config import MAX_QUESTION_LENGTH


def handle_question(question: str) -> str:
    """
    Run the original Amas pipeline with Dami's providers:
    Scope Detection -> Textbook Retrieval -> Optional Q&A Retrieval
    -> Context Formatting -> Socratic Prompt -> Gemini Generation.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    question = question.strip()
    if len(question) > MAX_QUESTION_LENGTH:
        raise ValueError(
            f"Question is too long. Maximum length is {MAX_QUESTION_LENGTH}."
        )

    # 1. Detect question scope (same separate classification stage as Amas).
    scope = detect_scope(question)

    # 2. Handle non-RAG cases.
    if scope == "PROMPT_INJECTION":
        return (
            "نمی‌توانم در تغییر دستورهای سیستم یا ارائه اطلاعات داخلی "
            "و محرمانه دامی کمک کنم."
        )

    if scope == "NON_SUBJECT":
        return (
            "این سؤال خارج از حوزه آموزشی و محدوده فعالیت دستیار دامی است."
        )

    if scope == "SUBJECT_OUT_OF_BOOK":
        return (
            "این موضوع به حوزه آموزشی و مهارت‌های زندگی مربوط است، اما "
            "در محدوده منابع کتابی فعلی دامی قرار ندارد."
        )

    # 3. BORDERLINE follows Amas behavior: attempt RAG rather than reject.
    # 4. Create one query embedding and reuse it across both retrieval lanes.
    # This avoids duplicate Google Embedding API calls for the same question.
    query_vector = embed_query(question)

    # 5. Retrieve textbook context.
    textbook_docs = retrieve_textbook(
        question, query_embedding=query_vector
    )

    # 6. Retrieve optional similar Q&A examples.
    # Disabled by default until Dami QA records/metadata are confirmed.
    qa_docs = (
        retrieve_examples(question, query_embedding=query_vector)
        if QA_RETRIEVAL_ENABLED
        else []
    )

    # 7. Format retrieved information.
    textbook_context = format_textbook_context(textbook_docs)
    few_shot_examples = format_few_shot_examples(qa_docs)

    # 8. Build the Socratic system prompt.
    system_prompt = build_system_prompt(
        question=question,
        textbook_context=textbook_context,
        few_shot_examples=few_shot_examples,
    )

    # 9. Generate the final answer.
    return generate_response(
        system_prompt=system_prompt,
        user_question=question,
    )
