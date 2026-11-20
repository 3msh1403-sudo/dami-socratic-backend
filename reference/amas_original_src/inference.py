from langchain_openai import ChatOpenAI

from .config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL_NAME,
)


def get_llm_model() -> ChatOpenAI:
    """
    Create an OpenAI-compatible LLM client.
    """

    return ChatOpenAI(
        model=LLM_MODEL_NAME,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
    )


def generate_response(
    system_prompt: str,
    user_question: str,
) -> str:
    """
    Generate a response from the LLM.
    """

    if not system_prompt or not system_prompt.strip():
        raise ValueError("System prompt cannot be empty.")

    if not user_question or not user_question.strip():
        raise ValueError("User question cannot be empty.")

    model = get_llm_model()

    response = model.invoke(
        [
            ("system", system_prompt.strip()),
            ("user", user_question.strip()),
        ]
    )

    if not response.content:
        raise RuntimeError("LLM returned an empty response.")

    return response.content.strip()