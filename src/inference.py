from __future__ import annotations

from .config import GEMINI_MODEL_NAME
from .gemini_client import post_gemini_json


def generate_text(
    *,
    model_name: str,
    system_prompt: str,
    user_text: str,
    temperature: float = 0.5,
    max_output_tokens: int = 1200,
    purpose: str = "chat",
) -> str:
    """Call Gemini generateContent, with project-key failover for quota/transient errors."""
    if not system_prompt or not system_prompt.strip():
        raise ValueError("System prompt cannot be empty.")
    if not user_text or not user_text.strip():
        raise ValueError("User text cannot be empty.")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent"
    )
    response = post_gemini_json(
        url,
        {
            "systemInstruction": {"parts": [{"text": system_prompt.strip()}]},
            "contents": [{"role": "user", "parts": [{"text": user_text.strip()}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        },
        purpose=purpose,
    )

    payload = response.json()
    candidates = payload.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini returned no candidates.")

    parts = candidates[0].get("content", {}).get("parts", [])
    result = "".join(
        part.get("text", "")
        for part in parts
        if isinstance(part, dict)
    ).strip()
    if not result:
        raise RuntimeError("Gemini returned an empty response.")
    return result


def generate_response(system_prompt: str, user_question: str) -> str:
    """Generate Dami's final Socratic response using the chat project."""
    return generate_text(
        model_name=GEMINI_MODEL_NAME,
        system_prompt=system_prompt,
        user_text=user_question,
        temperature=0.55,
        max_output_tokens=1200,
        purpose="chat",
    )



