from __future__ import annotations

import requests

from .config import GEMINI_API_KEY, GEMINI_MODEL_NAME, REQUEST_TIMEOUT


def generate_text(
    *,
    model_name: str,
    system_prompt: str,
    user_text: str,
    temperature: float = 0.5,
    max_output_tokens: int = 1200,
) -> str:
    """Call the Google Gemini generateContent REST API."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    if not system_prompt or not system_prompt.strip():
        raise ValueError("System prompt cannot be empty.")
    if not user_text or not user_text.strip():
        raise ValueError("User text cannot be empty.")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent"
    )
    response = requests.post(
        url,
        params={"key": GEMINI_API_KEY},
        json={
            "systemInstruction": {
                "parts": [{"text": system_prompt.strip()}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_text.strip()}],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        },
        timeout=REQUEST_TIMEOUT,
    )
    if not response.ok:
        raise RuntimeError(
            f"Gemini API error ({response.status_code}): "
            f"{response.text[:2000]}"
        )

    payload = response.json()
    candidates = payload.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini returned no candidates.")

    parts = (
        candidates[0]
        .get("content", {})
        .get("parts", [])
    )
    text = "".join(
        part.get("text", "")
        for part in parts
        if isinstance(part, dict)
    ).strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text


def generate_response(system_prompt: str, user_question: str) -> str:
    """Generate Dami's final Socratic response using Gemini."""
    return generate_text(
        model_name=GEMINI_MODEL_NAME,
        system_prompt=system_prompt,
        user_text=user_question,
        temperature=0.55,
        max_output_tokens=1200,
    )
