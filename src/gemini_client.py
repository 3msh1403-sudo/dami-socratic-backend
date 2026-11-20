from __future__ import annotations

import requests

from .config import REQUEST_TIMEOUT, gemini_keys_for

# Only these responses indicate that trying the other independent project
# may reasonably help. Model-not-found and invalid-request errors are not retried.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def post_gemini_json(url: str, payload: dict, *, purpose: str) -> requests.Response:
    """POST to Gemini, failing over to the other project on quota/transient errors."""
    keys = gemini_keys_for(purpose)
    if not keys:
        raise RuntimeError(f"No Gemini API key configured for {purpose}.")

    last_error = None
    for index, api_key in enumerate(keys):
        try:
            response = requests.post(
                url,
                params={"key": api_key},
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            last_error = f"Gemini network error: {exc}"
            if index < len(keys) - 1:
                continue
            raise RuntimeError(last_error) from exc

        if response.ok:
            return response

        if response.status_code in RETRYABLE_STATUS_CODES and index < len(keys) - 1:
            last_error = (
                f"Gemini API error ({response.status_code}): "
                f"{response.text[:1500]}"
            )
            continue

        raise RuntimeError(
            f"Gemini API error ({response.status_code}): {response.text[:2000]}"
        )

    raise RuntimeError(last_error or "Gemini request failed on all configured projects.")
