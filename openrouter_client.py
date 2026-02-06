"""Minimal OpenRouter API wrapper using requests."""

import requests


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT_SECONDS = 60


def call_openrouter(api_key, model, system_prompt, user_message, temperature=0.7):
    """Send a chat completion request to OpenRouter.

    Args:
        api_key: OpenRouter API key.
        model: LLM model identifier (e.g. "anthropic/claude-sonnet-4").
        system_prompt: The system prompt string.
        user_message: The user message string.
        temperature: Sampling temperature (0.0-1.0).

    Returns:
        The assistant's response text.

    Raises:
        RuntimeError: On API or network errors with a human-readable message.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": 1024,
    }

    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"OpenRouter request timed out after {TIMEOUT_SECONDS}s. "
            "The LLM may be overloaded — try again."
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not connect to OpenRouter. Check your internet connection."
        )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error calling OpenRouter: {e}")

    if response.status_code == 401:
        raise RuntimeError(
            "OpenRouter API key is invalid or missing. "
            "Get your key at https://openrouter.ai/keys"
        )
    if response.status_code == 402:
        raise RuntimeError(
            "OpenRouter account has insufficient credits. "
            "Add credits at https://openrouter.ai/credits"
        )
    if response.status_code == 429:
        raise RuntimeError(
            "OpenRouter rate limit exceeded. Wait a moment and try again."
        )
    if response.status_code != 200:
        raise RuntimeError(
            f"OpenRouter returned HTTP {response.status_code}: {response.text[:300]}"
        )

    data = response.json()

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise RuntimeError(
            f"Unexpected response format from OpenRouter: {str(data)[:300]}"
        )
