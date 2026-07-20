import requests

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT_SECONDS = 60


def call_openrouter(
    api_key: str,
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float = 0.7,
    image_data_uri: str | None = None,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if image_data_uri:
        user_content = [
            {"type": "text", "text": user_message},
            {"type": "image_url", "image_url": {"url": image_data_uri}},
        ]
    else:
        user_content = user_message

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": 1024,
    }

    try:
        response = requests.post(
            OPENROUTER_API_URL, headers=headers, json=payload, timeout=TIMEOUT_SECONDS
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"OpenRouter request timed out after {TIMEOUT_SECONDS}s. "
            "The LLM may be overloaded -- try again."
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not connect to OpenRouter. Check your internet connection."
        )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error calling OpenRouter: {e}")

    if response.status_code == 401:
        raise RuntimeError(
            "OpenRouter API key is invalid or missing. Check the OPENROUTER_API_KEY "
            "server configuration."
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

    try:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError):
        raise RuntimeError(
            f"Unexpected response format from OpenRouter: {response.text[:300]}"
        )
