import logging
import time
from dataclasses import dataclass

import requests

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
TIMEOUT_SECONDS = 60
MODELS_CACHE_TTL_SECONDS = 3600

logger = logging.getLogger("app.openrouter")

_models_cache: dict = {"models": None, "fetched_at": 0.0}


@dataclass
class OpenRouterResult:
    content: str
    cost: float | None


def call_openrouter(
    api_key: str,
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float = 0.7,
    image_data_uri: str | None = None,
) -> OpenRouterResult:
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
        "usage": {"include": True},
    }

    start = time.monotonic()
    try:
        response = requests.post(
            OPENROUTER_API_URL, headers=headers, json=payload, timeout=TIMEOUT_SECONDS
        )
    except requests.exceptions.Timeout:
        logger.warning("openrouter call timed out model=%s after=%ss", model, TIMEOUT_SECONDS)
        raise RuntimeError(
            f"OpenRouter request timed out after {TIMEOUT_SECONDS}s. "
            "The LLM may be overloaded -- try again."
        )
    except requests.exceptions.ConnectionError:
        logger.warning("openrouter call connection error model=%s", model)
        raise RuntimeError(
            "Could not connect to OpenRouter. Check your internet connection."
        )
    except requests.exceptions.RequestException as e:
        logger.warning("openrouter call network error model=%s error=%s", model, e)
        raise RuntimeError(f"Network error calling OpenRouter: {e}")

    duration_ms = (time.monotonic() - start) * 1000
    logger.info(
        "openrouter call model=%s status=%d duration_ms=%.1f",
        model, response.status_code, duration_ms,
    )

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
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("content is not a string")
    except (ValueError, KeyError, IndexError):
        raise RuntimeError(
            f"Unexpected response format from OpenRouter: {response.text[:300]}"
        )

    cost = data.get("usage", {}).get("cost")
    return OpenRouterResult(content=content, cost=cost)


def list_models() -> list[dict]:
    now = time.time()
    if (
        _models_cache["models"] is not None
        and (now - _models_cache["fetched_at"]) < MODELS_CACHE_TTL_SECONDS
    ):
        return _models_cache["models"]

    try:
        response = requests.get(OPENROUTER_MODELS_URL, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        raw_models = response.json()["data"]
    except (requests.exceptions.RequestException, ValueError, KeyError):
        if _models_cache["models"] is not None:
            return _models_cache["models"]
        raise RuntimeError("Could not fetch model list from OpenRouter.")

    models = [
        {
            "id": m["id"],
            "name": m.get("name", m["id"]),
            "supports_images": "image" in m.get("architecture", {}).get("input_modalities", []),
        }
        for m in raw_models
    ]
    _models_cache["models"] = models
    _models_cache["fetched_at"] = now
    return models
