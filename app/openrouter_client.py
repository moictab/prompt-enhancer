import json
import logging
import time
from dataclasses import dataclass
from typing import Iterator

import requests

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
# A 12-prompt story or variation set can take a couple of minutes to generate,
# and a non-streamed completion only arrives once it is complete.
TIMEOUT_SECONDS = 180
MODELS_TIMEOUT_SECONDS = 30
MODELS_CACHE_TTL_SECONDS = 3600
MAX_TOKENS = 8192

logger = logging.getLogger("app.openrouter")

_models_cache: dict = {"models": None, "fetched_at": 0.0}


@dataclass
class OpenRouterResult:
    content: str
    cost: float | None
    truncated: bool = False


@dataclass
class StreamEnd:
    """The last item stream_chat yields, once the reply is complete."""
    cost: float | None
    truncated: bool = False


def _headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _post(api_key: str, model: str, payload: dict, stream: bool = False) -> requests.Response:
    try:
        return requests.post(
            OPENROUTER_API_URL, headers=_headers(api_key), json=payload,
            timeout=TIMEOUT_SECONDS, stream=stream,
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


def _raise_for_status(response: requests.Response) -> None:
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


def call_openrouter(
    api_key: str,
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float = 0.7,
    image_data_uri: str | None = None,
) -> OpenRouterResult:
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
        "max_tokens": MAX_TOKENS,
        "usage": {"include": True},
    }

    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        start = time.monotonic()
        response = _post(api_key, model, payload)
        duration_ms = (time.monotonic() - start) * 1000
        logger.info(
            "openrouter call model=%s status=%d duration_ms=%.1f attempt=%d",
            model, response.status_code, duration_ms, attempt,
        )
        _raise_for_status(response)

        try:
            data = response.json()
            choice = data["choices"][0]
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("content is not a string")
        # TypeError/AttributeError: a null "choices"/"message" or a non-object body
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            if attempt < max_attempts:
                logger.warning(
                    "openrouter call returned malformed content model=%s attempt=%d, retrying",
                    model, attempt,
                )
                continue
            raise RuntimeError(
                f"Unexpected response format from OpenRouter: {response.text[:300]}"
            )

        cost = (data.get("usage") or {}).get("cost")
        truncated = choice.get("finish_reason") == "length"
        if truncated:
            logger.warning(
                "openrouter call truncated by max_tokens model=%s content_len=%d",
                model, len(content),
            )
        return OpenRouterResult(content=content, cost=cost, truncated=truncated)


def stream_chat(
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
) -> Iterator[str | StreamEnd]:
    """Stream a chat completion: yields each text delta as it arrives, then one StreamEnd.

    Raises RuntimeError (with a user-facing message) for HTTP errors, an error
    reported mid-stream, or a dropped connection. Unlike call_openrouter there is
    no retry: part of the reply may already have been shown.
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
        "usage": {"include": True},
        "stream": True,
    }
    start = time.monotonic()
    response = _post(api_key, model, payload, stream=True)
    with response:
        logger.info("openrouter stream model=%s status=%d", model, response.status_code)
        _raise_for_status(response)

        cost = None
        truncated = False
        received = 0
        try:
            # Server-sent events: "data: {json}" lines, ": comment" keep-alives, "data: [DONE]".
            for raw_line in response.iter_lines():
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("error"):
                    error = chunk["error"]
                    message = error.get("message") if isinstance(error, dict) else str(error)
                    raise RuntimeError(f"OpenRouter reported an error mid-reply: {message}")
                for choice in chunk.get("choices") or []:
                    delta = (choice.get("delta") or {}).get("content")
                    if isinstance(delta, str) and delta:
                        received += len(delta)
                        yield delta
                    if choice.get("finish_reason") == "length":
                        truncated = True
                usage = chunk.get("usage")
                if isinstance(usage, dict) and usage.get("cost") is not None:
                    cost = usage["cost"]
        except requests.exceptions.RequestException as e:
            logger.warning("openrouter stream dropped model=%s error=%s", model, e)
            raise RuntimeError("The connection to OpenRouter dropped mid-reply. Try again.")

    logger.info(
        "openrouter stream finished model=%s duration_ms=%.1f chars=%d cost=%s truncated=%s",
        model, (time.monotonic() - start) * 1000, received, cost, truncated,
    )
    yield StreamEnd(cost=cost, truncated=truncated)


def list_models() -> list[dict]:
    now = time.time()
    if (
        _models_cache["models"] is not None
        and (now - _models_cache["fetched_at"]) < MODELS_CACHE_TTL_SECONDS
    ):
        return _models_cache["models"]

    try:
        response = requests.get(OPENROUTER_MODELS_URL, timeout=MODELS_TIMEOUT_SECONDS)
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
