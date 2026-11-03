import json
import logging
import time
from dataclasses import dataclass, field
from typing import Iterator

import requests

from .request_log import RequestRecord

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
# A 12-prompt story or variation set can take a couple of minutes to generate,
# and a non-streamed completion only arrives once it is complete.
TIMEOUT_SECONDS = 180
MODELS_TIMEOUT_SECONDS = 30
MODELS_CACHE_TTL_SECONDS = 3600
MAX_TOKENS = 8192
# Every request asks OpenRouter to route only to providers that neither store the data
# nor train on it: the prompts and images sent here are personal.
PROVIDER_PREFERENCES = {"data_collection": "deny"}

logger = logging.getLogger("app.openrouter")

_models_cache: dict = {"models": None, "fetched_at": 0.0}


@dataclass
class OpenRouterResult:
    content: str
    cost: float | None
    truncated: bool = False
    request_id: str | None = None  # its entry in request_log


@dataclass
class StreamEnd:
    """The last item stream_chat yields, once the reply is complete."""
    cost: float | None
    truncated: bool = False
    request_id: str | None = None
    # [{"id", "name", "arguments" (the JSON text as sent)}] when the model called tools
    tool_calls: list[dict] = field(default_factory=list)


class OpenRouterError(RuntimeError):
    """A failed call, with the id of its entry in request_log."""

    def __init__(self, message: str, request_id: str | None = None):
        super().__init__(message)
        self.request_id = request_id


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
    if response.status_code == 404 and "data policy" in response.text.lower():
        raise RuntimeError(
            "No provider of this model meets the privacy requirement (no storing or training on "
            "the data). Pick another model."
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
    context: dict | None = None,
) -> OpenRouterResult:
    """context ({"mode": ..., plus anything worth linking}) goes to the request log."""
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
        "provider": PROVIDER_PREFERENCES,
    }

    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        record = RequestRecord(payload, context=context, attempt=attempt)
        start = time.monotonic()
        try:
            response = _post(api_key, model, payload)
        except RuntimeError as e:
            record.finish(error=str(e))
            raise OpenRouterError(str(e), record.id) from e
        duration_ms = (time.monotonic() - start) * 1000
        logger.info(
            "openrouter call model=%s status=%d duration_ms=%.1f attempt=%d request_id=%s",
            model, response.status_code, duration_ms, attempt, record.id,
        )
        try:
            _raise_for_status(response)
        except RuntimeError as e:
            record.finish(status=response.status_code, error=str(e), response_body=response.text)
            raise OpenRouterError(str(e), record.id) from e

        try:
            data = response.json()
            choice = data["choices"][0]
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("content is not a string")
        # TypeError/AttributeError: a null "choices"/"message" or a non-object body
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            error = f"Unexpected response format from OpenRouter: {response.text[:300]}"
            record.finish(status=response.status_code, error=error, response_body=response.text)
            if attempt < max_attempts:
                logger.warning(
                    "openrouter call returned malformed content model=%s attempt=%d, retrying",
                    model, attempt,
                )
                continue
            raise OpenRouterError(error, record.id)

        record.finish(status=response.status_code, response=data)
        cost = (data.get("usage") or {}).get("cost")
        truncated = choice.get("finish_reason") == "length"
        if truncated:
            logger.warning(
                "openrouter call truncated by max_tokens model=%s content_len=%d",
                model, len(content),
            )
        return OpenRouterResult(content=content, cost=cost, truncated=truncated, request_id=record.id)


def stream_chat(
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    context: dict | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | None = None,
) -> Iterator[str | StreamEnd]:
    """Stream a chat completion: yields each text delta as it arrives, then one StreamEnd.

    Raises OpenRouterError (with a user-facing message) for HTTP errors, an error
    reported mid-stream, or a dropped connection. Unlike call_openrouter there is
    no retry: part of the reply may already have been shown. However it ends --
    finished, failed, or abandoned by the caller -- the request and whatever text
    arrived go to the request log, with context ({"mode": ...}) attached.

    With tools (OpenAI-style function definitions), the model may call them instead
    of, or after, writing text: the calls arrive in pieces and come back assembled
    in StreamEnd.tool_calls, for the caller to run and answer in a new request.
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
        "usage": {"include": True},
        "provider": PROVIDER_PREFERENCES,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
    record = RequestRecord(payload, context=context)
    start = time.monotonic()
    chunks: list[str] = []
    calls: list[dict] = []
    reply: dict = {"finish_reason": None, "usage": None}
    status = None
    body = None
    error = None
    try:
        response = _post(api_key, model, payload, stream=True)
        status = response.status_code
        with response:
            logger.info("openrouter stream model=%s status=%d request_id=%s", model, status, record.id)
            if status != 200:
                body = response.text
            _raise_for_status(response)
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
                        problem = chunk["error"]
                        message = problem.get("message") if isinstance(problem, dict) else str(problem)
                        raise RuntimeError(f"OpenRouter reported an error mid-reply: {message}")
                    for key in ("id", "model", "provider"):
                        if chunk.get(key) and key not in reply:
                            reply[key] = chunk[key]
                    for choice in chunk.get("choices") or []:
                        delta = (choice.get("delta") or {}).get("content")
                        if isinstance(delta, str) and delta:
                            chunks.append(delta)
                            yield delta
                        _add_tool_call_pieces(calls, (choice.get("delta") or {}).get("tool_calls"))
                        if choice.get("finish_reason"):
                            reply["finish_reason"] = choice["finish_reason"]
                    if isinstance(chunk.get("usage"), dict):
                        reply["usage"] = chunk["usage"]
            except requests.exceptions.RequestException as e:
                logger.warning("openrouter stream dropped model=%s error=%s", model, e)
                raise RuntimeError("The connection to OpenRouter dropped mid-reply. Try again.")
    except GeneratorExit:
        error = "Abandoned before the reply finished: whoever asked for it went away."
        raise
    except RuntimeError as e:
        error = str(e)
        raise OpenRouterError(error, record.id) from e
    except Exception as e:
        error = f"Unexpected error while streaming: {e!r}"
        raise
    finally:
        response_log = {**reply, "content": "".join(chunks)}
        if calls:
            response_log["tool_calls"] = calls
        record.finish(status=status, response=response_log, error=error, response_body=body)

    usage = reply["usage"] or {}
    cost = usage.get("cost")
    truncated = reply["finish_reason"] == "length"
    logger.info(
        "openrouter stream finished model=%s duration_ms=%.1f chars=%d tool_calls=%d cost=%s truncated=%s "
        "request_id=%s",
        model, (time.monotonic() - start) * 1000, sum(map(len, chunks)), len(calls), cost, truncated, record.id,
    )
    yield StreamEnd(
        cost=cost, truncated=truncated, request_id=record.id,
        tool_calls=[c for c in calls if c["name"]],
    )


def _add_tool_call_pieces(calls: list[dict], pieces) -> None:
    """Merge one chunk's tool-call deltas: the first piece of a call brings its id and
    name, the following ones (same index) more of its arguments' JSON text."""
    if not isinstance(pieces, list):
        return
    for piece in pieces:
        if not isinstance(piece, dict):
            continue
        index = piece.get("index")
        if not isinstance(index, int):
            index = len(calls) if piece.get("id") or not calls else len(calls) - 1
        while len(calls) <= index:
            calls.append({"id": "", "name": "", "arguments": ""})
        call = calls[index]
        function = piece.get("function") or {}
        if piece.get("id"):
            call["id"] = piece["id"]
        if function.get("name"):
            call["name"] += function["name"] if call["name"] != function["name"] else ""
        if isinstance(function.get("arguments"), str):
            call["arguments"] += function["arguments"]


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
            "supports_tools": "tools" in (m.get("supported_parameters") or []),
        }
        for m in raw_models
    ]
    _models_cache["models"] = models
    _models_cache["fetched_at"] = now
    return models
