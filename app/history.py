import uuid
from datetime import datetime, timezone

from . import storage
from .config import get_settings


def _default_path() -> str:
    return f"{get_settings().data_dir}/history.jsonl"


def append_entry(
    mode: str,
    family_id: str,
    family_name: str,
    llm_model: str | None,
    vision_model: str | None,
    temperature: float,
    user_input: str,
    example_prompts: str,
    previous_prompt: str | None,
    positive_prompt: str,
    negative_prompt: str,
    path: str | None = None,
) -> dict:
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "family_id": family_id,
        "family_name": family_name,
        "llm_model": llm_model,
        "vision_model": vision_model,
        "temperature": temperature,
        "user_input": user_input,
        "example_prompts": example_prompts,
        "previous_prompt": previous_prompt,
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
    }
    storage.append_jsonl(path or _default_path(), entry)
    return entry


def list_entries(path: str | None = None) -> list[dict]:
    entries = storage.read_jsonl(path or _default_path())
    return list(reversed(entries))
