import threading
import uuid
from datetime import datetime, timezone

from . import storage
from .config import get_settings

# Appends and the read-modify-write of favorite/delete run on FastAPI's threadpool,
# so they must not interleave or a rewrite could drop a concurrently appended entry.
_lock = threading.Lock()


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
    prompts: list[dict],
    cost: float | None = None,
    params: dict | None = None,
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
        "prompts": prompts,
        "cost": cost,
        "params": params or {},
        "favorite": False,
    }
    with _lock:
        storage.append_jsonl(path or _default_path(), entry)
    return entry


def _normalize(entry: dict) -> dict:
    # Entries written before the "prompts" list existed kept a single pair at the top
    # level (or a story's pairs under "episodes") and had no cost/params/favorite.
    if "prompts" in entry:
        return entry
    normalized = dict(entry)
    episodes = normalized.pop("episodes", None)
    num_episodes = normalized.pop("num_episodes", None)
    positive = normalized.pop("positive_prompt", "")
    negative = normalized.pop("negative_prompt", "")
    normalized["prompts"] = episodes or [
        {"positive_prompt": positive, "negative_prompt": negative}
    ]
    normalized.setdefault("cost", None)
    normalized.setdefault("params", {"num_episodes": num_episodes} if num_episodes else {})
    normalized.setdefault("favorite", False)
    return normalized


def _read_all(path: str) -> list[dict]:
    return [_normalize(entry) for entry in storage.read_jsonl(path)]


def list_entries(path: str | None = None) -> list[dict]:
    return list(reversed(_read_all(path or _default_path())))


def set_favorite(entry_id: str, favorite: bool, path: str | None = None) -> dict | None:
    resolved_path = path or _default_path()
    with _lock:
        entries = _read_all(resolved_path)
        for entry in entries:
            if entry["id"] == entry_id:
                entry["favorite"] = favorite
                storage.write_jsonl(resolved_path, entries)
                return entry
    return None


def delete_entry(entry_id: str, path: str | None = None) -> bool:
    resolved_path = path or _default_path()
    with _lock:
        entries = _read_all(resolved_path)
        remaining = [e for e in entries if e["id"] != entry_id]
        if len(remaining) == len(entries):
            return False
        storage.write_jsonl(resolved_path, remaining)
        return True
