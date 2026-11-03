"""The Conversar bot's notes about the user, carried into every conversation."""
import uuid
from datetime import datetime, timezone

from . import storage
from .config import get_settings


def _default_path() -> str:
    return f"{get_settings().data_dir}/memory.json"


def list_notes(path: str | None = None) -> list[dict]:
    return storage.read_json(path or _default_path(), default=[])


def create_note(text: str, conversation_id: str | None = None, path: str | None = None) -> dict:
    resolved_path = path or _default_path()
    notes = list_notes(resolved_path)
    note = {
        "id": str(uuid.uuid4()),
        "text": text,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "conversation_id": conversation_id,
    }
    notes.append(note)
    storage.write_json(resolved_path, notes)
    return note


def update_note(note_id: str, text: str, path: str | None = None) -> dict | None:
    resolved_path = path or _default_path()
    notes = list_notes(resolved_path)
    for note in notes:
        if note["id"] == note_id:
            note["text"] = text
            storage.write_json(resolved_path, notes)
            return note
    return None


def delete_note(note_id: str, path: str | None = None) -> bool:
    resolved_path = path or _default_path()
    notes = list_notes(resolved_path)
    remaining = [n for n in notes if n["id"] != note_id]
    if len(remaining) == len(notes):
        return False
    storage.write_json(resolved_path, remaining)
    return True
