import uuid

from . import storage
from .config import get_settings


def _default_path() -> str:
    return f"{get_settings().data_dir}/characters.json"


def list_characters(path: str | None = None) -> list[dict]:
    return storage.read_json(path or _default_path(), default=[])


def create_character(name: str, text: str, path: str | None = None) -> dict:
    resolved_path = path or _default_path()
    all_characters = list_characters(resolved_path)
    character = {"id": str(uuid.uuid4()), "name": name, "text": text}
    all_characters.append(character)
    storage.write_json(resolved_path, all_characters)
    return character


def update_character(
    character_id: str, name: str, text: str, path: str | None = None
) -> dict | None:
    resolved_path = path or _default_path()
    all_characters = list_characters(resolved_path)
    for character in all_characters:
        if character["id"] == character_id:
            character["name"] = name
            character["text"] = text
            storage.write_json(resolved_path, all_characters)
            return character
    return None


def delete_character(character_id: str, path: str | None = None) -> bool:
    resolved_path = path or _default_path()
    all_characters = list_characters(resolved_path)
    remaining = [c for c in all_characters if c["id"] != character_id]
    if len(remaining) == len(all_characters):
        return False
    storage.write_json(resolved_path, remaining)
    return True
