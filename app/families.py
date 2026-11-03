import re
import unicodedata
import uuid

from . import storage
from .config import get_settings


def _key(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def family_for_folder(folder: str, path: str | None = None) -> dict | None:
    """The family a model folder stands for: models live in a folder named after their
    family ("krea-2\\x.safetensors" is Krea 2), compared by name ("Krea 2" -> "krea-2")."""
    key = _key(folder)
    return next((f for f in list_families(path) if key and _key(f["name"]) == key), None)


def covers_folder(family: dict, folder: str) -> bool:
    """Whether a family writes prompts for a model folder: its own, or a variant of it
    ("Krea 2 - Ages" covers krea-2)."""
    name, key = _key(family["name"]), _key(folder)
    if not key:
        return False
    if name == key:
        return True
    # "krea-2-ages" is a variant of krea-2; "krea-2" is another version of krea, not a variant.
    suffix = name[len(key) + 1:] if name.startswith(f"{key}-") else ""
    return bool(suffix) and not suffix[0].isdigit()


def _default_path() -> str:
    return f"{get_settings().data_dir}/families.json"


def list_families(path: str | None = None) -> list[dict]:
    return storage.read_json(path or _default_path(), default=[])


def get_family(family_id: str, path: str | None = None) -> dict | None:
    for family in list_families(path):
        if family["id"] == family_id:
            return family
    return None


def create_family(
    name: str, instructions: str, has_negative_prompt: bool, path: str | None = None
) -> dict:
    resolved_path = path or _default_path()
    all_families = list_families(resolved_path)
    family = {
        "id": str(uuid.uuid4()),
        "name": name,
        "instructions": instructions,
        "has_negative_prompt": has_negative_prompt,
    }
    all_families.append(family)
    storage.write_json(resolved_path, all_families)
    return family


def update_family(
    family_id: str,
    name: str,
    instructions: str,
    has_negative_prompt: bool,
    path: str | None = None,
) -> dict | None:
    resolved_path = path or _default_path()
    all_families = list_families(resolved_path)
    for family in all_families:
        if family["id"] == family_id:
            family["name"] = name
            family["instructions"] = instructions
            family["has_negative_prompt"] = has_negative_prompt
            storage.write_json(resolved_path, all_families)
            return family
    return None


def delete_family(family_id: str, path: str | None = None) -> bool:
    resolved_path = path or _default_path()
    all_families = list_families(resolved_path)
    remaining = [f for f in all_families if f["id"] != family_id]
    if len(remaining) == len(all_families):
        return False
    storage.write_json(resolved_path, remaining)
    return True
