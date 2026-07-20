import uuid

from . import storage
from .config import get_settings


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
