from . import storage
from .config import get_settings


def _default_path() -> str:
    return f"{get_settings().data_dir}/system_prompts.json"


def read_system_prompt(mode: str, path: str | None = None) -> str:
    data = storage.read_json(path or _default_path(), default={})
    return data.get(mode, "")


def write_system_prompt(mode: str, text: str, path: str | None = None) -> None:
    resolved_path = path or _default_path()
    data = storage.read_json(resolved_path, default={})
    data[mode] = text
    storage.write_json(resolved_path, data)
