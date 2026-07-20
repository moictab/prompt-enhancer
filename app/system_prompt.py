from pathlib import Path

from .config import get_settings


def _default_path() -> str:
    return f"{get_settings().data_dir}/system_prompt.txt"


def read_system_prompt(path: str | None = None) -> str:
    p = Path(path or _default_path())
    return p.read_text(encoding="utf-8") if p.exists() else ""


def write_system_prompt(text: str, path: str | None = None) -> None:
    p = Path(path or _default_path())
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
