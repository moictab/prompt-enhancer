import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, TextIO


def read_json(path: str, default: Any) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_atomically(path: str, suffix: str, write: Callable[[TextIO], None]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(p.parent), prefix=".tmp-", suffix=suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            write(f)
        os.replace(tmp_path, p)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def write_json(path: str, data: Any) -> None:
    _write_atomically(path, ".json", lambda f: json.dump(data, f, indent=2, ensure_ascii=False))


def append_jsonl(path: str, entry: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def write_jsonl(path: str, entries: list[dict]) -> None:
    def write(f: TextIO) -> None:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    _write_atomically(path, ".jsonl", write)


def read_jsonl(path: str) -> list:
    p = Path(path)
    if not p.exists():
        return []
    entries = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
