import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, TextIO

# On Windows a file can't be replaced while anyone has it open -- a reader in another thread
# (the chat polls a conversation while a run's results are stored), an antivirus -- nor opened
# at the instant it is replaced: "access denied" (PermissionError) for a moment. So both wait
# and try again, for up to ~10 s in all.
ATTEMPTS = 20
WAIT_SECONDS = 0.05


def _retrying(action: Callable[[], Any]) -> Any:
    for attempt in range(ATTEMPTS):
        try:
            return action()
        except PermissionError:
            if attempt == ATTEMPTS - 1:
                raise
            time.sleep(WAIT_SECONDS * (attempt + 1))


def read_json(path: str, default: Any) -> Any:
    p = Path(path)
    if not p.exists():
        return default

    def read() -> Any:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    return _retrying(read)


def _write_atomically(path: str, suffix: str, write: Callable[[TextIO], None]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(p.parent), prefix=".tmp-", suffix=suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            write(f)
        _retrying(lambda: os.replace(tmp_path, p))
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def write_json(path: str, data: Any) -> None:
    _write_atomically(path, ".json", lambda f: json.dump(data, f, indent=2, ensure_ascii=False))


def write_bytes(path: str, data: bytes) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(p.parent), prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        _retrying(lambda: os.replace(tmp_path, p))
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def remove(path: str) -> None:
    """Delete a file or a whole directory; a missing path is not an error."""
    p = Path(path)
    if p.is_dir():
        shutil.rmtree(p)
    elif p.exists():
        p.unlink()


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
