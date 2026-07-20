import json

from app import storage


def test_read_json_returns_default_when_file_missing(tmp_path):
    path = str(tmp_path / "missing.json")

    result = storage.read_json(path, default=[])

    assert result == []


def test_write_json_then_read_json_roundtrips(tmp_path):
    path = str(tmp_path / "data.json")

    storage.write_json(path, {"a": 1})

    assert storage.read_json(path, default=None) == {"a": 1}


def test_write_json_creates_parent_directories(tmp_path):
    path = str(tmp_path / "nested" / "dir" / "data.json")

    storage.write_json(path, [1, 2, 3])

    assert json.loads((tmp_path / "nested" / "dir" / "data.json").read_text()) == [1, 2, 3]


def test_write_json_does_not_leave_temp_files(tmp_path):
    path = str(tmp_path / "data.json")

    storage.write_json(path, {"a": 1})

    leftover = [p for p in tmp_path.iterdir() if p.name != "data.json"]
    assert leftover == []


def test_append_jsonl_then_read_jsonl_roundtrips(tmp_path):
    path = str(tmp_path / "history.jsonl")

    storage.append_jsonl(path, {"id": 1})
    storage.append_jsonl(path, {"id": 2})

    assert storage.read_jsonl(path) == [{"id": 1}, {"id": 2}]


def test_read_jsonl_returns_empty_list_when_file_missing(tmp_path):
    path = str(tmp_path / "missing.jsonl")

    assert storage.read_jsonl(path) == []
