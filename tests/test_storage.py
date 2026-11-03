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


def test_write_jsonl_replaces_file_contents(tmp_path):
    path = str(tmp_path / "history.jsonl")
    storage.append_jsonl(path, {"id": 1})
    storage.append_jsonl(path, {"id": 2})

    storage.write_jsonl(path, [{"id": 2, "favorite": True}])

    assert storage.read_jsonl(path) == [{"id": 2, "favorite": True}]
    assert [p.name for p in tmp_path.iterdir()] == ["history.jsonl"]


def test_write_bytes_creates_parent_directories_and_leaves_no_temp_files(tmp_path):
    path = tmp_path / "images" / "a.png"

    storage.write_bytes(str(path), b"\x89PNG")

    assert path.read_bytes() == b"\x89PNG"
    assert [p.name for p in path.parent.iterdir()] == ["a.png"]


def test_remove_deletes_files_and_directories_and_ignores_missing_paths(tmp_path):
    (tmp_path / "dir" / "sub").mkdir(parents=True)
    (tmp_path / "dir" / "sub" / "f.txt").write_text("x")
    (tmp_path / "file.json").write_text("{}")

    storage.remove(str(tmp_path / "dir"))
    storage.remove(str(tmp_path / "file.json"))
    storage.remove(str(tmp_path / "missing"))

    assert list(tmp_path.iterdir()) == []
