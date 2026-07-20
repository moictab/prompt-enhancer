from app import system_prompt


def test_read_system_prompt_returns_empty_string_when_missing(tmp_path):
    path = str(tmp_path / "system_prompt.txt")

    assert system_prompt.read_system_prompt(path=path) == ""


def test_write_then_read_roundtrips(tmp_path):
    path = str(tmp_path / "system_prompt.txt")

    system_prompt.write_system_prompt("You are an expert prompt engineer.", path=path)

    assert system_prompt.read_system_prompt(path=path) == "You are an expert prompt engineer."


def test_write_creates_parent_directories(tmp_path):
    path = str(tmp_path / "nested" / "system_prompt.txt")

    system_prompt.write_system_prompt("hello", path=path)

    assert (tmp_path / "nested" / "system_prompt.txt").read_text(encoding="utf-8") == "hello"
