from app import system_prompt


def test_read_system_prompt_returns_empty_string_when_missing(tmp_path):
    path = str(tmp_path / "system_prompts.json")

    assert system_prompt.read_system_prompt("generate", path=path) == ""


def test_write_then_read_roundtrips(tmp_path):
    path = str(tmp_path / "system_prompts.json")

    system_prompt.write_system_prompt("generate", "You are an expert prompt engineer.", path=path)

    assert system_prompt.read_system_prompt("generate", path=path) == "You are an expert prompt engineer."


def test_write_creates_parent_directories(tmp_path):
    path = str(tmp_path / "nested" / "system_prompts.json")

    system_prompt.write_system_prompt("generate", "hello", path=path)

    assert (tmp_path / "nested" / "system_prompts.json").exists()


def test_modes_are_stored_independently(tmp_path):
    path = str(tmp_path / "system_prompts.json")

    system_prompt.write_system_prompt("generate", "generate text", path=path)
    system_prompt.write_system_prompt("iterate", "iterate text", path=path)
    system_prompt.write_system_prompt("image", "image text", path=path)

    assert system_prompt.read_system_prompt("generate", path=path) == "generate text"
    assert system_prompt.read_system_prompt("iterate", path=path) == "iterate text"
    assert system_prompt.read_system_prompt("image", path=path) == "image text"


def test_writing_one_mode_does_not_clobber_another(tmp_path):
    path = str(tmp_path / "system_prompts.json")
    system_prompt.write_system_prompt("generate", "generate text", path=path)

    system_prompt.write_system_prompt("iterate", "iterate text", path=path)

    assert system_prompt.read_system_prompt("generate", path=path) == "generate text"
