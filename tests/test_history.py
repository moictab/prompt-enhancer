from app import history


def test_list_entries_empty_when_file_missing(tmp_path):
    path = str(tmp_path / "history.jsonl")

    assert history.list_entries(path=path) == []


def test_append_entry_persists_all_fields(tmp_path):
    path = str(tmp_path / "history.jsonl")

    entry = history.append_entry(
        mode="generate",
        family_id="fam-1",
        family_name="SDXL",
        llm_model="anthropic/claude-sonnet-4",
        vision_model=None,
        temperature=0.7,
        user_input="a cyberpunk samurai",
        example_prompts="",
        previous_prompt=None,
        positive_prompt="a cyberpunk samurai in neon rain",
        negative_prompt="blurry, deformed hands",
        path=path,
    )

    assert entry["mode"] == "generate"
    assert entry["family_name"] == "SDXL"
    assert entry["positive_prompt"] == "a cyberpunk samurai in neon rain"
    assert "id" in entry
    assert "timestamp" in entry


def test_list_entries_returns_newest_first(tmp_path):
    path = str(tmp_path / "history.jsonl")
    first = history.append_entry(
        mode="generate", family_id="f", family_name="SDXL", llm_model="m",
        vision_model=None, temperature=0.7, user_input="first", example_prompts="",
        previous_prompt=None, positive_prompt="first result", negative_prompt="",
        path=path,
    )
    second = history.append_entry(
        mode="generate", family_id="f", family_name="SDXL", llm_model="m",
        vision_model=None, temperature=0.7, user_input="second", example_prompts="",
        previous_prompt=None, positive_prompt="second result", negative_prompt="",
        path=path,
    )

    entries = history.list_entries(path=path)

    assert [e["id"] for e in entries] == [second["id"], first["id"]]
