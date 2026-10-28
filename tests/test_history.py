import json

from app import history


def _append(path, user_input="a cat", prompts=None, **overrides):
    fields = dict(
        mode="generate", family_id="f", family_name="SDXL", llm_model="m",
        vision_model=None, temperature=0.7, user_input=user_input, example_prompts="",
        previous_prompt=None,
        prompts=prompts or [{"positive_prompt": f"{user_input} result", "negative_prompt": ""}],
        path=path,
    )
    fields.update(overrides)
    return history.append_entry(**fields)


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
        prompts=[{"positive_prompt": "a cyberpunk samurai in neon rain", "negative_prompt": "blurry"}],
        cost=0.0021,
        params={"variants": 1},
        path=path,
    )

    assert entry["mode"] == "generate"
    assert entry["family_name"] == "SDXL"
    assert entry["prompts"] == [
        {"positive_prompt": "a cyberpunk samurai in neon rain", "negative_prompt": "blurry"}
    ]
    assert entry["cost"] == 0.0021
    assert entry["params"] == {"variants": 1}
    assert entry["favorite"] is False
    assert "id" in entry
    assert "timestamp" in entry
    assert history.list_entries(path=path) == [entry]


def test_append_entry_defaults_params_to_empty_dict(tmp_path):
    entry = _append(str(tmp_path / "history.jsonl"))

    assert entry["params"] == {}
    assert entry["cost"] is None


def test_list_entries_returns_newest_first(tmp_path):
    path = str(tmp_path / "history.jsonl")
    first = _append(path, "first")
    second = _append(path, "second")

    entries = history.list_entries(path=path)

    assert [e["id"] for e in entries] == [second["id"], first["id"]]


def test_list_entries_normalizes_legacy_single_prompt_entries(tmp_path):
    path = tmp_path / "history.jsonl"
    legacy = {
        "id": "old-1", "timestamp": "2026-07-20T10:00:00+00:00", "mode": "generate",
        "family_id": "f", "family_name": "SDXL", "llm_model": "m", "vision_model": None,
        "temperature": 0.7, "user_input": "a cat", "example_prompts": "", "previous_prompt": None,
        "positive_prompt": "a cat in a hat", "negative_prompt": "blurry",
        "episodes": None, "num_episodes": None,
    }
    path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    [entry] = history.list_entries(path=str(path))

    assert entry["prompts"] == [{"positive_prompt": "a cat in a hat", "negative_prompt": "blurry"}]
    assert entry["params"] == {}
    assert entry["cost"] is None
    assert entry["favorite"] is False
    assert "positive_prompt" not in entry
    assert "episodes" not in entry


def test_list_entries_normalizes_legacy_story_entries(tmp_path):
    path = tmp_path / "history.jsonl"
    episodes = [
        {"positive_prompt": "a quiet dinner", "negative_prompt": "blurry"},
        {"positive_prompt": "things escalate", "negative_prompt": "blurry"},
    ]
    legacy = {
        "id": "old-2", "timestamp": "2026-07-20T10:00:00+00:00", "mode": "story",
        "family_id": "f", "family_name": "SDXL", "llm_model": "m", "vision_model": None,
        "temperature": 0.7, "user_input": "dinner", "example_prompts": "", "previous_prompt": None,
        "positive_prompt": "", "negative_prompt": "", "episodes": episodes, "num_episodes": 2,
    }
    path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    [entry] = history.list_entries(path=str(path))

    assert entry["prompts"] == episodes
    assert entry["params"] == {"num_episodes": 2}


def test_set_favorite_updates_only_that_entry(tmp_path):
    path = str(tmp_path / "history.jsonl")
    first = _append(path, "first")
    second = _append(path, "second")

    updated = history.set_favorite(first["id"], True, path=path)

    assert updated["favorite"] is True
    by_id = {e["id"]: e for e in history.list_entries(path=path)}
    assert by_id[first["id"]]["favorite"] is True
    assert by_id[second["id"]]["favorite"] is False


def test_set_favorite_returns_none_for_unknown_entry(tmp_path):
    path = str(tmp_path / "history.jsonl")
    _append(path)

    assert history.set_favorite("nope", True, path=path) is None


def test_set_favorite_migrates_legacy_entries_on_rewrite(tmp_path):
    path = tmp_path / "history.jsonl"
    legacy = {
        "id": "old-1", "timestamp": "2026-07-20T10:00:00+00:00", "mode": "generate",
        "family_id": "f", "family_name": "SDXL", "llm_model": "m", "vision_model": None,
        "temperature": 0.7, "user_input": "a cat", "example_prompts": "", "previous_prompt": None,
        "positive_prompt": "a cat in a hat", "negative_prompt": "",
    }
    path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    history.set_favorite("old-1", True, path=str(path))

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["prompts"] == [{"positive_prompt": "a cat in a hat", "negative_prompt": ""}]
    assert stored["favorite"] is True


def test_delete_entry_removes_only_that_entry(tmp_path):
    path = str(tmp_path / "history.jsonl")
    first = _append(path, "first")
    second = _append(path, "second")

    assert history.delete_entry(first["id"], path=path) is True

    assert [e["id"] for e in history.list_entries(path=path)] == [second["id"]]


def test_delete_entry_returns_false_for_unknown_entry(tmp_path):
    path = str(tmp_path / "history.jsonl")
    _append(path)

    assert history.delete_entry("nope", path=path) is False
    assert len(history.list_entries(path=path)) == 1
