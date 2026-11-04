import json

from app.seed import ensure_seed_data


def test_ensure_seed_data_creates_families_with_sdxl_and_zit(tmp_path):
    ensure_seed_data(str(tmp_path))

    families = json.loads((tmp_path / "families.json").read_text())
    names = {f["name"] for f in families}
    assert names == {"SDXL", "Z-Image-Turbo"}

    sdxl = next(f for f in families if f["name"] == "SDXL")
    zit = next(f for f in families if f["name"] == "Z-Image-Turbo")
    assert sdxl["has_negative_prompt"] is True
    assert zit["has_negative_prompt"] is False


def test_ensure_seed_data_creates_the_bots_personality(tmp_path):
    ensure_seed_data(str(tmp_path))

    data = json.loads((tmp_path / "system_prompts.json").read_text())
    assert set(data.keys()) == {"conversation"}
    assert "partner in crime" in data["conversation"]
    # The personality alone: the prompt format is appended by app.chat, not editable.
    assert "<prompt>" not in data["conversation"]


def test_ensure_seed_data_is_idempotent(tmp_path):
    ensure_seed_data(str(tmp_path))
    ensure_seed_data(str(tmp_path))

    families = json.loads((tmp_path / "families.json").read_text())
    assert len(families) == 2


def test_ensure_seed_data_does_not_overwrite_existing_system_prompts(tmp_path):
    (tmp_path / "system_prompts.json").write_text('{"generate": "custom"}', encoding="utf-8")

    ensure_seed_data(str(tmp_path))

    data = json.loads((tmp_path / "system_prompts.json").read_text())
    assert data["generate"] == "custom"


def test_ensure_seed_data_fills_in_modes_missing_from_an_older_data_dir(tmp_path):
    (tmp_path / "system_prompts.json").write_text(
        '{"generate": "custom", "story": ""}', encoding="utf-8"
    )

    ensure_seed_data(str(tmp_path))

    data = json.loads((tmp_path / "system_prompts.json").read_text())
    # The old one-shot modes' texts stay as they were, unused.
    assert data["generate"] == "custom"
    assert data["story"] == ""
    assert "partner in crime" in data["conversation"]
