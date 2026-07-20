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


def test_ensure_seed_data_creates_empty_characters_file(tmp_path):
    ensure_seed_data(str(tmp_path))

    characters = json.loads((tmp_path / "characters.json").read_text())
    assert characters == []


def test_ensure_seed_data_creates_system_prompt_file(tmp_path):
    ensure_seed_data(str(tmp_path))

    text = (tmp_path / "system_prompt.txt").read_text(encoding="utf-8")
    assert "POSITIVE:" in text
    assert "NEGATIVE:" in text


def test_ensure_seed_data_is_idempotent(tmp_path):
    ensure_seed_data(str(tmp_path))
    ensure_seed_data(str(tmp_path))

    families = json.loads((tmp_path / "families.json").read_text())
    assert len(families) == 2


def test_ensure_seed_data_does_not_overwrite_existing_system_prompt(tmp_path):
    (tmp_path / "system_prompt.txt").write_text("custom prompt", encoding="utf-8")

    ensure_seed_data(str(tmp_path))

    assert (tmp_path / "system_prompt.txt").read_text(encoding="utf-8") == "custom prompt"
