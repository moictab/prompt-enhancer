from app import characters


def test_list_characters_empty_when_file_missing(tmp_path):
    path = str(tmp_path / "characters.json")

    assert characters.list_characters(path=path) == []


def test_create_character_persists_and_is_listed(tmp_path):
    path = str(tmp_path / "characters.json")

    created = characters.create_character("Warrior", "a fierce warrior", path=path)

    assert created["name"] == "Warrior"
    assert created["text"] == "a fierce warrior"
    assert "id" in created
    assert characters.list_characters(path=path) == [created]


def test_update_character_changes_fields(tmp_path):
    path = str(tmp_path / "characters.json")
    created = characters.create_character("Warrior", "a fierce warrior", path=path)

    updated = characters.update_character(
        created["id"], "Warrior v2", "an even fiercer warrior", path=path
    )

    assert updated["name"] == "Warrior v2"
    assert updated["text"] == "an even fiercer warrior"
    assert updated["id"] == created["id"]


def test_update_character_returns_none_when_not_found(tmp_path):
    path = str(tmp_path / "characters.json")

    assert characters.update_character("nonexistent-id", "x", "y", path=path) is None


def test_delete_character_removes_it(tmp_path):
    path = str(tmp_path / "characters.json")
    created = characters.create_character("Warrior", "a fierce warrior", path=path)

    result = characters.delete_character(created["id"], path=path)

    assert result is True
    assert characters.list_characters(path=path) == []


def test_delete_character_returns_false_when_not_found(tmp_path):
    path = str(tmp_path / "characters.json")

    assert characters.delete_character("nonexistent-id", path=path) is False
