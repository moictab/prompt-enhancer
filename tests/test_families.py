from app import families


def test_list_families_empty_when_file_missing(tmp_path):
    path = str(tmp_path / "families.json")

    assert families.list_families(path=path) == []


def test_create_family_persists_and_is_listed(tmp_path):
    path = str(tmp_path / "families.json")

    created = families.create_family("SDXL", "some rules", True, path=path)

    assert created["name"] == "SDXL"
    assert created["instructions"] == "some rules"
    assert created["has_negative_prompt"] is True
    assert "id" in created
    assert families.list_families(path=path) == [created]


def test_get_family_returns_none_when_not_found(tmp_path):
    path = str(tmp_path / "families.json")
    families.create_family("SDXL", "rules", True, path=path)

    assert families.get_family("nonexistent-id", path=path) is None


def test_get_family_returns_matching_family(tmp_path):
    path = str(tmp_path / "families.json")
    created = families.create_family("SDXL", "rules", True, path=path)

    assert families.get_family(created["id"], path=path) == created


def test_update_family_changes_fields(tmp_path):
    path = str(tmp_path / "families.json")
    created = families.create_family("SDXL", "rules", True, path=path)

    updated = families.update_family(
        created["id"], "SDXL v2", "new rules", False, path=path
    )

    assert updated["name"] == "SDXL v2"
    assert updated["instructions"] == "new rules"
    assert updated["has_negative_prompt"] is False
    assert updated["id"] == created["id"]


def test_update_family_returns_none_when_not_found(tmp_path):
    path = str(tmp_path / "families.json")

    assert families.update_family("nonexistent-id", "x", "y", True, path=path) is None


def test_delete_family_removes_it(tmp_path):
    path = str(tmp_path / "families.json")
    created = families.create_family("SDXL", "rules", True, path=path)

    result = families.delete_family(created["id"], path=path)

    assert result is True
    assert families.list_families(path=path) == []


def test_delete_family_returns_false_when_not_found(tmp_path):
    path = str(tmp_path / "families.json")

    assert families.delete_family("nonexistent-id", path=path) is False
