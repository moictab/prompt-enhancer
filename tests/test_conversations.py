from app import conversations


def _create(root, **overrides):
    fields = dict(family_id="f", llm_model="m", temperature=0.7, root=root)
    fields.update(overrides)
    return conversations.create(**fields)


def test_create_then_get_roundtrips(tmp_path):
    root = str(tmp_path)

    created = _create(root)

    assert conversations.get(created["id"], root=root) == created
    assert created["messages"] == []
    assert created["title"] == ""
    assert created["cost"] is None


def test_get_rejects_ids_that_could_escape_the_folder(tmp_path):
    assert conversations.get("../secrets", root=str(tmp_path)) is None
    assert conversations.get("nope", root=str(tmp_path)) is None


def test_append_message_stamps_it_and_adds_its_cost(tmp_path):
    root = str(tmp_path)
    conversation = _create(root)

    conversations.append_message(conversation["id"], {"role": "user", "content": "hi"}, root=root)
    conversations.append_message(conversation["id"], {"role": "assistant", "content": "a", "cost": 0.002}, root=root)
    updated = conversations.append_message(
        conversation["id"], {"role": "assistant", "content": "b", "cost": 0.001}, root=root
    )

    assert [m["content"] for m in updated["messages"]] == ["hi", "a", "b"]
    assert all("id" in m and "timestamp" in m for m in updated["messages"])
    assert updated["cost"] == 0.003
    assert updated["updated_at"] >= conversation["updated_at"]


def test_append_message_sets_the_title_only_once(tmp_path):
    root = str(tmp_path)
    conversation = _create(root)

    conversations.append_message(conversation["id"], {"role": "assistant", "content": "a"}, title="First", root=root)
    updated = conversations.append_message(
        conversation["id"], {"role": "assistant", "content": "b"}, title="Second", root=root
    )

    assert updated["title"] == "First"


def test_append_message_to_unknown_conversation_returns_none(tmp_path):
    assert conversations.append_message(
        "00000000-0000-0000-0000-000000000000", {"role": "user", "content": "hi"}, root=str(tmp_path)
    ) is None


def test_summary_falls_back_to_the_first_message_until_there_is_a_title(tmp_path):
    root = str(tmp_path)
    conversation = _create(root)
    long_text = "Quiero una foto de Lena en la playa al atardecer con una chaqueta vaquera y el pelo suelto"
    updated = conversations.append_message(conversation["id"], {"role": "user", "content": long_text}, root=root)

    title = conversations.summary(updated)["title"]

    assert title.startswith("Quiero una foto de Lena")
    assert title.endswith("…")
    assert len(title) <= conversations.FALLBACK_TITLE_LENGTH + 1


def test_list_summaries_is_most_recently_updated_first(tmp_path):
    root = str(tmp_path)
    older = _create(root)
    newer = _create(root)
    conversations.append_message(older["id"], {"role": "user", "content": "bump"}, root=root)

    summaries = conversations.list_summaries(root=root)

    assert [s["id"] for s in summaries] == [older["id"], newer["id"]]
    assert summaries[0]["message_count"] == 1
    assert "messages" not in summaries[0]


def test_list_summaries_is_empty_without_a_folder(tmp_path):
    assert conversations.list_summaries(root=str(tmp_path / "missing")) == []


def test_update_settings_changes_only_settings(tmp_path):
    root = str(tmp_path)
    conversation = _create(root)

    updated = conversations.update_settings(
        conversation["id"], {"llm_model": "other", "temperature": 0.2, "messages": ["nope"]}, root=root
    )

    assert updated["llm_model"] == "other"
    assert updated["temperature"] == 0.2
    assert updated["messages"] == []


def test_save_image_then_find_it(tmp_path):
    root = str(tmp_path)
    conversation = _create(root)

    image = conversations.save_image(conversation["id"], b"\x89PNG data", "image/png", root=root)
    path, content_type = conversations.image_file(conversation["id"], image["id"], root=root)

    assert image["content_type"] == "image/png"
    assert content_type == "image/png"
    assert open(path, "rb").read() == b"\x89PNG data"


def test_image_file_rejects_bad_ids(tmp_path):
    root = str(tmp_path)
    conversation = _create(root)

    assert conversations.image_file(conversation["id"], "../x", root=root) is None
    assert conversations.image_file(conversation["id"], "0" * 32, root=root) is None


def test_delete_removes_the_conversation_and_its_images(tmp_path):
    root = str(tmp_path)
    conversation = _create(root)
    conversations.save_image(conversation["id"], b"data", "image/webp", root=root)

    assert conversations.delete(conversation["id"], root=root) is True

    assert conversations.get(conversation["id"], root=root) is None
    assert list(tmp_path.iterdir()) == []
    assert conversations.delete(conversation["id"], root=root) is False
