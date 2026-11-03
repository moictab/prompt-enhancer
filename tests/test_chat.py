from unittest.mock import patch

from app import chat

FAMILY = {"id": "f", "name": "SDXL", "instructions": "  family rules  ", "has_negative_prompt": True}


def _conversation(*messages):
    return {"id": "c", "title": "", "messages": list(messages)}


def _user(content, images=None):
    return {"role": "user", "content": content, "images": images or []}


def _assistant(content):
    return {"role": "assistant", "content": content}


@patch("app.chat.system_prompt.read_system_prompt")
def test_system_prompt_puts_personality_first_then_mechanics_and_family(mock_read):
    mock_read.return_value = "You are a grumpy art director."

    result = chat.build_system_prompt(FAMILY, [], [], needs_title=False)

    mock_read.assert_called_once_with("conversation")
    assert result.startswith("You are a grumpy art director.\n\n## How This Conversation Works\n")
    assert "<prompt>\nPOSITIVE:" in result
    assert result.endswith("## Model Family Rules\nfamily rules")
    assert "<title>" not in result


@patch("app.chat.system_prompt.read_system_prompt", return_value="")
def test_system_prompt_keeps_the_mechanics_when_the_personality_is_empty(mock_read):
    result = chat.build_system_prompt(FAMILY, [], [], needs_title=False)

    assert result.startswith("## How This Conversation Works\n")


@patch("app.chat.system_prompt.read_system_prompt", return_value="persona")
def test_system_prompt_asks_for_a_title_only_when_needed(mock_read):
    assert "<title>" in chat.build_system_prompt(FAMILY, [], [], needs_title=True)
    assert "<title>" not in chat.build_system_prompt(FAMILY, [], [], needs_title=False)


@patch("app.chat.system_prompt.read_system_prompt", return_value="persona")
def test_system_prompt_lists_characters_and_notes(mock_read):
    result = chat.build_system_prompt(
        FAMILY,
        [{"name": "Lena", "text": "auburn hair, freckles"}],
        [{"text": "Likes hard light"}, {"text": "Hates lens flare"}],
        needs_title=False,
    )

    assert "## Characters\n" in result
    assert "- Lena: auburn hair, freckles" in result
    assert result.endswith(
        "## Your Notes About The User\n" + chat.NOTES_INTRO + "\n- Likes hard light\n- Hates lens flare"
    )


def test_llm_messages_number_prompt_blocks_across_the_conversation():
    conversation = _conversation(
        _user("a witch"),
        _assistant("Here:\n<prompt>\nPOSITIVE: a witch\nNEGATIVE:\n</prompt>\nOr:\n<prompt>\nPOSITIVE: b\n</prompt>"),
        _user("darker"),
        _assistant('<prompt id="9">\nPOSITIVE: c\n</prompt>'),
    )

    messages = chat.build_llm_messages(conversation, load_image=lambda image: None)

    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert '<prompt id="1">' in messages[1]["content"]
    assert '<prompt id="2">' in messages[1]["content"]
    assert messages[3]["content"].startswith('<prompt id="3">')
    assert messages[0]["content"] == "a witch"


def test_llm_messages_attach_only_the_most_recent_images():
    images = [{"id": f"img{i}", "content_type": "image/png"} for i in range(6)]
    conversation = _conversation(
        _user("first batch", images[:3]),
        _assistant("nice"),
        _user("", images[3:]),
    )

    messages = chat.build_llm_messages(conversation, load_image=lambda image: f"data:{image['id']}")

    # 6 images, only the last 4 travel: img2 from the first message, img3-5 from the last.
    first, last = messages[0]["content"], messages[2]["content"]
    assert first[0] == {
        "type": "text",
        "text": "first batch\n\n[2 images shared here earlier, no longer attached]",
    }
    assert [part["image_url"]["url"] for part in first[1:]] == ["data:img2"]
    assert last[0] == {"type": "text", "text": "(image attached)"}
    assert [part["image_url"]["url"] for part in last[1:]] == ["data:img3", "data:img4", "data:img5"]


def test_llm_messages_mention_an_image_whose_file_is_gone():
    conversation = _conversation(_user("look", [{"id": "gone", "content_type": "image/png"}]))

    messages = chat.build_llm_messages(conversation, load_image=lambda image: None)

    assert messages[0]["content"] == "look\n\n[1 image shared here earlier, no longer attached]"


def test_parse_reply_splits_text_and_prompt_blocks_in_order():
    reply = chat.parse_reply(
        "Two takes.\n\n<prompt>\nPOSITIVE: a witch at dawn\nNEGATIVE: blurry\n</prompt>\n\n"
        "The second is moodier:\n<prompt>\nPOSITIVE: a witch at night\nNEGATIVE: blurry\n</prompt>\n\nWhich one?",
        has_negative_prompt=True,
    )

    assert reply.parts == [
        {"type": "text", "text": "Two takes."},
        {"type": "prompt", "positive_prompt": "a witch at dawn", "negative_prompt": "blurry"},
        {"type": "text", "text": "The second is moodier:"},
        {"type": "prompt", "positive_prompt": "a witch at night", "negative_prompt": "blurry"},
        {"type": "text", "text": "Which one?"},
    ]
    assert reply.prompts == [
        {"positive_prompt": "a witch at dawn", "negative_prompt": "blurry"},
        {"positive_prompt": "a witch at night", "negative_prompt": "blurry"},
    ]


def test_parse_reply_extracts_title_and_notes_and_keeps_them_out_of_the_content():
    reply = chat.parse_reply(
        "<title> Witch at  dawn </title>\nGot it. <remember>Prefers  hard light</remember>\n"
        "<prompt>\nPOSITIVE: a witch\nNEGATIVE:\n</prompt>\n<remember>Uses SDXL mostly</remember>",
        has_negative_prompt=True,
    )

    assert reply.title == "Witch at dawn"
    assert reply.notes == ["Prefers hard light", "Uses SDXL mostly"]
    assert "<remember>" not in reply.content
    assert "<title>" not in reply.content
    assert reply.content.startswith("Got it.")
    # The prompt block stays in the content, which is what the LLM sees next turn.
    assert "<prompt>" in reply.content


def test_parse_reply_ends_an_unclosed_title_or_note_at_its_line():
    reply = chat.parse_reply(
        "<title>Night market\nHere is my take. <remember>likes neon\nMore text.</remember>",
        has_negative_prompt=True,
    )

    assert reply.title == "Night market"
    assert reply.notes == ["likes neon"]
    assert reply.parts == [{"type": "text", "text": "Here is my take.\nMore text."}]


def test_parse_reply_keeps_an_unclosed_prompt_block_cut_off_by_the_length_limit():
    reply = chat.parse_reply("Here:\n<prompt>\nPOSITIVE: a long prompt that gets cu", has_negative_prompt=True)

    assert reply.prompts == [{"positive_prompt": "a long prompt that gets cu", "negative_prompt": ""}]


def test_parse_reply_drops_code_fences_around_blocks():
    reply = chat.parse_reply(
        "Try this:\n```\n<prompt>\nPOSITIVE: a cat\nNEGATIVE:\n</prompt>\n```\nThoughts?",
        has_negative_prompt=True,
    )

    assert [p["type"] for p in reply.parts] == ["text", "prompt", "text"]
    assert "```" not in reply.content


def test_parse_reply_forces_empty_negative_when_the_family_has_none():
    reply = chat.parse_reply("<prompt>\nPOSITIVE: a cat\nNEGATIVE: blurry\n</prompt>", has_negative_prompt=False)

    assert reply.prompts == [{"positive_prompt": "a cat", "negative_prompt": ""}]


def test_parse_reply_without_tags_is_plain_text():
    reply = chat.parse_reply("What mood are you after?", has_negative_prompt=True)

    assert reply.parts == [{"type": "text", "text": "What mood are you after?"}]
    assert reply.prompts == []
    assert reply.notes == []
    assert reply.title == ""
