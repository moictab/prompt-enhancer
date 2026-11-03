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
def test_system_prompt_puts_personality_first_family_rules_next_and_mechanics_last(mock_read):
    mock_read.return_value = "You are a grumpy art director."

    result = chat.build_system_prompt(FAMILY, [], [], needs_title=False)

    mock_read.assert_called_once_with("conversation")
    assert result.startswith(
        "You are a grumpy art director.\n\n## Model Family Rules: SDXL\n" + chat.FAMILY_RULES_INTRO + "\n\nfamily rules"
    )
    # Last, and declared to win: a family written for one-shot modes says
    # "output only the prompt", which a conversation must not follow.
    mechanics_at = result.index("## How This Conversation Works\n")
    assert mechanics_at > result.index("family rules")
    assert result.endswith(chat.CONVERSATION_MECHANICS)
    assert "take precedence over anything above" in result
    assert "<prompt>\nPOSITIVE:" in result
    assert "<title>" not in result


@patch("app.chat.system_prompt.read_system_prompt", return_value="persona")
def test_system_prompt_tells_the_bot_to_keep_terse_notes_and_change_them_without_asking(mock_read):
    result = chat.build_system_prompt(FAMILY, [], [], needs_title=False)

    assert "Change them in your reply, right away and without asking" in result
    assert "telegraphic" in result
    assert '<remember id="n">note</remember> rewrites note n' in result
    assert '<forget id="n"/> deletes note n' in result
    # No ComfyUI: nothing about running workflows.
    assert "run_workflow" not in result
    assert "## Workflow Library" not in result


@patch("app.chat.system_prompt.read_system_prompt", return_value="persona")
def test_system_prompt_lists_the_workflow_library_and_how_to_run_when_comfy_is_available(mock_read):
    result = chat.build_system_prompt(
        FAMILY, [], [], needs_title=True,
        workflows=[{"name": "krea-2/txt2img", "description": "Text to image."}, {"name": "bare", "description": ""}],
    )

    assert "## Workflow Library\n- krea-2/txt2img: Text to image.\n- bare\n\n## How This" in result
    assert chat.COMFY_MECHANICS in result
    # Still last: the title rule.
    assert result.endswith(chat.TITLE_RULE)


@patch("app.chat.system_prompt.read_system_prompt", return_value="persona")
def test_system_prompt_shows_an_empty_library(mock_read):
    result = chat.build_system_prompt(FAMILY, [], [], needs_title=False, workflows=[])

    assert "## Workflow Library\n(empty)" in result


@patch("app.chat.system_prompt.read_system_prompt", return_value="")
def test_system_prompt_starts_with_the_family_rules_when_the_personality_is_empty(mock_read):
    result = chat.build_system_prompt(FAMILY, [], [], needs_title=False)

    assert result.startswith("## Model Family Rules: SDXL\n")
    assert "## How This Conversation Works\n" in result


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
    # Numbered, so the bot can rewrite or forget one by its number.
    notes = "## Your Notes About The User\n[1] Likes hard light\n[2] Hates lens flare"
    assert notes in result
    assert result.index("## Characters") < result.index(notes) < result.index("## How This Conversation Works")


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
    # Every image is numbered across the conversation, attached or not.
    first, last = messages[0]["content"], messages[2]["content"]
    assert first == [
        {"type": "text", "text": "first batch"},
        {"type": "text", "text": "Image 1 (shared earlier, no longer attached):"},
        {"type": "text", "text": "Image 2 (shared earlier, no longer attached):"},
        {"type": "text", "text": "Image 3:"},
        {"type": "image_url", "image_url": {"url": "data:img2"}},
    ]
    assert [part.get("text") for part in last if part["type"] == "text"] == ["Image 4:", "Image 5:", "Image 6:"]
    assert [part["image_url"]["url"] for part in last if part["type"] == "image_url"] == [
        "data:img3", "data:img4", "data:img5",
    ]


def test_llm_messages_mention_an_image_whose_file_is_gone():
    conversation = _conversation(_user("look", [{"id": "gone", "content_type": "image/png"}]))

    messages = chat.build_llm_messages(conversation, load_image=lambda image: None)

    assert messages[0]["content"] == "look\n\nImage 1 (shared earlier, no longer attached):"


def test_llm_messages_give_each_image_its_metadata():
    metadata = {"format": "PNG", "width": 832, "height": 1216, "generator": "ComfyUI", "has_workflow": True,
                "positive": "a witch", "loras": [{"name": "ink.safetensors", "strength": 0.5, "strength_clip": 0.5}],
                "settings": {"seed": 7, "steps": 8}}
    conversation = _conversation(_user("mira", [{"id": "a", "content_type": "image/png", "metadata": metadata}]))

    messages = chat.build_llm_messages(conversation, load_image=lambda image: "data:a")

    assert messages[0]["content"][1] == {
        "type": "text",
        "text": 'Image 1:\nImage 1 metadata (PNG, 832x1216, made with ComfyUI; its ComfyUI workflow can be used as '
                '"image:1"):\n- Positive: a witch\n- LoRAs: ink.safetensors (0.5)\n- Settings: seed: 7, steps: 8',
    }


def _run(n=1, jobs=None):
    return {"n": n, "workflow": "krea-2/txt2img", "status": "done",
            "jobs": jobs or [{"status": "success", "seed": 11, "images": ["r1"]}]}


def test_llm_messages_show_finished_runs_as_automatic_when_nothing_follows():
    conversation = {
        **_conversation(
            _user("hazla"),
            _assistant("Lanzada."),
            {"role": "results", "run": 1, "content": "", "images": [{"id": "r1", "content_type": "image/png"}]},
        ),
        "runs": [_run()],
    }

    messages = chat.build_llm_messages(conversation, load_image=lambda image: "data:r1")

    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[2]["content"][0] == {
        "type": "text",
        "text": f"[Run 1 (krea-2/txt2img) finished: image 1 (seed 11).] {chat.AUTOMATIC_NOTE}",
    }
    assert messages[2]["content"][2]["image_url"]["url"] == "data:r1"


def test_llm_messages_merge_results_with_the_users_next_message():
    failed = _run(jobs=[{"status": "error", "error": "CUDA out of memory", "images": []}])
    conversation = {
        **_conversation(
            _user("hazla"),
            _assistant("Lanzada."),
            {"role": "results", "run": 1, "content": "", "images": []},
            _user("¿qué ha pasado?"),
        ),
        "runs": [failed],
    }

    messages = chat.build_llm_messages(conversation, load_image=lambda image: None)

    # One user turn, and no "automatic" claim now that the user has written.
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[2]["content"] == (
        "[Run 1 (krea-2/txt2img) produced no images. 1 of 1 job failed: CUDA out of memory.]\n\n¿qué ha pasado?"
    )


def _tool_reply():
    return {
        "role": "assistant",
        "content": "Voy.\n\nLanzada.",
        "steps": [
            {"content": "Voy.", "tool_calls": [
                {"id": "c1", "name": "inspect_workflow", "arguments": '{"workflow": "w"}'},
                {"id": "c2", "name": "run_workflow", "arguments": '{"workflow": "w"}'},
            ], "results": ['{"nodes": []}', '{"run": 1}']},
            {"content": "Lanzada.", "tool_calls": [], "results": []},
        ],
    }


def test_llm_messages_replay_tool_calls_and_omit_old_listings():
    conversation = _conversation(_user("hazla"), _tool_reply(), _user("otra"))

    messages = chat.build_llm_messages(conversation, load_image=lambda image: None, tools=True)

    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "tool", "assistant", "user"]
    assert messages[1]["tool_calls"][1] == {
        "id": "c2", "type": "function", "function": {"name": "run_workflow", "arguments": '{"workflow": "w"}'},
    }
    assert messages[2] == {"role": "tool", "tool_call_id": "c1", "content": '{"note": "Result omitted from an '
                           'earlier turn; call the tool again if you need it."}'}
    assert messages[3] == {"role": "tool", "tool_call_id": "c2", "content": '{"run": 1}'}
    assert messages[4] == {"role": "assistant", "content": "Lanzada."}


def test_llm_messages_summarize_tool_calls_when_tools_are_off():
    conversation = _conversation(_user("hazla"), _tool_reply())

    messages = chat.build_llm_messages(conversation, load_image=lambda image: None, tools=False)

    assert [m["role"] for m in messages] == ["user", "assistant", "assistant"]
    assert messages[1]["content"].startswith("Voy.\n\n[Tools used: inspect_workflow(")


def test_llm_messages_append_the_current_turns_steps_in_full():
    conversation = _conversation(_user("hazla"))
    steps = [{"content": "", "tool_calls": [{"id": "c1", "name": "inspect_workflow", "arguments": "{}"}],
              "results": ['{"nodes": [1]}']}]

    messages = chat.build_llm_messages(conversation, load_image=lambda image: None, tools=True, current_steps=steps)

    assert messages[1]["content"] is None
    assert messages[2] == {"role": "tool", "tool_call_id": "c1", "content": '{"nodes": [1]}'}


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


def test_parse_reply_reads_rewrites_and_deletions_of_existing_notes():
    reply = chat.parse_reply(
        'Anotado. <remember id="2">likes: cool  light, grain</remember> <forget id=3/>'
        "<forget id=\"4\"></forget><forget id='3'> <remember>krea-2: ink LoRA &lt;0.3</remember>\nSigo.",
        has_negative_prompt=True,
    )

    assert reply.notes == ["krea-2: ink LoRA &lt;0.3"]
    assert reply.note_updates == [(2, "likes: cool light, grain")]
    assert reply.note_forgets == [3, 4]
    assert reply.parts == [{"type": "text", "text": "Anotado.\nSigo."}]


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


def test_parse_reply_turns_a_bare_prompt_into_a_card():
    reply = chat.parse_reply(
        "Here it is:\nPOSITIVE: masterpiece, best quality, 1girl, solo. A witch reads.\n\nWhat do you think?",
        has_negative_prompt=False,
    )

    assert reply.parts == [
        {"type": "text", "text": "Here it is:"},
        {"type": "prompt", "positive_prompt": "masterpiece, best quality, 1girl, solo. A witch reads.", "negative_prompt": ""},
        {"type": "text", "text": "What do you think?"},
    ]
    # Stored in the block format, so the model sees it that way next turn.
    assert "<prompt>\nPOSITIVE: masterpiece" in reply.content


def test_parse_reply_joins_a_bare_negative_paragraph_and_unbolds_labels():
    reply = chat.parse_reply(
        "**POSITIVE:** a cat on a roof\n\n**NEGATIVE:** blurry, watermark\n\nAnother idea:\n\nPOSITIVE: a dog",
        has_negative_prompt=True,
    )

    assert reply.prompts == [
        {"positive_prompt": "a cat on a roof", "negative_prompt": "blurry, watermark"},
        {"positive_prompt": "a dog", "negative_prompt": ""},
    ]
    assert [p["type"] for p in reply.parts] == ["prompt", "text", "prompt"]


def test_parse_reply_leaves_bare_labels_alone_when_there_are_blocks():
    reply = chat.parse_reply(
        "<prompt>\nPOSITIVE: a cat\n</prompt>\n\nPOSITIVE: words are cheap",
        has_negative_prompt=True,
    )

    assert reply.prompts == [{"positive_prompt": "a cat", "negative_prompt": ""}]


def test_parse_reply_ignores_positive_in_the_middle_of_a_sentence():
    reply = chat.parse_reply("El POSITIVE: que te pasé antes funciona mejor.", has_negative_prompt=True)

    assert reply.prompts == []


def test_parse_reply_without_tags_is_plain_text():
    reply = chat.parse_reply("What mood are you after?", has_negative_prompt=True)

    assert reply.parts == [{"type": "text", "text": "What mood are you after?"}]
    assert reply.prompts == []
    assert reply.notes == []
    assert reply.title == ""
