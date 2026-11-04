import json
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

    result = chat.build_system_prompt(FAMILY)

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
    result = chat.build_system_prompt(FAMILY)

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
        FAMILY,
        workflows=[{"name": "krea-2/txt2img", "description": "Text to image.", "model": "krea-2\\k.safetensors"},
                   {"name": "bare", "description": ""}],
    )

    assert "## Workflow Library\n- krea-2/txt2img on krea-2\\k.safetensors: Text to image.\n- bare\n\n## How This" in result
    assert result.endswith(chat.COMFY_MECHANICS)


@patch("app.chat.system_prompt.read_system_prompt", return_value="persona")
def test_system_prompt_shows_an_empty_library(mock_read):
    result = chat.build_system_prompt(FAMILY, workflows=[])

    assert "## Workflow Library\n(empty)" in result


@patch("app.chat.system_prompt.read_system_prompt", return_value="")
def test_system_prompt_starts_with_the_family_rules_when_the_personality_is_empty(mock_read):
    result = chat.build_system_prompt(FAMILY)

    assert result.startswith("## Model Family Rules: SDXL\n")
    assert "## How This Conversation Works\n" in result


@patch("app.chat.system_prompt.read_system_prompt", return_value="persona")
def test_a_title_is_asked_for_only_while_there_is_none_and_not_in_the_system_prompt(mock_read):
    # Asked in the turn's context: the system prompt stays the same once the title is there.
    assert "<title>" not in chat.build_system_prompt(FAMILY)
    assert chat.turn_context([], None, needs_title=True) == f"{chat.CONTEXT_HEADING}\n\n{chat.TITLE_RULE}"
    assert chat.turn_context([], None, needs_title=False) is None


@patch("app.chat.system_prompt.read_system_prompt", return_value="persona")
def test_the_notes_go_with_the_latest_message_not_the_system_prompt(mock_read):
    result = chat.build_system_prompt(FAMILY)

    # The notes and the installed models change every turn: not in the (cached) system prompt.
    assert "## Your Notes" not in result
    context = chat.turn_context([{"text": "Likes hard light"}, {"text": "Hates lens flare"}], models="LoRAs:\n- x")
    # Numbered, so the bot can rewrite or forget one by its number.
    assert context == (chat.CONTEXT_HEADING + "\n\n## Your Notes About The User\n[1] Likes hard light\n"
                       "[2] Hates lens flare\n\n## Installed Models\nLoRAs:\n- x")
    assert chat.turn_context([], None) is None
    conversation = _conversation(_user("hola"))
    messages = chat.build_llm_messages(conversation, load_image=lambda image: None, context=context)
    assert messages[-1]["content"] == [{"type": "text", "text": "hola"}, {"type": "text", "text": context}]
    assert messages.volatile_tail == 1


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


def test_a_prompt_block_names_its_model_and_keeps_it_when_replayed():
    reply = chat.parse_reply(
        "Va:\n<prompt model=\"krea-2\\gonzalomo_v1.0.safetensors\">\nPOSITIVE: a witch\n</prompt>\n"
        "<prompt>\nPOSITIVE: an owl\n</prompt>",
        has_negative_prompt=False,
    )

    assert reply.parts[1] == {"type": "prompt", "positive_prompt": "a witch", "negative_prompt": "",
                              "model": "krea-2\\gonzalomo_v1.0.safetensors"}
    assert "model" not in reply.parts[2]
    assert reply.prompts == [{"positive_prompt": "a witch", "negative_prompt": ""},
                             {"positive_prompt": "an owl", "negative_prompt": ""}]
    messages = chat.build_llm_messages(_conversation(_user("a witch"), _assistant(reply.content)),
                                       load_image=lambda image: None)
    assert '<prompt id="1" model="krea-2\\gonzalomo_v1.0.safetensors">\nPOSITIVE: a witch' in messages[1]["content"]
    assert '<prompt id="2">\nPOSITIVE: an owl' in messages[1]["content"]


def _attached(messages):
    return [p["image_url"]["url"] for m in messages if isinstance(m["content"], list)
            for p in m["content"] if p["type"] == "image_url"]


def test_llm_messages_attach_recent_images_in_a_window_that_moves_in_steps():
    images = [{"id": f"img{i}", "content_type": "image/png"} for i in range(1, 25)]
    load = lambda image: f"data:{image['id']}"  # noqa: E731

    def up_to(count):
        return _conversation(_user("first batch", images[:3]), _assistant("nice"),
                             *[m for image in images[3:count] for m in (_user("", [image]), _assistant("ok"))])

    # Up to 15 images, all travel.
    assert _attached(chat.build_llm_messages(up_to(15), load)) == [f"data:img{i}" for i in range(1, 16)]
    # At 16 the window steps 8 ahead: images 9 on.
    sixteen = chat.build_llm_messages(up_to(16), load)
    assert _attached(sixteen) == [f"data:img{i}" for i in range(9, 17)]
    assert sixteen[0]["content"].startswith("first batch\n\nImage 1 (shared earlier, no longer attached):")
    # And stays there while images arrive (what earlier requests sent doesn't change: the
    # prompt cache keeps serving it), until 24.
    twenty_three = chat.build_llm_messages(up_to(23), load)
    assert twenty_three[:len(sixteen)] == list(sixteen)
    assert _attached(chat.build_llm_messages(up_to(24), load))[0] == "data:img17"


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


def _long_conversation(rounds):
    """Rounds of: the user asks, the bot answers with a long block and launches it, the run's results."""
    messages, runs = [], []
    for n in range(1, rounds + 1):
        positive = f"Photograph {n}: " + "a witch in a misty forest at dusk, " * 60
        messages += [
            _user(f"ronda {n}"),
            {"role": "assistant", "content": f"Exploro la ronda {n}. " + "Razono mucho. " * 80
                + f"\n\n<prompt>POSITIVE: {positive}</prompt>",
             "parts": [{"type": "prompt", "positive_prompt": positive, "negative_prompt": ""}],
             "steps": [{"content": f"Exploro la ronda {n}. " + "Razono mucho. " * 80
                        + f"\n\n<prompt>POSITIVE: {positive}</prompt>",
                        "tool_calls": [{"id": f"c{n}", "name": "run_workflow", "arguments": json.dumps(
                            {"workflow": "krea-2/txt2img", "changes": {"positive": positive, "aspect": "2:3"}})}],
                        "results": [json.dumps({"run": n})]}]},
            {"role": "results", "run": n, "content": "", "images": [{"id": f"r{n}", "content_type": "image/png"}]},
        ]
        runs.append({"n": n, "workflow": "krea-2/txt2img", "status": "done",
                     "jobs": [{"status": "success", "seed": n, "images": [f"r{n}"]}]})
    return {**_conversation(*messages), "runs": runs}


def test_a_call_summarized_in_text_names_the_prompt_it_runs_instead_of_repeating_it():
    messages = chat.build_llm_messages(_long_conversation(1), load_image=lambda image: None, tools=False)
    reply = messages[1]["content"]
    assert '[Tools used: run_workflow({"workflow": "krea-2/txt2img", "changes": {"positive": "(prompt 1)", ' \
           '"aspect": "2:3"}})]' in reply
    assert reply.count("a witch in a misty forest") == 60  # once: in the block


def test_a_long_conversations_older_part_goes_in_brief_and_stays_put_between_moves():
    load = lambda image: None  # noqa: E731
    short = chat.build_llm_messages(_long_conversation(10), load)
    assert chat.compacted_until(_long_conversation(10)["messages"]) == 0
    assert chat.COMPACTED_NOTE not in short[0]["content"]

    conversation = _long_conversation(40)
    cut = chat.compacted_until(conversation["messages"])
    assert 0 < cut <= len(conversation["messages"]) - chat.COMPACT_MIN_MESSAGES
    messages = chat.build_llm_messages(conversation, load)
    assert messages[0]["content"].startswith(chat.COMPACTED_NOTE) and "ronda 1" in messages[0]["content"]
    first_reply = messages[1]["content"]
    assert first_reply.startswith("Exploro la ronda 1.") and len(first_reply) < 700
    assert '<prompt id="1">Photograph 1: a witch' in first_reply and "…</prompt>" in first_reply
    assert '"positive": "(prompt 1)"' in first_reply
    assert "[Run 1 (krea-2/txt2img) finished: image 1 (seed 1).]" in messages[2]["content"]
    assert "no longer attached" not in messages[2]["content"]  # a results line names its images already
    # The latest messages go in full.
    assert messages[-2]["content"].count("Razono mucho.") == 80
    size = lambda replay: sum(len(m["content"]) for m in replay if isinstance(m["content"], str))  # noqa: E731
    with patch.object(chat, "COMPACT_ABOVE", 10**9):
        whole = chat.build_llm_messages(conversation, load)
    assert size(messages) < size(whole) / 2

    # The boundary decided for earlier messages never changes, and moves only now and then: what was
    # sent stays the same from turn to turn (the prompt cache) until it does.
    cuts = [chat.compacted_until(_long_conversation(n)["messages"]) for n in range(1, 41)]
    assert cuts == sorted(cuts) and len(set(cuts)) < 10
    stable = [n for n in range(2, 41) if cuts[n - 1] == cuts[n - 2]]
    for n in stable[-3:]:
        before = chat.build_llm_messages(_long_conversation(n - 1), load)
        after = chat.build_llm_messages(_long_conversation(n), load)
        assert list(after[:len(before) - 1]) == list(before[:-1])


def test_get_prompt_gives_an_earlier_block_in_full():
    conversation = _long_conversation(3)
    found = chat.prompt_by_number(conversation, 2)
    assert found["number"] == 2 and found["positive"].startswith("Photograph 2: a witch")
    assert chat.prompt_by_number(conversation, 9) is None


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
