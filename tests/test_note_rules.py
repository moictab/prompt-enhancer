"""Notes stand on their own in any conversation: one short claim, no image numbers. One that
breaks that isn't kept -- never cut -- and the bot is told why, once, in its next turn."""
from unittest.mock import patch

from app import chat, conversations, memory

from .conftest import message_text
from .test_api_conversar_comfy import Script, _conversation, _family, _send

LONG = "krea-2: " + "the flash blowout works with the analog LoRA at 0.6 and the realism stack, " * 4


def test_a_long_note_or_one_citing_an_image_is_refused_not_cut():
    reply = chat.parse_reply(
        f"Hecho.<remember>krea-2: negations don't suppress (2 of 2)</remember><remember>{LONG}</remember>"
        '<remember id="3">krea-2: the leather spreads to the hand holding it (#20, #21)</remember>', True)
    assert reply.notes == ["krea-2: negations don't suppress (2 of 2)"]
    assert reply.note_updates == []
    assert [(reason, number) for _, reason, number in reply.refused_notes] == [("too_long", None), ("image_number", 3)]
    assert reply.refused_notes[0][0] == " ".join(LONG.split())  # whole, as written
    assert chat.parse_reply("<remember>likes: #ff0000 red accents</remember>", True).notes  # a colour isn't an image


def test_the_bot_hears_once_which_notes_werent_kept(api_client, auth_headers, tmp_path):
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]
    script = Script((f"Apuntado.<remember>{LONG}</remember><remember>likes: dusk light</remember>", []),
                    ("<remember>krea-2: flash blowout works (2 of 2)</remember>Reescrita.", []), ("Vale.", []))
    with patch("app.conversation_turn.stream_chat", script):
        _send(api_client, auth_headers, conversation_id, "hola")
        _send(api_client, auth_headers, conversation_id, "sigue")
        _send(api_client, auth_headers, conversation_id, "y ahora")

    first = conversations.get(conversation_id)["messages"][1]
    assert first["notes_refused"] == [{"text": " ".join(LONG.split()), "reason": "too_long"}]
    assert [n["text"] for n in memory.list_notes()] == ["likes: dusk light", "krea-2: flash blowout works (2 of 2)"]
    told, later = (message_text(call["messages"][-1]) for call in script.calls[1:])
    assert "Notes You Wrote That Weren't Kept" in told and "too long" in told and "one claim per note" in told
    assert "Weren't Kept" not in later
