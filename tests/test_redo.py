"""Rehacer: the user discards the bot's last reply and has it answer again."""
from unittest.mock import patch

from app import chat, conversations, history, memory, runs, workflows

from .conftest import message_text
from .test_api_conversar_comfy import RUN_CALL, Script, _conversation, _events, _family, _send

FIRST = ("<remember>likes fog</remember>Voy con niebla:\n<prompt>\nPOSITIVE: a witch at dawn\n</prompt>", [RUN_CALL])


def test_a_discarded_reply_takes_its_runs_notes_and_history_with_it(api_client, auth_headers, tmp_path, fake_comfy):
    workflows.ensure_seeded()
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]
    kept = memory.create_note("prefers 2:3")
    script = Script(FIRST, ('Mejor sin niebla, a pleno sol.<remember id="1">prefers 3:2</remember>', []))

    with patch("app.conversation_turn.stream_chat", script):
        _send(api_client, auth_headers, conversation_id, "hazme una bruja")
        first = conversations.get(conversation_id)["messages"][-1]
        assert first["history_id"] and [n["text"] for n in first["notes"]] == ["likes fog"]
        events = _events(api_client.post(f"/api/conversations/{conversation_id}/redo", auth=auth_headers))
    runs.wait_until_idle()

    assert events[-1]["type"] == "done"
    stored = conversations.get(conversation_id)
    assert [m["role"] for m in stored["messages"]] == ["user", "assistant"]
    assert stored["messages"][0]["redo"] == 1
    assert stored["messages"][-1]["content"].startswith("Mejor sin niebla")
    # Its run was taken off ComfyUI's queue, and its images won't come into the chat.
    run = stored["runs"][0]
    assert run["discarded"] is True and run["status"] == "cancelled"
    assert sorted(fake_comfy.cancelled) == ["prompt-1", "prompt-2"]
    # Its note went, its history entry too; the new reply's change stands.
    assert [n["text"] for n in memory.list_notes()] == ["prefers 3:2"] and memory.list_notes()[0]["id"] == kept["id"]
    assert all(e["id"] != first["history_id"] for e in history.list_entries())
    # The bot is told it is answering again.
    assert chat.REDO_RULE in message_text(script.calls[-1]["messages"][-1])


def test_a_rewritten_note_comes_back_and_only_a_reply_can_be_redone(api_client, auth_headers, tmp_path, fake_comfy):
    workflows.ensure_seeded()
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]
    note = memory.create_note("prefers 2:3")
    script = Script(('<remember id="1">prefers 1:1</remember><forget id="1"/>Hecho.', []), ("Otra vez.", []))

    assert api_client.post(f"/api/conversations/{conversation_id}/redo", auth=auth_headers).status_code == 409
    with patch("app.conversation_turn.stream_chat", script):
        _send(api_client, auth_headers, conversation_id, "olvida eso")
        assert memory.list_notes() == []
        api_client.post(f"/api/conversations/{conversation_id}/redo", auth=auth_headers)
    assert [n["text"] for n in memory.list_notes()] == ["prefers 2:3"]
    assert memory.list_notes()[0]["id"] != note["id"]  # forgotten, then written again
