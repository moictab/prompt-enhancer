"""Ordenar la memoria: the bot proposes changes to its notes, checked; the user applies the ones
picked, and can undo them."""
import json

from app import claude_code, memory, memory_tidy


def _notes():
    a = memory.create_note("krea-2: negations don't suppress (#13)", scope={"scope": "family", "family": "krea-2"})
    b = memory.create_note("Kim: look = selfie móvil amateur; ropa = camiseta negra; odia maquillaje corrido",
                           conversation_id="c1", scope={"scope": "topic", "topic": "Kim"})
    c = memory.create_note("anima: base checkpoint cfg 5 is sharper", scope={"scope": "family", "family": "anima"})
    d = memory.create_note("anima: base checkpoint at cfg 5 gives more detail than turbo",
                           scope={"scope": "family", "family": "anima"})
    e = memory.create_note("legs pass through chairs on anima")  # general, but it's anima's
    f = memory.create_note("omar-dogan style: closes clothes, discarded")
    memory.confirm_note(c["id"])
    memory.confirm_note(d["id"])
    return a, b, c, d, e, f


PROPOSAL = {"operations": [
    {"op": "rewrite", "note": 1, "text": "krea-2: negations don't suppress the trait (2 of 2)", "why": "citaba #13"},
    {"op": "split", "note": 2, "why": "tres cosas", "into": [
        {"text": "Kim: amateur phone selfie look", "topic": "Kim"},
        {"text": "Kim: faded black band tee", "topic": "Kim"},
        {"text": "Kim: hates smudged makeup", "topic": "Kim"}]},
    {"op": "merge", "notes": [3, 4], "text": "anima: base checkpoint at cfg 5 sharper than turbo (2 notes)",
     "why": "duplicadas"},
    {"op": "move", "note": 5, "scope": "family", "family": "anima", "why": "es de Anima"},
    {"op": "forget", "note": 6, "why": "superada"},
    {"op": "rewrite", "note": 1, "text": "again", "why": "used twice: dropped"},
    {"op": "rewrite", "note": 6, "text": "cites #4", "why": "image number: dropped"},
    {"op": "move", "note": 5, "family": "sd15", "why": "no such folder: dropped"},
    {"op": "forget", "note": 99, "why": "no such note: dropped"},
]}


def test_a_proposal_is_checked_and_only_what_holds_is_offered(monkeypatch):
    _notes()
    asked = []
    monkeypatch.setattr(memory_tidy, "_ask", lambda engine, model, prompt: (
        asked.append((engine, model, prompt)) or "Here:\n```json\n" + json.dumps(PROPOSAL) + "\n```", 0.02))
    monkeypatch.setattr("app.downloads.family_folders", lambda: {"loras": ["anima", "krea-2"]})

    plan = memory_tidy.propose("claude-code", "sonnet")

    assert [op["kind"] for op in plan["operations"]] == ["rewrite", "split", "merge", "move", "forget"]
    assert plan["dropped"] == 4 and plan["notes"] == 6 and memory_tidy.pending()["id"] == plan["id"]
    engine, model, prompt = asked[0]
    assert (engine, model) == ("claude-code", "sonnet")
    assert "[2] (topic Kim; " in prompt and "one of: anima, krea-2" in prompt and "the topics so far: Kim" in prompt
    split = plan["operations"][1]
    assert [a["label"] for a in split["after"]] == ["topic Kim"] * 3 and split["before"][0]["label"] == "topic Kim"
    assert len(memory.list_notes()) == 6  # nothing applied yet


def test_applying_keeps_history_and_can_be_undone(monkeypatch):
    a, b, c, d, e, f = _notes()
    monkeypatch.setattr(memory_tidy, "_ask", lambda *args: (json.dumps(PROPOSAL), None))
    monkeypatch.setattr("app.downloads.family_folders", lambda: {"loras": ["anima", "krea-2"]})
    memory_tidy.propose("openrouter", "")
    # Edited meanwhile: its operation no longer applies.
    memory.update_note(f["id"], "omar-dogan style: closes clothes; maybe at 0.4")

    done = memory_tidy.apply([0, 1, 2, 3, 4])

    assert done == {"applied": 4, "skipped": 1, "before": 6, "after": 7}
    notes = memory.list_notes()
    texts = [n["text"] for n in notes]
    assert texts[0] == "krea-2: negations don't suppress the trait (2 of 2)" and notes[0]["id"] == a["id"]
    assert texts[1:4] == ["Kim: amateur phone selfie look", "Kim: faded black band tee", "Kim: hates smudged makeup"]
    assert all(n["conversation_id"] == "c1" and n["created_at"] == b["created_at"] for n in notes[1:4])
    merged = notes[4]
    assert merged["confirmations"] == 2 and merged["created_at"] == c["created_at"] and merged["family"] == "anima"
    moved = next(n for n in notes if n["id"] == e["id"])
    assert (moved["scope"], moved["family"]) == ("family", "anima")
    assert memory_tidy.pending() is None and memory_tidy.can_undo()

    since = memory.create_note("likes dusk light")
    assert memory_tidy.undo() == {"notes": 7}
    restored = memory.list_notes()
    assert [n["id"] for n in restored] == [a["id"], b["id"], c["id"], d["id"], e["id"], f["id"], since["id"]]
    assert not memory_tidy.can_undo()


def test_ajustes_drives_it_end_to_end(api_client, auth_headers, monkeypatch):
    _notes()
    monkeypatch.setattr(memory_tidy, "_ask", lambda *args: (json.dumps(PROPOSAL), 0.01))
    monkeypatch.setattr("app.downloads.family_folders", lambda: {"loras": ["anima", "krea-2"]})
    assert api_client.get("/api/admin/memory/tidy", auth=auth_headers).json() == {"plan": None, "can_undo": False}
    plan = api_client.post("/api/admin/memory/tidy", auth=auth_headers,
                           json={"engine": "claude-code", "model": "sonnet"}).json()
    assert len(plan["operations"]) == 5
    assert api_client.get("/api/admin/memory/tidy", auth=auth_headers).json()["plan"]["id"] == plan["id"]
    assert api_client.post("/api/admin/memory/tidy/apply", auth=auth_headers,
                           json={"accept": [4]}).json()["after"] == 5
    assert api_client.post("/api/admin/memory/tidy/undo", auth=auth_headers).json() == {"notes": 6}
    assert api_client.post("/api/admin/memory/tidy/undo", auth=auth_headers).status_code == 409
    monkeypatch.setattr(memory_tidy, "_ask", lambda *args: ("I can't do that.", None))
    failed = api_client.post("/api/admin/memory/tidy", auth=auth_headers, json={"engine": "openrouter"})
    assert failed.status_code == 502 and "JSON" in failed.json()["detail"]


def test_claude_code_answers_a_one_off_request_without_tools_or_session(monkeypatch):
    launched = {}

    class Done:
        returncode = 0

        def communicate(self, prompt, timeout=None):
            launched["prompt"] = prompt
            return json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "{}",
                               "total_cost_usd": 0.03, "duration_ms": 900}) + "\n", ""

    def spawn(args, env, cwd):
        launched["args"] = args
        return Done()

    monkeypatch.setattr(claude_code, "_spawn", spawn)
    text, info = claude_code.ask("system", "the notes", "sonnet", {"mode": "memory_tidy"})
    assert text == "{}" and info["estimated_cost"] == 0.03 and launched["prompt"] == "the notes"
    args = launched["args"]
    assert args[args.index("--tools") + 1] == "" and "--no-session-persistence" in args
    assert "--mcp-config" not in args and "--strict-mcp-config" in args
