"""Conversar's bot played by Claude Code: the CLI run per turn, locked down to the bot's tools
over MCP, its output turned into the chat's reply (faked here, line by line, as stream-json)."""
import json
from unittest.mock import patch

import pytest

from app import bot_seat, chat_tools, claude_code, comfy, conversations, families, runs, workflows

from .test_api_conversar_comfy import Script, _events, _family, _send

MODEL = "krea-2\\krea-2-turbo-official-comfy-int8-convrot_v1.0.safetensors"
RUN_INPUT = {"workflow": "krea-2/txt2img", "changes": {"positive": "a witch at dawn"}}
HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


# --- a fake Claude Code -----------------------------------------------------------------------------

def init(*tools, session="s"):
    return {"type": "system", "subtype": "init", "session_id": session,
            "tools": list(tools) or ["mcp__pe__run_workflow", "mcp__pe__get_image"]}


def text(content):
    """What the CLI prints for a text block: its deltas as they stream, then the block."""
    half = len(content) // 2
    return [*({"type": "stream_event", "parent_tool_use_id": None,
               "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": piece}}}
              for piece in (content[:half], content[half:]) if piece),
            {"type": "assistant", "parent_tool_use_id": None, "message": {"content": [{"type": "text", "text": content}]}}]


def tool(call_id, name, arguments):
    """A call to the app's tools: the tool_use block, the call reaching the app's MCP endpoint, its result."""
    return [{"type": "assistant", "parent_tool_use_id": None, "message": {"content": [
                {"type": "tool_use", "id": call_id, "name": f"mcp__pe__{name}", "input": arguments}]}},
            ("call", name, arguments),
            {"type": "user", "parent_tool_use_id": None, "message": {"content": [
                {"type": "tool_result", "tool_use_id": call_id, "content": "(the result)"}]}}]


def result(cost=0.05, error=None, session="s"):
    return {"type": "result", "subtype": "error_during_execution" if error else "success", "is_error": bool(error),
            "result": error or "done", "total_cost_usd": cost, "num_turns": 2, "duration_ms": 1200,
            "stop_reason": "end_turn", "session_id": session, "usage": {"input_tokens": 10, "output_tokens": 5}}


RATE_LIMIT = {"type": "rate_limit_event", "rate_limit_info": {"status": "allowed", "unifiedWindows": {
    "five_hour": {"utilization": 0.07}, "seven_day": {"utilization": 0.18}}}}


class Stdin:
    def __init__(self):
        self.written = ""

    def write(self, data):
        self.written += data

    def close(self):
        pass


class FakeProcess:
    def __init__(self, script, token):
        self.stdin, self.returncode, self.script, self.token = Stdin(), 0, script, token

        class _Empty:
            def read(self):
                return ""

        self.stderr = _Empty()
        self.stdout = self._lines()

    def _lines(self):
        for item in self.script:
            if isinstance(item, tuple):  # the CLI calls the app's MCP endpoint (in process here)
                _, name, arguments = item
                bot_seat._seats[self.token].run(name, arguments)
                continue
            yield json.dumps(item) + "\n"

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.returncode = -9


class FakeClaude:
    def __init__(self, *scripts):
        self.scripts = list(scripts)
        self.launches = []

    def __call__(self, args, env, cwd):
        files = {flag: args[args.index(flag) + 1] for flag in ("--mcp-config", "--system-prompt-file")}
        mcp = json.loads(open(files["--mcp-config"], encoding="utf-8").read())
        url = mcp["mcpServers"]["pe"]["url"]
        process = FakeProcess([item for part in self.scripts.pop(0) for item in (part if isinstance(part, list)
                                                                                 else [part])],
                              url.rsplit("/", 1)[-1])
        self.launches.append({"args": args, "env": env, "cwd": cwd, "url": url, "process": process,
                              "system": open(files["--system-prompt-file"], encoding="utf-8").read()})
        return process

    def sent(self, index=-1) -> str:
        """The text of the message a launch was given."""
        line = json.loads(self.launches[index]["process"].stdin.written)
        return "\n".join(b["text"] for b in line["message"]["content"] if b["type"] == "text")


@pytest.fixture
def claude(monkeypatch):
    monkeypatch.setattr(claude_code, "available", lambda: True)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-must-not-reach-it")
    monkeypatch.setenv("CLAUDE_CODE_CONFIG_DIR", "C:/claude-pe")

    def install(*scripts):
        fake = FakeClaude(*scripts)
        monkeypatch.setattr(claude_code, "_spawn", fake)
        return fake

    return install


def _claude_conversation(api_client, auth_headers, family, **settings):
    response = api_client.post("/api/conversations", auth=auth_headers, json={
        "family_id": family["id"], "llm_model": "vendor/model", "engine": "claude-code", "claude_model": "sonnet",
        **settings})
    assert response.status_code == 200, response.text
    return response.json()


# --- tests ------------------------------------------------------------------------------------------

def test_a_turn_runs_claude_code_locked_down_and_its_reply_lands_like_the_bots(api_client, auth_headers, tmp_path,
                                                                                fake_comfy, claude):
    workflows.ensure_seeded()
    fake_comfy.model_files["diffusion_models"] = [MODEL]
    conversation = _claude_conversation(api_client, auth_headers, _family(tmp_path))
    fake = claude([init(), text(f'<title>Bruja</title>Va:\n<prompt model="{MODEL}">\nPOSITIVE: a witch at dawn\n'
                                '</prompt>'),
                   tool("c1", "run_workflow", RUN_INPUT), text("Lanzada."), RATE_LIMIT, result()])

    events = _events(_send(api_client, auth_headers, conversation["id"]))

    assert [e["type"] for e in events] == ["delta", "delta", "tool", "delta", "delta", "done"]
    assert events[2]["part"] == {"type": "run", "run": 1, "workflow": "krea-2/txt2img", "count": 1}
    [launch] = fake.launches
    args = launch["args"]
    # Locked down: no built-in tools, only the app's MCP server, no settings files, nothing that asks.
    assert args[args.index("--tools") + 1] == "" and "--strict-mcp-config" in args
    assert args[args.index("--allowedTools") + 1] == "mcp__pe__*"
    assert args[args.index("--permission-mode") + 1] == "dontAsk"
    assert args[args.index("--setting-sources") + 1] == "" and args[args.index("--model") + 1] == "sonnet"
    session_id = args[args.index("--session-id") + 1]
    # Its own account: the configuration folder set, no API key to use instead.
    assert launch["env"]["CLAUDE_CONFIG_DIR"] == "C:/claude-pe" and "ANTHROPIC_API_KEY" not in launch["env"]
    assert launch["url"].startswith("http://127.0.0.1:8090/mcp/bot/")
    # The bot's instructions, as on OpenRouter, plus how it works through Claude Code.
    assert "## Model Family Rules: Krea" in launch["system"] and "## Workflow Library" in launch["system"]
    assert "mcp__pe__" in launch["system"] and "<prompt>" in launch["system"]
    assert "Hazme una bruja" in fake.sent() and "no title yet" in fake.sent()

    message = events[-1]["message"]
    assert [p["type"] for p in message["parts"]] == ["text", "prompt", "run", "text"]
    assert message["parts"][1]["model"] == MODEL
    [first, second] = message["steps"]
    assert first["tool_calls"] == [{"id": "c1", "name": "run_workflow", "arguments": json.dumps(RUN_INPUT)}]
    assert json.loads(first["results"][0])["run"] == 1 and second["content"] == "Lanzada."
    assert message["engine"] == "claude-code" and message["claude_code"]["estimated_cost"] == 0.05
    assert message["cost"] is None  # the subscription pays: not the conversation's cost
    stored = conversations.get(conversation["id"])
    assert stored["title"] == "Bruja" and stored["cost"] is None
    assert stored["claude_code"]["session_id"] == session_id
    assert stored["claude_code"]["rate_limit"]["unifiedWindows"]["five_hour"]["utilization"] == 0.07
    assert stored["runs"][0]["base_models"] == [MODEL] and len(fake_comfy.submitted) == 1
    # The turn's tools closed with it.
    assert not bot_seat._seats


def test_later_turns_resume_the_session_and_send_only_what_changed(api_client, auth_headers, tmp_path, fake_comfy,
                                                                   claude):
    workflows.ensure_seeded()
    family = _family(tmp_path)
    other = families.create_family("SDXL", "SDXL rules here", True, path=str(tmp_path / "families.json"))
    conversation = _claude_conversation(api_client, auth_headers, family)
    fake = claude([init(), text("<title>Bruja</title>\n<prompt>\nPOSITIVE: a witch\n</prompt>"), result()],
                  [init(), text("Otra."), result()],
                  [init(), text("Con SDXL."), result()])
    _send(api_client, auth_headers, conversation["id"], "hola")
    _send(api_client, auth_headers, conversation["id"], "otra")

    session_id = fake.launches[0]["args"][fake.launches[0]["args"].index("--session-id") + 1]
    resumed = fake.launches[1]["args"]
    assert resumed[resumed.index("--resume") + 1] == session_id and "--session-id" not in resumed
    second = fake.sent(1)
    assert claude_code.UNCHANGED_NOTE in second  # the notes and models it already has
    assert "Your previous reply's prompt blocks are numbered 1." in second
    assert "SDXL rules here" not in second

    # A family switched since the session began: its rules come in the message.
    api_client.patch(f"/api/conversations/{conversation['id']}", auth=auth_headers, json={"family_id": other["id"]})
    _send(api_client, auth_headers, conversation["id"], "pasemos a SDXL")
    third = fake.sent(2)
    assert claude_code.CHANGED_HEADING.splitlines()[0] in third and "SDXL rules here" in third


def test_a_session_that_starts_late_gets_the_conversation_so_far(api_client, auth_headers, tmp_path, fake_comfy,
                                                                 claude):
    conversation_id = api_client.post("/api/conversations", auth=auth_headers, json={
        "family_id": _family(tmp_path)["id"], "llm_model": "vendor/model"}).json()["id"]
    with patch("app.conversation_turn.stream_chat", Script(("<title>Gato</title>Un gato naranja, ¿vale?", []))):
        _send(api_client, auth_headers, conversation_id, "quiero un gato")
    api_client.patch(f"/api/conversations/{conversation_id}", auth=auth_headers, json={"engine": "claude-code"})
    fake = claude([init(), text("Sigo con el gato."), result()])

    _send(api_client, auth_headers, conversation_id, "más grande")

    sent = fake.sent()
    assert sent.startswith(claude_code.HISTORY_INTRO)
    assert "[The user]\nquiero un gato" in sent and "[You]\nUn gato naranja, ¿vale?" in sent
    assert sent.index(claude_code.HISTORY_END) < sent.index("más grande")
    assert conversations.get(conversation_id)["messages"][-1]["content"] == "Sigo con el gato."


def test_failures_say_what_to_do_and_store_nothing(api_client, auth_headers, tmp_path, fake_comfy, claude):
    conversation = _claude_conversation(api_client, auth_headers, _family(tmp_path))
    claude([init(), result(error="Invalid API key · Please run /login")],
           [init("Bash", "mcp__pe__run_workflow"), text("never"), result()])

    signed_out = _events(_send(api_client, auth_headers, conversation["id"], "hola"))[-1]
    assert signed_out["type"] == "error" and "claude auth login" in signed_out["detail"]
    assert "C:/claude-pe" in signed_out["detail"] and signed_out["request_id"]

    api_client.post(f"/api/conversations/{conversation['id']}/reply", auth=auth_headers)
    stored = conversations.get(conversation["id"])
    assert [m["role"] for m in stored["messages"]] == ["user"] and "claude_code" not in stored
    # A session that came up with a tool it mustn't have was stopped before it did anything.
    log = api_client.get("/api/requests?errors=true", auth=auth_headers).json()
    assert any("mustn't have (Bash)" in (r.get("error") or "") for r in log["items"])


def test_the_files_it_is_given_are_found_from_its_own_folder(tmp_path, monkeypatch):
    """DATA_DIR is relative by default ("data"), and the CLI runs in the workspace folder: the paths
    it is given must not depend on where it runs."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", "data")
    seat = bot_seat.open_seat("c", None, tools_allowed=False)
    try:
        files = claude_code._write_files(claude_code._folder("turns", seat.token), "system", seat)
        args = claude_code.args_for("opus", "s", False, files)
    finally:
        bot_seat.close_seat(seat)
    workspace = claude_code._folder("workspace")
    assert workspace.is_absolute()
    for flag in ("--mcp-config", "--settings", "--system-prompt-file"):
        given = args[args.index(flag) + 1]
        assert (workspace / given).is_file(), flag  # an absolute path joined to the workspace is itself


def test_a_cli_that_quits_before_reading_says_why(api_client, auth_headers, tmp_path, monkeypatch):
    """The message is written while the CLI starts up; if it ends first, the write fails (a broken
    pipe), and the reason is what it said, not that it couldn't start."""
    conversation = _claude_conversation(api_client, auth_headers, _family(tmp_path))
    monkeypatch.setattr(claude_code, "available", lambda: True)

    class Closed:
        def write(self, data):
            raise BrokenPipeError(32, "Broken pipe")

        def close(self):
            raise BrokenPipeError(32, "Broken pipe")

    class Said:
        def read(self):
            return "Error: the configuration file is not valid JSON\n"

    class Quit:
        stdin, stderr, stdout, returncode = Closed(), Said(), iter(()), 1

        def wait(self, timeout=None):
            return 1

        def kill(self):
            pass

    monkeypatch.setattr(claude_code, "_spawn", lambda args, env, cwd: Quit())
    failed = _events(_send(api_client, auth_headers, conversation["id"], "hola"))[-1]
    assert failed["type"] == "error"
    assert failed["detail"] == "Claude Code stopped (exit code 1): Error: the configuration file is not valid JSON."
    assert "Is it installed" not in failed["detail"]


def test_the_seat_serves_one_turns_tools_and_closes_with_it(api_client, auth_headers, tmp_path, fake_comfy, monkeypatch):
    workflows.ensure_seeded()
    monkeypatch.setattr(runs.tracker, "start", lambda: None)
    conversation_id = api_client.post("/api/conversations", auth=auth_headers, json={
        "family_id": _family(tmp_path)["id"], "llm_model": "vendor/model"}).json()["id"]

    def rpc(token, method, params=None, number=1):
        return api_client.post(f"/mcp/bot/{token}", headers=HEADERS, json={
            "jsonrpc": "2.0", "id": number, "method": method, "params": params or {}})

    with api_client:
        seat = bot_seat.open_seat(conversation_id, chat_tools.ToolBox(conversation_id, comfy.client()), driven=False)
        assert rpc(seat.token, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                             "clientInfo": {"name": "test", "version": "0"}}).status_code == 200
        names = {t["name"] for t in rpc(seat.token, "tools/list", number=2).json()["result"]["tools"]}
        assert {"run_workflow", "calibrate_lora", "get_image", "get_balance_data"} <= names
        called = rpc(seat.token, "tools/call", {"name": "run_workflow", "arguments": RUN_INPUT}, number=3).json()
        assert json.loads(called["result"]["content"][0]["text"])["run"] == 1 and len(fake_comfy.submitted) == 1
        assert seat.claim("run_workflow", RUN_INPUT)["part"]["type"] == "run"

        # A turn that mustn't run anything (an automatic look, taking stock) gets an error back.
        quiet = bot_seat.open_seat(conversation_id, chat_tools.ToolBox(conversation_id, comfy.client()),
                                   driven=False, tools_allowed=False)
        refused = rpc(quiet.token, "tools/call", {"name": "run_workflow", "arguments": RUN_INPUT}, number=4).json()
        assert refused["result"]["isError"] and len(fake_comfy.submitted) == 1

        bot_seat.close_seat(seat)
        bot_seat.close_seat(quiet)
        assert rpc(seat.token, "tools/list").status_code == 404
        assert rpc("not-a-turn", "tools/list").status_code == 404
