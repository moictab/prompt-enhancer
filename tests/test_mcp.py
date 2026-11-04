"""The MCP endpoint: an agent uses the app's features as tools, behind the same credentials."""
import base64
import json
from unittest.mock import patch

import pytest

from app import families, mcp_server, runs, workflows
from app.openrouter_client import OpenRouterResult, StreamEnd

from .conftest import message_text, png_bytes

HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
MODEL = "krea-2\\krea-2-turbo-official-comfy-int8-convrot_v1.0.safetensors"


class Agent:
    """An MCP client over the test client: JSON-RPC posts to /mcp."""

    def __init__(self, client, auth):
        self.client, self.auth, self.ids = client, auth, 0
        initialized = self.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                              "clientInfo": {"name": "test", "version": "0"}})
        self.version = initialized["result"]["protocolVersion"]

    def rpc(self, method, params=None):
        self.ids += 1
        headers = {**HEADERS, **({"MCP-Protocol-Version": self.version} if self.ids > 1 else {})}
        response = self.client.post("/mcp", headers=headers, auth=self.auth,
                                    json={"jsonrpc": "2.0", "id": self.ids, "method": method, "params": params or {}})
        assert response.status_code == 200, response.text
        return response.json()

    def raw(self, name, **arguments):
        return self.rpc("tools/call", {"name": name, "arguments": arguments})["result"]

    def call(self, name, **arguments):
        result = self.raw(name, **arguments)
        assert not result.get("isError"), result["content"][0]["text"]
        structured = result.get("structuredContent")
        if structured is not None:
            return structured.get("result", structured) if set(structured) == {"result"} else structured
        return json.loads(result["content"][0]["text"])


class Script:
    """A fake stream_chat that plays one scripted reply per request: (text, tool_calls)."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, api_key, model, messages, temperature, context=None, tools=None, tool_choice=None):
        self.calls.append({"messages": messages, "context": context})
        text, tool_calls = self.replies.pop(0)
        if text:
            yield text
        yield StreamEnd(cost=0.01, tool_calls=tool_calls)


@pytest.fixture
def agent(api_client, auth_headers, monkeypatch):
    # The lifespan runs the MCP server; the tests drive the run tracker by hand.
    monkeypatch.setattr(runs.tracker, "start", lambda: None)
    monkeypatch.setattr(mcp_server, "WAIT_POLL_SECONDS", 0.01)
    with api_client:
        yield Agent(api_client, auth_headers)


def test_the_endpoint_takes_the_apps_credentials_and_lists_every_feature(api_client, auth_headers, agent):
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    assert api_client.post("/mcp", json=body, headers=HEADERS).status_code == 401
    assert api_client.post("/mcp", json=body, headers=HEADERS, auth=("admin", "wrong")).status_code == 401

    tools = {t["name"]: t for t in agent.rpc("tools/list")["result"]["tools"]}
    for name in ("generate_prompt", "prompt_from_image", "write_story", "make_variations", "start_conversation",
                 "send_message", "wait_for_conversation", "get_image", "react_to_image", "search_history",
                 "save_family", "save_character", "save_note", "set_system_prompt", "save_workflow",
                 "list_installed_models", "list_requests", "take_stock", "get_balance_data"):
        assert name in tools
    assert tools["delete_conversation"]["annotations"]["destructiveHint"] is True
    assert tools["get_conversation"]["annotations"]["readOnlyHint"] is True


@patch("app.routes.api.call_openrouter")
def test_an_agent_writes_a_prompt_for_a_family_named_loosely(mock_call, agent, tmp_path):
    families.create_family("Krea 2", "rules", False, path=str(tmp_path / "families.json"))
    mock_call.return_value = OpenRouterResult(content="POSITIVE: a witch at dawn", cost=0.002)

    made = agent.call("generate_prompt", idea="una bruja al amanecer", family="krea-2")

    assert made["prompts"] == [{"positive_prompt": "a witch at dawn", "negative_prompt": ""}]
    assert made["family_name"] == "Krea 2"
    assert agent.call("search_history", query="witch")["total"] == 1

    refused = agent.raw("generate_prompt", idea="x", family="Flux")
    assert refused["isError"] and 'There is no family "Flux". Families: ' in refused["content"][0]["text"]


def test_an_agent_converses_launches_and_looks_at_the_images(agent, tmp_path, fake_comfy):
    workflows.ensure_seeded()
    fake_comfy.model_files["diffusion_models"] = [MODEL]
    families.create_family("Krea", "rules", False, path=str(tmp_path / "families.json"))
    run_call = {"id": "c1", "name": "run_workflow",
                "arguments": json.dumps({"workflow": "krea-2/txt2img", "changes": {"positive": "a witch at dawn"}})}
    script = Script(
        ('<title>Bruja</title>Va:\n<prompt model="krea-2\\krea-2-turbo-official-comfy-int8-convrot_v1.0">\n'
         'POSITIVE: a witch at dawn\n</prompt>', [run_call]),
        ('<review image="1" verdict="ok">clean</review>Ha salido bien: #1.', []),  # its look at the results
        ("Me alegro de que te guste.", []),
    )

    with patch("app.conversation_turn.stream_chat", script):
        started = agent.call("start_conversation", message="Hazme una bruja", family="Krea")
        conversation_id = started["id"]
        [reply] = [m for m in started["messages"] if m["role"] == "assistant"]
        # The prompt shows the model it ran on: the one the bot named (resolved to the installed file).
        assert reply["prompts"] == [{"number": 1, "positive": "a witch at dawn", "negative": "", "models": [MODEL]}]
        assert reply["runs_launched"] == [1] and started["runs"][0]["status"] == "pending"
        assert "wait_for_conversation" in started["next"]

        fake_comfy.finish("prompt-1", "witch.png")
        runs.tracker.poll_once()
        runs.wait_until_idle()
        arrived = agent.call("wait_for_conversation", conversation_id=conversation_id)
        assert arrived["still_working"] is False
        assert [m["role"] for m in arrived["messages"]] == ["assistant", "results", "assistant"]
        results = arrived["messages"][1]
        assert results["images"][0]["number"] == 1
        assert results["images"][0]["review"] == {"verdict": "ok", "note": "clean"}
        assert arrived["runs"][0] | {"seeds": None} == {
            "n": 1, "workflow": "krea-2/txt2img", "status": "done", "count": 1, "base_models": [MODEL],
            "size": "2:3 portrait, 1 MP", "seeds": None, "images": [1], "errors": []}

        seen = agent.raw("get_image", conversation_id=conversation_id, number=1)
        assert seen["content"][0]["type"] == "image"
        assert base64.b64decode(seen["content"][0]["data"]) == fake_comfy.files["witch.png"]

        reacted = agent.call("react_to_image", conversation_id=conversation_id, number=1, rating="up")
        assert reacted["reaction"]["rating"] == "up"
        agent.call("send_message", conversation_id=conversation_id)

    assert "[Reactions: image 1 liked]" in message_text(script.calls[-1]["messages"][-1])
    assert agent.call("get_conversation", conversation_id=conversation_id, last_messages=1)["messages"][0]["text"] \
        == "Me alegro de que te guste."

    # Looking back over the conversation: the evidence alone, or the bot's own balance of it.
    evidence = agent.call("get_balance_data", conversation_id=conversation_id)
    assert evidence["images"] == 1 and evidence["liked"] == 1 and "user liked; you: ok" in evidence["text"]
    with patch("app.conversation_turn.stream_chat", Script(("Lo mejor: #1.", []))):
        took = agent.call("take_stock", conversation_id=conversation_id, focus="la luz")
    asked, answered = took["messages"]
    assert asked["took_stock"]["images"] == 1 and asked["text"] == "la luz"
    assert answered["text"] == "Lo mejor: #1."


def test_a_message_refused_before_anything_is_stored_leaves_no_conversation_behind(agent, tmp_path):
    families.create_family("SDXL", "rules", True, path=str(tmp_path / "families.json"))

    refused = agent.raw("start_conversation", message="mira", family="SDXL", images=["bm90IGFuIGltYWdl"])
    assert refused["isError"] and "PNG, JPEG or WebP" in refused["content"][0]["text"]

    shared = base64.b64encode(png_bytes()).decode()
    with patch("app.conversation_turn.stream_chat", Script(("Bonita.", []))):
        agent.call("start_conversation", message="mira", family="SDXL", images=[f"data:image/png;base64,{shared}"])
    assert len(agent.call("list_conversations")) == 1
