"""The bot's seat, over MCP: the tools Conversar's bot has (chat_tools.TOOLS, run by the same
ToolBox: the same isolation, the same checks, the same run cards), plus two lookups -- an image
of the conversation again, the evidence for taking stock -- served to a Claude Code turn
(claude_code.py) at /mcp/bot/<token>.

A token is one turn's: it opens that conversation's tools only, and only while the turn lasts,
so the Claude Code process can do nothing else with it -- no other conversation, none of the
app's settings. The app's Basic Auth doesn't cover the route (the CLI is handed this URL, not
the password): the token is the credential, unguessable and short-lived.

A tool call comes in over HTTP while the turn's own thread is still reading what the CLI
prints. The call waits (briefly) until that thread has read the tool_use it answers, so the
reply's text before it -- the <prompt> block naming the model the prompt runs on, the title
being written -- is known when the tool runs, and its card goes into the chat after that text.
"""
import base64
import io
import json
import logging
import secrets
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import anyio
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.routing import Route

from . import balance, catalog, chat, chat_tools, conversations, image_metadata

logger = logging.getLogger("app.conversations")

PATH = "/mcp/bot/{token}"
# How long a call waits for the turn to read the tool_use it answers, and the turn for its result.
SEEN_WAIT_SECONDS = 5
DONE_WAIT_SECONDS = 5
# Tools that only look: not part of the reply as stored (nor replayed on OpenRouter, which
# doesn't have them).
LOOKUPS = ("get_image", "get_balance_data", "get_prompt")
# Zooming in (get_image with regions): crops of the original, at least ZOOM_MIN_SIDE a side (a small
# crop scaled up, so the model looks at it closely) and at most ZOOM_MAX_SIDE (what it takes).
MAX_REGIONS = 6
ZOOM_MIN_SIDE = 768
ZOOM_MAX_SIDE = 1568
MIN_REGION_PERCENT = 2

LOOKUP_TOOLS = [
    types.Tool(
        name="get_image",
        description="Look at an image of this conversation again, by its number (#N): the ones still attached "
                    "come with the messages; this shows any of them, with the generation metadata it carries. "
                    "With regions, zoom in instead: each region comes cropped from the original at its full "
                    "detail (hands, faces, joints, anything doubtful).",
        input_schema={"type": "object", "properties": {
            "number": {"type": "integer", "minimum": 1, "description": "The image's number in the conversation."},
            "full_size": {"type": "boolean", "description": "The original instead of the reduced copy."},
            "regions": {"type": "array", "maxItems": MAX_REGIONS, "description": (
                "Parts to zoom in on, each in percent of the image's width and height from its top left "
                "corner, e.g. a hand low on the left: {\"x\": 5, \"y\": 60, \"width\": 25, \"height\": 25}."),
                "items": {"type": "object", "properties": {
                    "x": {"type": "number"}, "y": {"type": "number"},
                    "width": {"type": "number"}, "height": {"type": "number"}},
                    "required": ["x", "y", "width", "height"]}},
        }, "required": ["number"]},
    ),
    types.Tool(
        name="get_balance_data",
        description="Every image the runs made in this conversation, with its recipe (workflow, the proposal "
                    "whose prompt it ran, base model, LoRAs and weights, sampler settings, size, seed), your "
                    "review and the user's reaction, then counts per ingredient.",
        input_schema={"type": "object", "properties": {}},
    ),
    *[types.Tool(name=t["function"]["name"], description=t["function"]["description"],
                 input_schema=t["function"]["parameters"]) for t in chat_tools.TOOLS
      if t["function"]["name"] == "get_prompt"],
]


def _key(arguments: dict) -> str:
    return json.dumps(arguments or {}, sort_keys=True, ensure_ascii=False)


@dataclass
class Seat:
    """One turn's tools: its conversation, its ToolBox (None without ComfyUI), and what the
    turn has read of the reply so far."""

    conversation_id: str
    toolbox: chat_tools.ToolBox | None
    # A turn reads the CLI's output (claude_code.run): calls wait for it to read each tool_use.
    driven: bool = True
    # False: the turn mustn't run anything (an automatic look at results, taking stock); lookups still work.
    tools_allowed: bool = True
    token: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    text: str = ""  # the reply's text before the latest call read
    seen: list[str] = field(default_factory=list)  # calls read, not yet run: "name key"
    done: list[dict] = field(default_factory=list)  # calls run, not yet claimed: {"name", "key", "result", "part"}
    condition: threading.Condition = field(default_factory=threading.Condition)

    def saw_call(self, name: str, arguments: dict, text: str) -> None:
        """The turn read a tool_use (the reply's text up to it)."""
        with self.condition:
            self.text = text
            self.seen.append(f"{name} {_key(arguments)}")
            self.condition.notify_all()

    def _wait_seen(self, name: str, arguments: dict) -> None:
        if not self.driven:
            return
        wanted = f"{name} {_key(arguments)}"
        deadline = time.monotonic() + SEEN_WAIT_SECONDS
        with self.condition:
            while wanted not in self.seen:
                left = deadline - time.monotonic()
                if left <= 0:
                    logger.warning("tool call not seen in the turn's output conversation_id=%s tool=%s",
                                   self.conversation_id, name)
                    return
                self.condition.wait(left)
            self.seen.remove(wanted)

    def run(self, name: str, arguments: dict) -> tuple[dict, dict | None]:
        """Run one of the bot's tools (in a worker thread): (result for Claude, part for the chat)."""
        self._wait_seen(name, arguments)
        if self.toolbox is None:
            result, part = {"error": "There is no ComfyUI here: no workflows can run."}, None
        elif not self.tools_allowed:
            result = {"error": "Not on this turn: look and comment, don't run anything."}
            part = {"type": "tool", "tool": name, "error": "no en este turno"}
        else:
            self.toolbox.reply_text = self.text
            title = chat.TITLE_PATTERN.search(self.text)
            self.toolbox.title_hint = " ".join(title.group(1).split()) if title else ""
            result, part = self.toolbox.run(name, json.dumps(arguments, ensure_ascii=False))
        with self.condition:
            self.done.append({"name": name, "key": _key(arguments), "result": result, "part": part})
            self.condition.notify_all()
        return result, part

    def claim(self, name: str, arguments: dict) -> dict | None:
        """The run call whose result the turn just read: {"result", "part"}; None if it never ran here."""
        key = _key(arguments)
        deadline = time.monotonic() + DONE_WAIT_SECONDS
        with self.condition:
            while True:
                for index, item in enumerate(self.done):
                    if item["name"] == name and item["key"] == key:
                        return self.done.pop(index)
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self.condition.wait(left)


_seats: dict[str, Seat] = {}
_seats_guard = threading.Lock()


def open_seat(conversation_id: str, toolbox: chat_tools.ToolBox | None, *, driven: bool = True,
              tools_allowed: bool = True) -> Seat:
    seat = Seat(conversation_id, toolbox, driven=driven, tools_allowed=tools_allowed)
    with _seats_guard:
        _seats[seat.token] = seat
    return seat


def close_seat(seat: Seat) -> None:
    with _seats_guard:
        _seats.pop(seat.token, None)


def url(seat: Seat, base: str) -> str:
    return f"{base}{PATH.format(token=seat.token)}"


# --- the MCP server ---------------------------------------------------------------------------------

def _seat_for(ctx) -> Seat | None:
    request = getattr(ctx, "request", None)
    path = request.url.path if request is not None else ""
    return _seats.get(path.rstrip("/").rsplit("/", 1)[-1])


def _tool_definitions(seat: Seat) -> list[types.Tool]:
    tools = []
    if seat.toolbox is not None:
        for tool in chat_tools.TOOLS:
            function = tool["function"]
            if function["name"] in LOOKUPS:  # listed with the lookups, with or without ComfyUI
                continue
            tools.append(types.Tool(name=function["name"], description=function["description"],
                                    input_schema=function["parameters"]))
    return tools + LOOKUP_TOOLS


async def _list_tools(ctx, params) -> types.ListToolsResult:
    seat = _seat_for(ctx)
    return types.ListToolsResult(tools=_tool_definitions(seat) if seat else [])


def _text(payload, error: bool = False) -> types.CallToolResult:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)], is_error=error)


def _zoomed(path: str, regions) -> list | str:
    """Each region of the image at path, cropped from it and sized for a close look; or what's wrong."""
    from PIL import Image
    if not isinstance(regions, list) or not regions:
        return "regions must be a list of {x, y, width, height}, in percent."
    out: list = []
    with Image.open(path) as original:
        original.load()
        width, height = original.size
        for region in regions[:MAX_REGIONS]:
            try:
                x, y, w, h = (float(region[k]) for k in ("x", "y", "width", "height"))
            except (KeyError, TypeError, ValueError):
                return "Each region needs x, y, width and height, in percent."
            x, y = min(max(x, 0.0), 100 - MIN_REGION_PERCENT), min(max(y, 0.0), 100 - MIN_REGION_PERCENT)
            w, h = min(max(w, MIN_REGION_PERCENT), 100 - x), min(max(h, MIN_REGION_PERCENT), 100 - y)
            box = (round(x * width / 100), round(y * height / 100),
                   round((x + w) * width / 100), round((y + h) * height / 100))
            crop = original.crop(box).convert("RGB")
            side = max(crop.size)
            scale = ZOOM_MIN_SIDE / side if side < ZOOM_MIN_SIDE else min(1.0, ZOOM_MAX_SIDE / side)
            if scale != 1.0:
                crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                                   Image.LANCZOS)
            data = io.BytesIO()
            crop.save(data, "JPEG", quality=92)
            out.append(types.TextContent(type="text", text=(
                f"Region x {x:g}% y {y:g}%, {w:g}% x {h:g}% of the image ({box[2] - box[0]}x{box[3] - box[1]} px "
                f"of the original, shown at {crop.width}x{crop.height}):")))
            out.append(types.ImageContent(type="image", data=base64.b64encode(data.getvalue()).decode("ascii"),
                                          mime_type="image/jpeg"))
    return out


def _image(seat: Seat, number, full_size: bool = False, regions=None) -> types.CallToolResult:
    conversation = conversations.get(seat.conversation_id) or {"messages": []}
    image = chat.image_by_number(conversation["messages"], number) if isinstance(number, int) else None
    if image is None:
        return _text({"error": f"There is no image {number} in this conversation."}, error=True)
    if regions is not None:
        original = conversations.image_file(seat.conversation_id, image["id"])
        if original is None:
            return _text({"error": f"Image {number} is gone."}, error=True)
        zoomed = _zoomed(original[0], regions)
        if isinstance(zoomed, str):
            return _text({"error": zoomed}, error=True)
        return types.CallToolResult(content=[types.TextContent(type="text", text=f"Image {number}, zoomed in:"),
                                             *zoomed])
    found = conversations.image_file(seat.conversation_id, image["id"], small=not full_size)
    if found is None:
        return _text({"error": f"Image {number} is gone."}, error=True)
    path, content_type = found
    with open(path, "rb") as handle:
        data = base64.b64encode(handle.read()).decode("ascii")
    label = f"Image {number}"
    if image.get("metadata"):
        installed = catalog.load()["entries"] or None
        label += ":\n" + image_metadata.describe(image["metadata"], number, installed)
    for key in ("review", "reaction"):
        if image.get(key):
            label += f"\n{key}: " + json.dumps({k: v for k, v in image[key].items() if k not in ("at", "sent")},
                                               ensure_ascii=False)
    return types.CallToolResult(content=[types.TextContent(type="text", text=label),
                                         types.ImageContent(type="image", data=data, mime_type=content_type)])


async def _call_tool(ctx, params) -> types.CallToolResult:
    seat = _seat_for(ctx)
    if seat is None:
        return _text({"error": "This turn is over."}, error=True)
    name, arguments = params.name, dict(params.arguments or {})
    if name == "get_image":
        return await anyio.to_thread.run_sync(lambda: _image(seat, arguments.get("number"),
                                                             bool(arguments.get("full_size")),
                                                             arguments.get("regions")))
    if name == "get_balance_data":
        found = balance.ledger(conversations.get(seat.conversation_id) or {"id": seat.conversation_id,
                                                                           "messages": [], "runs": []})
        return _text(found.text if found else "The runs here have made no images yet.")
    if name == "get_prompt":
        conversation = conversations.get(seat.conversation_id) or {"messages": []}
        number = arguments.get("number")
        found = chat.prompt_by_number(conversation, number) if isinstance(number, int) else None
        return _text(found or {"error": f"There is no prompt {number} in this conversation."}, error=not found)
    if not any(t["function"]["name"] == name for t in chat_tools.TOOLS):
        return _text({"error": f"There is no tool {name}."}, error=True)
    result, _ = await anyio.to_thread.run_sync(seat.run, name, arguments)
    return _text(result, error="error" in result)


_server = Server("prompt-enhancer-bot", version="1.0.0", on_list_tools=_list_tools, on_call_tool=_call_tool)
_state: dict = {"manager": None}


class _Endpoint:
    """An ASGI app (an instance, so Starlette doesn't take it for a request handler): the token,
    then the MCP session manager of the current lifespan."""

    async def __call__(self, scope, receive, send) -> None:
        token = scope["path"].rstrip("/").rsplit("/", 1)[-1]
        manager = _state["manager"]
        if token not in _seats or manager is None:
            status, detail = (404, "No such turn") if token not in _seats else (503, "Not running")
            body = json.dumps({"detail": detail}).encode()
            await send({"type": "http.response.start", "status": status,
                        "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": body})
            return
        await manager.handle_request(scope, receive, send)


def attach(app) -> None:
    app.router.routes.append(Route(PATH, endpoint=_Endpoint()))


@asynccontextmanager
async def running():
    """The session manager runs once: a fresh one per lifespan. DNS rebinding protection is off:
    the CLI calls 127.0.0.1, and the token is the boundary."""
    manager = StreamableHTTPSessionManager(
        app=_server, stateless=True, json_response=True,
        security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    async with manager.run():
        _state["manager"] = manager
        try:
            yield
        finally:
            _state["manager"] = None
