"""The app's features as MCP tools, for agents that work without the frontend.

Model Context Protocol over streamable HTTP at /mcp (stateless, JSON responses), behind the
same Basic Auth as the rest of the app: an agent sends `Authorization: Basic …`. Every tool
calls the app's own HTTP API in-process (httpx2's ASGITransport -- httpx2 is the mcp SDK's own
HTTP client), so what an agent can do, the checks it goes through and the errors it gets are
the frontend's; a refused call comes back as a tool error carrying the API's detail.

`attach(app)` adds the route; `running()`, entered in the app's lifespan, builds the MCP HTTP
app and runs its session manager (which can run only once, hence a fresh one per lifespan).
"""
import base64
import binascii
import json
import logging
import re
import secrets
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

import anyio
import httpx2
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import Field
from starlette.routing import Route

from . import chat
from .comfy_graph import Graph
from .config import get_settings

PATH = "/mcp"
WAIT_POLL_SECONDS = 3.0
MAX_WAIT_SECONDS = 1800
IMAGE_KINDS = {b"\x89PNG": "png", b"\xff\xd8\xff": "jpeg"}

INSTRUCTIONS = """Prompt Enhancer crafts prompts for AI image models in Conversar, a chat with a \
prompt-crafting bot that can also run ComfyUI workflows on the owner's PC and look at the images.

- Conversar: start_conversation (with a model family, list_families), then send_message. The bot replies with prompts (numbered across \
the conversation, each naming the base model it is for) and may launch ComfyUI runs; their images \
arrive later: call wait_for_conversation, then get_image to see them, react_to_image to tell the \
bot what worked. Image N of a conversation is written #N in messages. When a conversation is \
done, finish_conversation ends it, and the bot first looks back over all its runs (what worked, \
the best recipe) and fixes its notes; get_balance_data gives the same evidence without asking it.
- The prompts proposed in every conversation (and those of the one-shot modes the app had before) are in \
the history (search_history).
- Everything else the app's settings page does is here too: families, the bot's notes \
about the user, its personality, the ComfyUI workflow library, installed models, and the log of \
every request sent to OpenRouter.
- Every conversation message spends the owner's OpenRouter credit (or their Claude subscription), \
and runs use the owner's GPU: don't call them in loops without reason."""

READ = ToolAnnotations(read_only_hint=True, open_world_hint=False)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)
LLM = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=True)
DELETE = ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=False)

def _server() -> MCPServer:
    # MCPServer sets up logging for the whole process (logging.basicConfig on the root logger),
    # which would print every app log line twice: undo exactly that, the app has its own.
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    made = MCPServer("prompt-enhancer", title="Prompt Enhancer", instructions=INSTRUCTIONS, version="2.0.0",
                     log_level="WARNING")
    for handler in root.handlers[:]:
        if handler not in handlers:
            root.removeHandler(handler)
    root.setLevel(level)
    return made


server = _server()

_state: dict[str, Any] = {"app": None, "http_app": None}

Family = Annotated[str, Field(description="Model family, by name (e.g. 'Krea 2', 'SDXL') or id: see list_families.")]
ImageData = Annotated[str, Field(description="A PNG, JPEG or WebP image, base64-encoded (a data: URI works too).")]
Temperature = Annotated[float, Field(ge=0, le=2, description="Sampling temperature.")]


# --- wiring ---------------------------------------------------------------------------------------

def attach(app) -> None:
    """Serve the MCP endpoint at PATH; the tools call `app`'s API."""
    _state["app"] = app
    app.router.routes.append(Route(PATH, endpoint=_Endpoint()))


@asynccontextmanager
async def running():
    http_app = server.streamable_http_app(
        streamable_http_path=PATH,
        stateless_http=True,
        json_response=True,
        # The app listens on the home network under whatever address it has; the Basic Auth
        # is the boundary (a rebinding page can't send the credentials).
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    async with server.session_manager.run():
        _state["http_app"] = http_app
        try:
            yield
        finally:
            _state["http_app"] = None


class _Endpoint:
    """An ASGI app (an instance, so Starlette doesn't take it for a request handler): the Basic
    Auth, then the MCP HTTP app of the current lifespan."""

    async def __call__(self, scope, receive, send) -> None:
        if not _authorized(scope):
            await _respond(send, 401, {"detail": "Invalid credentials"},
                           [(b"www-authenticate", b'Basic realm="prompt-enhancer"')])
            return
        http_app = _state["http_app"]
        if http_app is None:
            await _respond(send, 503, {"detail": "The MCP server isn't running"})
            return
        await http_app(scope, receive, send)


def _authorized(scope) -> bool:
    header = dict(scope.get("headers") or []).get(b"authorization", b"").decode("latin-1")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "basic":
        return False
    try:
        username, _, password = base64.b64decode(token, validate=True).decode("utf-8").partition(":")
    except (binascii.Error, UnicodeDecodeError):
        return False
    settings = get_settings()
    return (secrets.compare_digest(username.encode(), settings.admin_username.encode())
            & secrets.compare_digest(password.encode(), settings.admin_password.encode()))


async def _respond(send, status: int, body: dict, headers: list | None = None) -> None:
    data = json.dumps(body).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), *(headers or [])]})
    await send({"type": "http.response.body", "body": data})


# --- the app's API, in-process --------------------------------------------------------------------

async def _call(method: str, path: str, **kwargs) -> httpx2.Response:
    settings = get_settings()
    transport = httpx2.ASGITransport(app=_state["app"])
    async with httpx2.AsyncClient(transport=transport, base_url="http://prompt-enhancer", timeout=None,
                                  auth=(settings.admin_username, settings.admin_password)) as client:
        response = await client.request(method, path, **kwargs)
    if response.status_code >= 400:
        raise ToolError(_detail(response))
    return response


async def _json(method: str, path: str, **kwargs) -> Any:
    response = await _call(method, path, **kwargs)
    return response.json() if response.content else None


def _detail(response: httpx2.Response) -> str:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    if isinstance(detail, list):  # a validation error: where and why
        detail = "; ".join(f"{'.'.join(str(p) for p in item.get('loc', [])[1:])}: {item.get('msg')}" for item in detail)
    return f"{detail or response.text or 'Request failed'} (HTTP {response.status_code})"


def _reply(response: httpx2.Response) -> dict:
    """The outcome of a streamed reply (NDJSON events): the stored message, or a tool error."""
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    final = next((e for e in reversed(events) if e.get("type") in ("done", "error")), None)
    if final is None:
        raise ToolError("The reply ended without an answer; get_conversation shows what was stored.")
    if final["type"] == "error":
        request = f" (request {final['request_id']}: get_request shows it)" if final.get("request_id") else ""
        raise ToolError(f"The reply failed: {final.get('detail')}{request}")
    return final


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


async def _family_id(family: str) -> str:
    families = await _json("GET", "/api/admin/families")
    found = next((f for f in families if f["id"] == family or _key(f["name"]) == _key(family)), None)
    if found is None:
        raise ToolError(f'There is no family "{family}". Families: {", ".join(f["name"] for f in families)}.')
    return found["id"]


async def _find(path: str, what: str, name_or_id: str) -> dict:
    items = await _json("GET", path)
    found = next((i for i in items if i["id"] == name_or_id or i.get("name") == name_or_id), None)
    if found is None:
        raise ToolError(f'There is no {what} "{name_or_id}".')
    return found


def _image_file(data: str, index: int = 1) -> tuple[str, bytes, str]:
    data = (data or "").strip()
    if data.startswith("data:"):
        data = data.partition(",")[2]
    try:
        raw = base64.b64decode(data, validate=False)
    except (binascii.Error, ValueError):
        raise ToolError("An image must be base64-encoded.")
    kind = next((k for magic, k in IMAGE_KINDS.items() if raw.startswith(magic)), None)
    if kind is None and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        kind = "webp"
    if kind is None:
        raise ToolError("An image must be a PNG, JPEG or WebP file.")
    return f"image-{index}.{'jpg' if kind == 'jpeg' else kind}", raw, f"image/{kind}"


def _defaults(llm_model: str | None) -> str:
    return llm_model or get_settings().default_llm_model


# --- conversations, as an agent reads them ---------------------------------------------------------

def _squash(text: str | None) -> str:
    return " ".join((text or "").split()).lower()


def _image_view(image: dict, numbers: dict[str, int]) -> dict:
    view = {"number": numbers.get(image["id"]), "id": image["id"]}
    for key in ("review", "reaction"):
        if image.get(key):
            view[key] = {k: v for k, v in image[key].items() if k not in ("at", "sent")}
    if image.get("metadata"):
        view["has_generation_metadata"] = True
    return view


def _conversation_view(conversation: dict, since: int = 0, last: int | None = None) -> dict:
    messages = conversation.get("messages") or []
    runs = conversation.get("runs") or []
    numbers = chat.image_numbers(messages)
    start = max(since, len(messages) - last) if last else since
    ran: dict[str, list[str]] = {}
    for run in runs:
        positive = _squash((run.get("changes") or {}).get("positive"))
        if positive and run.get("kind") != "calibration":
            ran.setdefault(positive, [])
            ran[positive] += [m for m in run.get("base_models") or [] if m not in ran[positive]]
    proposal = 0
    shown = []
    for index, message in enumerate(messages):
        role = message["role"]
        view: dict[str, Any] = {"index": index, "role": role, "at": message.get("timestamp")}
        if role == "assistant":
            texts, prompt_views, launched = [], [], []
            for part in message.get("parts") or []:
                if part["type"] == "text":
                    texts.append(part["text"])
                elif part["type"] == "prompt":
                    proposal += 1
                    model = ran.get(_squash(part["positive_prompt"])) or ([part["model"]] if part.get("model") else [])
                    prompt_views.append({"number": proposal, "positive": part["positive_prompt"],
                                         "negative": part["negative_prompt"], "models": model})
                elif part["type"] == "run":
                    launched.append(part["run"])
                elif part["type"] == "family":
                    texts.append(f"[Switched to the {part['family']} family]")
                elif part["type"] == "tool" and part.get("error"):
                    texts.append(f"[{part['tool']} failed: {part['error']}]")
            view.update(text="\n\n".join(texts), prompts=prompt_views, runs_launched=launched,
                        notes=message.get("notes") or [], cost=message.get("cost"))
            for key in ("automatic", "error", "truncated"):
                if message.get(key):
                    view[key] = message[key]
        elif role == "results":
            view.update(run=message.get("run"), images=[_image_view(i, numbers) for i in message.get("images") or []])
        else:
            view.update(text=message.get("content") or "",
                        images=[_image_view(i, numbers) for i in message.get("images") or []])
            if message.get("reactions"):
                view["reactions"] = message["reactions"]
            if message.get("balance"):
                view["took_stock"] = message.get("balance_counts") or True  # asked the bot to take stock
        if index >= start:
            shown.append(view)
    return {
        "id": conversation["id"],
        "title": conversation.get("title") or "",
        "family_id": conversation.get("family_id"),
        "llm_model": conversation.get("llm_model"),
        "temperature": conversation.get("temperature"),
        "auto_comment": conversation.get("auto_comment", True),
        "initiative_images": conversation.get("initiative_images", 24),
        "cost": conversation.get("cost"),
        "replying": conversation.get("replying", False),
        "commenting": conversation.get("commenting", False),
        "autonomy": conversation.get("autonomy"),
        "finished_at": conversation.get("finished_at"),
        "message_count": len(messages),
        "messages": shown,
        "runs": [_run_view(r, numbers) for r in runs],
    }


def _run_view(run: dict, numbers: dict[str, int]) -> dict:
    jobs = run.get("jobs") or []
    view = {"n": run["n"], "workflow": run.get("workflow"), "status": run.get("status"), "count": run.get("count"),
            "base_models": run.get("base_models") or [], "size": run.get("size"),
            "seeds": [j.get("seed") for j in jobs],
            "images": [numbers[i] for j in jobs for i in j.get("images") or [] if i in numbers],
            "errors": sorted({j["error"] for j in jobs if j.get("error")})}
    if run.get("kind") == "calibration":
        view.update(kind="calibration", lora=run.get("lora"),
                    weights=[j.get("calibration_weight") for j in jobs])
    return view


def _busy(conversation: dict) -> bool:
    return bool(conversation.get("replying") or conversation.get("commenting")
                or any(r.get("status") == "pending" for r in conversation.get("runs") or [])
                or (conversation.get("autonomy") or {}).get("active"))


async def _conversation(conversation_id: str) -> dict:
    return await _json("GET", f"/api/conversations/{conversation_id}")


async def _send(conversation_id: str, message: str, images: list[str], path: str = "messages",
                field: str = "content") -> dict:
    files = [("images", _image_file(data, i)) for i, data in enumerate(images or [], 1)]
    before = len((await _conversation(conversation_id))["messages"])
    response = await _call("POST", f"/api/conversations/{conversation_id}/{path}",
                           data={field: message or ""}, files=files or None)
    _reply(response)
    conversation = await _conversation(conversation_id)
    view = _conversation_view(conversation, since=before)
    pending = [r["n"] for r in view["runs"] if r["status"] == "pending"]
    if pending:
        view["next"] = "Runs launched: their images arrive later. Call wait_for_conversation."
    return view


# --- models -------------------------------------------------------------------------------------

@server.tool(annotations=READ)
async def list_llm_models(
    search: Annotated[str, Field(description="Words in the model's id or name, optional.")] = "",
    images: Annotated[bool, Field(description="Only models that take images.")] = False,
    tools: Annotated[bool, Field(description="Only models that call tools (Conversar's bot runs ComfyUI "
                                             "only with one).")] = False,
    limit: Annotated[int, Field(ge=1, le=500)] = 50,
) -> dict:
    """The OpenRouter models the app can use, and the default it uses when none is given."""
    words = search.lower().split()
    models = [m for m in await _json("GET", "/api/openrouter-models")
              if all(w in f"{m['id']} {m.get('name', '')}".lower() for w in words)
              and (not images or m.get("supports_images")) and (not tools or m.get("supports_tools"))]
    settings = get_settings()
    return {"default_model": settings.default_llm_model, "total": len(models), "models": models[:limit]}


# --- history --------------------------------------------------------------------------------------

@server.tool(annotations=READ)
async def search_history(
    query: Annotated[str, Field(description="Words to find in ideas, prompts and settings, optional.")] = "",
    mode: Annotated[Literal["generate", "image", "story", "variations", "conversation", "iterate"] | None,
                    Field(description="Only entries of this mode.")] = None,
    family: Annotated[str | None, Field(description="Only entries of this family (name).")] = None,
    favorites_only: bool = False,
    limit: Annotated[int, Field(ge=1, le=200)] = 20,
    offset: Annotated[int, Field(ge=0)] = 0,
) -> dict:
    """Past generations, newest first, with their prompts and cost (prompts proposed in Conversar included)."""
    words = query.lower().split()
    entries = [
        e for e in await _json("GET", "/api/history")
        if (not mode or e.get("mode") == mode)
        and (not family or _key(e.get("family_name") or "") == _key(family))
        and (not favorites_only or e.get("favorite"))
        and all(w in json.dumps([e.get("user_input"), e.get("prompts"), e.get("params")], ensure_ascii=False).lower()
                for w in words)
    ]
    return {"total": len(entries), "entries": entries[offset:offset + limit]}


@server.tool(annotations=WRITE)
async def set_history_favorite(entry_id: str, favorite: bool = True) -> dict:
    """Star or unstar a history entry."""
    return await _json("PATCH", f"/api/history/{entry_id}", json={"favorite": favorite})


@server.tool(annotations=DELETE)
async def delete_history_entry(entry_id: str) -> dict:
    """Delete a history entry for good."""
    return await _json("DELETE", f"/api/history/{entry_id}")


# --- Conversar ------------------------------------------------------------------------------------

@server.tool(annotations=READ)
async def list_conversations() -> list[dict]:
    """Conversations, most recently updated first: title, what is under way in each, cost."""
    return await _json("GET", "/api/conversations")


@server.tool(annotations=LLM)
async def start_conversation(
    message: Annotated[str, Field(description="The first message to the bot, in any language.")],
    family: Family,
    images: Annotated[list[str], Field(max_length=4, description=(
        "Up to 4 images to share, base64-encoded; images made by ComfyUI keep their workflow, which "
        "the bot can rerun."))] = [],
    llm_model: Annotated[str | None, Field(description="OpenRouter model; it runs ComfyUI only if it calls "
                                                       "tools.")] = None,
    temperature: Temperature = 0.7,
    auto_comment: Annotated[bool, Field(description="Whether the bot looks at a run's images by itself when "
                                                    "they arrive (one request more per run).")] = True,
    initiative_images: Annotated[int, Field(ge=0, le=100, description=(
        "How many images the bot may make on its own after each message -- launching, looking and launching the "
        "next batch without waiting -- (0: only what it is asked for)."))] = 24,
) -> dict:
    """Start a conversation with the prompt-crafting bot and get its first reply."""
    conversation = await _json("POST", "/api/conversations", json={
        "family_id": await _family_id(family), "llm_model": _defaults(llm_model), "temperature": temperature,
        "auto_comment": auto_comment, "initiative_images": initiative_images})
    try:
        return await _send(conversation["id"], message, images)
    except ToolError:
        if not (await _conversation(conversation["id"]))["messages"]:
            await _call("DELETE", f"/api/conversations/{conversation['id']}")  # refused before anything was stored
        raise


@server.tool(annotations=LLM)
async def send_message(
    conversation_id: str,
    message: Annotated[str, Field(description="The message; may be empty when it only carries reactions "
                                              "(react_to_image) or images. A Civitai image link "
                                              "(civitai.com/images/N) brings that image and its recipe.")] = "",
    images: Annotated[list[str], Field(max_length=4, description="Up to 4 images, base64-encoded.")] = [],
) -> dict:
    """Send a message in a conversation and wait for the bot's reply (its prompts, the runs it launched)."""
    return await _send(conversation_id, message, images)


@server.tool(annotations=LLM)
async def finish_conversation(
    conversation_id: str,
    focus: Annotated[str, Field(max_length=2000, description=(
        "Optional: what the bot's balance should look at above all (\"sobre todo la luz\"), in any language."))] = "",
) -> dict:
    """End a conversation ("Terminar" in the chat; writing again reopens it). When its runs made images, the bot
    first takes stock of it: the app hands it every image with its recipe, the bot's review and the owner's
    reaction, and counts per ingredient; the bot says what worked best and worst and why, the best recipe and
    what to try next, and brings its notes in line with that (the note changes come back with the reply). One
    reply, no runs launched."""
    conversation = await _conversation(conversation_id)
    has_images = any(m["role"] == "results" and m.get("images") for m in conversation["messages"])
    if not has_images:
        await _json("POST", f"/api/conversations/{conversation_id}/finish", data={"focus": focus})
        return _conversation_view(await _conversation(conversation_id), last=1)
    return await _send(conversation_id, focus, [], path="finish", field="focus")


@server.tool(annotations=READ)
async def get_balance_data(conversation_id: str) -> dict:
    """The evidence for taking stock of a conversation, without asking its bot: every image the runs made there
    (by number) with its recipe -- workflow, the proposal whose prompt it ran, base model, LoRAs and weights,
    sampler settings, size, seed --, the bot's review and the owner's reaction; then counts per ingredient
    (liked, disliked, unrated, flawed). As text, plus the totals."""
    return await _json("GET", f"/api/conversations/{conversation_id}/balance")


@server.tool(annotations=READ)
async def get_conversation(
    conversation_id: str,
    last_messages: Annotated[int, Field(ge=0, description="How many of the latest messages to include "
                                                          "(0 for all).")] = 20,
) -> dict:
    """A conversation as it stands: its settings, messages (the bot's prompts numbered, with the base model each
    is for; images numbered, with the bot's review and the owner's reaction), runs, and what is under way."""
    return _conversation_view(await _conversation(conversation_id), last=last_messages or None)


@server.tool(annotations=READ)
async def wait_for_conversation(
    conversation_id: str,
    timeout_seconds: Annotated[int, Field(ge=1, le=MAX_WAIT_SECONDS)] = 600,
    since_message: Annotated[int | None, Field(ge=0, description=(
        "Return the messages from this index on (message_count from an earlier call); by default, everything "
        "after your last message."))] = None,
) -> dict:
    """Wait until nothing is under way in a conversation -- runs finished, the bot done commenting on their
    images, its work on its own (keep_going) over -- and return the messages that arrived."""
    idle_polls = 0
    with anyio.move_on_after(timeout_seconds) as scope:
        while True:
            conversation = await _conversation(conversation_id)
            # Twice in a row: a run that just finished hands over to the bot's comment.
            idle_polls = 0 if _busy(conversation) else idle_polls + 1
            if idle_polls >= 2:
                break
            await anyio.sleep(WAIT_POLL_SECONDS)
    conversation = await _conversation(conversation_id)
    if since_message is None:
        users = [i for i, m in enumerate(conversation["messages"]) if m["role"] == "user"]
        since_message = users[-1] + 1 if users else 0
    view = _conversation_view(conversation, since=since_message)
    view["still_working"] = scope.cancelled_caught
    return view


@server.tool(annotations=READ)
async def get_image(
    conversation_id: str,
    number: Annotated[int, Field(ge=1, description="The image's number in the conversation (#N).")],
    full_size: Annotated[bool, Field(description="The original instead of the reduced copy.")] = False,
) -> Image:
    """See image #N of a conversation."""
    messages = (await _conversation(conversation_id))["messages"]
    image = chat.image_by_number(messages, number)
    if image is None:
        raise ToolError(f"There is no image {number} in this conversation.")
    response = await _call("GET", f"/api/conversations/{conversation_id}/images/{image['id']}",
                           params={"small": "false" if full_size else "true"})
    kind = response.headers.get("content-type", "image/png").split(";")[0].removeprefix("image/")
    return Image(data=response.content, format=kind)


@server.tool(annotations=WRITE)
async def react_to_image(
    conversation_id: str,
    number: Annotated[int, Field(ge=1, description="The image's number (#N).")],
    rating: Annotated[Literal["up", "down"] | None, Field(description="Liked, disliked, or None to clear.")],
    comment: Annotated[str, Field(description="Why; required for a thumbs down.")] = "",
) -> dict:
    """React to an image a run made. Reactions reach the bot with the next message (send_message, which may
    then be empty)."""
    messages = (await _conversation(conversation_id))["messages"]
    image = chat.image_by_number(messages, number)
    if image is None:
        raise ToolError(f"There is no image {number} in this conversation.")
    stored = await _json("PUT", f"/api/conversations/{conversation_id}/images/{image['id']}/reaction",
                         json={"rating": rating, "comment": comment})
    return _image_view(stored, chat.image_numbers(messages))


@server.tool(annotations=LLM)
async def retry_reply(conversation_id: str) -> dict:
    """Ask the bot again to answer the last message, after a failed reply, or to comment on results it
    hasn't commented on."""
    before = len((await _conversation(conversation_id))["messages"])
    _reply(await _call("POST", f"/api/conversations/{conversation_id}/reply"))
    return _conversation_view(await _conversation(conversation_id), since=before)


@server.tool(annotations=WRITE)
async def update_conversation(
    conversation_id: str,
    title: str | None = None,
    family: Annotated[str | None, Field(description="Family name or id.")] = None,
    llm_model: str | None = None,
    temperature: Annotated[float | None, Field(ge=0, le=2)] = None,
    auto_comment: bool | None = None,
    initiative_images: Annotated[int | None, Field(ge=0, le=100)] = None,
) -> dict:
    """Change a conversation's title or settings; what is left out stays as it is."""
    changes: dict[str, Any] = {"title": title, "llm_model": llm_model, "temperature": temperature,
                               "auto_comment": auto_comment, "initiative_images": initiative_images}
    if family is not None:
        changes["family_id"] = await _family_id(family)
    await _json("PATCH", f"/api/conversations/{conversation_id}",
                json={k: v for k, v in changes.items() if v is not None})
    return _conversation_view(await _conversation(conversation_id), last=1)


@server.tool(annotations=DELETE)
async def delete_conversation(conversation_id: str) -> dict:
    """Delete a conversation with its images for good (its pending runs are cancelled)."""
    return await _json("DELETE", f"/api/conversations/{conversation_id}")


@server.tool(annotations=WRITE)
async def cancel_run(conversation_id: str, run: Annotated[int, Field(ge=1)]) -> dict:
    """Cancel a pending ComfyUI run (it also ends the bot's work on its own)."""
    return await _json("POST", f"/api/conversations/{conversation_id}/runs/{run}/cancel")


@server.tool(annotations=WRITE)
async def stop_autonomy(conversation_id: str) -> dict:
    """Stop the bot working on its own toward a goal (keep_going); runs already launched go on."""
    return await _json("POST", f"/api/conversations/{conversation_id}/autonomy/stop")


@server.tool(annotations=WRITE)
async def retry_run_downloads(conversation_id: str, run: Annotated[int, Field(ge=1)]) -> dict:
    """Fetch again the images of a run whose downloads from ComfyUI were given up."""
    return await _json("POST", f"/api/conversations/{conversation_id}/runs/{run}/retry-downloads")


@server.tool(annotations=READ)
async def get_run_workflow(
    conversation_id: str,
    run: Annotated[int, Field(ge=1)],
    job: Annotated[int, Field(ge=1, description="Which of the run's jobs (one per image).")] = 1,
) -> dict:
    """The exact API-format ComfyUI workflow a run sent."""
    return await _json("GET", f"/api/conversations/{conversation_id}/runs/{run}/workflow", params={"job": job})


# --- families, notes, the bot's personality --------------------------------------------------

@server.tool(annotations=READ)
async def list_families() -> list[dict]:
    """The model families prompts are written for, with their rules and whether they use a negative prompt."""
    return await _json("GET", "/api/admin/families")


@server.tool(annotations=WRITE)
async def save_family(
    name: str,
    instructions: Annotated[str, Field(description="How to write prompts for this family's models.")],
    has_negative_prompt: bool,
    family_id: Annotated[str | None, Field(description="To change an existing one; omit to create.")] = None,
) -> dict:
    """Create a model family, or change one."""
    body = {"name": name, "instructions": instructions, "has_negative_prompt": has_negative_prompt}
    if family_id:
        return await _json("PUT", f"/api/admin/families/{family_id}", json=body)
    return await _json("POST", "/api/admin/families", json=body)


@server.tool(annotations=DELETE)
async def delete_family(family: Family) -> dict:
    """Delete a model family."""
    return await _json("DELETE", f"/api/admin/families/{await _family_id(family)}")


@server.tool(annotations=READ)
async def list_notes() -> list[dict]:
    """The Conversar bot's notes about the owner's tastes and what works with each model."""
    return await _json("GET", "/api/admin/memory")


@server.tool(annotations=WRITE)
async def save_note(
    text: str,
    note_id: Annotated[str | None, Field(description="To rewrite an existing note; omit to add one.")] = None,
) -> dict:
    """Add a note to the bot's memory, or rewrite one."""
    if note_id:
        return await _json("PUT", f"/api/admin/memory/{note_id}", json={"text": text})
    return await _json("POST", "/api/admin/memory", json={"text": text})


@server.tool(annotations=DELETE)
async def delete_note(note_id: str) -> dict:
    """Delete a note from the bot's memory."""
    return await _json("DELETE", f"/api/admin/memory/{note_id}")


@server.tool(annotations=READ)
async def get_personality(
    original: Annotated[bool, Field(description="The text the app ships with, instead of the current one.")] = False,
) -> dict:
    """The Conversar bot's personality (Ajustes): who it is and how it talks. The rest of its instructions
    -- how to deliver prompts, notes, tools -- are the app's own."""
    return await _json("GET", "/api/admin/system-prompt/conversation" + ("/default" if original else ""))


@server.tool(annotations=WRITE)
async def set_personality(text: str) -> dict:
    """Replace the Conversar bot's personality, for both engines (OpenRouter and Claude Code). Write it in English."""
    return await _json("PUT", "/api/admin/system-prompt/conversation", json={"text": text})


# --- ComfyUI: workflows and installed models ------------------------------------------------------

@server.tool(annotations=READ)
async def list_workflows() -> list[dict]:
    """The ComfyUI workflow library the bot runs from: name, description, base model, default size."""
    out = []
    for item in await _json("GET", "/api/admin/workflows"):
        graph = Graph(item["workflow"])
        out.append({"id": item["id"], "name": item["name"], "description": item.get("description", ""),
                    "base_models": graph.base_models(), "size": graph.shape()})
    return out


@server.tool(annotations=READ)
async def get_workflow(workflow: Annotated[str, Field(description="Name or id.")]) -> dict:
    """A library workflow in full (API format)."""
    return await _find("/api/admin/workflows", "workflow", workflow)


@server.tool(annotations=WRITE)
async def save_workflow(
    name: Annotated[str, Field(description="Usually '<family folder>/<what it does>', e.g. 'krea-2/txt2img'.")],
    workflow: Annotated[dict, Field(description="The workflow in ComfyUI's API format.")],
    description: str = "",
    workflow_id: Annotated[str | None, Field(description="To replace an existing one; omit to add.")] = None,
) -> dict:
    """Add a workflow to the library, or replace one."""
    body = {"name": name, "description": description, "workflow": workflow}
    if workflow_id:
        return await _json("PUT", f"/api/admin/workflows/{workflow_id}", json=body)
    return await _json("POST", "/api/admin/workflows", json=body)


@server.tool(annotations=DELETE)
async def delete_workflow(workflow: Annotated[str, Field(description="Name or id.")]) -> dict:
    """Remove a workflow from the library."""
    found = await _find("/api/admin/workflows", "workflow", workflow)
    return await _json("DELETE", f"/api/admin/workflows/{found['id']}")


@server.tool(annotations=READ)
async def extract_workflow_from_image(image: ImageData) -> dict:
    """The API-format workflow embedded in an image ComfyUI generated, with its generation metadata."""
    return await _json("POST", "/api/admin/workflows/extract", files=[("image", _image_file(image))])


@server.tool(annotations=READ)
async def list_installed_models(
    search: Annotated[str, Field(description="Words in the file name, optional.")] = "",
    folder: Annotated[Literal["checkpoints", "diffusion_models", "loras"] | None, Field()] = None,
) -> dict:
    """The checkpoints, diffusion models and LoRAs installed in ComfyUI, with the owner's description of each,
    LoRA base models and trigger words, past use and how the images made with each went."""
    view = await _json("GET", "/api/admin/catalog")
    words = search.lower().split()
    view["items"] = [i for i in view["items"] if (not folder or i["folder"] == folder)
                     and all(w in i["file"].lower() for w in words)]
    return view


@server.tool(annotations=WRITE)
async def refresh_installed_models() -> dict:
    """Read the installed models and LoRAs from ComfyUI again (the app does it daily by itself)."""
    view = await _json("POST", "/api/admin/catalog/refresh")
    return {"refreshed_at": view["refreshed_at"], "items": len(view["items"])}


@server.tool(annotations=WRITE)
async def describe_model(
    file: Annotated[str, Field(description="The file as list_installed_models names it.")],
    text: Annotated[str, Field(description="What it does and how to use it; '' clears it.")],
) -> dict:
    """Write the owner's description of a model or LoRA, which the bot reads."""
    return await _json("PUT", "/api/admin/catalog/description", json={"file": file, "text": text})


# --- the request log ------------------------------------------------------------------------------

@server.tool(annotations=READ)
async def list_requests(
    limit: Annotated[int, Field(ge=1, le=200)] = 50,
    before: Annotated[str | None, Field(description="The last id of the previous page, for the next one.")] = None,
    mode: Annotated[str | None, Field(description="e.g. 'conversation', 'generate', 'variations'.")] = None,
    errors_only: bool = False,
) -> dict:
    """The requests sent to OpenRouter, newest first: mode, model, tokens, cost, timing, errors."""
    params = {"limit": limit, "errors": str(errors_only).lower()}
    params.update({k: v for k, v in (("before", before), ("mode", mode)) if v})
    return await _json("GET", "/api/requests", params=params)


@server.tool(annotations=READ)
async def get_request(request_id: str) -> dict:
    """One request to OpenRouter in full: exactly what was sent and what came back."""
    return await _json("GET", f"/api/requests/{request_id}")
