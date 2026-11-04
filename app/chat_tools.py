"""The tools Conversar's bot calls to work with ComfyUI, and what running each one does.

Every tool returns (result, part): the result goes back to the LLM as JSON (English,
like everything it reads), the part is what the chat shows for that call -- data the
frontend labels in Spanish, never text for the user. Errors are results too
({"error": ...}), so the bot can read them and fix its call within the same turn.
"""
import base64
import json
import logging
import random
import re

import requests

from . import catalog, chat, comfy, comfy_policy, conversations, families, prompts, runs, workflows
from .comfy_graph import ASPECTS, SAMPLER_SETTINGS, Graph, GraphError, is_api_workflow, normalize_aspect

logger = logging.getLogger("app.chat_tools")

MAX_RUN_COUNT = 4
# Working on its own (keep_going): images it may make and dollars it may spend on its replies.
AUTONOMY_IMAGES = 8
AUTONOMY_MAX_IMAGES = 100
AUTONOMY_COST = 1.0
# The budget between the user's messages (conversations.initiative): images per conversation's setting,
# up to this many, and this much money on the bot's own replies.
INITIATIVE_MAX_IMAGES = 100
INITIATIVE_COST = 1.0
AUTONOMY_MAX_COST = 10.0
MAX_LISTED_MODELS = 80
MODEL_FOLDERS = list(comfy.MODEL_FOLDERS)
MAX_SEED = 2 ** 48
CALIBRATION_WEIGHTS = [0, 0.25, 0.5, 0.75, 1.0]
MAX_CALIBRATION_WEIGHTS = 6
# comfy-gateway's panel shows a job's tag cut at about 30 characters: the conversation first, at most this long.
TAG_TITLE_CHARS = 40
# Of a reply the user asked for, how many images go to ComfyUI at high priority (they are waiting
# for them); the rest of its batch, at normal.
WAITED_FOR_IMAGES = 2
# Calibrations run at idle priority: only when nothing else wants the GPU, so they may wait long.
CALIBRATION_TIMEOUT_SECONDS = 12 * 3600

WORKFLOW_PARAM = {
    "type": "string",
    "description": 'A library workflow by name, "image:N" for the workflow embedded in image N of this '
                   'conversation, or "run:N" for the exact workflow of run N.',
}
GRAPH_PARAM = {
    "type": "object",
    "description": 'Instead of workflow: a whole workflow you write, in ComfyUI\'s API format -- {"<node id>": '
                   '{"class_type": ..., "inputs": {...}}}, a link written as ["<node id>", <output index>] -- ending '
                   "in a SaveImage node. Only node types available here (describe_nodes lists them and shows each "
                   "one's inputs).",
}
# The model folder each base model input loads from.
LOADER_FOLDERS = {"ckpt_name": "checkpoints", "unet_name": "diffusion_models"}
MAX_DESCRIBED_NODES = 12
MAX_CHOICES = 40

CHANGES_PARAM = {
    "type": "object",
    "description": "Edits applied to a copy of the workflow (the original stays as it is): first add, remove "
                   "and set, then the shortcuts. Leave out whatever stays the same.",
    "properties": {
        "positive": {"type": "string", "description": "The positive prompt, replacing the workflow's."},
        "negative": {"type": "string", "description": "The negative prompt, if the workflow has one."},
        "model": {
            "type": "string",
            "description": "The base model: an installed checkpoint or diffusion model (exact file name), loaded where "
                           "the workflow loads its own. It must suit the loader: a checkpoint where it loads one, a "
                           "diffusion model where it loads one -- of the same family, since the workflow's text "
                           "encoders and VAE are for it.",
        },
        "seed": {
            "type": "string",
            "description": '"random" (the default: a new seed for every image), "keep" (the workflow\'s own '
                           'seed, to reproduce an image) or a number (consecutive numbers when count > 1).',
        },
        "steps": {"type": "integer"},
        "cfg": {"type": "number"},
        "sampler_name": {"type": "string"},
        "scheduler": {"type": "string"},
        "denoise": {"type": "number"},
        "aspect": {
            "type": "string",
            "enum": list(ASPECTS),
            "description": "The image's proportions, at the workflow's own pixel count (what its model is made "
                           "for), in any workflow that makes an image from scratch.",
        },
        "width": {"type": "integer", "description": "With height: an exact size, only when the user asks for one."},
        "height": {"type": "integer"},
        "megapixels": {"type": "number",
                       "description": "Only when the user asks for bigger or smaller images (workflows with a "
                                      "ResolutionSelector)."},
        "loras": {
            "type": "array",
            "description": "The whole LoRA stack, replacing the current one (an empty list removes them all). "
                           "Exact file names.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "strength": {"type": "number"},
                    "strength_clip": {"type": "number", "description": "Defaults to strength."},
                },
                "required": ["name"],
            },
        },
        "input_images": {
            "type": "array",
            "description": "Conversation images fed to the workflow's image inputs (a LoadImage node, or a node "
                           "with a base64_data input).",
            "items": {
                "type": "object",
                "properties": {"node": {"type": "string"}, "image": {"type": "integer", "description": "Its number"}},
                "required": ["node", "image"],
            },
        },
        "set": {
            "type": "array",
            "description": 'Set any node input: a literal value, or a link written as ["node id", output index].',
            "items": {
                "type": "object",
                "properties": {"node": {"type": "string"}, "input": {"type": "string"},
                               "value": {"description": "The new value"}},
                "required": ["node", "input", "value"],
            },
        },
        "add": {
            "type": "array",
            "description": "New nodes. Give each an id of your choice (it must be new) to link to it from other "
                           "inputs or from set.",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "class_type": {"type": "string"},
                               "inputs": {"type": "object"}, "title": {"type": "string"}},
                "required": ["class_type"],
            },
        },
        "remove": {
            "type": "array",
            "description": "Node ids to take out. Whatever used a removed node's output is wired to its matching "
                           "input, like ComfyUI's bypass.",
            "items": {"type": "string"},
        },
    },
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_workflow",
            "description": "Launch a ComfyUI workflow on the user's machine -- an existing one with changes, or one "
                           "you write (graph). Returns at once with the run's number; its images arrive later in the "
                           "conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "workflow": WORKFLOW_PARAM,
                    "graph": GRAPH_PARAM,
                    "changes": CHANGES_PARAM,
                    "count": {"type": "integer", "description": f"Images to make, 1-{MAX_RUN_COUNT} (default 1), "
                                                                "each its own job and seed."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_workflow",
            "description": "A workflow's nodes, with their values and links, and what it does: prompts, sampler "
                           "settings, models, LoRAs, size, image inputs.",
            "parameters": {"type": "object", "properties": {"workflow": WORKFLOW_PARAM}, "required": ["workflow"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_nodes",
            "description": "The node types you can use here; with types, what each takes (inputs: type, default, "
                           "range, choices) and gives (outputs, by index) -- before writing or rewiring nodes.",
            "parameters": {
                "type": "object",
                "properties": {"types": {"type": "array", "items": {"type": "string"},
                                         "description": f"Node class types, at most {MAX_DESCRIBED_NODES}; leave out "
                                                        "to list the available ones."}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_comfy_models",
            "description": "The model files installed in ComfyUI for a folder, optionally filtered.",
            "parameters": {
                "type": "object",
                "properties": {
                    "folder": {"type": "string", "enum": MODEL_FOLDERS},
                    "search": {"type": "string", "description": "Words that must all appear in the file name."},
                },
                "required": ["folder"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_workflow",
            "description": "Keep a workflow -- an existing one with changes, or one you wrote (graph) -- in the "
                           "library under a new name, only when the user wants it kept. replace: true overwrites the "
                           "library workflow with that name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "workflow": WORKFLOW_PARAM,
                    "graph": GRAPH_PARAM,
                    "name": {"type": "string", "description": 'Like the others, e.g. "krea-2/txt2img-ink".'},
                    "description": {"type": "string", "description": "One English sentence: what it does."},
                    "changes": CHANGES_PARAM,
                    "replace": {"type": "boolean"},
                },
                "required": ["name", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "use_family",
            "description": "Switch this conversation to another model family, to write prompts by its rules: when the "
                           "user asks for one, or before working with a model of another family. Returns its rules; "
                           "the conversation keeps that family.",
            "parameters": {
                "type": "object",
                "properties": {"family": {"type": "string", "description": "A family's name, as listed."}},
                "required": ["family"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calibrate_lora",
            "description": "Show what a LoRA does and from which weight it breaks: one image per weight, the same "
                           "prompt and seed, on a workflow of the LoRA's family. Runs at idle priority (when the GPU "
                           "is free), so it can take a while; review its images weight by weight when they arrive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lora": {"type": "string", "description": "The LoRA's file name, as installed."},
                    "prompt": {"type": "string", "description": "The positive prompt for every image, by the "
                                                                "family's rules; include its trigger words."},
                    "workflow": {"type": "string", "description": "The workflow to calibrate on (default: the "
                                                                  "library's plain text-to-image of the LoRA's family)."},
                    "weights": {"type": "array", "items": {"type": "number"},
                                "description": f"Default {CALIBRATION_WEIGHTS}; at most {MAX_CALIBRATION_WEIGHTS}."},
                    "keep_other_loras": {"type": "boolean", "description": "Keep the workflow's other LoRAs (the "
                                                                          "default: how it behaves in the usual stack); "
                                                                          "false shows it alone."},
                },
                "required": ["lora", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_run",
            "description": "Cancel a run still waiting or in progress.",
            "parameters": {"type": "object", "properties": {"run": {"type": "integer"}}, "required": ["run"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "prepare_download",
            "description": (
                "Prepare a download from Civitai of a checkpoint or LoRA the conversation needs and isn't installed: "
                "one a shared Civitai image was made with (its Civitai version), a Civitai model link the user "
                "pasted, or a file hash a reference image gives. It only prepares it: the chat shows the user what "
                "it is, its size and where it would go, and the user decides whether to download it. Don't launch "
                "with it until it is installed (you'll be told)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "version_id": {"type": "integer", "description": "The Civitai model version id."},
                    "url": {"type": "string", "description": "A civitai.com model link (its newest version, unless "
                                                             "the link names one with modelVersionId)."},
                    "hash": {"type": "string", "description": "The file's hash (AutoV2 or SHA256)."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "keep_going",
            "description": (
                "Take on a bigger task of your own, with a goal the user set ('until it looks right', 'keep "
                "going up to 60 images', 'try a lot of things and show me the best'): it replaces the budget "
                "you have between the user's messages until the goal is met or this budget is spent -- "
                "max_images images made, or max_cost dollars spent on your replies. Each batch's images come "
                "back to you, and you launch the next yourself, without waiting for the user. Use the user's "
                "limits; without any, the defaults, and say so. Call it again to change the goal or the limits "
                "(what was used so far still counts), or with stop: true when the user asks you to stop."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "What counts as done, in a sentence."},
                    "max_images": {"type": "integer", "description": f"Default {AUTONOMY_IMAGES}, at most "
                                                                     f"{AUTONOMY_MAX_IMAGES}."},
                    "max_cost": {"type": "number", "description": f"Dollars; default {AUTONOMY_COST:g}, at most "
                                                                  f"{AUTONOMY_MAX_COST:g}."},
                    "stop": {"type": "boolean"},
                },
            },
        },
    },
]


class ToolError(Exception):
    pass


def _number(value, kind=float):
    try:
        return kind(value)
    except (TypeError, ValueError):
        raise ToolError(f"expected a number, got {value!r}")


def _usage(working: dict) -> dict:
    return {"images": f"{working['images']} of {working['max_images']}",
            "dollars": f"{working['cost']:.2f} of {working['max_cost']:.2f}"}


def _squash(text) -> str:
    return " ".join(str(text or "").split()).casefold()


def _link_or_value(value):
    if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int) and not isinstance(value[1], bool):
        return [str(value[0]), value[1]]
    return value


def node_errors_text(node_errors: dict) -> dict:
    out = {}
    for node_id, info in (node_errors or {}).items():
        if not isinstance(info, dict):
            continue
        messages = []
        for error in info.get("errors") or []:
            if isinstance(error, dict):
                text = error.get("message") or ""
                if error.get("details"):
                    text = f"{text}: {error['details']}"
                messages.append(text)
        out[node_id] = {"class_type": info.get("class_type"), "errors": messages}
    return out


def describe_node(info: dict) -> dict:
    """A node type's definition (ComfyUI's /object_info entry) as the bot reads it: inputs with
    their type, default, range and choices, and outputs by index. Choices that are files of
    ComfyUI's input folder (LoadImage lists every image there, other projects' too) never reach
    it -- the isolation (comfy_policy) -- and long lists are cut."""
    def spec(raw) -> dict:
        kind, options = (raw[0], raw[1] if len(raw) > 1 and isinstance(raw[1], dict) else {}) \
            if isinstance(raw, list) and raw else (raw, {})
        choices = kind if isinstance(kind, list) else options.get("options") if kind == "COMBO" else None
        out = {"type": "COMBO" if choices is not None else kind}
        for key in ("default", "min", "max", "step", "multiline", "tooltip"):
            value = options.get(key)
            if value is not None and not (isinstance(value, str) and comfy_policy.looks_like_file(value)):
                out[key] = value
        if isinstance(choices, list):
            if any(isinstance(c, str) and comfy_policy.looks_like_file(c) for c in choices):
                out["choices"] = "images: only this conversation's, fed with input_images"
            else:
                out["choices"] = choices[:MAX_CHOICES]
                if len(choices) > MAX_CHOICES:
                    out["more_choices"] = f"{len(choices) - MAX_CHOICES} more (for model files: list_comfy_models)"
        return out

    inputs = info.get("input") or {}
    return {
        **({"description": info["description"]} if info.get("description") else {}),
        "inputs": {group: {name: spec(raw) for name, raw in (inputs.get(group) or {}).items()}
                   for group in ("required", "optional") if inputs.get(group)},
        "outputs": [{"index": i, "type": kind, "name": name}
                    for i, (kind, name) in enumerate(zip(info.get("output") or [],
                                                         info.get("output_name") or info.get("output") or []))],
    }


class ToolBox:
    def __init__(self, conversation_id: str, client: comfy.Comfy):
        self.conversation_id = conversation_id
        self.client = client
        # The title the bot is giving the conversation in the reply being made, if any:
        # a first run may come before the title is stored, and names the output folder.
        self.title_hint = ""
        # The reply being made, so far: the prompt blocks it wrote before a call say which model
        # each prompt is for.
        self.reply_text = ""
        # A turn the bot takes by itself (its look at a run's results): what it launches then goes behind
        # what someone asked for in ComfyUI's queue.
        self.automatic = False
        # In a turn someone asked for, the images they are waiting for go first; the rest of a big
        # batch goes behind other conversations' few (comfy.PRIORITY_REST).
        self.waited_for = WAITED_FOR_IMAGES

    definitions = TOOLS

    def run(self, name: str, arguments: str) -> tuple[dict, dict]:
        try:
            args = json.loads(arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError
        except ValueError:
            return {"error": "The arguments weren't a JSON object."}, {"type": "tool", "tool": name,
                                                                      "error": "argumentos ilegibles"}
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return {"error": f"There is no tool {name}."}, {"type": "tool", "tool": name, "error": "desconocida"}
        try:
            return handler(**{k: v for k, v in args.items() if isinstance(k, str)})
        except TypeError as e:
            error = f"Bad arguments: {e}"
        except (ToolError, GraphError, workflows.WorkflowError, comfy_policy.PolicyError) as e:
            error = str(e)
        except comfy.Unreachable as e:
            error = str(e)
        except requests.exceptions.Timeout:
            error = ("ComfyUI accepted the connection but isn't answering: it may be busy loading a model, "
                     "or frozen. Try again in a while; if it lasts, ComfyUI needs a restart.")
        except comfy.ComfyError as e:
            result = {"error": f"ComfyUI rejected it: {e}"}
            if e.node_errors:
                result["node_errors"] = node_errors_text(e.node_errors)
            return result, {"type": "tool", "tool": name, "workflow": args.get("workflow"), "error": str(e)}
        except Exception as e:  # a bug or an unexpected answer: tell the bot, keep the turn alive
            logger.exception("tool failed conversation_id=%s tool=%s", self.conversation_id, name)
            error = f"Unexpected error: {e!r}"
        return {"error": error}, {"type": "tool", "tool": name, "workflow": args.get("workflow"), "error": error}

    # --- workflows -------------------------------------------------------------------------------

    def _conversation(self) -> dict:
        conversation = conversations.get(self.conversation_id)
        if conversation is None:
            raise ToolError("The conversation is gone.")
        return conversation

    def _resolve(self, reference: str) -> tuple[Graph, str]:
        reference = (reference or "").strip()
        image = re.fullmatch(r"image\s*:?\s*(\d+)", reference, re.IGNORECASE)
        run = re.fullmatch(r"run\s*:?\s*(\d+)", reference, re.IGNORECASE)
        if image:
            number = int(image.group(1))
            found = chat.image_by_number(self._conversation()["messages"], number)
            if found is None:
                raise ToolError(f"There is no image {number}.")
            workflow = conversations.image_workflow(self.conversation_id, found["id"])
            if not is_api_workflow(workflow):
                hint = (" It embeds only the editor graph."
                        if (found.get("metadata") or {}).get("editor_workflow_only") else "")
                raise ToolError(f"Image {number} has no runnable ComfyUI workflow.{hint}")
            return Graph(workflow), f"image:{number}"
        if run:
            number = int(run.group(1))
            graphs = conversations.run_workflows(self.conversation_id, number)
            if not graphs:
                raise ToolError(f"There is no run {number}.")
            return Graph(graphs[0]), f"run:{number}"
        entry = workflows.find_by_name(reference)
        if entry is None:
            names = ", ".join(w["name"] for w in workflows.list_workflows()) or "none"
            raise ToolError(f'There is no workflow "{reference}". Library: {names}; or image:N, run:N.')
        return Graph(entry["workflow"]), entry["name"]

    def _source(self, workflow, graph) -> tuple[Graph, str]:
        """The workflow a call works on: a reference (_resolve), or a whole one the bot wrote."""
        if graph is None:
            if not workflow:
                raise ToolError("Give workflow (a name, image:N or run:N) or graph (a workflow you write).")
            return self._resolve(str(workflow))
        if workflow:
            raise ToolError("Give workflow or graph, not both.")
        if not is_api_workflow(graph):
            raise ToolError('graph must be a workflow in API format: {"<node id>": {"class_type": ..., '
                            '"inputs": {...}}}.')
        written = Graph({str(node_id): {**node, "inputs": {k: _link_or_value(v)
                                                           for k, v in (node.get("inputs") or {}).items()}}
                         for node_id, node in graph.items()})
        broken = written.broken_links()
        if broken:
            raise ToolError(f"Links to nodes that don't exist: {', '.join(broken)}")
        if not written.find("SaveImage"):
            raise ToolError("Your workflow saves nothing: end it in a SaveImage node.")
        return written, "custom"

    def _set_model(self, graph: Graph, name: str) -> str:
        """Load the installed base model `name` where the workflow loads its own; returns the file."""
        loaders = graph.base_model_loaders()
        if not loaders:
            raise ToolError("This workflow loads no base model to change (no checkpoint or diffusion model loader).")
        kinds = {key for _, key in loaders}
        if len(kinds) > 1:
            raise ToolError("This workflow loads both a checkpoint and a diffusion model: change the one you mean "
                            "with set.")
        folder = LOADER_FOLDERS[kinds.pop()]
        other = "diffusion_models" if folder == "checkpoints" else "checkpoints"
        noun = {"checkpoints": "a checkpoint", "diffusion_models": "a diffusion model"}
        file = catalog.resolve_file(str(name), folder, self.client)
        if file not in self.client.models(folder):
            elsewhere = catalog.resolve_file(str(name), other, self.client)
            if elsewhere in self.client.models(other):
                raise ToolError(f"{elsewhere} is {noun[other]}, but this workflow loads {noun[folder]}: use a "
                                f"workflow that loads {noun[other]} (from the library, an image or a run), or write "
                                "one.")
            raise ToolError(f'"{name}" isn\'t an installed checkpoint or diffusion model: check list_comfy_models.')
        if folder == "diffusion_models":
            current, new = graph.base_model_folder(), catalog.family_of(file)
            if current and new and current != new:
                raise ToolError(f"{file} is a {new} model, and this workflow's text encoders and VAE are for {current}: "
                                f"use a {new} workflow (from the library, an image or a run), or write one.")
        graph.set_base_model(file)
        return file

    def _declared_model(self, graph: Graph) -> str | None:
        """The model the bot's <prompt> block says the prompt this workflow runs is for: in the
        reply being made, or else in an earlier one -- the latest block with this prompt."""
        positive = _squash(graph.generation_info().get("positive"))
        if not positive:
            return None
        for match in reversed(list(chat.PROMPT_PATTERN.finditer(self.reply_text))):
            parsed, _ = prompts.parse_response(match.group(2), True)
            if _squash(parsed) == positive:
                return chat.block_model(match.group(1)) or None
        for message in reversed(self._conversation()["messages"]):
            for part in reversed(message.get("parts") or []) if message["role"] == "assistant" else []:
                if part.get("type") == "prompt" and _squash(part["positive_prompt"]) == positive:
                    return part.get("model")
        return None

    def _apply(self, graph: Graph, changes: dict, base: str | None) -> list[str]:
        """Apply the changes to graph; returns what was done, in short."""
        if not isinstance(changes, dict):
            raise ToolError("changes must be an object")
        done = []
        added = changes.get("add") or []
        for node in added:
            if not isinstance(node, dict):
                raise ToolError("each item of add must be an object")
            inputs = {k: _link_or_value(v) for k, v in (node.get("inputs") or {}).items()}
            node_id = graph.add_node(node.get("class_type"), inputs, node.get("title"), node.get("id"))
            done.append(f"added {node_id} ({node.get('class_type')})")
        for node_id in changes.get("remove") or []:
            node_id = str(node_id)
            try:
                graph.bypass(node_id)
                done.append(f"bypassed {node_id}")
            except GraphError:
                dangling = graph.remove_node(node_id)
                done.append(f"removed {node_id}" + (f", leaving {len(dangling)} input(s) unlinked" if dangling else ""))
        for item in changes.get("set") or []:
            if not isinstance(item, dict) or "node" not in item or "input" not in item:
                raise ToolError("each item of set needs node, input and value")
            graph.set_input(str(item["node"]), str(item["input"]), _link_or_value(item.get("value")))
            done.append(f"set {item['node']}.{item['input']}")
        broken = graph.broken_links()
        if broken:
            raise ToolError(f"Links to nodes that don't exist: {', '.join(broken)}")
        if changes.get("model"):
            done.append(f"model {self._set_model(graph, changes['model'])}")
        if "positive" in changes or "negative" in changes:
            graph.set_prompt(changes.get("positive"), changes.get("negative"))
            done.append("prompt")
        sampler = {k: changes[k] for k in SAMPLER_SETTINGS if changes.get(k) is not None}
        if sampler:
            for key in ("steps",):
                if key in sampler:
                    sampler[key] = _number(sampler[key], int)
            for key in ("cfg", "denoise"):
                if key in sampler:
                    sampler[key] = _number(sampler[key])
            graph.set_sampler(**sampler)
            done.append(", ".join(f"{k} {v}" for k, v in sampler.items()))
        aspect = changes.get("aspect") or changes.get("aspect_ratio")  # aspect_ratio: the older name
        if aspect is not None:
            graph.set_aspect(str(aspect))
            done.append(f"aspect {normalize_aspect(aspect)}")
        if any(changes.get(k) is not None for k in ("width", "height", "megapixels")):
            width, height = changes.get("width"), changes.get("height")
            graph.set_size(
                width=_number(width, int) if width is not None else None,
                height=_number(height, int) if height is not None else None,
                megapixels=_number(changes["megapixels"]) if changes.get("megapixels") is not None else None,
            )
            done.append("size")
        if changes.get("loras") is not None:
            if not isinstance(changes["loras"], list):
                raise ToolError("loras must be a list")
            stack = [
                {**item, "name": catalog.resolve_file(item["name"], "loras", self.client)}
                if isinstance(item, dict) and isinstance(item.get("name"), str) else item
                for item in changes["loras"]
            ]
            graph.set_loras(stack)
            done.append(f"{len(changes['loras'])} LoRA(s)")
        for item in changes.get("input_images") or []:
            if base is None:
                raise ToolError("input_images only apply when running a workflow")
            done.append(self._feed_image(graph, base, item))
        return done

    def _feed_image(self, graph: Graph, base: str, item: dict) -> str:
        if not isinstance(item, dict):
            raise ToolError("each input image needs node and image")
        node_id, number = str(item.get("node")), _number(item.get("image"), int)
        node = graph.node(node_id)
        image = chat.image_by_number(self._conversation()["messages"], number)
        found = conversations.image_file(self.conversation_id, image["id"]) if image else None
        if found is None:
            raise ToolError(f"There is no image {number}.")
        path, content_type = found
        with open(path, "rb") as f:
            data = f.read()
        if "base64_data" in node["inputs"]:
            node["inputs"]["base64_data"] = base64.b64encode(data).decode("ascii")
        elif node["class_type"] in ("LoadImage", "LoadImageMask") or "image" in node["inputs"]:
            filename = comfy_policy.upload_name(self.conversation_id, image["id"], path.rsplit(".", 1)[-1])
            node["inputs"]["image"] = self.client.upload_image(base, data, filename, content_type)
        else:
            raise ToolError(f"Node {node_id} ({node['class_type']}) takes no image")
        return f"image {number} into {node_id}"

    @staticmethod
    def _seeds(seed, count: int, graph: Graph) -> list:
        text = str(seed).strip().lower() if seed is not None else "random"
        if text == "keep":
            return [graph.seed()] * count
        if text in ("", "random"):
            return [random.randint(1, MAX_SEED) for _ in range(count)]
        start = _number(text, int)
        return [start + i for i in range(count)]

    # --- the tools -------------------------------------------------------------------------------

    def _tool_run_workflow(self, workflow: str | None = None, changes: dict | None = None, count: int = 1,
                           graph: dict | None = None) -> tuple[dict, dict]:
        count = max(1, min(MAX_RUN_COUNT, _number(count or 1, int)))
        changes = changes or {}
        graph, label = self._source(workflow, graph)
        base = self.client.pick_base()
        done = self._apply(graph, changes, base)
        if not changes.get("model") and graph.base_model_loaders():
            # The prompt runs on the model its block names, whatever model the workflow came with.
            declared = self._declared_model(graph)
            if declared and declared not in graph.base_models():
                try:
                    done.append(f"model {self._set_model(graph, declared)}, as your <prompt> block says")
                except ToolError as e:
                    raise ToolError(f"Your <prompt> block says {declared}, which can't run here: {e} Pass "
                                    "changes.model to run it on another model, or another workflow.")
        # On the workflow as it will be sent: the bot's changes may have swapped the base model.
        switched = self._follow_family(graph)
        if switched:
            return switched
        conversation = self._conversation()
        folder = conversations.output_folder(self.conversation_id, self.title_hint)
        number = len(conversation.get("runs") or []) + 1  # the number add_run will give it
        graph.set_output_prefix(comfy.output_prefix(folder, number))
        comfy_policy.check(graph, conversation)  # the isolation: before anything reaches ComfyUI
        seed_mode = str(changes.get("seed", "random")).strip().lower()
        seeds = self._seeds(changes.get("seed"), count, graph)
        variants = []
        for seed in seeds:
            job_graph = graph.copy()
            if seed_mode != "keep" and seed is not None:
                job_graph.set_seed(seed)
            job_graph.set_output_prefix(comfy.output_prefix(folder, number, seed))
            variants.append((job_graph, seed, {}))
        size = graph.shape()
        run = self._submit_run(label, changes, base, folder, variants, run_fields={"size": size} if size else None)
        result = {"run": run["n"], "status": "queued", "images": run["count"], "seeds": seeds[:run["count"]],
                  "model": ", ".join(run["base_models"]), "applied": done}
        if size:
            result["size"] = size
        return result, {"type": "run", "run": run["n"], "workflow": label, "count": run["count"]}

    def _priority(self) -> str:
        """The next image's place in ComfyUI's queue, unless the run sets its own."""
        if self.automatic:
            return comfy.PRIORITY_ON_ITS_OWN
        if self.waited_for > 0:
            self.waited_for -= 1
            return comfy.PRIORITY
        return comfy.PRIORITY_REST

    def _tag(self, number: int, calibrating: str | None = None) -> str:
        """What comfy-gateway's panel shows for the run's jobs: the conversation, the run and who
        it is for -- "Bruja en el bosque · ejecución 3 · por su cuenta"."""
        conversation = self._conversation()
        title = " ".join((conversation.get("title") or self.title_hint
                          or conversations.display_title(conversation) or "Conversar").split())
        if len(title) > TAG_TITLE_CHARS:
            title = title[:TAG_TITLE_CHARS - 1].rstrip() + "…"
        if calibrating:
            why = f"calibración de {re.split(r'[\\/]', catalog.short_name(calibrating))[-1]}"
        else:
            why = "por su cuenta" if self.automatic else "tras tu mensaje"
        return f"{title} · ejecución {number} · {why}"

    def _submit_run(self, label: str, changes: dict, base: str, folder: str, variants: list,
                    priority: str | None = None, run_fields: dict | None = None) -> dict:
        """Send one job per (graph, seed, job fields) and record the run. priority: every job's, or
        by default each one's by who is waiting for it (_priority)."""
        self._check_budget(len(variants))
        number = len(self._conversation().get("runs") or []) + 1  # the number add_run will give it
        tag = self._tag(number, (run_fields or {}).get("lora") if (run_fields or {}).get("kind") == "calibration"
                        else None)
        jobs, sent, priorities = [], [], []
        for job_graph, seed, fields in variants:
            payload = job_graph.to_api()
            job_priority = priority or self._priority()
            try:
                prompt_id = self.client.submit(base, payload, tag=tag, priority=job_priority)
            except comfy.ComfyError:
                if not jobs:
                    raise
                break  # the first went through: keep what was sent
            jobs.append({"prompt_id": prompt_id, "base_url": base, "seed": seed, "status": "pending",
                         "images": [], "error": None, "models": job_graph.models(),
                         "loras": [{"name": lora["name"], "strength": lora.get("strength")}
                                   for lora in job_graph.loras()], **fields})
            sent.append(payload)
            priorities.append(job_priority)
        run = conversations.add_run(
            self.conversation_id,
            {"workflow": label, "changes": changes, "count": len(jobs), "status": "pending", "jobs": jobs,
             "base_models": variants[0][0].base_models(),  # every job's graph differs only in its seed
             "output_folder": f"{comfy.OUTPUT_SUBFOLDER}/{folder}", **(run_fields or {})},
            sent,
        )
        if run is None:
            raise ToolError("The conversation is gone.")
        conversations.add_autonomy_usage(self.conversation_id, images=len(jobs))
        runs.tracker.watch(self.conversation_id, run["n"])
        logger.info("run launched conversation_id=%s run=%s workflow=%s jobs=%d priority=%s base=%s",
                    self.conversation_id, run["n"], label, len(jobs), ",".join(dict.fromkeys(priorities)), base)
        return run

    def _tool_use_family(self, family: str) -> tuple[dict, dict]:
        wanted = families._key(family)
        target = next((f for f in families.list_families() if families._key(f["name"]) == wanted), None)
        if target is None:
            names = ", ".join(f["name"] for f in families.list_families())
            raise ToolError(f'There is no family "{family}". Families: {names}.')
        current = families.get_family(self._conversation()["family_id"])
        if current and current["id"] == target["id"]:
            return ({"family": target["name"], "note": "Already this conversation's family."},
                    {"type": "tool", "tool": "use_family", "family": target["name"]})
        conversations.update_settings(self.conversation_id, {"family_id": target["id"]})
        logger.info("conversation family switched conversation_id=%s family=%s", self.conversation_id, target["name"])
        return ({"family": target["name"], "uses_negative_prompt": target["has_negative_prompt"],
                 "family_rules": target["instructions"]},
                {"type": "family", "family": target["name"], "previous": (current or {}).get("name"),
                 "reason": "asked"})

    def _calibration_workflow(self, folder: str) -> str:
        candidates = [w["name"] for w in workflows.list_workflows()
                      if Graph(w["workflow"]).base_model_folder() == folder]
        plain = [n for n in candidates
                 if "txt2img" in n and not any(x in n for x in ("detail", "upscale", "hires"))]
        if not (plain or candidates):
            raise ToolError(f"No library workflow runs a {folder} model: pass one with workflow.")
        return (plain or candidates)[0]

    def _tool_calibrate_lora(self, lora: str, prompt: str, workflow: str | None = None, weights=None,
                             keep_other_loras: bool = True) -> tuple[dict, dict]:
        lora_file = catalog.resolve_file(str(lora), "loras", self.client)
        folder = catalog.family_of(lora_file)
        workflow = workflow or self._calibration_workflow(folder)
        weights = list(dict.fromkeys(_number(w) for w in (weights or CALIBRATION_WEIGHTS)))
        if not 2 <= len(weights) <= MAX_CALIBRATION_WEIGHTS or any(not -2 <= w <= 3 for w in weights):
            raise ToolError(f"Give 2 to {MAX_CALIBRATION_WEIGHTS} weights between -2 and 3")
        graph, label = self._resolve(workflow)
        base = self.client.pick_base()
        self._apply(graph, {"positive": prompt}, base)
        switched = self._follow_family(graph)
        if switched:
            return switched
        chain = graph.lora_chain()
        others = [{"name": c["name"], "strength": c["strength"],
                   **({"strength_clip": c["strength_clip"]} if c.get("strength_clip") is not None else {})}
                  for c in chain if c["name"] != lora_file] if keep_other_loras else []
        conversation = self._conversation()
        out_folder = conversations.output_folder(self.conversation_id, self.title_hint)
        number = len(conversation.get("runs") or []) + 1
        seed = random.randint(1, MAX_SEED)
        variants = []
        for weight in weights:
            job_graph = graph.copy()
            job_graph.set_loras(others + [{"name": lora_file, "strength": weight}])
            job_graph.set_seed(seed)
            job_graph.set_output_prefix(f"{comfy.output_prefix(out_folder, number, seed)}_w-{weight:g}")
            variants.append((job_graph, seed, {"calibration_weight": weight}))
        comfy_policy.check(variants[0][0], conversation)
        run = self._submit_run(label, {"positive": prompt}, base, out_folder, variants, priority="idle",
                               run_fields={"kind": "calibration", "lora": lora_file,
                                           "timeout_seconds": CALIBRATION_TIMEOUT_SECONDS})
        result = {"run": run["n"], "status": "queued at idle priority: it runs when the GPU is free", "lora": lora_file,
                  "workflow": label, "weights": weights[:run["count"]], "seed": seed,
                  "other_loras": [o["name"] for o in others]}
        return result, {"type": "run", "run": run["n"], "workflow": label, "count": run["count"],
                        "calibration": lora_file}

    def _follow_family(self, graph: Graph) -> tuple[dict, dict] | None:
        """Prompts follow the rules of the family of the model they run on. When the workflow
        (as it will be sent) runs a base model of another family than the conversation's, the
        conversation switches to it, and the run waits: the bot gets that family's rules to
        rewrite the prompt by, and runs again."""
        folder = graph.base_model_folder()
        if folder is None:
            return None
        conversation = self._conversation()
        current = families.get_family(conversation["family_id"])
        if current is not None and families.covers_folder(current, folder):
            return None
        target = families.family_for_folder(folder)
        if target is None:
            return None  # no family for that folder: nothing better to go by
        conversations.update_settings(self.conversation_id, {"family_id": target["id"]})
        logger.info("conversation family follows the model conversation_id=%s folder=%s family=%s",
                    self.conversation_id, folder, target["name"])
        result = {
            "error": f"Not run yet: this workflow's base model is in {folder}, a {target['name']} model, so "
                     f"this conversation now follows the {target['name']} family. Rewrite the prompt by its rules "
                     "below (in a <prompt> block too) and run it again.",
            "family": target["name"],
            "uses_negative_prompt": target["has_negative_prompt"],
            "family_rules": target["instructions"],
        }
        return result, {"type": "family", "family": target["name"], "previous": (current or {}).get("name"),
                        "reason": "model"}

    def _tool_inspect_workflow(self, workflow: str) -> tuple[dict, dict]:
        graph, label = self._resolve(workflow)
        slots = graph.prompt_slots()
        entry = workflows.find_by_name(label)
        folder = graph.base_model_folder()
        family = families.family_for_folder(folder) if folder else None
        result = {
            "workflow": label,
            **({"family": family["name"]} if family else {}),
            **({"description": entry["description"]} if entry and entry.get("description") else {}),
            "does": graph.generation_info(),
            **({"size": graph.shape()} if graph.shape() else {}),
            "prompt_inputs": {role: [f"{n}.{i}" for n, i in slots[role]] for role in ("positive", "negative")},
            "image_inputs": graph.image_inputs() + [n for n, v in graph.nodes.items()
                                                    if "base64_data" in v["inputs"]],
            "nodes": graph.summary(),
        }
        return result, {"type": "tool", "tool": "inspect_workflow", "workflow": label}

    def _tool_describe_nodes(self, types: list | None = None) -> tuple[dict, dict]:
        allowed = comfy_policy.allowed_types(self._conversation())
        if not types:
            return ({"available": sorted(allowed),
                     "note": "Pass types to see what a node takes and gives."},
                    {"type": "tool", "tool": "describe_nodes", "types": []})
        if not isinstance(types, list):
            raise ToolError("types must be a list of node class types")
        wanted = list(dict.fromkeys(str(t) for t in types))[:MAX_DESCRIBED_NODES]
        nodes = {}
        for class_type in wanted:
            if class_type not in allowed:
                nodes[class_type] = {"error": "Not a node type available here (describe_nodes without types lists them)."}
                continue
            info = self.client.node_info(class_type)
            nodes[class_type] = describe_node(info) if info else {"error": "ComfyUI has no such node type."}
        return {"nodes": nodes}, {"type": "tool", "tool": "describe_nodes", "types": wanted}

    def _tool_list_comfy_models(self, folder: str, search: str = "") -> tuple[dict, dict]:
        if folder not in MODEL_FOLDERS:
            raise ToolError(f"folder must be one of: {', '.join(MODEL_FOLDERS)}")
        names = self.client.models(folder)
        words = [w for w in re.split(r"[\s,]+", (search or "").casefold()) if w]
        data, file_stats = catalog.load(), catalog.stats()
        matches = [n for n in names if all(w in n.casefold() for w in words)
                   or any(w in " ".join(catalog.details(n, data, file_stats).get("triggers", [])).casefold()
                          for w in words)]
        if folder in catalog.FOLDERS:
            files = []
            for name in matches[:MAX_LISTED_MODELS]:
                known = catalog.details(name, data, file_stats)
                files.append({"file": name, **known} if known else name)
        else:
            files = matches[:MAX_LISTED_MODELS]
        result = {"folder": folder, "total": len(matches), "files": files}
        if len(matches) > MAX_LISTED_MODELS:
            result["note"] = f"Only the first {MAX_LISTED_MODELS}: search to narrow it down."
        return result, {"type": "tool", "tool": "list_comfy_models", "folder": folder, "search": search or "",
                        "found": len(matches)}

    def _tool_save_workflow(self, name: str, description: str, workflow: str | None = None,
                            changes: dict | None = None, replace: bool = False,
                            graph: dict | None = None) -> tuple[dict, dict]:
        graph, label = self._source(workflow, graph)
        changes = changes or {}
        self._apply(graph, {k: v for k, v in changes.items() if k not in ("input_images", "seed")}, None)
        if str(changes.get("seed", "")).strip().lstrip("-").isdigit():
            graph.set_seed(int(changes["seed"]))
        # A saved workflow widens the node types allowed: it may only use those allowed already.
        comfy_policy.check(graph, self._conversation(), files=False)
        existing = workflows.find_by_name(name)
        if existing and replace:
            saved = workflows.update(existing["id"], name, description, graph.to_api())
        elif existing:
            raise ToolError(f'There is already a workflow named "{existing["name"]}": pick another name, or pass '
                            "replace: true if the user wants it overwritten.")
        else:
            saved = workflows.create(name, description, graph.to_api())
        return ({"saved": saved["name"], "replaced": bool(existing)},
                {"type": "tool", "tool": "save_workflow", "workflow": label, "name": saved["name"],
                 "replaced": bool(existing)})

    def _tool_keep_going(self, goal: str = "", max_images=None, max_cost=None, stop: bool = False) -> tuple[dict, dict]:
        if stop:
            ended = conversations.end_autonomy(self.conversation_id, "user")
            return ({"stopped": True, "used": _usage(ended)} if ended else {"note": "You weren't working on your own."},
                    {"type": "tool", "tool": "keep_going", "stop": True})
        goal = " ".join(str(goal or "").split())
        if not goal:
            raise ToolError("Say the goal: what counts as done.")
        images = min(AUTONOMY_MAX_IMAGES, max(1, _number(max_images, int))) if max_images is not None \
            else AUTONOMY_IMAGES
        cost = min(AUTONOMY_MAX_COST, max(0.05, _number(max_cost))) if max_cost is not None else AUTONOMY_COST
        started = conversations.start_autonomy(self.conversation_id, goal, images, round(cost, 2))
        if started is None:
            raise ToolError("The conversation is gone.")
        return ({"working_on_your_own": True, "goal": goal, "max_images": images, "max_cost": started["max_cost"],
                 "used": _usage(started),
                 "next": "Launch the first run now; each run's images will come back to you."},
                {"type": "tool", "tool": "keep_going", "goal": goal, "max_images": images,
                 "max_cost": started["max_cost"]})

    def _check_budget(self, count: int) -> None:
        """While the bot works on its own, a launch must fit what's left of the budget -- in
        images, and in money for the look at its results that will follow."""
        working = conversations.autonomy(self._conversation())
        if working is None:
            return
        left = working["max_images"] - working["images"]
        if left <= 0:
            raise ToolError(f"The budget's images are spent ({working['images']} of {working['max_images']}): "
                            "don't launch more; sum up for the user.")
        if count > left:
            raise ToolError(f"Only {left} image{'s' if left > 1 else ''} left in the budget: launch at most that.")
        if working["cost"] + working.get("last_turn_cost", 0) > working["max_cost"]:
            raise ToolError(f"The budget's money (${working['cost']:.2f} of ${working['max_cost']:.2f}) wouldn't "
                            "cover looking at another run's images: don't launch more; sum up for the user.")

    def _tool_prepare_download(self, version_id=None, url=None, hash=None) -> tuple[dict, dict]:
        from . import catalog, civitai, downloads
        try:
            record = downloads.prepare(self.conversation_id, {"version_id": version_id, "url": url, "hash": hash},
                                       catalog.load()["entries"] or None)
        except (downloads.DownloadError, civitai.CivitaiError) as e:
            raise ToolError(str(e))
        where = f"{record['folder']}\\{record['family'] or '(a family folder the user picks)'}\\{record['name']}"
        result = {"download": record["n"], "status": record["status"], "model": record["model_name"],
                  "version": record["version_name"], "type": record["model_type"], "base_model": record["base_model"],
                  "size_mb": round(record["size_bytes"] / 1e6), "would_install_as": f"{where}.safetensors"}
        if record["status"] == "proposed":
            result["note"] = "Waiting for the user to accept it in the chat; nothing is downloaded until then."
        return result, {"type": "download", "download": record["n"]}

    def _tool_cancel_run(self, run: int) -> tuple[dict, dict]:
        number = _number(run, int)
        found = conversations.get_run(self.conversation_id, number)
        if found is None:
            raise ToolError(f"There is no run {number}.")
        if found["status"] != "pending":
            return ({"run": number, "status": found["status"], "note": "It had already finished."},
                    {"type": "tool", "tool": "cancel_run", "run": number})
        cancelled = runs.cancel(self.conversation_id, number)
        return ({"run": number, "status": (cancelled or found)["status"]},
                {"type": "tool", "tool": "cancel_run", "run": number})
