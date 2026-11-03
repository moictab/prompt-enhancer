"""A ComfyUI workflow in API format, and the edits the Conversar bot makes to it.

API format: `{id: {"class_type": ..., "inputs": {name: value | [source_id, output_index]}, "_meta": {"title"}}}`.
Ids can be compound ("136:102", flattened subgraphs) or non-numeric ("101n"). A link is a two-item list
`[id (str), index (int)]`; anything else is a literal value.

Low-level edits (add, remove, bypass, set inputs) cover any change; shortcuts cover the usual ones (the
LoRA stack, the prompt texts, seed, sampler settings, size, output prefix). `generation_info()` reads
the same things back: it is what an image's embedded workflow says about how it was made.

Adapted from image-generation's app/graph/graph.py.
"""
import copy
import re
from typing import Any, Iterable

LORA_LOADERS = ("LoraLoader", "LoraLoaderModelOnly")
SAMPLERS = ("KSampler", "KSamplerAdvanced", "FaceDetailer", "SamplerCustom", "SamplerCustomAdvanced")
MAIN_SAMPLERS = ("KSampler", "KSamplerAdvanced", "SamplerCustom")
TEXT_NODES = ("PrimitiveStringMultiline", "PrimitiveString")
SAMPLER_SETTINGS = ("steps", "cfg", "sampler_name", "scheduler", "denoise")
# Inputs that name the model files a workflow loads.
MODEL_INPUTS = ("ckpt_name", "unet_name", "model_name", "clip_name", "clip_name1", "clip_name2", "vae_name")
# The base model itself: a checkpoint, or a diffusion model loaded on its own.
BASE_MODEL_INPUTS = ("ckpt_name", "unet_name")
# The proportions an image can take (set_aspect), each with its option in ComfyUI's ResolutionSelector.
ASPECTS = {
    "1:1": "1:1 (Square)",
    "2:3": "2:3 (Portrait Photo)",
    "3:2": "3:2 (Photo)",
    "3:4": "3:4 (Portrait Standard)",
    "4:3": "4:3 (Standard)",
    "9:16": "9:16 (Portrait Widescreen)",
    "16:9": "16:9 (Widescreen)",
    "21:9": "21:9 (Ultrawide)",
}
LATENT_MULTIPLE = 16  # a latent's sides: what every model family here takes
SCALE_MULTIPLE = 8

# Without node schemas: which input usually carries each kind of output through a node.
_PASSTHROUGH_INPUTS = {
    0: ("model", "samples", "latent", "latent_image", "image", "images", "conditioning", "positive"),
    1: ("clip",),
}


class GraphError(ValueError):
    pass


def is_link(value: Any) -> bool:
    return (isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)
            and isinstance(value[1], int) and not isinstance(value[1], bool))


def is_api_workflow(data: Any) -> bool:
    """Whether data looks like an API-format workflow (not the editor's {"nodes": [...], "links": [...]})."""
    return (isinstance(data, dict) and bool(data)
            and all(isinstance(n, dict) and isinstance(n.get("class_type"), str) for n in data.values()))


def is_editor_workflow(data: Any) -> bool:
    return isinstance(data, dict) and isinstance(data.get("nodes"), list) and "links" in data


def _id_key(node_id: str) -> tuple:
    return tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in node_id.split(":"))


def normalize_aspect(value: Any) -> str | None:
    """"2:3", "2:3 (Portrait Photo)" or "2/3" -> "2:3"; None if it isn't one of ASPECTS."""
    match = re.match(r"\s*(\d+)\s*[:/]\s*(\d+)", str(value or ""))
    key = f"{match.group(1)}:{match.group(2)}" if match else None
    return key if key in ASPECTS else None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _fit(area: int, w: float, h: float, multiple: int) -> tuple[int, int]:
    """Width and height in proportion w:h with about `area` pixels, in multiples of `multiple`."""
    width = (area * w / h) ** 0.5
    return (max(multiple, round(width / multiple) * multiple),
            max(multiple, round(width * h / w / multiple) * multiple))


def _orientation(width: float, height: float) -> str:
    return "square" if width == height else "portrait" if width < height else "landscape"


def _nearest_aspect(width: int, height: int) -> str | None:
    for key in ASPECTS:
        w, h = (int(n) for n in key.split(":"))
        if abs(width / height - w / h) <= 0.03 * w / h:
            return key
    return None


class Graph:
    def __init__(self, nodes: dict | None = None):
        if not isinstance(nodes, dict):
            raise GraphError("a workflow must be an object of nodes")
        self.nodes: dict[str, dict] = copy.deepcopy(nodes)
        for node_id, node in self.nodes.items():
            if not isinstance(node, dict) or not isinstance(node.get("class_type"), str):
                raise GraphError(f"node {node_id} has no class_type")
            if not isinstance(node.setdefault("inputs", {}), dict):
                raise GraphError(f"node {node_id} has malformed inputs")

    def to_api(self) -> dict:
        return copy.deepcopy(self.nodes)

    def copy(self) -> "Graph":
        return Graph(self.nodes)

    # --- queries ---------------------------------------------------------------------------------

    def node(self, node_id: str) -> dict:
        if node_id not in self.nodes:
            raise GraphError(f"there is no node {node_id}")
        return self.nodes[node_id]

    def find(self, *class_types: str) -> list[str]:
        return [i for i, n in self.nodes.items() if n["class_type"] in class_types]

    def links(self) -> Iterable[tuple[str, str, str, int]]:
        """(destination, input, source, output index) of every link."""
        for dst, node in self.nodes.items():
            for name, value in node["inputs"].items():
                if is_link(value):
                    yield dst, name, value[0], value[1]

    def consumers(self, node_id: str, output: int | None = None) -> list[tuple[str, str, int]]:
        return [(dst, name, idx) for dst, name, src, idx in self.links()
                if src == node_id and (output is None or idx == output)]

    def title(self, node_id: str) -> str:
        return (self.nodes[node_id].get("_meta") or {}).get("title", "")

    def broken_links(self) -> list[str]:
        return [f"{dst}.{name} -> {src}" for dst, name, src, _ in self.links() if src not in self.nodes]

    def summary(self, max_literal: int = 120) -> list[dict]:
        """The graph in compact form for the bot: each node with its values and its links."""
        out = []
        for node_id in sorted(self.nodes, key=_id_key):
            node = self.nodes[node_id]
            values, links = {}, {}
            for name, value in node["inputs"].items():
                if is_link(value):
                    links[name] = f"{value[0]}.{value[1]}"
                elif isinstance(value, str) and len(value) > max_literal:
                    values[name] = value[:max_literal] + f"... ({len(value)} characters)"
                else:
                    values[name] = value
            entry = {"id": node_id, "class_type": node["class_type"]}
            if self.title(node_id) and self.title(node_id) != node["class_type"]:
                entry["title"] = self.title(node_id)
            if values:
                entry["values"] = values
            if links:
                entry["links"] = links
            out.append(entry)
        return out

    def main_sampler(self) -> str | None:
        found = self.find(*MAIN_SAMPLERS) or self.find(*SAMPLERS)
        return sorted(found, key=_id_key)[0] if found else None

    def _literal(self, node_id: str, name: str, depth: int = 0) -> Any:
        """An input's value, following a link to a primitive-like node that holds it."""
        value = self.nodes.get(node_id, {}).get("inputs", {}).get(name)
        if not is_link(value):
            return value
        if depth > 4 or value[0] not in self.nodes:
            return None
        source = self.nodes[value[0]]["inputs"]
        for key in (name, "value", "seed", "int", "float", "string", "text"):
            if key in source:
                return self._literal(value[0], key, depth + 1)
        return None

    def loras(self) -> list[dict]:
        """Every LoRA the workflow loads, with its strengths: LoraLoader nodes and
        stack nodes whose inputs hold {"lora": ..., "strength": ...} entries (rgthree's Power Lora Loader)."""
        found = []
        for node_id in sorted(self.nodes, key=_id_key):
            node = self.nodes[node_id]
            inputs = node["inputs"]
            if node["class_type"] in LORA_LOADERS and isinstance(inputs.get("lora_name"), str):
                entry = {"node": node_id, "name": inputs["lora_name"], "strength": inputs.get("strength_model")}
                if node["class_type"] == "LoraLoader":
                    entry["strength_clip"] = inputs.get("strength_clip")
                found.append(entry)
                continue
            for value in inputs.values():
                if isinstance(value, dict) and isinstance(value.get("lora"), str) and value.get("lora") != "None":
                    if value.get("on") is False:
                        continue
                    found.append({"node": node_id, "name": value["lora"], "strength": value.get("strength")})
        return found

    def models(self) -> list[str]:
        names = []
        for node_id in sorted(self.nodes, key=_id_key):
            for key in MODEL_INPUTS:
                value = self.nodes[node_id]["inputs"].get(key)
                if isinstance(value, str) and value and value not in names:
                    names.append(value)
        return names

    def base_models(self) -> list[str]:
        """The checkpoints and diffusion models it loads (not text encoders, VAEs or LoRAs)."""
        names = []
        for node_id in sorted(self.nodes, key=_id_key):
            for key in BASE_MODEL_INPUTS:
                value = self.nodes[node_id]["inputs"].get(key)
                if isinstance(value, str) and value and value not in names:
                    names.append(value)
        return names

    def base_model_folder(self) -> str | None:
        """The folder the base model sits in ("krea-2" for "krea-2\\x.safetensors"), which names its
        family; None when there is no base model in a folder, or base models from several."""
        folders = set()
        for node in self.nodes.values():
            for key in BASE_MODEL_INPUTS:
                value = node["inputs"].get(key)
                if isinstance(value, str) and value:
                    parts = re.split(r"[\\/]", value)
                    folders.add(parts[0] if len(parts) > 1 else "")
        folders.discard("")
        return folders.pop() if len(folders) == 1 else None

    def size(self) -> str | None:
        for node_id in self._latents():
            width, height = self._literal(node_id, "width"), self._literal(node_id, "height")
            if isinstance(width, int) and isinstance(height, int):
                return f"{width}x{height}"
        for node_id in self.find("ResolutionSelector"):
            inputs = self.nodes[node_id]["inputs"]
            if inputs.get("aspect_ratio") is not None:
                return f"{inputs.get('aspect_ratio')}, {inputs.get('megapixels')} MP"
        return None

    def shape(self) -> str | None:
        """The size of the image it makes, as the bot reads it: "2:3 portrait, 1 MP" (a ResolutionSelector)
        or "832x1248 (2:3 portrait)" (a fixed size); None when it takes its size from an input image."""
        for node_id in self._latents():
            width, height = self._literal(node_id, "width"), self._literal(node_id, "height")
            if _is_int(width) and _is_int(height) and width > 0 and height > 0:
                aspect = _nearest_aspect(width, height)
                return f"{width}x{height} ({f'{aspect} ' if aspect else ''}{_orientation(width, height)})"
        for node_id in self.find("ResolutionSelector"):
            inputs = self.nodes[node_id]["inputs"]
            key = normalize_aspect(inputs.get("aspect_ratio"))
            if key:
                w, h = (int(n) for n in key.split(":"))
                megapixels = inputs.get("megapixels")
                pixels = f", {megapixels:g} MP" if isinstance(megapixels, (int, float)) else ""
                return f"{key} {_orientation(w, h)}{pixels}"
        return None

    def generation_info(self) -> dict:
        """What the workflow says about the image it makes: prompts, sampler settings, models, LoRAs, size."""
        slots = self.prompt_slots()
        texts = {}
        for role in ("positive", "negative"):
            values = [self.nodes[n]["inputs"].get(i) for n, i in slots[role]]
            values = [v for v in values if isinstance(v, str) and v.strip()]
            if values:
                texts[role] = "\n\n".join(dict.fromkeys(values))
        settings = {}
        sampler = self.main_sampler()
        if sampler:
            for key in ("seed", "noise_seed", *SAMPLER_SETTINGS):
                value = self._literal(sampler, key)
                if value is not None and not isinstance(value, (dict, list)):
                    settings["seed" if key == "noise_seed" else key] = value
        size = self.size()
        if size:
            settings["size"] = size
        return {**texts, "settings": settings, "models": self.models(),
                "loras": [{k: v for k, v in lora.items() if k != "node"} for lora in self.loras()]}

    # --- low-level edits -------------------------------------------------------------------------

    def new_id(self) -> str:
        top = [int(m.group(0)) for i in self.nodes if (m := re.match(r"\d+", i))]
        return str(max(top, default=0) + 1)

    def add_node(self, class_type: str, inputs: dict | None = None, title: str | None = None,
                 node_id: str | None = None) -> str:
        if not isinstance(class_type, str) or not class_type:
            raise GraphError("a new node needs a class_type")
        node_id = str(node_id) if node_id else self.new_id()
        if node_id in self.nodes:
            raise GraphError(f"node {node_id} already exists")
        self.nodes[node_id] = {"class_type": class_type, "inputs": dict(inputs or {})}
        if title:
            self.nodes[node_id]["_meta"] = {"title": title}
        return node_id

    def set_input(self, node_id: str, name: str, value: Any) -> None:
        if is_link(value) and value[0] not in self.nodes:
            raise GraphError(f"the link points to a node that doesn't exist: {value[0]}")
        self.node(node_id)["inputs"][name] = value

    def remove_input(self, node_id: str, name: str) -> None:
        self.node(node_id)["inputs"].pop(name, None)

    def remove_node(self, node_id: str) -> list[tuple[str, str]]:
        """Remove the node. Returns the inputs left without a link (to reconnect)."""
        self.node(node_id)
        dangling = [(dst, name) for dst, name, _ in self.consumers(node_id)]
        for dst, name in dangling:
            del self.nodes[dst]["inputs"][name]
        del self.nodes[node_id]
        return dangling

    def bypass(self, node_id: str) -> None:
        """Remove the node, wiring whatever used its outputs to its matching inputs (like ComfyUI's Bypass)."""
        node = self.node(node_id)
        for dst, name, index in self.consumers(node_id):
            source = next((node["inputs"][n] for n in _PASSTHROUGH_INPUTS.get(index, ())
                           if is_link(node["inputs"].get(n))), None)
            if source is None:
                raise GraphError(f"can't bypass {node_id} ({node['class_type']}): its output {index} has no "
                                 "matching input; remove it and rewire with 'set' instead")
            self.nodes[dst]["inputs"][name] = list(source)
        del self.nodes[node_id]

    # --- shortcuts: LoRAs ------------------------------------------------------------------------

    def lora_chain(self) -> list[dict]:
        """The LoRA stack in order, from the model loader towards the sampler."""
        ids = set(self.find(*LORA_LOADERS))
        if not ids:
            return []
        heads = [i for i in ids if not (is_link(self.nodes[i]["inputs"].get("model"))
                                        and self.nodes[i]["inputs"]["model"][0] in ids)]
        if len(heads) != 1:
            raise GraphError("the LoRAs don't form a single chain: change them with 'set', 'add' and 'remove'")
        chain, current = [], heads[0]
        while current is not None:
            node = self.nodes[current]
            chain.append({"node": current, "name": node["inputs"].get("lora_name"),
                          "strength": node["inputs"].get("strength_model"),
                          "strength_clip": node["inputs"].get("strength_clip"),
                          "model_only": node["class_type"] == "LoraLoaderModelOnly"})
            following = [dst for dst, name, _ in self.consumers(current) if dst in ids and name == "model"]
            if len(following) > 1:
                raise GraphError("the LoRAs branch: change them with 'set', 'add' and 'remove'")
            current = following[0] if following else None
        if len(chain) != len(ids):
            raise GraphError("some LoRAs sit outside the main chain: change them with 'set', 'add' and 'remove'")
        return chain

    def _model_roots(self, chain: list[dict]) -> tuple[list, list | None]:
        if chain:
            head = self.nodes[chain[0]["node"]]["inputs"]
            return head["model"], head.get("clip")
        samplers = [i for i in self.find("KSampler", "KSamplerAdvanced")
                    if is_link(self.nodes[i]["inputs"].get("model"))]
        if not samplers:
            raise GraphError("there is no KSampler to take the model from")
        model = self.nodes[samplers[0]]["inputs"]["model"]
        encoders = [i for i in self.find("CLIPTextEncode") if is_link(self.nodes[i]["inputs"].get("clip"))]
        clip = self.nodes[encoders[0]]["inputs"]["clip"] if encoders else None
        return model, clip

    def set_loras(self, stack: list[dict]) -> list[str]:
        """Replace the LoRA stack. Each item: {"name", "strength", "strength_clip" (optional, = strength)}.
        Whatever used the old stack's model or clip uses the new one's."""
        chain = self.lora_chain()
        root_model, root_clip = self._model_roots(chain)
        old = {c["node"] for c in chain}
        model_users, clip_users = [], []
        for dst, name, src, idx in list(self.links()):
            if dst in old:
                continue
            if (src in old and idx == 0) or (not old and [src, idx] == root_model):
                model_users.append((dst, name))
            elif (src in old and idx == 1) or (not old and root_clip is not None and [src, idx] == root_clip):
                clip_users.append((dst, name))
        for node_id in old:
            del self.nodes[node_id]
        model, clip, created = root_model, root_clip, []
        for item in stack:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"]:
                raise GraphError("each LoRA needs a file name in 'name'")
            strength = float(item.get("strength", 1.0))
            if clip is None:
                node_id = self.add_node("LoraLoaderModelOnly", {
                    "lora_name": item["name"], "strength_model": strength, "model": list(model)})
                model = [node_id, 0]
            else:
                node_id = self.add_node("LoraLoader", {
                    "lora_name": item["name"], "strength_model": strength,
                    "strength_clip": float(item.get("strength_clip", strength)),
                    "model": list(model), "clip": list(clip)})
                model, clip = [node_id, 0], [node_id, 1]
            created.append(node_id)
        for dst, name in model_users:
            self.nodes[dst]["inputs"][name] = list(model)
        for dst, name in clip_users:
            self.nodes[dst]["inputs"][name] = list(clip)
        return created

    # --- shortcuts: prompt, seed, sampler, size, output ------------------------------------------

    def _text_source(self, node_id: str) -> tuple[str, str] | None:
        """Follow a conditioning link to the node holding the literal text."""
        seen = set()
        while node_id in self.nodes and node_id not in seen:
            seen.add(node_id)
            node = self.nodes[node_id]
            inputs = node["inputs"]
            if node["class_type"] in TEXT_NODES:
                return node_id, "value"
            if "text" in inputs:
                if is_link(inputs["text"]):
                    node_id = inputs["text"][0]
                    continue
                return node_id, "text"
            if node["class_type"] == "ConditioningZeroOut":
                return None
            following = next((v for k, v in inputs.items()
                              if is_link(v) and k in ("conditioning", "conditioning_1", "positive")), None)
            if following is None:
                return None
            node_id = following[0]
        return None

    def prompt_slots(self) -> dict[str, list[tuple[str, str]]]:
        """Where the main sampler's positive and negative text live. With no negative (zeroed with
        ConditioningZeroOut) its list is empty. Texts only other samplers use (a detailer with a prompt
        of its own) go under "other" and are left alone when the prompt changes.

        Roles come from the main sampler: some workflows feed a detailer the negative as its positive,
        so reading every sampler alike would mix the roles up."""
        main = self.find(*MAIN_SAMPLERS) or self.find(*SAMPLERS)
        slots: dict[str, list[tuple[str, str]]] = {"positive": [], "negative": [], "other": []}
        for node_id in main:
            for role in ("positive", "negative"):
                value = self.nodes[node_id]["inputs"].get(role)
                if is_link(value):
                    source = self._text_source(value[0])
                    if source and source not in slots[role]:
                        slots[role].append(source)
        known = set(slots["positive"]) | set(slots["negative"])
        for node_id in self.find(*SAMPLERS):
            if node_id in main:
                continue
            for role in ("positive", "negative"):
                value = self.nodes[node_id]["inputs"].get(role)
                source = self._text_source(value[0]) if is_link(value) else None
                if source and source not in known and source not in slots["other"]:
                    slots["other"].append(source)
        return slots

    def set_prompt(self, positive: str | None = None, negative: str | None = None) -> None:
        slots = self.prompt_slots()
        if positive is not None:
            if not slots["positive"]:
                raise GraphError("can't find where the positive prompt goes: use 'set'")
            for node_id, name in slots["positive"]:
                self.nodes[node_id]["inputs"][name] = positive
        if negative is not None:
            if not slots["negative"] and negative.strip():
                raise GraphError("this workflow has no negative prompt (it is zeroed out)")
            for node_id, name in slots["negative"]:
                self.nodes[node_id]["inputs"][name] = negative

    def set_seed(self, seed: int) -> None:
        """Every sampler's seed (the main one and the detailers')."""
        for node in self.nodes.values():
            for key in ("seed", "noise_seed"):
                if key in node["inputs"] and not is_link(node["inputs"][key]):
                    node["inputs"][key] = int(seed)

    def seed(self) -> Any:
        sampler = self.main_sampler()
        if sampler is None:
            return None
        value = self._literal(sampler, "seed")
        return value if value is not None else self._literal(sampler, "noise_seed")

    def set_sampler(self, **values: Any) -> None:
        """steps, cfg, sampler_name, scheduler or denoise of the main sampler."""
        unknown = set(values) - set(SAMPLER_SETTINGS)
        if unknown:
            raise GraphError(f"unknown sampler settings: {sorted(unknown)}")
        sampler = self.main_sampler()
        if sampler is None:
            raise GraphError("there is no sampler")
        for key, value in values.items():
            if key not in self.nodes[sampler]["inputs"]:
                raise GraphError(f"the main sampler ({self.nodes[sampler]['class_type']}) has no {key}: use 'set'")
            self.nodes[sampler]["inputs"][key] = value

    def _latents(self) -> list[str]:
        return sorted((i for i, n in self.nodes.items()
                       if n["class_type"].startswith("Empty") and "Latent" in n["class_type"]), key=_id_key)

    def set_aspect(self, aspect: str) -> None:
        """The image's proportions, at the workflow's own pixel count (what its model is made for): the
        ResolutionSelector's option, or the empty latent's size recomputed for about the same area. A rescale
        to a fixed size further on (a hires pass) takes the new proportions too, at its own pixel count."""
        key = normalize_aspect(aspect)
        if key is None:
            raise GraphError(f"unknown aspect {aspect!r}: use one of {', '.join(ASPECTS)}")
        ratio = tuple(int(n) for n in key.split(":"))
        sized = False
        for node_id in self.find("ResolutionSelector"):
            self.nodes[node_id]["inputs"]["aspect_ratio"] = ASPECTS[key]
            sized = True
        for node_id in self._latents():
            inputs = self.nodes[node_id]["inputs"]
            width, height = inputs.get("width"), inputs.get("height")
            if _is_int(width) and _is_int(height) and width > 0 and height > 0:
                inputs["width"], inputs["height"] = _fit(width * height, *ratio, LATENT_MULTIPLE)
                sized = True
        if not sized:
            raise GraphError("this workflow takes its size from an input image: it has no size of its own")
        self._reshape_fixed_scales(*ratio)

    def _reshape_fixed_scales(self, w: float, h: float) -> None:
        """A rescale to a fixed size (a hires pass to 1560x1560) would squash an image of other
        proportions: it takes the image's, at its own pixel count."""
        for node_id in self.find("ImageScale"):
            inputs = self.nodes[node_id]["inputs"]
            width, height = inputs.get("width"), inputs.get("height")
            if _is_int(width) and _is_int(height) and width > 0 and height > 0:
                inputs["width"], inputs["height"] = _fit(width * height, w, h, SCALE_MULTIPLE)

    def set_size(self, *, width: int | None = None, height: int | None = None, aspect_ratio: str | None = None,
                 megapixels: float | None = None) -> None:
        """An exact size (width and height), or aspect ratio and megapixels if the graph uses ResolutionSelector."""
        if (width is None) != (height is None):
            raise GraphError("width and height go together")
        if width is not None:
            latents = self._latents()
            if not latents:
                raise GraphError("there is no empty latent node to size: use 'set'")
            for latent_id in latents:
                inputs = self.nodes[latent_id]["inputs"]
                selectors = {v[0] for k, v in inputs.items() if k in ("width", "height") and is_link(v)}
                inputs["width"], inputs["height"] = int(width), int(height)
                for selector in selectors:
                    if selector in self.nodes and not self.consumers(selector):
                        del self.nodes[selector]
            self._reshape_fixed_scales(int(width), int(height))
        if aspect_ratio is not None or megapixels is not None:
            selectors = self.find("ResolutionSelector")
            if not selectors:
                raise GraphError("this graph doesn't use ResolutionSelector: give width and height")
            for selector in selectors:
                if aspect_ratio is not None:
                    self.nodes[selector]["inputs"]["aspect_ratio"] = ASPECTS.get(normalize_aspect(aspect_ratio),
                                                                                  aspect_ratio)
                if megapixels is not None:
                    self.nodes[selector]["inputs"]["megapixels"] = float(megapixels)

    def set_output_prefix(self, prefix: str) -> None:
        for node_id in self.find("SaveImage"):
            self.nodes[node_id]["inputs"]["filename_prefix"] = prefix

    def image_inputs(self) -> list[str]:
        return sorted(self.find("LoadImage"), key=_id_key)
