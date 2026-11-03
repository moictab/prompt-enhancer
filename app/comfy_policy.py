"""What a workflow sent from Conversar may touch: the isolation of the bot.

The bot only reaches ComfyUI through its tools, and the tools only send workflows; but a
workflow is a program -- a node can load any image from ComfyUI's output folder, or a file
by path. So every workflow is checked before it is sent (and before the bot saves one to
the library), and the bot may only work with:
- what the user gave it: the library workflows (curated in Ajustes) and the workflows
  embedded in images the user shared in this conversation;
- what it made: its own runs;
- images from this conversation, which reach ComfyUI only as uploads named after the
  conversation (or as base64 data).

Concretely:
- Node types: only those in the library, in workflows the user shared here, or in
  ALWAYS_ALLOWED (core nodes that load nothing but model files). Bot-saved workflows
  can't add new types, since save_workflow goes through the same check.
- NEVER_ALLOWED wins over all of that: nodes that read ComfyUI's folders, files by path
  or URLs.
- File references: any short string value that looks like a file (an image, video or
  text extension), a path or a URL must be one of this conversation's uploads.

Nothing else is reachable either: the ComfyUI client only asks /history and /view about
the jobs the bot itself sent, never lists the queue or the history.
"""
import re

from . import conversations, workflows
from .comfy import UPLOAD_SUBFOLDER
from .comfy_graph import Graph, is_api_workflow

# ComfyUI core nodes that only process data or load model files.
ALWAYS_ALLOWED = frozenset({
    "KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced", "KSamplerSelect", "BasicScheduler",
    "RandomNoise", "CFGGuider", "BasicGuider", "DisableNoise",
    "CheckpointLoaderSimple", "UNETLoader", "CLIPLoader", "DualCLIPLoader", "VAELoader", "LoraLoader",
    "LoraLoaderModelOnly", "UpscaleModelLoader", "ControlNetLoader", "CLIPVisionLoader",
    "CLIPTextEncode", "CLIPSetLastLayer", "ConditioningCombine", "ConditioningConcat", "ConditioningSetArea",
    "ConditioningSetMask", "ConditioningZeroOut", "ConditioningSetTimestepRange", "ControlNetApplyAdvanced",
    "CLIPVisionEncode", "FluxGuidance", "InpaintModelConditioning",
    "EmptyLatentImage", "EmptySD3LatentImage", "LatentUpscale", "LatentUpscaleBy", "LatentComposite",
    "SetLatentNoiseMask", "VAEDecode", "VAEEncode", "VAEDecodeTiled", "VAEEncodeTiled", "VAEEncodeForInpaint",
    "ImageScale", "ImageScaleBy", "ImageScaleToTotalPixels", "ImageUpscaleWithModel", "ImageCrop", "ImageInvert",
    "ImageBlur", "ImageSharpen", "ImageBlend", "ImagePadForOutpaint", "ImageCompositeMasked",
    "InvertMask", "MaskComposite", "GrowMask", "SolidMask", "JoinImageWithAlpha", "SplitImageWithAlpha",
    "ModelSamplingDiscrete", "ModelSamplingAuraFlow", "ModelSamplingSD3", "ModelSamplingFlux",
    "DifferentialDiffusion",
    "PrimitiveString", "PrimitiveStringMultiline", "PrimitiveInt", "PrimitiveFloat", "PrimitiveBoolean",
    "SaveImage", "PreviewImage",
    # Read a file from ComfyUI's input folder: allowed, but only this conversation's uploads (checked below).
    "LoadImage", "LoadImageMask",
})

# Nodes that read ComfyUI's folders, files by path, or URLs: never, whoever put them in a workflow.
NEVER_ALLOWED = frozenset({
    "LoadImageOutput", "LoadLatent", "LoadAudio", "LoadVideo", "SaveLatent", "SaveAudio", "SaveVideo",
    "Image Load", "Load Image Batch", "Load Text File", "Save Text File", "Text File History Loader",
    "Image Save", "Load Cache", "Save Cache",
    "VHS_LoadImages", "VHS_LoadImagesPath", "VHS_LoadVideo", "VHS_LoadVideoPath", "VHS_LoadAudio",
    "LoadImagesFromDir //Inspire", "LoadImageListFromDir //Inspire", "LoadImageFromUrl",
    "Load Image From Url (mtb)", "LoadImageFromPath", "LoadImagesFromPath", "easy loadImagesForLoop",
})

FILE_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".mp4", ".webm", ".mov", ".mkv", ".avi",
    ".wav", ".mp3", ".flac", ".txt", ".json", ".csv", ".latent", ".npy", ".npz",
)
# Long strings are data (base64), not references.
MAX_REFERENCE_LENGTH = 1024
PATH_START = re.compile(r"^(?:/|\\|~[/\\]|[A-Za-z]:[/\\])")
DATA_INPUTS = ("base64_data",)


class PolicyError(ValueError):
    pass


def upload_name(conversation_id: str, image_id: str, extension: str) -> str:
    """The file name an image of this conversation gets in ComfyUI's input folder."""
    return f"{conversation_id}-{image_id}.{extension}"


def _own_upload(value: str, conversation: dict) -> bool:
    match = re.fullmatch(
        rf"{UPLOAD_SUBFOLDER}/{re.escape(conversation['id'])}-([0-9a-f]{{32}})\.(?:png|jpg|webp)", value
    )
    if not match:
        return False
    return any(img["id"] == match.group(1) for m in conversation["messages"] for img in m.get("images") or [])


def _looks_like_file(value: str) -> bool:
    if not value or len(value) > MAX_REFERENCE_LENGTH or any(c.isspace() for c in value):
        return False
    lowered = value.lower()
    return "://" in lowered or bool(PATH_START.match(value)) or lowered.endswith(FILE_EXTENSIONS)


def allowed_types(conversation: dict) -> set[str]:
    """Node types this conversation may use: the library's, those of workflows the user
    shared in it, and the core set -- never the forbidden ones."""
    types = set(ALWAYS_ALLOWED)
    for entry in workflows.list_workflows():
        types |= {node["class_type"] for node in entry["workflow"].values()}
    for message in conversation["messages"]:
        if message["role"] != "user":
            continue
        for image in message.get("images") or []:
            shared = conversations.image_workflow(conversation["id"], image["id"])
            if is_api_workflow(shared):
                types |= {node["class_type"] for node in shared.values()}
    return types - NEVER_ALLOWED


def problems(graph: Graph, conversation: dict, files: bool = True) -> list[str]:
    """Why this workflow can't be sent from this conversation (empty when it can)."""
    allowed = allowed_types(conversation)
    found = []
    for node_id, node in sorted(graph.nodes.items()):
        class_type = node["class_type"]
        if class_type in NEVER_ALLOWED:
            found.append(f"node {node_id} ({class_type}) is never allowed: it reads files outside this conversation")
            continue
        if class_type not in allowed:
            found.append(f"node {node_id} ({class_type}) isn't a node type available here")
            continue
        if not files:
            continue
        for name, value in node["inputs"].items():
            if name in DATA_INPUTS or not isinstance(value, str) or not _looks_like_file(value):
                continue
            if not _own_upload(value, conversation):
                found.append(f"node {node_id} ({class_type}) input {name} refers to {value!r}, which isn't an "
                             "image from this conversation (feed one with input_images)")
    return found


def check(graph: Graph, conversation: dict, files: bool = True) -> None:
    found = problems(graph, conversation, files)
    if found:
        raise PolicyError(
            "This workflow reaches outside what this conversation may use: " + "; ".join(found) + ". Node types "
            "available here are those of the library workflows, of workflows in images the user shared in this "
            "conversation, and common core nodes; anything that reads ComfyUI's folders, paths or URLs never is."
        )
