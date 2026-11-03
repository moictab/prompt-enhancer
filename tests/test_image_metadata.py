import io
import json

from PIL import Image

from app import image_metadata

from .conftest import png_bytes

COMFY_GRAPH = {
    "3": {"class_type": "KSampler", "inputs": {
        "seed": 1234, "steps": 25, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0,
        "model": ["10", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "illustrious\\wai-v14.0.safetensors"}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 832, "height": 1216, "batch_size": 1}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "1girl, witch, forest", "clip": ["10", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "lowres, bad hands", "clip": ["10", 1]}},
    "10": {"class_type": "LoraLoader", "inputs": {"lora_name": "detail.safetensors", "strength_model": 0.7,
                                                  "strength_clip": 0.7, "model": ["4", 0], "clip": ["4", 1]}},
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "x", "images": ["8", 0]}},
}

A1111 = (
    "masterpiece, a witch reading <lora:inkstyle:0.8>\n"
    "Negative prompt: blurry, lowres\n"
    'Steps: 30, Sampler: DPM++ 2M, CFG scale: 7, Seed: 42, Size: 832x1216, Model: wai-v14, '
    'Lora hashes: "inkstyle: abc123, other: def456", Civitai resources: [{"type":"lora","weight":0.8}], Version: f1'
)


def test_reads_a_comfyui_png_and_returns_its_workflow():
    metadata, workflow = image_metadata.read(png_bytes(prompt=COMFY_GRAPH, workflow={"nodes": [], "links": []}))

    assert workflow == COMFY_GRAPH
    assert metadata["generator"] == "ComfyUI"
    assert metadata["has_workflow"] is True
    assert metadata["format"] == "PNG"
    assert (metadata["width"], metadata["height"]) == (64, 48)
    assert metadata["positive"] == "1girl, witch, forest"
    assert metadata["negative"] == "lowres, bad hands"
    assert metadata["models"] == ["illustrious\\wai-v14.0.safetensors"]
    assert metadata["loras"] == [{"name": "detail.safetensors", "strength": 0.7, "strength_clip": 0.7}]
    assert metadata["settings"] == {"seed": 1234, "steps": 25, "cfg": 6.5, "sampler_name": "dpmpp_2m",
                                    "scheduler": "karras", "denoise": 1.0, "size": "832x1216"}


def test_reads_a1111_parameters():
    metadata, workflow = image_metadata.read(png_bytes(parameters=A1111))

    assert workflow is None
    assert metadata["generator"] == "A1111"
    assert metadata["positive"] == "masterpiece, a witch reading <lora:inkstyle:0.8>"
    assert metadata["negative"] == "blurry, lowres"
    assert metadata["loras"] == [{"name": "inkstyle", "strength": 0.8}]
    assert metadata["models"] == ["wai-v14"]
    settings = metadata["settings"]
    assert settings["Steps"] == "30"
    assert settings["Sampler"] == "DPM++ 2M"
    assert settings["Lora hashes"] == "inkstyle: abc123, other: def456"
    assert settings["Civitai resources"] == '[{"type":"lora","weight":0.8}]'
    assert settings["Version"] == "f1"


def test_reads_parameters_from_jpeg_exif_user_comment():
    exif = Image.Exif()
    exif.get_ifd(0x8769)[0x9286] = b"UNICODE\x00" + A1111.encode("utf-16-be")
    out = io.BytesIO()
    Image.new("RGB", (40, 30)).save(out, "JPEG", exif=exif.tobytes())

    metadata, _ = image_metadata.read(out.getvalue())

    assert metadata["format"] == "JPEG"
    assert metadata["negative"] == "blurry, lowres"
    assert metadata["settings"]["Seed"] == "42"


def test_reads_comfyui_webp_exif():
    exif = Image.Exif()
    exif[0x0110] = "prompt:" + json.dumps(COMFY_GRAPH)
    out = io.BytesIO()
    Image.new("RGB", (40, 30)).save(out, "WEBP", exif=exif.tobytes())

    metadata, workflow = image_metadata.read(out.getvalue())

    assert workflow == COMFY_GRAPH
    assert metadata["format"] == "WEBP"
    assert metadata["positive"] == "1girl, witch, forest"


def test_an_editor_only_workflow_is_flagged_but_not_returned():
    metadata, workflow = image_metadata.read(png_bytes(workflow={"nodes": [{"id": 1}], "links": []}))

    assert workflow is None
    assert metadata["editor_workflow_only"] is True


def test_images_without_generation_metadata_or_not_images_give_nothing():
    assert image_metadata.read(png_bytes()) == (None, None)
    assert image_metadata.read(png_bytes(Software="GIMP")) == (None, None)
    assert image_metadata.read(b"\x89PNG not really") == (None, None)


def test_describe_writes_the_metadata_for_the_llm():
    metadata, _ = image_metadata.read(png_bytes(prompt=COMFY_GRAPH))

    text = image_metadata.describe(metadata, 3)

    assert text.splitlines() == [
        'Image 3 metadata (PNG, 64x48, made with ComfyUI; its ComfyUI workflow can be used as "image:3"):',
        "- Positive: 1girl, witch, forest",
        "- Negative: lowres, bad hands",
        "- Models: illustrious\\wai-v14.0.safetensors",
        "- LoRAs: detail.safetensors (0.7)",
        "- Settings: seed: 1234, steps: 25, cfg: 6.5, sampler_name: dpmpp_2m, scheduler: karras, denoise: 1, "
        "size: 832x1216",
    ]


def test_small_copy_only_for_large_images():
    assert image_metadata.small_copy(png_bytes(size=(800, 600))) is None

    small = image_metadata.small_copy(png_bytes(size=(3000, 2000)))

    image = Image.open(io.BytesIO(small))
    assert image.format == "JPEG"
    assert max(image.size) == image_metadata.SMALL_MAX_SIDE
