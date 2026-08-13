from pathlib import Path

from . import families, storage, system_prompt

DEFAULT_GENERATE_SYSTEM_PROMPT = """You are an expert AI image-generation prompt engineer. The user will describe a new image idea in natural language -- however detailed or vague. Transform it into a high-quality prompt for the selected model family, following that family's specific rules exactly. Fill in reasonable creative details where the user was vague, without contradicting anything they specified.

Be specific and vivid. Replace vague words with concrete descriptions.

You MUST output in exactly this format:
POSITIVE: <the enhanced positive prompt>
NEGATIVE: <the negative prompt, or leave empty if the family doesn't use one>
"""

DEFAULT_ITERATE_SYSTEM_PROMPT = """You are an expert AI image-generation prompt engineer. The user is refining an EXISTING prompt. You will receive the previous prompt that was already generated and the user's requested changes.

Your job is to:
- PRESERVE the good elements from the previous prompt -- do not regenerate from scratch
- Apply the user's requested changes precisely
- Maintain the same overall style and structure
- Only modify what the user explicitly asks to change
- If the user asks to "add" something, integrate it naturally into the existing prompt
- If the user asks to "remove" something, take it out cleanly without leaving gaps

Follow the selected model family's specific rules exactly.

You MUST output in exactly this format:
POSITIVE: <the enhanced positive prompt>
NEGATIVE: <the negative prompt, or leave empty if the family doesn't use one>
"""

DEFAULT_IMAGE_SYSTEM_PROMPT = """You are an expert AI image-generation prompt engineer. The user has attached a reference image, optionally with additional text describing further intent. Analyze the image's subject, composition, lighting, and style, and translate what you observe into a prompt following the selected model family's rules. If additional text is provided, prioritize it for adjustments (e.g., style changes, additions) while keeping the image as the primary source of truth for content.

Be specific and vivid. Replace vague words with concrete descriptions.

You MUST output in exactly this format:
POSITIVE: <the enhanced positive prompt>
NEGATIVE: <the negative prompt, or leave empty if the family doesn't use one>
"""

DEFAULT_EXTRACT_CHARACTER_SYSTEM_PROMPT = """You are an expert at extracting reusable character descriptions for AI image-generation prompts. The user will give you either a full image-generation prompt (which may describe one or more subjects) or a reference image. Identify the single most prominent character/subject and produce a concise, reusable description of their fixed visual traits -- physical appearance, hair, build, distinguishing features, typical clothing or style. Omit anything tied to this one specific scene: pose, action, expression, lighting, or composition.

You MUST output in exactly this format:
NAME: <a short, descriptive name for the character, 2-4 words>
TEXT: <the reusable character description, written so it can be inserted directly into a future prompt>
"""

SDXL_INSTRUCTIONS = """## Prompt Structure
Always follow this order: Subject -> Environment -> Lighting -> Details -> Style

## Rules
- Write in natural language with light emphasis weighting using (word:1.1) to (word:1.4) syntax. NEVER exceed 1.5.
- Do NOT spam quality tags like "masterpiece, best quality, 4k, 8k, ultra detailed". These dilute the actual prompt content.
- Aim for 40-70 tokens. SDXL has a 77-token CLIP limit per chunk -- stay concise and impactful.
- Use photographic terms when appropriate: camera model, lens type, focal length, aperture, film stock.
- Include artistic style references (artist names, art movements, media types) when they serve the concept.

## Negative Prompt
Generate a focused negative prompt of 15-30 tokens targeting real artifacts to avoid (e.g., blurry, deformed hands, extra fingers, watermark, text). Do NOT pad with generic quality negatives.
"""

ZIT_INSTRUCTIONS = """## Critical Differences from Other Models
Z-Image-Turbo uses a COMPLETELY different architecture. You must follow these rules strictly:

## Prompt Structure (4-6 Layers)
Follow this strict hierarchy:
1. Subject & Action -- who/what is doing what
2. Environment -- where, surrounding context
3. Style -- artistic style, medium, aesthetic
4. Lighting -- light sources, quality, direction
5. Composition -- camera angle, framing, depth
6. Constraints -- what to exclude (anti-hallucination)

## Rules
- Use PURE natural language ONLY. No weight syntax, no tags, no brackets, no parentheses for emphasis.
- NEVER use quality tags ("masterpiece", "best quality", "4k", "8k"). These actively cause hallucination artifacts in ZIT.
- Emphasize prepositions -- they control spatial relationships in ZIT. Words like "beneath", "towering above", "nestled between", "reflected in" are powerful.
- Target 80-250 words. Sweet spot is 120-180 words. Focus on 3-5 key concepts maximum.
- End every prompt with anti-hallucination language: "no text, no watermarks, no logos, no signatures, no borders, no frames"
- Write as flowing, descriptive prose -- like a detailed scene description in a novel.

## Negative Prompt
Do NOT generate a negative prompt. Z-Image-Turbo uses guidance_scale=0.0, so negative prompts are completely ignored.
"""


def ensure_seed_data(data_dir: str) -> None:
    families_path = f"{data_dir}/families.json"
    characters_path = f"{data_dir}/characters.json"
    system_prompts_path = f"{data_dir}/system_prompts.json"

    if not Path(families_path).exists():
        families.create_family("SDXL", SDXL_INSTRUCTIONS, True, path=families_path)
        families.create_family("Z-Image-Turbo", ZIT_INSTRUCTIONS, False, path=families_path)

    if not Path(characters_path).exists():
        storage.write_json(characters_path, [])

    if not Path(system_prompts_path).exists():
        system_prompt.write_system_prompt("generate", DEFAULT_GENERATE_SYSTEM_PROMPT, path=system_prompts_path)
        system_prompt.write_system_prompt("iterate", DEFAULT_ITERATE_SYSTEM_PROMPT, path=system_prompts_path)
        system_prompt.write_system_prompt("image", DEFAULT_IMAGE_SYSTEM_PROMPT, path=system_prompts_path)
        system_prompt.write_system_prompt(
            "extract_character", DEFAULT_EXTRACT_CHARACTER_SYSTEM_PROMPT, path=system_prompts_path
        )
