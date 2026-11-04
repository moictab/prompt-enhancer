from pathlib import Path

from . import families, storage, workflows

# Only the bot's personality: chat.py appends the mechanics the app relies on
# (how to deliver a prompt, take a note, title the conversation), so editing
# this text in Ajustes can't break them.
DEFAULT_CONVERSATION_PERSONALITY = """You are the user's partner in crime for AI image prompts: a seasoned, opinionated collaborator with a photographer's and an art director's eye, not an assistant waiting for orders.

- Propose concrete prompts early instead of asking a string of questions; ask only when the intent is genuinely unclear.
- Say briefly why you made the important choices, and be honest when an idea won't work or when you would do it differently.
- Build on what the user shares: what they liked, what failed, the images they send.
- Be direct and concise. No flattery, no filler.
"""

# One editable system prompt per mode. Only Conversar's is left: the one-shot modes of the
# old Crear page are gone (their texts stay, unused, in deployments that had them).
DEFAULT_SYSTEM_PROMPTS = {
    "conversation": DEFAULT_CONVERSATION_PERSONALITY,
}

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
    system_prompts_path = f"{data_dir}/system_prompts.json"

    if not Path(families_path).exists():
        families.create_family("SDXL", SDXL_INSTRUCTIONS, True, path=families_path)
        families.create_family("Z-Image-Turbo", ZIT_INSTRUCTIONS, False, path=families_path)

    workflows.ensure_seeded(f"{data_dir}/workflows.json")

    # System prompts are seeded per mode rather than per file: a data dir created
    # before a mode existed would otherwise run that mode with an empty system
    # prompt. Modes already present -- edited or not -- are never touched.
    existing = storage.read_json(system_prompts_path, default={})
    missing = {mode: text for mode, text in DEFAULT_SYSTEM_PROMPTS.items() if mode not in existing}
    if missing:
        storage.write_json(system_prompts_path, {**existing, **missing})
