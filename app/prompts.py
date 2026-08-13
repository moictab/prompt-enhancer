import re

from . import system_prompt


def build_system_prompt(mode: str, family: dict) -> str:
    return f"{system_prompt.read_system_prompt(mode).strip()}\n\n{family['instructions'].strip()}"


CHARACTERS_INSTRUCTION = (
    "Incluye a los siguientes personajes en el prompt final tal como se describen a "
    "continuacion, sin modificar ni parafrasear su texto. Completa de forma coherente "
    "cualquier detalle de cada personaje que no este especificado."
)


def build_user_message(
    mode: str,
    user_input: str,
    previous_prompt: str | None = None,
    example_prompts: str | None = None,
    characters: list[dict] | None = None,
) -> str:
    parts = []
    if mode == "iterate":
        parts.append(f"## Prompt Previo\n{previous_prompt}")
        parts.append(f"## Cambios Solicitados\n{user_input}")
    elif mode == "image":
        parts.append("## Imagen\nSe adjunta una imagen de referencia para este prompt.")
        if user_input and user_input.strip():
            parts.append(f"## Idea\n{user_input.strip()}")
    else:
        parts.append(f"## Idea\n{user_input}")

    if characters:
        character_lines = "\n".join(f"- {c['name']}: {c['text']}" for c in characters)
        parts.append(f"## Personajes\n{CHARACTERS_INSTRUCTION}\n\n{character_lines}")

    if example_prompts and example_prompts.strip():
        parts.append(f"## Prompts de Ejemplo\n{example_prompts.strip()}")

    return "\n\n".join(parts)


def parse_response(response: str, has_negative_prompt: bool) -> tuple[str, str]:
    positive = ""
    negative = ""

    response_upper = response.upper()
    pos_idx = response_upper.find("POSITIVE:")
    neg_idx = response_upper.find("NEGATIVE:")

    if pos_idx != -1:
        pos_start = pos_idx + len("POSITIVE:")
        if neg_idx != -1 and neg_idx > pos_idx:
            positive = response[pos_start:neg_idx].strip()
        else:
            positive = response[pos_start:].strip()
        if neg_idx != -1:
            neg_start = neg_idx + len("NEGATIVE:")
            negative = response[neg_start:].strip()
    else:
        lines = response.strip().split("\n")
        preamble_pattern = re.compile(
            r"^(sure|here|i |i'|of course|certainly|let me|this is|note:|hope|feel free)",
            re.IGNORECASE,
        )
        while lines and preamble_pattern.match(lines[0].strip()):
            lines.pop(0)
        while lines and preamble_pattern.match(lines[-1].strip()):
            lines.pop()
        positive = "\n".join(lines).strip()

    if not has_negative_prompt:
        negative = ""

    return (positive, negative)


def parse_character_response(response: str) -> tuple[str, str]:
    name = ""
    text = ""

    response_upper = response.upper()
    name_idx = response_upper.find("NAME:")
    text_idx = response_upper.find("TEXT:")

    if name_idx != -1:
        name_start = name_idx + len("NAME:")
        if text_idx != -1 and text_idx > name_idx:
            name = response[name_start:text_idx].strip()
        else:
            name = response[name_start:].strip()
        if text_idx != -1:
            text_start = text_idx + len("TEXT:")
            text = response[text_start:].strip()
    else:
        text = response.strip()

    return (name, text)
