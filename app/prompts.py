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
    num_episodes: int | None = None,
) -> str:
    parts = []
    if mode == "iterate":
        parts.append(f"## Prompt Previo\n{previous_prompt}")
        parts.append(f"## Cambios Solicitados\n{user_input}")
    elif mode == "image":
        parts.append("## Imagen\nSe adjunta una imagen de referencia para este prompt.")
        if user_input and user_input.strip():
            parts.append(f"## Idea\n{user_input.strip()}")
    elif mode == "story":
        parts.append(f"## Historia\n{user_input}")
        parts.append(f"## Numero de Episodios\nGenera exactamente {num_episodes} episodios siguiendo el formato indicado.")
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


POSITIVE_PATTERN = re.compile(r"POSITIVE\s*:", re.IGNORECASE)
NEGATIVE_PATTERN = re.compile(r"NEGATIVE\s*:", re.IGNORECASE)


def _strip_trailing_episode_header(text: str) -> str:
    # Whatever heading style the model used to separate episodes ("EPISODIO 2:",
    # "Scene 3:", a markdown "## Episode 4" ...) can leak into the end of the
    # previous episode's text, since parsing only anchors on POSITIVE:/NEGATIVE:.
    # Peel off a short, colon-terminated dangling line like that if present.
    stripped = text.strip()
    body, separator, tail = stripped.rpartition("\n\n")
    if separator and tail and len(tail) <= 80 and "\n" not in tail and tail.endswith(":"):
        return body.rstrip()
    return stripped


def parse_story_response(response: str, has_negative_prompt: bool) -> list[tuple[str, str]]:
    positive_matches = list(POSITIVE_PATTERN.finditer(response))
    if not positive_matches:
        return [parse_response(response, has_negative_prompt)]

    negative_matches = list(NEGATIVE_PATTERN.finditer(response))

    episodes = []
    for i, pos_match in enumerate(positive_matches):
        pos_start = pos_match.end()
        next_pos_start = (
            positive_matches[i + 1].start() if i + 1 < len(positive_matches) else len(response)
        )
        neg_match = next(
            (m for m in negative_matches if pos_match.end() <= m.start() < next_pos_start), None
        )
        if neg_match:
            positive = response[pos_start:neg_match.start()].strip()
            negative = _strip_trailing_episode_header(response[neg_match.end():next_pos_start])
        else:
            positive = _strip_trailing_episode_header(response[pos_start:next_pos_start])
            negative = ""
        if not has_negative_prompt:
            negative = ""
        episodes.append((positive, negative))
    return episodes


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
