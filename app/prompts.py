import re

from . import system_prompt


def build_system_prompt(mode: str, family: dict) -> str:
    return f"{system_prompt.read_system_prompt(mode).strip()}\n\n{family['instructions'].strip()}"


CHARACTERS_INSTRUCTION = (
    "Incluye a los siguientes personajes en el prompt final tal como se describen a "
    "continuacion, sin modificar ni parafrasear su texto. Completa de forma coherente "
    "cualquier detalle de cada personaje que no este especificado."
)


# Aspects of a shot that the Variaciones mode can hold fixed or vary. Keys are
# what the API accepts; labels are what both the UI and the LLM see.
VARIATION_AXES = {
    "outfit": "Ropa y accesorios",
    "setting": "Escenario y situación",
    "lighting": "Iluminación",
    "pose": "Pose y acción",
    "camera": "Encuadre y cámara",
    "expression": "Expresión",
}


def _variation_axes_parts(axes: list[dict]) -> list[str]:
    fixed_lines = []
    varying_lines = []
    for axis in axes:
        label = VARIATION_AXES[axis["key"]]
        value = (axis.get("value") or "").strip()
        if axis["vary"]:
            varying_lines.append(f"- {label} (pistas: {value})" if value else f"- {label}")
        else:
            fixed_lines.append(
                f"- {label}: {value}" if value
                else f"- {label}: elige uno y mantenlo identico en todas"
            )

    parts = []
    if fixed_lines:
        parts.append(
            "## Aspectos Fijos (identicos en todas las variaciones)\n" + "\n".join(fixed_lines)
        )
    if varying_lines:
        parts.append(
            "## Aspectos que Varian (distintos en cada variacion)\n" + "\n".join(varying_lines)
        )
    return parts


def build_user_message(
    mode: str,
    user_input: str,
    previous_prompt: str | None = None,
    example_prompts: str | None = None,
    characters: list[dict] | None = None,
    count: int | None = None,
    aesthetic: str | None = None,
    axes: list[dict] | None = None,
    variant: tuple[int, int] | None = None,
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
        parts.append(f"## Numero de Episodios\nGenera exactamente {count} episodios siguiendo el formato indicado.")
    elif mode == "variations":
        if user_input and user_input.strip():
            parts.append(f"## Idea Base\n{user_input.strip()}")
        if aesthetic and aesthetic.strip():
            parts.append(f"## Estetica (identica en todas las variaciones)\n{aesthetic.strip()}")
        parts.extend(_variation_axes_parts(axes or []))
        parts.append(f"## Numero de Variaciones\nGenera exactamente {count} variaciones siguiendo el formato indicado.")
    else:
        parts.append(f"## Idea\n{user_input}")

    if characters:
        character_lines = "\n".join(f"- {c['name']}: {c['text']}" for c in characters)
        parts.append(f"## Personajes\n{CHARACTERS_INSTRUCTION}\n\n{character_lines}")

    if example_prompts and example_prompts.strip():
        parts.append(f"## Prompts de Ejemplo\n{example_prompts.strip()}")

    if variant and variant[1] > 1:
        index, total = variant
        parts.append(
            f"## Alternativa\nSe estan generando {total} alternativas independientes para "
            f"esta misma peticion; esta es la numero {index}. Toma tus propias decisiones "
            "creativas (composicion, detalles, enfoque) para que sea una interpretacion "
            "propia, sin contradecir nada de lo especificado."
        )

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


def _is_block_header(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 80:
        return False
    return (
        line.startswith("#")
        or line.strip("*_ ").endswith(":")
        or (line.startswith(("**", "__")) and line.endswith(("**", "__")))
    )


def _strip_trailing_block_header(text: str) -> str:
    # Whatever heading style the model used to separate blocks ("EPISODIO 2:",
    # "**VARIACION 3:**", a markdown "## Episode 4" ...) can leak into the end of
    # the previous block's text, since parsing only anchors on POSITIVE:/NEGATIVE:.
    # Peel off a short dangling heading line like that if present -- or return ""
    # when the heading is all there is (an empty NEGATIVE: right before it).
    stripped = text.strip()
    if "\n" not in stripped and _is_block_header(stripped):
        return ""
    body, separator, tail = stripped.rpartition("\n")
    if separator and _is_block_header(tail):
        return body.rstrip()
    return stripped


def parse_multi_response(response: str, has_negative_prompt: bool) -> list[tuple[str, str]]:
    positive_matches = list(POSITIVE_PATTERN.finditer(response))
    if not positive_matches:
        return [parse_response(response, has_negative_prompt)]

    negative_matches = list(NEGATIVE_PATTERN.finditer(response))

    blocks = []
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
            negative = _strip_trailing_block_header(response[neg_match.end():next_pos_start])
        else:
            positive = _strip_trailing_block_header(response[pos_start:next_pos_start])
            negative = ""
        if not has_negative_prompt:
            negative = ""
        blocks.append((positive, negative))
    return blocks


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
