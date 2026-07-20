import re

ITERATION_ADDENDUM = """

## Iteration Mode
The user is refining an EXISTING prompt. You will receive:
1. The previous prompt that was already generated
2. The user's requested changes

Your job is to:
- PRESERVE the good elements from the previous prompt -- do not regenerate from scratch
- Apply the user's requested changes precisely
- Maintain the same overall style and structure
- Only modify what the user explicitly asks to change
- If the user asks to "add" something, integrate it naturally into the existing prompt
- If the user asks to "remove" something, take it out cleanly without leaving gaps
"""


def build_system_prompt(global_system_prompt: str, family: dict, is_iteration: bool = False) -> str:
    prompt = f"{global_system_prompt.strip()}\n\n{family['instructions'].strip()}"
    if is_iteration:
        prompt += ITERATION_ADDENDUM
    return prompt


def build_user_message(
    mode: str,
    user_input: str,
    previous_prompt: str | None = None,
    example_prompts: str | None = None,
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
