from unittest.mock import patch

from app.prompts import build_system_prompt, build_user_message, parse_response


@patch("app.prompts.system_prompt.read_system_prompt")
def test_build_system_prompt_combines_mode_prompt_and_family(mock_read):
    mock_read.return_value = "global rules"
    family = {"instructions": "family rules"}

    result = build_system_prompt("generate", family)

    assert result == "global rules\n\nfamily rules"
    mock_read.assert_called_once_with("generate")


@patch("app.prompts.system_prompt.read_system_prompt")
def test_build_system_prompt_uses_iterate_mode_prompt(mock_read):
    mock_read.return_value = "iterate rules"
    family = {"instructions": "family rules"}

    result = build_system_prompt("iterate", family)

    assert result == "iterate rules\n\nfamily rules"
    mock_read.assert_called_once_with("iterate")


def test_build_user_message_generate_mode():
    result = build_user_message("generate", "a cyberpunk samurai")

    assert result == "## Idea\na cyberpunk samurai"


def test_build_user_message_generate_mode_with_example_prompts():
    result = build_user_message("generate", "a cyberpunk samurai", example_prompts="cinematic, moody")

    assert result == "## Idea\na cyberpunk samurai\n\n## Prompts de Ejemplo\ncinematic, moody"


def test_build_user_message_iterate_mode():
    result = build_user_message("iterate", "add lightning", previous_prompt="a samurai in rain")

    assert result == "## Prompt Previo\na samurai in rain\n\n## Cambios Solicitados\nadd lightning"


def test_build_user_message_with_characters_adds_section_with_instruction():
    result = build_user_message(
        "generate", "a cyberpunk samurai",
        characters=[{"name": "Kaito", "text": "a stoic ronin with a scarred left eye"}],
    )

    assert result == (
        "## Idea\na cyberpunk samurai\n\n"
        "## Personajes\n"
        "Incluye a los siguientes personajes en el prompt final tal como se describen a "
        "continuacion, sin modificar ni parafrasear su texto. Completa de forma coherente "
        "cualquier detalle de cada personaje que no este especificado.\n\n"
        "- Kaito: a stoic ronin with a scarred left eye"
    )


def test_build_user_message_with_multiple_characters_lists_each_on_its_own_line():
    result = build_user_message(
        "generate", "a cyberpunk samurai",
        characters=[
            {"name": "Kaito", "text": "a stoic ronin with a scarred left eye"},
            {"name": "Mika", "text": "a neon-haired hacker in a trench coat"},
        ],
    )

    assert result.endswith(
        "- Kaito: a stoic ronin with a scarred left eye\n"
        "- Mika: a neon-haired hacker in a trench coat"
    )


def test_build_user_message_omits_characters_section_when_none_selected():
    result = build_user_message("generate", "a cyberpunk samurai", characters=[])

    assert "Personajes" not in result


def test_build_user_message_omits_characters_section_when_not_passed():
    result = build_user_message("generate", "a cyberpunk samurai")

    assert "Personajes" not in result


def test_build_user_message_image_mode_without_extra_text():
    result = build_user_message("image", "")

    assert result == "## Imagen\nSe adjunta una imagen de referencia para este prompt."


def test_build_user_message_image_mode_with_extra_text():
    result = build_user_message("image", "make it night time")

    assert result == (
        "## Imagen\nSe adjunta una imagen de referencia para este prompt.\n\n"
        "## Idea\nmake it night time"
    )


def test_parse_response_with_markers():
    response = "POSITIVE: a cyberpunk samurai\nNEGATIVE: blurry, extra fingers"

    positive, negative = parse_response(response, has_negative_prompt=True)

    assert positive == "a cyberpunk samurai"
    assert negative == "blurry, extra fingers"


def test_parse_response_forces_empty_negative_when_family_does_not_use_one():
    response = "POSITIVE: a flowing scene\nNEGATIVE: blurry"

    positive, negative = parse_response(response, has_negative_prompt=False)

    assert positive == "a flowing scene"
    assert negative == ""


def test_parse_response_falls_back_when_markers_missing():
    response = "Sure, here's your prompt:\na cyberpunk samurai in neon rain\nHope this helps!"

    positive, negative = parse_response(response, has_negative_prompt=True)

    assert positive == "a cyberpunk samurai in neon rain"
    assert negative == ""
