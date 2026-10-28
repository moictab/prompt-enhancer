from unittest.mock import patch

from app.prompts import (
    build_system_prompt,
    build_user_message,
    parse_character_response,
    parse_response,
    parse_multi_response,
)


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


def test_build_user_message_story_mode():
    result = build_user_message("story", "two characters have dinner", count=4)

    assert result == (
        "## Historia\ntwo characters have dinner\n\n"
        "## Numero de Episodios\nGenera exactamente 4 episodios siguiendo el formato indicado."
    )


def test_build_user_message_story_mode_with_characters():
    result = build_user_message(
        "story", "two characters have dinner", count=2,
        characters=[{"name": "Kaito", "text": "a stoic ronin with a scarred left eye"}],
    )

    assert "## Personajes" in result
    assert "Kaito" in result
    assert result.index("## Historia") < result.index("## Personajes")


def test_parse_multi_response_splits_episodes_in_order():
    response = (
        "EPISODIO 1:\nPOSITIVE: two friends sit down to dinner\nNEGATIVE: blurry\n\n"
        "EPISODIO 2:\nPOSITIVE: the mood turns flirtatious\nNEGATIVE: blurry\n\n"
        "EPISODIO 3:\nPOSITIVE: cards are dealt for strip poker\nNEGATIVE: blurry"
    )

    episodes = parse_multi_response(response, has_negative_prompt=True)

    assert episodes == [
        ("two friends sit down to dinner", "blurry"),
        ("the mood turns flirtatious", "blurry"),
        ("cards are dealt for strip poker", "blurry"),
    ]


def test_parse_multi_response_is_case_insensitive_on_episode_marker():
    response = "episodio 1:\nPOSITIVE: a quiet dinner\nNEGATIVE: \n\nepisodio 2:\nPOSITIVE: things escalate\nNEGATIVE: "

    episodes = parse_multi_response(response, has_negative_prompt=True)

    assert [positive for positive, _ in episodes] == ["a quiet dinner", "things escalate"]


def test_parse_multi_response_forces_empty_negative_when_family_does_not_use_one():
    response = "EPISODIO 1:\nPOSITIVE: a flowing scene\nNEGATIVE: blurry"

    episodes = parse_multi_response(response, has_negative_prompt=False)

    assert episodes == [("a flowing scene", "")]


def test_parse_multi_response_falls_back_to_single_episode_when_markers_missing():
    response = "POSITIVE: just one scene\nNEGATIVE: blurry"

    episodes = parse_multi_response(response, has_negative_prompt=True)

    assert episodes == [("just one scene", "blurry")]


def test_parse_multi_response_splits_by_repeated_markers_without_episode_headers():
    # Some models don't reliably emit the literal "EPISODIO N:" heading, so the
    # parser must not depend on it -- only on the POSITIVE:/NEGATIVE: pairs.
    response = (
        "Here is the story:\n\n"
        "POSITIVE: two friends sit down to dinner\nNEGATIVE: blurry\n\n"
        "POSITIVE: the mood turns flirtatious\nNEGATIVE: blurry\n\n"
        "POSITIVE: cards are dealt for strip poker\nNEGATIVE: blurry"
    )

    episodes = parse_multi_response(response, has_negative_prompt=True)

    assert episodes == [
        ("two friends sit down to dinner", "blurry"),
        ("the mood turns flirtatious", "blurry"),
        ("cards are dealt for strip poker", "blurry"),
    ]


def test_parse_multi_response_does_not_bleed_negative_across_episodes():
    response = (
        "POSITIVE: first scene\nNEGATIVE: blurry\n\n"
        "POSITIVE: second scene without its own negative"
    )

    episodes = parse_multi_response(response, has_negative_prompt=True)

    assert episodes == [
        ("first scene", "blurry"),
        ("second scene without its own negative", ""),
    ]


def test_parse_multi_response_strips_markdown_and_bold_headings_without_negatives():
    response = (
        "### Variación 1\nPOSITIVE: first look\n\n"
        "**VARIACION 2:**\nPOSITIVE: second look\n"
        "### Variación 3\nPOSITIVE: third look"
    )

    blocks = parse_multi_response(response, has_negative_prompt=False)

    assert [positive for positive, _ in blocks] == ["first look", "second look", "third look"]


def test_parse_multi_response_does_not_return_heading_as_empty_negative():
    response = (
        "VARIACION 1:\nPOSITIVE: first look\nNEGATIVE:\n\n"
        "VARIACION 2:\nPOSITIVE: second look\nNEGATIVE: blurry"
    )

    blocks = parse_multi_response(response, has_negative_prompt=True)

    assert blocks == [("first look", ""), ("second look", "blurry")]


def test_parse_multi_response_keeps_ordinary_last_lines():
    response = "POSITIVE: a quiet harbour at dawn,\nmist over the water\nNEGATIVE: blurry"

    blocks = parse_multi_response(response, has_negative_prompt=True)

    assert blocks == [("a quiet harbour at dawn,\nmist over the water", "blurry")]


def test_parse_character_response_with_markers():
    response = "NAME: Kaito the Ronin\nTEXT: a stoic ronin with a scarred left eye and silver hair"

    name, text = parse_character_response(response)

    assert name == "Kaito the Ronin"
    assert text == "a stoic ronin with a scarred left eye and silver hair"


def test_parse_character_response_is_case_insensitive():
    response = "name: Mika\ntext: a neon-haired hacker in a trench coat"

    name, text = parse_character_response(response)

    assert name == "Mika"
    assert text == "a neon-haired hacker in a trench coat"


def test_parse_character_response_falls_back_to_empty_name_when_markers_missing():
    response = "Just a plain description of the character with no markers."

    name, text = parse_character_response(response)

    assert name == ""
    assert text == "Just a plain description of the character with no markers."
