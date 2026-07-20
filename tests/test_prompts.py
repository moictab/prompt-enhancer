from app.prompts import build_system_prompt, build_user_message, parse_response


def test_build_system_prompt_combines_global_and_family():
    family = {"instructions": "family rules"}

    result = build_system_prompt("global rules", family, is_iteration=False)

    assert result == "global rules\n\nfamily rules"


def test_build_system_prompt_appends_iteration_addendum():
    family = {"instructions": "family rules"}

    result = build_system_prompt("global rules", family, is_iteration=True)

    assert result.startswith("global rules\n\nfamily rules")
    assert "Iteration Mode" in result
    assert "PRESERVE the good elements" in result


def test_build_user_message_generate_mode():
    result = build_user_message("generate", "a cyberpunk samurai")

    assert result == "## Idea\na cyberpunk samurai"


def test_build_user_message_generate_mode_with_example_prompts():
    result = build_user_message("generate", "a cyberpunk samurai", example_prompts="cinematic, moody")

    assert result == "## Idea\na cyberpunk samurai\n\n## Prompts de Ejemplo\ncinematic, moody"


def test_build_user_message_iterate_mode():
    result = build_user_message("iterate", "add lightning", previous_prompt="a samurai in rain")

    assert result == "## Prompt Previo\na samurai in rain\n\n## Cambios Solicitados\nadd lightning"


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
