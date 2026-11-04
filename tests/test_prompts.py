from app.prompts import parse_response


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
