from unittest.mock import patch

from app import characters, families, history
from app.openrouter_client import OpenRouterResult


def _create_family(tmp_path, has_negative_prompt=True):
    return families.create_family(
        "SDXL", "family instructions", has_negative_prompt,
        path=str(tmp_path / "families.json"),
    )


VARIATIONS_RESPONSE = (
    "VARIATION 1:\nPOSITIVE: Mara in a red raincoat on a rooftop\nNEGATIVE: blurry\n\n"
    "VARIATION 2:\nPOSITIVE: Mara in a linen dress in a market\nNEGATIVE: blurry\n\n"
    "VARIATION 3:\nPOSITIVE: Mara in a leather jacket in a diner\nNEGATIVE: blurry"
)

AXES = [
    {"key": "outfit", "vary": True, "value": "ropa de calle"},
    {"key": "setting", "vary": True, "value": ""},
    {"key": "lighting", "vary": False, "value": "luz de neon rosa"},
    {"key": "camera", "vary": False, "value": ""},
]


def _payload(family, **overrides):
    payload = {
        "family_id": family["id"],
        "llm_model": "m",
        "aesthetic": "fotografia analogica, grano de 35mm",
        "axes": AXES,
        "count": 3,
    }
    payload.update(overrides)
    return payload


def test_variations_requires_auth(api_client):
    response = api_client.post("/api/variations", json={
        "family_id": "x", "llm_model": "m", "axes": AXES,
    })

    assert response.status_code == 401


@patch("app.routes.api.call_openrouter")
def test_variations_returns_parsed_prompts(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = OpenRouterResult(content=VARIATIONS_RESPONSE, cost=0.004)

    response = api_client.post("/api/variations", json=_payload(family), auth=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "variations"
    assert [p["positive_prompt"] for p in body["prompts"]] == [
        "Mara in a red raincoat on a rooftop",
        "Mara in a linen dress in a market",
        "Mara in a leather jacket in a diner",
    ]
    assert all(p["negative_prompt"] == "blurry" for p in body["prompts"])
    assert body["count_mismatch"] is False
    assert body["cost"] == 0.004


@patch("app.routes.api.call_openrouter")
def test_variations_sends_constants_and_axes_to_llm(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    character = characters.create_character(
        "Mara", "a tall woman with a silver bob haircut", path=str(tmp_path / "characters.json")
    )
    mock_call.return_value = OpenRouterResult(content=VARIATIONS_RESPONSE, cost=0.004)

    api_client.post(
        "/api/variations",
        json=_payload(family, user_input="retratos urbanos", character_ids=[character["id"]]),
        auth=auth_headers,
    )

    kwargs = mock_call.call_args.kwargs
    message = kwargs["user_message"]
    assert "## Base Idea\nretratos urbanos" in message
    assert "fotografia analogica, grano de 35mm" in message
    assert "a tall woman with a silver bob haircut" in message
    assert "- Lighting: luz de neon rosa" in message
    assert "- Framing and camera: choose one and keep it identical in every variation" in message
    assert "- Outfit and accessories (hints: ropa de calle)" in message
    assert "- Setting and situation\n" in message
    assert "Generate exactly 3 variations" in message
    assert mock_call.call_count == 1


@patch("app.routes.api.call_openrouter")
def test_variations_flags_count_mismatch(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = OpenRouterResult(content=VARIATIONS_RESPONSE, cost=0.004)

    body = api_client.post(
        "/api/variations", json=_payload(family, count=5), auth=auth_headers
    ).json()

    assert len(body["prompts"]) == 3
    assert body["count_mismatch"] is True


@patch("app.routes.api.call_openrouter")
def test_variations_suppresses_negative_when_family_has_none(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path, has_negative_prompt=False)
    mock_call.return_value = OpenRouterResult(content=VARIATIONS_RESPONSE, cost=0.004)

    body = api_client.post("/api/variations", json=_payload(family), auth=auth_headers).json()

    assert all(p["negative_prompt"] == "" for p in body["prompts"])


@patch("app.routes.api.call_openrouter")
def test_variations_appends_history_with_params(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = OpenRouterResult(content=VARIATIONS_RESPONSE, cost=0.004)

    api_client.post("/api/variations", json=_payload(family), auth=auth_headers)

    [entry] = history.list_entries(path=str(tmp_path / "history.jsonl"))
    assert entry["mode"] == "variations"
    assert entry["params"]["count"] == 3
    assert entry["params"]["aesthetic"] == "fotografia analogica, grano de 35mm"
    assert [{k: a[k] for k in ("key", "vary", "value")} for a in entry["params"]["axes"]] == AXES
    assert [a["label"] for a in entry["params"]["axes"]][:2] == [
        "Ropa y accesorios", "Escenario y situación",
    ]
    assert len(entry["prompts"]) == 3


@patch("app.routes.api.call_openrouter")
def test_variations_sends_reference_prompts_to_llm(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = OpenRouterResult(content=VARIATIONS_RESPONSE, cost=0.004)

    api_client.post(
        "/api/variations",
        json=_payload(family, reference_prompts=["  Mara on a rooftop  ", "", "Mara in a diner"]),
        auth=auth_headers,
    )

    message = mock_call.call_args.kwargs["user_message"]
    assert "## Reference Prompts" in message
    # Blank references are dropped before numbering.
    assert "### Reference 1\nMara on a rooftop\n\n### Reference 2\nMara in a diner" in message
    assert "Reference 3" not in message


@patch("app.routes.api.call_openrouter")
def test_variations_accepts_reference_prompts_as_the_only_constant(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = OpenRouterResult(content=VARIATIONS_RESPONSE, cost=0.004)

    response = api_client.post(
        "/api/variations",
        json=_payload(family, aesthetic="", reference_prompts=["Mara on a rooftop"]),
        auth=auth_headers,
    )

    assert response.status_code == 200


@patch("app.routes.api.call_openrouter")
def test_variations_stores_reference_prompts_in_history_params(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = OpenRouterResult(content=VARIATIONS_RESPONSE, cost=0.004)

    body = api_client.post(
        "/api/variations",
        json=_payload(family, reference_prompts=["Mara on a rooftop ", "  "]),
        auth=auth_headers,
    ).json()

    assert body["params"]["reference_prompts"] == ["Mara on a rooftop"]
    [entry] = history.list_entries(path=str(tmp_path / "history.jsonl"))
    assert entry["params"]["reference_prompts"] == ["Mara on a rooftop"]


@patch("app.routes.api.call_openrouter")
def test_variations_without_references_leaves_them_out_of_params(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = OpenRouterResult(content=VARIATIONS_RESPONSE, cost=0.004)

    body = api_client.post("/api/variations", json=_payload(family), auth=auth_headers).json()

    assert "reference_prompts" not in body["params"]
    assert "Reference" not in mock_call.call_args.kwargs["user_message"]


def test_variations_rejects_too_many_reference_prompts(api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)

    response = api_client.post(
        "/api/variations",
        json=_payload(family, reference_prompts=[f"prompt {i}" for i in range(13)]),
        auth=auth_headers,
    )

    assert response.status_code == 422


def test_variations_rejects_unknown_axis(api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)

    response = api_client.post(
        "/api/variations",
        json=_payload(family, axes=[{"key": "weather", "vary": True}]),
        auth=auth_headers,
    )

    assert response.status_code == 400
    assert "weather" in response.json()["detail"]


def test_variations_rejects_when_nothing_varies(api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)

    response = api_client.post(
        "/api/variations",
        json=_payload(family, axes=[{"key": "outfit", "vary": False}]),
        auth=auth_headers,
    )

    assert response.status_code == 400


def test_variations_rejects_when_nothing_is_kept_constant(api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)

    response = api_client.post(
        "/api/variations",
        json=_payload(family, aesthetic="  ", user_input="", character_ids=["missing"]),
        auth=auth_headers,
    )

    assert response.status_code == 400


def test_variations_rejects_unknown_family(api_client, auth_headers):
    response = api_client.post(
        "/api/variations",
        json={"family_id": "nope", "llm_model": "m", "aesthetic": "x", "axes": AXES},
        auth=auth_headers,
    )

    assert response.status_code == 404


def test_variations_rejects_count_out_of_range(api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)

    assert api_client.post(
        "/api/variations", json=_payload(family, count=1), auth=auth_headers
    ).status_code == 422
    assert api_client.post(
        "/api/variations", json=_payload(family, count=13), auth=auth_headers
    ).status_code == 422


@patch("app.routes.api.call_openrouter")
def test_variations_returns_502_on_openrouter_error(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.side_effect = RuntimeError("OpenRouter rate limit exceeded. Wait a moment and try again.")

    response = api_client.post("/api/variations", json=_payload(family), auth=auth_headers)

    assert response.status_code == 502
