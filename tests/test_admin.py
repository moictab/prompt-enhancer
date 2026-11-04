def test_admin_families_requires_auth(api_client):
    response = api_client.get("/api/admin/families")

    assert response.status_code == 401


def test_create_list_update_delete_family(api_client, auth_headers):
    create_response = api_client.post(
        "/api/admin/families",
        json={"name": "SDXL", "instructions": "rules", "has_negative_prompt": True},
        auth=auth_headers,
    )
    assert create_response.status_code == 200
    family_id = create_response.json()["id"]

    list_response = api_client.get("/api/admin/families", auth=auth_headers)
    assert len(list_response.json()) == 1

    update_response = api_client.put(
        f"/api/admin/families/{family_id}",
        json={"name": "SDXL v2", "instructions": "new rules", "has_negative_prompt": False},
        auth=auth_headers,
    )
    assert update_response.status_code == 200
    assert update_response.json()["name"] == "SDXL v2"

    delete_response = api_client.delete(f"/api/admin/families/{family_id}", auth=auth_headers)
    assert delete_response.status_code == 200
    assert api_client.get("/api/admin/families", auth=auth_headers).json() == []


def test_update_unknown_family_returns_404(api_client, auth_headers):
    response = api_client.put(
        "/api/admin/families/nonexistent",
        json={"name": "x", "instructions": "y", "has_negative_prompt": True},
        auth=auth_headers,
    )

    assert response.status_code == 404


def test_delete_unknown_family_returns_404(api_client, auth_headers):
    response = api_client.delete("/api/admin/families/nonexistent", auth=auth_headers)

    assert response.status_code == 404


def test_get_and_update_the_bots_personality(api_client, auth_headers):
    for mode in ["conversation"]:
        get_response = api_client.get(f"/api/admin/system-prompt/{mode}", auth=auth_headers)
        assert get_response.status_code == 200
        assert get_response.json() == {"text": ""}

        put_response = api_client.put(
            f"/api/admin/system-prompt/{mode}",
            json={"text": f"You are an expert at {mode}."},
            auth=auth_headers,
        )
        assert put_response.status_code == 200

        assert api_client.get(f"/api/admin/system-prompt/{mode}", auth=auth_headers).json() == {
            "text": f"You are an expert at {mode}."
        }


def test_the_one_shot_modes_system_prompts_are_gone(api_client, auth_headers):
    for mode in ["generate", "image", "story", "variations", "extract_character"]:
        assert api_client.get(f"/api/admin/system-prompt/{mode}", auth=auth_headers).status_code == 400


def test_retired_iterate_system_prompt_mode_is_rejected(api_client, auth_headers):
    response = api_client.get("/api/admin/system-prompt/iterate", auth=auth_headers)

    assert response.status_code == 400


def test_create_list_update_delete_memory_note(api_client, auth_headers):
    created = api_client.post(
        "/api/admin/memory", json={"text": "  Prefers hard light  "}, auth=auth_headers
    ).json()
    assert created["text"] == "Prefers hard light"

    assert [n["text"] for n in api_client.get("/api/admin/memory", auth=auth_headers).json()] == [
        "Prefers hard light"
    ]

    updated = api_client.put(
        f"/api/admin/memory/{created['id']}", json={"text": "Prefers soft light"}, auth=auth_headers
    ).json()
    assert updated["text"] == "Prefers soft light"

    assert api_client.delete(f"/api/admin/memory/{created['id']}", auth=auth_headers).status_code == 200
    assert api_client.get("/api/admin/memory", auth=auth_headers).json() == []


def test_memory_rejects_blank_notes_and_unknown_ids(api_client, auth_headers):
    assert api_client.post("/api/admin/memory", json={"text": " "}, auth=auth_headers).status_code == 400
    assert api_client.put(
        "/api/admin/memory/nope", json={"text": "x"}, auth=auth_headers
    ).status_code == 404
    assert api_client.delete("/api/admin/memory/nope", auth=auth_headers).status_code == 404


def test_get_system_prompt_rejects_unknown_mode(api_client, auth_headers):
    response = api_client.get("/api/admin/system-prompt/bogus", auth=auth_headers)

    assert response.status_code == 400


def test_update_system_prompt_rejects_unknown_mode(api_client, auth_headers):
    response = api_client.put(
        "/api/admin/system-prompt/bogus", json={"text": "x"}, auth=auth_headers
    )

    assert response.status_code == 400


def test_get_default_system_prompt_returns_seed_text_without_touching_saved_prompt(
    api_client, auth_headers
):
    from app.seed import DEFAULT_CONVERSATION_PERSONALITY

    api_client.put(
        "/api/admin/system-prompt/conversation", json={"text": "custom"}, auth=auth_headers
    )

    response = api_client.get("/api/admin/system-prompt/conversation/default", auth=auth_headers)

    assert response.status_code == 200
    assert response.json() == {"text": DEFAULT_CONVERSATION_PERSONALITY}
    assert api_client.get(
        "/api/admin/system-prompt/conversation", auth=auth_headers
    ).json() == {"text": "custom"}


def test_get_default_system_prompt_rejects_unknown_mode(api_client, auth_headers):
    response = api_client.get("/api/admin/system-prompt/bogus/default", auth=auth_headers)

    assert response.status_code == 400
