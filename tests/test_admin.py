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


def test_create_list_update_delete_character(api_client, auth_headers):
    create_response = api_client.post(
        "/api/admin/characters",
        json={"name": "Warrior", "text": "a fierce warrior"},
        auth=auth_headers,
    )
    assert create_response.status_code == 200
    character_id = create_response.json()["id"]

    update_response = api_client.put(
        f"/api/admin/characters/{character_id}",
        json={"name": "Warrior v2", "text": "an even fiercer warrior"},
        auth=auth_headers,
    )
    assert update_response.status_code == 200
    assert update_response.json()["name"] == "Warrior v2"

    delete_response = api_client.delete(f"/api/admin/characters/{character_id}", auth=auth_headers)
    assert delete_response.status_code == 200
    assert api_client.get("/api/admin/characters", auth=auth_headers).json() == []


def test_get_and_update_system_prompt_for_each_mode(api_client, auth_headers):
    for mode in ["generate", "iterate", "image", "extract_character"]:
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


def test_system_prompt_modes_are_independent(api_client, auth_headers):
    api_client.put(
        "/api/admin/system-prompt/generate", json={"text": "generate text"}, auth=auth_headers
    )

    assert api_client.get("/api/admin/system-prompt/iterate", auth=auth_headers).json() == {
        "text": ""
    }


def test_get_system_prompt_rejects_unknown_mode(api_client, auth_headers):
    response = api_client.get("/api/admin/system-prompt/bogus", auth=auth_headers)

    assert response.status_code == 400


def test_update_system_prompt_rejects_unknown_mode(api_client, auth_headers):
    response = api_client.put(
        "/api/admin/system-prompt/bogus", json={"text": "x"}, auth=auth_headers
    )

    assert response.status_code == 400
