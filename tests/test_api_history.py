from app import history


def _append_entry(path, user_input):
    return history.append_entry(
        mode="generate", family_id="f", family_name="SDXL", llm_model="m",
        vision_model=None, temperature=0.7, user_input=user_input, example_prompts="",
        previous_prompt=None,
        prompts=[{"positive_prompt": f"{user_input} result", "negative_prompt": ""}],
        path=path,
    )


def test_patch_history_entry_sets_favorite(api_client, auth_headers, tmp_path):
    entry = _append_entry(str(tmp_path / "history.jsonl"), "a cat")

    response = api_client.patch(
        f"/api/history/{entry['id']}", json={"favorite": True}, auth=auth_headers
    )

    assert response.status_code == 200
    assert response.json()["favorite"] is True
    assert api_client.get("/api/history", auth=auth_headers).json()[0]["favorite"] is True


def test_patch_history_entry_returns_404_for_unknown_entry(api_client, auth_headers):
    response = api_client.patch("/api/history/nope", json={"favorite": True}, auth=auth_headers)

    assert response.status_code == 404


def test_delete_history_entry_removes_it(api_client, auth_headers, tmp_path):
    path = str(tmp_path / "history.jsonl")
    keep = _append_entry(path, "keep")
    drop = _append_entry(path, "drop")

    response = api_client.delete(f"/api/history/{drop['id']}", auth=auth_headers)

    assert response.status_code == 200
    assert [e["id"] for e in api_client.get("/api/history", auth=auth_headers).json()] == [keep["id"]]


def test_delete_history_entry_returns_404_for_unknown_entry(api_client, auth_headers):
    response = api_client.delete("/api/history/nope", auth=auth_headers)

    assert response.status_code == 404


def test_history_mutations_require_auth(api_client):
    assert api_client.patch("/api/history/x", json={"favorite": True}).status_code == 401
    assert api_client.delete("/api/history/x").status_code == 401
