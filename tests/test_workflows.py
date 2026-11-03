import pytest

from app import workflows

from .conftest import png_bytes

GRAPH = {"1": {"class_type": "KSampler", "inputs": {"seed": 1}}}


def test_seeded_once_from_the_owners_workflows(tmp_path):
    path = str(tmp_path / "workflows.json")

    workflows.ensure_seeded(path)

    names = [w["name"] for w in workflows.list_workflows(path)]
    assert "krea-2/txt2img" in names and "illustrious/txt2img" in names
    assert len(names) == 7
    workflows.delete(workflows.find_by_name("KREA-2/TXT2IMG", path)["id"], path)
    workflows.ensure_seeded(path)  # the file exists: nothing comes back
    assert len(workflows.list_workflows(path)) == 6


def test_crud_with_unique_names_and_api_format_only(tmp_path):
    path = str(tmp_path / "workflows.json")
    created = workflows.create("  mine/test ", " Does it. ", GRAPH, path)

    assert (created["name"], created["description"]) == ("mine/test", "Does it.")
    with pytest.raises(workflows.WorkflowError, match="already"):
        workflows.create("MINE/test", "", GRAPH, path)
    with pytest.raises(workflows.WorkflowError, match="Export"):
        workflows.create("other", "", {"nodes": [], "links": []}, path)
    with pytest.raises(workflows.WorkflowError):
        workflows.create(" ", "", GRAPH, path)

    updated = workflows.update(created["id"], "mine/test", "Changed.", GRAPH, path)
    assert updated["description"] == "Changed."
    assert workflows.update("nope", "x", "", GRAPH, path) is None
    assert workflows.delete(created["id"], path)
    assert not workflows.delete(created["id"], path)


def test_admin_routes(api_client, auth_headers):
    created = api_client.post("/api/admin/workflows", auth=auth_headers,
                              json={"name": "mine/a", "description": "A.", "workflow": GRAPH})
    assert created.status_code == 200
    workflow_id = created.json()["id"]

    assert api_client.post("/api/admin/workflows", auth=auth_headers,
                           json={"name": "mine/a", "workflow": GRAPH}).status_code == 400
    listed = api_client.get("/api/admin/workflows", auth=auth_headers).json()
    assert [w["name"] for w in listed] == ["mine/a"]
    assert api_client.put(f"/api/admin/workflows/{workflow_id}", auth=auth_headers,
                          json={"name": "mine/b", "workflow": GRAPH}).json()["name"] == "mine/b"
    assert api_client.put("/api/admin/workflows/nope", auth=auth_headers,
                          json={"name": "x", "workflow": GRAPH}).status_code == 404
    assert api_client.delete(f"/api/admin/workflows/{workflow_id}", auth=auth_headers).status_code == 200


def test_admin_extracts_the_workflow_from_a_comfyui_image(api_client, auth_headers):
    response = api_client.post("/api/admin/workflows/extract", auth=auth_headers,
                               files={"image": ("x.png", png_bytes(prompt=GRAPH), "image/png")})

    assert response.status_code == 200
    assert response.json()["workflow"] == GRAPH

    plain = api_client.post("/api/admin/workflows/extract", auth=auth_headers,
                            files={"image": ("x.png", png_bytes(), "image/png")})
    assert plain.status_code == 400
