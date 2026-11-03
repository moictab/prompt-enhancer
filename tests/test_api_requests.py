import base64

from app import request_log

PNG_URI = "data:image/png;base64," + base64.b64encode(b"\x89PNG bytes").decode()


def _log(payload, **outcome):
    record = request_log.RequestRecord(payload, context={"mode": "conversation", "conversation_id": "c1"})
    record.finish(**outcome)
    return record.id


def test_requests_require_auth(api_client):
    assert api_client.get("/api/requests").status_code == 401


def test_list_get_and_image(api_client, auth_headers):
    content = [{"type": "text", "text": "mira"}, {"type": "image_url", "image_url": {"url": PNG_URI}}]
    request_id = _log(
        {"model": "m", "stream": True, "messages": [{"role": "user", "content": content}]},
        status=200, response={"content": "Bonita", "usage": {"cost": 0.01}},
    )

    listing = api_client.get("/api/requests", auth=auth_headers).json()
    [item] = listing["items"]
    assert item["id"] == request_id
    assert item["preview"] == "mira"
    assert item["cost"] == 0.01
    assert listing["has_more"] is False
    assert listing["storage_bytes"] > 0

    entry = api_client.get(f"/api/requests/{request_id}", auth=auth_headers).json()
    marker = entry["request"]["messages"][0]["content"][1]["image_url"]["url"]
    image = api_client.get(f"/api/requests/images/{marker.split(':', 1)[1]}", auth=auth_headers)
    assert image.status_code == 200
    assert image.content == b"\x89PNG bytes"


def test_list_filters(api_client, auth_headers):
    _log({"model": "m", "messages": []}, status=200)
    failed = _log({"model": "m", "messages": []}, status=404, error="HTTP 404")

    errors = api_client.get("/api/requests?errors=true", auth=auth_headers).json()["items"]
    other_mode = api_client.get("/api/requests?mode=generate", auth=auth_headers).json()["items"]

    assert [i["id"] for i in errors] == [failed]
    assert other_mode == []


def test_unknown_or_malformed_ids(api_client, auth_headers):
    assert api_client.get("/api/requests/20261103-090000-00000001", auth=auth_headers).status_code == 404
    assert api_client.get("/api/requests/nope", auth=auth_headers).status_code == 404
    assert api_client.get("/api/requests?before=../x", auth=auth_headers).status_code == 400
    assert api_client.get("/api/requests/images/nope.png", auth=auth_headers).status_code == 404
