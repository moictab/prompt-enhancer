import base64

from app import request_log, storage

PNG = b"\x89PNG fake image bytes"
PNG_URI = "data:image/png;base64," + base64.b64encode(PNG).decode()


def _payload(text="hola", **extra):
    return {"model": "vendor/model", "messages": [{"role": "user", "content": text}], **extra}


def _log(root, payload=None, context=None, **outcome):
    record = request_log.RequestRecord(payload or _payload(), context=context, root=root)
    record.finish(**outcome)
    return record.id


def test_finish_writes_the_exact_payload_and_outcome(tmp_path):
    root = str(tmp_path)
    payload = _payload(temperature=0.4, max_tokens=8192)
    response = {"id": "gen-1", "choices": [{"message": {"content": "POSITIVE: a cat"}}], "usage": {"cost": 0.002}}

    request_id = _log(root, payload, {"mode": "generate"}, status=200, response=response)

    entry = request_log.get(request_id, root=root)
    assert request_log.ID_PATTERN.match(request_id)
    assert entry["request"] == payload
    assert entry["response"] == response
    assert entry["mode"] == "generate"
    assert entry["context"] == {}
    assert entry["status"] == 200
    assert entry["error"] is None
    assert entry["model"] == "vendor/model"
    assert entry["duration_ms"] >= 0


def test_finish_only_writes_once(tmp_path):
    root = str(tmp_path)
    record = request_log.RequestRecord(_payload(), root=root)

    record.finish(status=200)
    record.finish(status=500)

    [day_file] = (tmp_path).glob("*.jsonl")
    assert len(storage.read_jsonl(str(day_file))) == 1


def test_images_are_stored_once_and_replaced_by_a_marker(tmp_path):
    root = str(tmp_path)
    content = [
        {"type": "text", "text": "mira"},
        {"type": "image_url", "image_url": {"url": PNG_URI}},
        {"type": "image_url", "image_url": {"url": PNG_URI}},
    ]

    request_id = _log(root, {"model": "m", "messages": [{"role": "user", "content": content}]}, status=200)

    logged = request_log.get(request_id, root=root)["request"]["messages"][0]["content"]
    marker = logged[1]["image_url"]["url"]
    assert marker.startswith(request_log.IMAGE_MARKER)
    assert logged[2]["image_url"]["url"] == marker
    name = marker[len(request_log.IMAGE_MARKER):]
    path, content_type = request_log.image_file(name, root=root)
    assert open(path, "rb").read() == PNG
    assert content_type == "image/png"
    assert len(list((tmp_path / "images").iterdir())) == 1


def test_a_logging_failure_never_raises(tmp_path):
    record = request_log.RequestRecord({"model": "m", "messages": [object()]}, root=str(tmp_path))

    record.finish(status=200)  # not JSON-serializable: reported, not raised


def test_summary_reads_both_plain_and_streamed_responses(tmp_path):
    root = str(tmp_path)
    plain = _log(
        root, _payload("una bruja"), {"mode": "generate"}, status=200,
        response={"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
                  "usage": {"cost": 0.002, "prompt_tokens": 120, "completion_tokens": 40}},
    )
    streamed = _log(
        root, _payload("la 3 de noche", stream=True), {"mode": "conversation", "conversation_id": "c1"},
        status=200, response={"content": "Vale", "finish_reason": "length", "usage": {"cost": 0.01}},
    )

    by_id = {s["id"]: s for s in request_log.list_summaries(root=root)["items"]}

    assert by_id[plain]["cost"] == 0.002
    assert by_id[plain]["prompt_tokens"] == 120
    assert by_id[plain]["finish_reason"] == "stop"
    assert by_id[plain]["preview"] == "una bruja"
    assert by_id[streamed]["stream"] is True
    assert by_id[streamed]["finish_reason"] == "length"
    assert by_id[streamed]["context"] == {"conversation_id": "c1"}
    assert request_log.response_text(request_log.get(streamed, root=root)) == "Vale"


def _write(root, request_id, mode="generate", error=None):
    day = request_id[:8]
    storage.append_jsonl(
        f"{root}/{day[:4]}-{day[4:6]}-{day[6:8]}.jsonl",
        {"id": request_id, "timestamp": "t", "mode": mode, "error": error, "request": _payload(), "response": None},
    )


def test_list_is_newest_first_across_days_and_pages_with_a_cursor(tmp_path):
    root = str(tmp_path)
    ids = [
        "20261101-090000-00000001",
        "20261101-100000-00000002",
        "20261102-090000-00000003",
        "20261103-090000-00000004",
        "20261103-100000-00000005",
    ]
    for request_id in ids:
        _write(root, request_id)

    first = request_log.list_summaries(limit=2, root=root)
    second = request_log.list_summaries(limit=2, before=first["items"][-1]["id"], root=root)
    third = request_log.list_summaries(limit=2, before=second["items"][-1]["id"], root=root)

    assert [s["id"] for s in first["items"]] == ids[:2:-1][:2]
    assert first["has_more"] is True
    assert [s["id"] for s in second["items"]] == [ids[2], ids[1]]
    assert [s["id"] for s in third["items"]] == [ids[0]]
    assert third["has_more"] is False


def test_list_filters_by_mode_and_errors(tmp_path):
    root = str(tmp_path)
    _write(root, "20261103-090000-00000001", mode="generate")
    _write(root, "20261103-090000-00000002", mode="conversation")
    _write(root, "20261103-090000-00000003", mode="conversation", error="HTTP 404")

    assert [s["id"][-1] for s in request_log.list_summaries(mode="conversation", root=root)["items"]] == ["3", "2"]
    assert [s["id"][-1] for s in request_log.list_summaries(errors_only=True, root=root)["items"]] == ["3"]


def test_get_and_image_file_reject_bad_names(tmp_path):
    root = str(tmp_path)

    assert request_log.get("../../etc/passwd", root=root) is None
    assert request_log.get("20261103-090000-00000001", root=root) is None
    assert request_log.image_file("../x.png", root=root) is None


def test_storage_bytes_counts_logs_and_images(tmp_path):
    root = str(tmp_path)
    assert request_log.storage_bytes(root=root) == 0

    _log(root, {"model": "m", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": PNG_URI}}]}]})

    assert request_log.storage_bytes(root=root) > len(PNG)
