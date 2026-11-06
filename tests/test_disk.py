"""Disk space: each model file in the models folder against its use (comfy-gateway's record), to tell
what can go -- and deleting what the owner picks."""
import os
import time
from datetime import date, timedelta

from app import catalog, disk, gateway_history, workflows
from app.comfy_graph import Graph

DAY = 86400
TODAY = date.today()


def _file(root, folder, name, size=1000, days_old=100):
    path = root / folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    old = time.time() - days_old * DAY
    os.utime(path, (old, old))  # a copy keeps the old write time and gets a new creation time
    return path


def _job(names, days_ago, loras=()):
    return {"status": "success", "finished_at": time.time() - days_ago * DAY,
            "params": {"models": list(names), "loras": [{"name": n, "strength": 0.6} for n in loras]},
            "project": "manual", "tag": "not kept", "outputs": ["not kept"]}


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(disk, "_fresh", {"at": 0.0, "usage": None})  # no reading left from another test
    root = tmp_path / "models"
    monkeypatch.setenv("COMFY_MODELS_DIR", str(root))
    monkeypatch.setenv("COMFY_GATEWAY_URL", "http://gateway.test")
    workflows.ensure_seeded()
    in_library = Graph(workflows.find_by_name("krea-2/txt2img")["workflow"]).base_models()[0]
    _file(root, "diffusion_models", in_library, days_old=200)
    _file(root, "checkpoints", "sdxl\\forgotten-mix_v1.safetensors", size=5000, days_old=120)
    (root / "checkpoints" / "sdxl" / "forgotten-mix_v1.civitai.info").write_text("{}")
    (root / "checkpoints" / "sdxl" / "forgotten-mix_v1-alt.safetensors").write_bytes(b"y")
    _file(root, "loras", "krea-2\\style-daily_v1.safetensors", days_old=90)
    _file(root, "loras", "krea-2\\style-once_v1.safetensors", days_old=90)
    _file(root, "loras", "krea-2\\style-old-love_v1.safetensors", days_old=90)
    _file(root, "loras", "krea-2\\style-fresh_v1.safetensors", days_old=3)
    _file(root, "loras", "krea-2\\notes.txt", days_old=90)  # not a model
    jobs = [_job([in_library], d, loras=["krea-2\\style-daily_v1.safetensors"]) for d in range(1, 30)]
    jobs += [_job([in_library], 20, loras=["krea-2\\style-once_v1.safetensors"]),
             _job([in_library], 50, loras=["krea-2\\style-old-love_v1.safetensors"]),
             _job([in_library], 59), {"status": "error", "params": {"models": ["sdxl\\forgotten-mix_v1.safetensors"]}}]
    fetched = []
    monkeypatch.setattr(gateway_history, "fetch", lambda url: fetched.append(url) or gateway_history.aggregate(jobs))
    return root, in_library, fetched


def test_each_file_is_judged_by_its_use_over_the_time_it_was_there(tmp_path, monkeypatch):
    root, in_library, fetched = _setup(tmp_path, monkeypatch)
    report = disk.report()
    states = {f["name"]: f for f in report["files"]}
    assert fetched == ["http://gateway.test"]  # the record of use, read from the gateway
    assert set(states) == {in_library, "sdxl\\forgotten-mix_v1.safetensors", "sdxl\\forgotten-mix_v1-alt.safetensors",
                           "krea-2\\style-daily_v1.safetensors", "krea-2\\style-once_v1.safetensors",
                           "krea-2\\style-old-love_v1.safetensors", "krea-2\\style-fresh_v1.safetensors"}
    assert states[in_library]["state"] == "library" and "krea-2/txt2img" in states[in_library]["library"]
    forgotten = states["sdxl\\forgotten-mix_v1.safetensors"]
    assert forgotten["state"] == "unused" and forgotten["jobs"] == 0 and forgotten["bytes"] == 5000
    assert forgotten["on_disk_since"] == (TODAY - timedelta(days=120)).isoformat()
    assert states["krea-2\\style-daily_v1.safetensors"]["state"] == "used"
    once = states["krea-2\\style-once_v1.safetensors"]
    assert once["state"] == "rare" and once["jobs"] == 1 and once["observed_days"] == 59
    assert states["krea-2\\style-old-love_v1.safetensors"]["state"] == "stale"
    assert states["krea-2\\style-fresh_v1.safetensors"]["state"] == "new"  # too early to judge
    assert report["records_from"] == (TODAY - timedelta(days=59)).isoformat()
    assert report["candidate_bytes"] == 5000 + 1000 + 1000  # the -alt file was written just now: new
    assert report["files"][0]["name"] == "sdxl\\forgotten-mix_v1.safetensors"  # biggest first
    # Read again after a few minutes, or when asked; the catalog's past use (what the bot sees, with
    # no project, tag or output) only once a day.
    disk.report()
    assert len(fetched) == 1
    kept = catalog.load()["history"]["krea-2\\style-daily_v1.safetensors"]
    assert kept["jobs"] == 29 and set(kept) <= {"jobs", "weight", "used_with", "last_used"}
    imported = catalog.load()["history_source"]["imported_at"]
    disk.report(refresh=True)
    assert len(fetched) == 2 and catalog.load()["history_source"]["imported_at"] == imported
    monkeypatch.setattr(disk, "_fresh", {**disk._fresh, "at": time.time() - disk.FRESH_SECONDS - 1})
    disk.report()
    assert len(fetched) == 3 and report["usage_read_at"]


def test_the_owner_deletes_a_file_with_its_side_files_but_not_what_the_library_loads(
        api_client, auth_headers, tmp_path, monkeypatch):
    root, in_library, _ = _setup(tmp_path, monkeypatch)
    listed = api_client.get("/api/admin/disk", auth=auth_headers).json()
    assert listed["configured"] and listed["free_bytes"] > 0 and len(listed["files"]) == 7

    gone = api_client.delete("/api/admin/disk/files", auth=auth_headers,
                             params={"folder": "checkpoints", "name": "sdxl\\forgotten-mix_v1.safetensors"})
    assert gone.status_code == 200, gone.text
    assert sorted(gone.json()["removed"]) == ["forgotten-mix_v1.civitai.info", "forgotten-mix_v1.safetensors"]
    left = sorted(p.name for p in (root / "checkpoints" / "sdxl").iterdir())
    assert left == ["forgotten-mix_v1-alt.safetensors"]  # another file that merely starts the same

    refused = api_client.delete("/api/admin/disk/files", auth=auth_headers,
                                params={"folder": "diffusion_models", "name": in_library})
    assert refused.status_code == 409 and "krea-2/txt2img" in refused.json()["detail"]
    assert (root / "diffusion_models" / in_library).exists()
    for folder, name in (("loras", "..\\checkpoints\\sdxl\\forgotten-mix_v1-alt.safetensors"),
                         ("loras", "krea-2\\notes.txt"), ("text_encoders", "x.safetensors"),
                         ("loras", "krea-2\\nothing.safetensors")):
        assert api_client.delete("/api/admin/disk/files", auth=auth_headers,
                                 params={"folder": folder, "name": name}).status_code == 400
    assert (root / "checkpoints" / "sdxl" / "forgotten-mix_v1-alt.safetensors").exists()


def test_without_the_models_folder_there_is_no_report(api_client, auth_headers):
    assert api_client.get("/api/admin/disk", auth=auth_headers).json() == {"configured": False}
