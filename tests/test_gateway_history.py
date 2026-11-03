import io
import json

from app import catalog, gateway_history

SECRET = "a very private prompt"
INK = "krea-2\\style-ink_v1.safetensors"
REAL = "krea-2\\base-real_v3.safetensors"
UNET = "krea-2\\krea-2-turbo.safetensors"


def _job(loras, models=(UNET,), status="success", at=1793000000.0):
    return {
        "id": "x", "project": "secret-project", "status": status, "finished_at": at,
        "params": {"loras": [{"name": n, "strength": s} for n, s in loras], "models": list(models),
                   "prompt": SECRET, "steps": 8},
        "extra_data": {"prompt": SECRET}, "outputs": [{"filename": "private-output"}],
    }


JOBS = [
    _job([(INK, 0.3), (REAL, 0.5)]),
    _job([(INK, 0.5), (REAL, 0.5)], at=1793500000.0),
    _job([(INK, 0.1)]),
    _job([(INK, 2.0)], status="error"),  # failed jobs don't count
    "not a job",
    {"status": "success", "params": None},
]


def test_aggregate_keeps_only_numbers_and_file_names():
    usage = gateway_history.aggregate(JOBS)

    assert usage["jobs"] == 3
    assert usage["files"][INK] == {
        "jobs": 3, "weight": {"median": 0.3, "min": 0.1, "max": 0.5},
        "used_with": [[REAL, 2]], "last_used": "2026-11-01",
    }
    assert usage["files"][UNET] == {"jobs": 3, "last_used": "2026-11-01"}
    assert (usage["from"], usage["to"]) == ("2026-10-26", "2026-11-01")
    # Nothing from a job but numbers and file names survives.
    text = json.dumps(usage)
    for private in (SECRET, "secret-project", "private-output"):
        assert private not in text


def test_aggregate_and_import_from_the_command_line(tmp_path, monkeypatch, capsys):
    source, out = tmp_path / "jobs.json", tmp_path / "usage.json"
    source.write_text(json.dumps(JOBS), encoding="utf-8")

    assert gateway_history.main(["aggregate", str(source), str(out)]) == 0
    printed = capsys.readouterr().out
    assert "jobs counted: 3; files: 3" in printed and SECRET not in printed

    monkeypatch.setattr("sys.stdin", io.StringIO(out.read_text(encoding="utf-8")))
    assert gateway_history.main(["import", "-"]) == 0
    assert "imported past use of 3 files" in capsys.readouterr().out

    data = catalog.load()
    assert data["history"][INK]["jobs"] == 3
    assert data["history_source"]["jobs"] == 3
    assert catalog.details(INK)["past_use"]["weight"]["median"] == 0.3


def test_import_keeps_only_known_fields(tmp_path):
    catalog.set_history({"files": {INK: {"jobs": 2, "prompt": SECRET}}, "jobs": 2})

    assert catalog.load()["history"] == {INK: {"jobs": 2}}


def test_past_use_goes_into_the_prompt_section(fake_comfy):
    catalog.refresh(fake_comfy)
    catalog.set_history(gateway_history.aggregate(JOBS))

    section = catalog.prompt_section()

    assert "style-ink_v1 (used in 3 past jobs at ~0.3 (0.1-0.5))" in section
    assert "base-real_v3 (used in 2 past jobs at ~0.5)" in section
    # A refresh from ComfyUI keeps it.
    catalog.refresh(fake_comfy)
    assert catalog.load()["history"][INK]["jobs"] == 3
