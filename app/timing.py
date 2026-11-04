"""How long things take on the GPU, so the bot can weigh fast drafts against detailed passes.

Each job records how long ComfyUI took to make it (`seconds`, from its /history: execution start to
end, queue wait aside). From the latest RECENT_JOBS jobs of each library workflow -- a run of
"run:N" counts for the library workflow it came from; a shared image's workflow or one the bot
wrote for none -- comes a typical time per image, computed at most once a day (data/
workflow_timing.json) and shown rounded next to each workflow in the library, so the system prompt
(and its cache) changes rarely. A workflow not measured yet says how much work it is instead (its
sampling passes, a model upscale). And the turn's context says how many of the app's images are in
ComfyUI's queue right now (queue_text, from the run tracker's latest poll).
"""
import re
import statistics
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import conversations, storage
from .comfy_graph import SAMPLERS, Graph
from .config import get_settings

MAX_AGE = timedelta(hours=24)
RECENT_JOBS = 40
MIN_JOBS = 3  # fewer measured jobs than this: not shown as a time yet
_lock = threading.Lock()


def _path() -> str:
    return str(Path(get_settings().data_dir) / "workflow_timing.json")


def library_name(run: dict, runs_by_n: dict[int, dict]) -> str | None:
    """The library workflow a run comes from (following run:N back), or None."""
    workflow, seen = str(run.get("workflow") or ""), set()
    while workflow.startswith("run:"):
        match = re.fullmatch(r"run:(\d+)", workflow)
        n = int(match.group(1)) if match else None
        if n is None or n in seen or n not in runs_by_n:
            return None
        seen.add(n)
        workflow = str(runs_by_n[n].get("workflow") or "")
    if not workflow or workflow.startswith("image:") or workflow == "custom":
        return None
    return workflow


def compute(root: str | None = None) -> dict:
    """Seconds per image of each library workflow, over its latest RECENT_JOBS measured jobs."""
    measured: dict[str, list[tuple[str, float]]] = {}
    directory = Path(conversations._root(root))
    for path in directory.glob("*.json") if directory.is_dir() else []:
        if not conversations.is_valid_id(path.stem):
            continue
        conversation = storage.read_json(str(path), default=None) or {}
        runs = {r["n"]: r for r in conversation.get("runs") or [] if isinstance(r.get("n"), int)}
        for run in runs.values():
            name = library_name(run, runs)
            if name is None:
                continue
            for job in run.get("jobs") or []:
                if job.get("status") == "success" and isinstance(job.get("seconds"), (int, float)):
                    measured.setdefault(name, []).append((job.get("finished_at") or "", float(job["seconds"])))
    workflows = {}
    for name, jobs in measured.items():
        latest = [s for _, s in sorted(jobs)[-RECENT_JOBS:]]
        workflows[name] = {"seconds": round(statistics.median(latest), 1), "jobs": len(latest)}
    every = [s for jobs in measured.values() for _, s in jobs]
    return {"computed_at": datetime.now(timezone.utc).isoformat(), "workflows": workflows,
            "typical": round(statistics.median(every), 1) if every else None}


def load() -> dict:
    """The measured times, computed again when older than MAX_AGE."""
    with _lock:
        stored = storage.read_json(_path(), default=None) or {}
        try:
            fresh = datetime.now(timezone.utc) - datetime.fromisoformat(stored["computed_at"]) < MAX_AGE
        except (KeyError, TypeError, ValueError):
            fresh = False
        if not fresh:
            stored = compute()
            storage.write_json(_path(), stored)
        return stored


def _rounded(seconds: float) -> str:
    if seconds < 90:
        return f"~{max(10, round(seconds / 10) * 10)} s"
    return f"~{round(seconds / 30) / 2:g} min"


def work(graph: Graph) -> str:
    """How much a workflow does, for one not measured yet: its sampling passes, a model upscale."""
    passes = len(graph.find(*SAMPLERS))
    parts = [f"{passes} sampling pass{'es' if passes != 1 else ''}"] if passes else []
    if graph.find("ImageUpscaleWithModel"):
        parts.append("a model upscale")
    return " and ".join(parts)


def speed(name: str, graph: Graph, measured: dict) -> str | None:
    """A library workflow's line: its measured time per image, or else how much work it is."""
    found = (measured.get("workflows") or {}).get(name)
    if found and found["jobs"] >= MIN_JOBS:
        return f"{_rounded(found['seconds'])} an image"
    done = work(graph)
    return f"not timed yet; {done}" if done else None


def queue_text(conversation_id: str, waiting: dict[str, int], typical: float | None) -> str | None:
    """The app's images in ComfyUI's queue now (the run tracker's count per conversation)."""
    total = sum(waiting.values())
    if not total:
        return None
    here = waiting.get(conversation_id, 0)
    work_left = f", about {max(1, round(total * typical / 60))} minutes of work" if typical else ""
    return (f"## ComfyUI Queue\n{total} image{'s' if total != 1 else ''} of this app waiting or being made right now "
            f"({here} from this conversation){work_left}, besides whatever other projects send. Weigh it when "
            "deciding how many images to launch and how heavy: a long queue makes the user wait for each answer.")
