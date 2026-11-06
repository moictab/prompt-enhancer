"""Past use of each model and LoRA, from comfy-gateway's job history -- as aggregates only.

The gateway's history holds every job's parameters, prompts and outputs; the user's line is
that the bot may know how much and how each file was used, never what was made. So this reads
only `status`, `finished_at` and, from `params`, `loras` ([{name, strength}]) and `models`,
and keeps per file: how many successful jobs used it, its weights (median, min, max), the
LoRAs it was most used with, and the day it was last used. Nothing else from a job --
prompts, workflows, outputs, projects -- is read or kept.

    python -m app.gateway_history aggregate jobs.json usage.json   # where the history is
    python -m app.gateway_history import usage.json                 # where the app's data is ("-": stdin)

Both print counts only. The import replaces the catalog's past use (catalog.set_history).
"""
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

MAX_USED_WITH = 5


def _day(timestamp) -> str | None:
    try:
        return datetime.fromtimestamp(float(timestamp), timezone.utc).date().isoformat()
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def aggregate(jobs: list) -> dict:
    """{"files": {name: {jobs, weight?, used_with?, last_used?}}, "jobs", "from", "to"}."""
    counts: Counter = Counter()
    weights: dict[str, list[float]] = defaultdict(list)
    pairs: dict[str, Counter] = defaultdict(Counter)
    last: dict[str, float] = {}
    times = []
    used = 0
    for job in jobs:
        if not isinstance(job, dict) or job.get("status") != "success":
            continue
        params = job.get("params") if isinstance(job.get("params"), dict) else {}
        loras = [lora for lora in params.get("loras") or []
                 if isinstance(lora, dict) and isinstance(lora.get("name"), str) and lora["name"]]
        lora_names = [lora["name"] for lora in loras]
        models = {m for m in params.get("models") or [] if isinstance(m, str) and m} - set(lora_names)
        if not loras and not models:
            continue
        used += 1
        at = job.get("finished_at")
        if isinstance(at, (int, float)):
            times.append(at)
        for name in set(lora_names) | models:
            counts[name] += 1
            if isinstance(at, (int, float)):
                last[name] = max(last.get(name, 0), at)
        for lora in loras:
            strength = lora.get("strength")
            if isinstance(strength, (int, float)) and not isinstance(strength, bool):
                weights[lora["name"]].append(float(strength))
            pairs[lora["name"]].update(n for n in set(lora_names) if n != lora["name"])
    files = {}
    for name, count in counts.items():
        entry: dict = {"jobs": count}
        if weights.get(name):
            values = weights[name]
            entry["weight"] = {"median": round(statistics.median(values), 3),
                               "min": round(min(values), 3), "max": round(max(values), 3)}
        if pairs.get(name):
            entry["used_with"] = [[other, n] for other, n in pairs[name].most_common(MAX_USED_WITH)]
        if name in last:
            entry["last_used"] = _day(last[name])
        files[name] = entry
    return {"files": files, "jobs": used,
            "from": _day(min(times)) if times else None, "to": _day(max(times)) if times else None}


MAX_JOBS = 1_000_000
FETCH_TIMEOUT = 60


def fetch(base_url: str) -> dict:
    """The gateway's successful jobs, aggregated as above, read straight from it: `/gateway/jobs` (an
    admin endpoint the gateway answers only on its own PC). Its rows carry no prompts or images, but
    projects, tags and output names: none of that is kept, only what aggregate() reads."""
    import requests
    response = requests.get(f"{base_url.rstrip('/')}/gateway/jobs", params={"status": "success", "limit": MAX_JOBS},
                            timeout=FETCH_TIMEOUT)
    response.raise_for_status()
    jobs = response.json()
    return aggregate(jobs if isinstance(jobs, list) else [])


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[0] == "aggregate":
        with open(argv[1], encoding="utf-8") as f:
            jobs = json.load(f)
        usage = aggregate(jobs if isinstance(jobs, list) else [])
        with open(argv[2], "w", encoding="utf-8") as f:
            json.dump(usage, f, ensure_ascii=False)
        print(f"jobs read: {len(jobs)}; jobs counted: {usage['jobs']}; files: {len(usage['files'])}; "
              f"span: {usage['from']} to {usage['to']}")
        return 0
    if len(argv) == 2 and argv[0] == "import":
        from . import catalog
        raw = sys.stdin.read() if argv[1] == "-" else open(argv[1], encoding="utf-8").read()
        usage = json.loads(raw)
        data = catalog.set_history(usage)
        installed = sum(1 for name in usage["files"] if name in data["entries"])
        print(f"imported past use of {len(usage['files'])} files ({installed} installed now), "
              f"from {usage['jobs']} jobs, {usage['from']} to {usage['to']}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
