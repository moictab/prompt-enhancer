"""Runs: workflows the Conversar bot launched on ComfyUI, followed until their images arrive.

A run is one run_workflow call: 1-4 jobs (one per image, each with its own seed), stored
in the conversation's "runs" list. The tracker thread polls /history for every pending
job; when a job finishes its images are downloaded into the conversation, and when the
whole run has, a "results" message carries them into the chat. If nothing else is
pending in that conversation, the bot then gets an automatic turn to comment on them
(on_finished, set by conversation_turn). A run cancelled by the user gets no comment.
"""
import logging
import threading
import time
from datetime import datetime, timezone

from . import comfy, conversations
from .config import get_settings

logger = logging.getLogger("app.runs")

POLL_SECONDS = 2.0
# A job given up for running too long (job_timeout_seconds); its images may come later.
TIMED_OUT = "No result after"
# A download that fails is tried again on the next polls (ComfyUI can stall for a while)
# before the job is given up as failed.
MAX_DOWNLOAD_ATTEMPTS = 5
DOWNLOAD_FAILED = "Couldn't download"
FINAL = ("success", "error", "interrupted", "cancelled")
CONTENT_TYPES = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def job_timeout_seconds(run: dict) -> int:
    """How long a run's jobs may go without a result once they start: the run's own limit (12 h
    for calibrations, which wait for a free GPU) or COMFY_JOB_TIMEOUT_MINUTES."""
    return run.get("timeout_seconds") or get_settings().comfy_job_timeout_minutes * 60


def _still_on_time(record: dict | None, timeout: float) -> bool:
    """A job past its run's time limit that the gateway says (its record, `Comfy.gateway_job`) is still
    waiting its turn, or running for less than the limit: the limit counts from when a job starts (the
    gateway's rule), since waiting in a long queue is no failure -- 13 jobs were given up on 2026-11-04
    while still queued, and ComfyUI made their images for nothing. Waiting is "queued" (the gateway's
    queue, where a job bumped by a more urgent one goes back, its start forgotten) and "dispatched"
    (handed to ComfyUI, not started: milliseconds, unless something launched in ComfyUI itself goes
    first -- on 2026-11-05 a job that had waited 50 minutes was given up in that second). The gateway
    doesn't notice a ComfyUI hung on a job: that's what the limit is for. A gateway that doesn't answer
    (restarting) gives no reason to give up: it's asked again on the next poll."""
    if not record:
        return False
    status = record.get("status")
    if status in ("queued", "dispatched", comfy.GATEWAY_UNKNOWN["status"]):  # unknown: can't tell yet
        return True
    started = record.get("started_at") or record.get("dispatched_at")
    return status == "running" and (not isinstance(started, (int, float)) or time.time() - started < timeout)


def _age_seconds(run: dict) -> float:
    try:
        created = datetime.fromisoformat(run.get("retried_at") or run["created_at"])
    except (KeyError, ValueError):
        return 0.0
    return (datetime.now(timezone.utc) - created).total_seconds()


def _run_status(jobs: list[dict]) -> str:
    statuses = {j["status"] for j in jobs}
    if "pending" in statuses:
        return "pending"
    if statuses == {"cancelled"} or ("cancelled" in statuses and "success" not in statuses):
        return "cancelled"
    if statuses == {"success"}:
        return "done"
    return "done" if "success" in statuses else "error"


class RunTracker:
    def __init__(self):
        self._pending: set[tuple[str, int]] = set()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        # Called (conversation_id, run) once a run is final and its results message is stored.
        self.on_finished = None
        # Jobs still waiting or being made, per conversation, as of the latest poll (timing.queue_text).
        self.waiting: dict[str, int] = {}

    def watch(self, conversation_id: str, n: int) -> None:
        with self._lock:
            self._pending.add((conversation_id, n))

    def start(self) -> None:
        for conversation_id, run in conversations.pending_runs():
            self.watch(conversation_id, run["n"])
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="run-tracker", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(POLL_SECONDS):
            try:
                self.poll_once()
            except Exception:
                logger.exception("run tracker poll failed")

    def poll_once(self) -> None:
        with self._lock:
            pending = sorted(self._pending)
        client = comfy.client()
        waiting: dict[str, int] = {}
        for conversation_id, n in pending:
            run = conversations.get_run(conversation_id, n)
            if run is None or run["status"] != "pending" or client is None:
                self._forget(conversation_id, n)
                continue
            for index, job in enumerate(run["jobs"]):
                if job["status"] == "pending":
                    self._check_job(client, conversation_id, run, index, job)
            run = conversations.get_run(conversation_id, n)
            if run is not None and run["status"] == "pending" and all(j["status"] in FINAL for j in run["jobs"]):
                self._finish(conversation_id, n)
            elif run is not None:
                left = sum(1 for j in run["jobs"] if j["status"] == "pending")
                if left:
                    waiting[conversation_id] = waiting.get(conversation_id, 0) + left
        self.waiting = waiting

    def _forget(self, conversation_id: str, n: int) -> None:
        with self._lock:
            self._pending.discard((conversation_id, n))

    def _check_job(self, client: comfy.Comfy, conversation_id: str, run: dict, index: int, job: dict) -> None:
        try:
            result = comfy.result_from_history(client.history(job["base_url"], job["prompt_id"]))
        except (comfy.ComfyError, *comfy.CONNECT_ERRORS, OSError) as e:
            logger.info("run history unavailable conversation_id=%s run=%s error=%s", conversation_id, run["n"], e)
            result = None
        except Exception as e:  # requests' other errors (timeouts, bad JSON)
            logger.info("run history failed conversation_id=%s run=%s error=%r", conversation_id, run["n"], e)
            result = None
        if result is None:
            timeout = job_timeout_seconds(run)
            overdue = _age_seconds(run) > timeout
            if not (overdue or run.get("retried_at")):
                return
            # Past the limit, or fetched again: ask the gateway (one more request, so not on every
            # poll), which also keeps what a job saved when ComfyUI's /history forgets it.
            record = client.gateway_job(job["base_url"], job["prompt_id"])
            result = comfy.result_from_gateway(record)
            if result is None:
                if overdue and not _still_on_time(record, timeout):
                    self._set_job(conversation_id, run["n"], index, status="error",
                                  error=f"{TIMED_OUT} {timeout // 60} minutes")
                    logger.info("run job given up conversation_id=%s run=%s job=%d prompt_id=%s timeout=%ss",
                                conversation_id, run["n"], index, job["prompt_id"], timeout)
                    # Stopped too: hung, it holds ComfyUI; left waiting, it would run for nothing.
                    try:
                        client.cancel(job["base_url"], job["prompt_id"],
                                      comfy.CANCEL_TIMED_OUT.format(minutes=timeout // 60))
                    except Exception as e:
                        logger.info("run job cancel failed prompt_id=%s error=%r", job["prompt_id"], e)
                return
        images, error = [], result.error
        workflows = conversations.run_workflows(conversation_id, run["n"])
        workflow = workflows[index] if index < len(workflows) else None
        downloaded, failed = [], []
        for output in result.outputs:
            try:
                downloaded.append((output, *client.view(job["base_url"], output["filename"], output["subfolder"],
                                                        output["type"])))
            except Exception as e:
                logger.warning("run output download failed conversation_id=%s run=%s file=%s error=%r",
                               conversation_id, run["n"], output["filename"], e)
                failed.append(output["filename"])
        if failed:
            attempts = job.get("download_attempts", 0) + 1
            if attempts < MAX_DOWNLOAD_ATTEMPTS:
                # Nothing is stored until every image of the job comes down: try again next poll.
                conversations.update_run(conversation_id, run["n"],
                                         lambda r: r["jobs"][index].update(download_attempts=attempts))
                return
            error = error or f"{DOWNLOAD_FAILED} {', '.join(failed)}"
        for output, data, content_type in downloaded:
            extension = output["filename"].rsplit(".", 1)[-1].lower()
            content_type = content_type if content_type in CONTENT_TYPES.values() else CONTENT_TYPES.get(extension)
            if content_type is None:
                continue  # not an image we can show (a video, a mask...)
            image = conversations.save_image(conversation_id, data, content_type, workflow=workflow or {})
            images.append(image)
        status = result.status if images or result.status != "success" else "error"
        if status == "error" and not error:
            error = "The workflow finished without saving any image"
        self._set_job(conversation_id, run["n"], index, status=status, error=error, images=images,
                      seconds=result.seconds)
        logger.info("run job finished conversation_id=%s run=%s job=%d status=%s images=%d",
                    conversation_id, run["n"], index, status, len(images))

    def _set_job(self, conversation_id: str, n: int, index: int, status: str, error: str | None = None,
                 images: list[dict] | None = None, seconds: float | None = None) -> None:
        def change(run: dict) -> None:
            job = run["jobs"][index]
            if job["status"] != "pending":
                return
            job["status"] = status
            job["error"] = error
            job["images"] = [image["id"] for image in images or []]
            job["finished_at"] = _now()
            if seconds is not None:
                job["seconds"] = seconds  # how long ComfyUI took making it (timing.py)
            # The image records travel with the run until the results message takes them.
            run.setdefault("_images", []).extend(images or [])
        conversations.update_run(conversation_id, n, change)

    def _finish(self, conversation_id: str, n: int) -> None:
        final: dict = {}

        def change(run: dict) -> None:
            run["status"] = _run_status(run["jobs"])
            run["finished_at"] = _now()
            final.update(run)
            final["images"] = run.pop("_images", [])
        conversations.update_run(conversation_id, n, change)
        self._forget(conversation_id, n)
        if not final:
            return
        logger.info("run finished conversation_id=%s run=%s status=%s", conversation_id, n, final["status"])
        if final["status"] == "cancelled" and not final["images"]:
            return
        if self.on_finished is not None:
            threading.Thread(
                target=self._safely, args=(self.on_finished, conversation_id, final), daemon=True,
                name=f"run-finished-{n}",
            ).start()

    @staticmethod
    def _safely(callback, conversation_id: str, run: dict) -> None:
        try:
            callback(conversation_id, run)
        except Exception:
            logger.exception("run finished callback failed conversation_id=%s run=%s", conversation_id, run["n"])

    def pending_in(self, conversation_id: str) -> bool:
        with self._lock:
            return any(cid == conversation_id for cid, _ in self._pending)


tracker = RunTracker()


def _fetchable(job: dict) -> bool:
    """A job whose images ComfyUI may well have: they didn't come down, or the job was given up for
    taking too long and finished later (as 14 did, given up while they waited their turn)."""
    return job["status"] == "error" and (job.get("error") or "").startswith((DOWNLOAD_FAILED, TIMED_OUT))


def _has_result(client: comfy.Comfy | None, job: dict) -> bool:
    """Whether the job's result can be had now: ComfyUI's /history (kept until ComfyUI restarts), or
    the gateway's record of what it saved."""
    if client is None:
        return False
    try:
        if comfy.result_from_history(client.history(job["base_url"], job["prompt_id"])) is not None:
            return True
    except Exception as e:  # unreachable, stalled: the gateway may still know
        logger.info("run history unavailable prompt_id=%s error=%r", job["prompt_id"], e)
    return comfy.result_from_gateway(client.gateway_job(job["base_url"], job["prompt_id"])) is not None


def can_retry_downloads(run: dict) -> bool:
    return run["status"] != "pending" and any(_fetchable(j) for j in run["jobs"])


def retry_downloads(conversation_id: str, n: int) -> dict | None:
    """Fetch again the images of a run's jobs whose download failed or that were given up for
    taking too long (ComfyUI keeps the images): those jobs go back to pending, and when they come
    down the run's results message gets its images in place, with no new comment from the bot.
    None when there is nothing to fetch."""
    run = conversations.get_run(conversation_id, n)
    if run is None or not can_retry_downloads(run):
        return None
    client = comfy.client()
    # A job given up for its time is fetched only if it did finish: one that never did would only
    # wait out the limit again.
    again = [index for index, job in enumerate(run["jobs"]) if _fetchable(job) and (
        job["error"].startswith(DOWNLOAD_FAILED) or _has_result(client, job))]
    if not again:
        return None

    def change(run: dict) -> None:
        for index in again:
            job = run["jobs"][index]
            job.update(status="pending", error=None, download_attempts=0)
            job.pop("finished_at", None)
        run["status"] = "pending"
        run["retried_at"] = _now()
        run.pop("finished_at", None)

    updated = conversations.update_run(conversation_id, n, change)
    tracker.watch(conversation_id, n)
    logger.info("run downloads retried conversation_id=%s run=%s", conversation_id, n)
    return updated


def cancel(conversation_id: str, n: int, reason: str = "") -> dict | None:
    """Cancel a run's pending jobs (taken off the queue, or stopped if running). reason: why, in a few
    words in Spanish: it goes with the cancellation to the gateway's panel and stays on the run
    (`cancel_reason`)."""
    run = conversations.get_run(conversation_id, n)
    if run is None:
        return None
    reason = " ".join((reason or "").split())[:comfy.MAX_CANCEL_REASON]
    if reason:
        conversations.update_run(conversation_id, n, lambda r: r.update(cancel_reason=reason))
    client = comfy.client()
    for index, job in enumerate(run["jobs"]):
        if job["status"] != "pending":
            continue
        if client is not None:
            try:
                client.cancel(job["base_url"], job["prompt_id"], reason)
            except Exception as e:
                logger.warning("run cancel failed conversation_id=%s run=%s error=%r", conversation_id, n, e)
        tracker._set_job(conversation_id, n, index, status="cancelled", error="Cancelled")
    run = conversations.get_run(conversation_id, n)
    if run is not None and run["status"] == "pending":
        tracker._finish(conversation_id, n)
    return conversations.get_run(conversation_id, n)


def prioritize(conversation_id: str, n: int) -> dict | None:
    """Put a run's jobs still waiting in the gateway's queue ahead of everything else waiting
    (comfy.PRIORITY_AHEAD), for the user working in that conversation now. Jobs already in ComfyUI
    go on as they are. The run keeps `priority` and `prioritized_at`; the answer says how many
    jobs moved (`raised`), and `errors` what failed."""
    run = conversations.get_run(conversation_id, n)
    client = comfy.client()
    if run is None or client is None:
        return None
    raised, errors = 0, []
    for job in run["jobs"]:
        if job["status"] != "pending":
            continue
        try:
            raised += client.prioritize(job["base_url"], job["prompt_id"])
        except Exception as e:
            logger.warning("run prioritize failed conversation_id=%s run=%s error=%r", conversation_id, n, e)
            errors.append(str(e))
    if raised:
        conversations.update_run(conversation_id, n, lambda r: r.update(priority=comfy.PRIORITY_AHEAD,
                                                                        prioritized_at=_now()))
    logger.info("run prioritized conversation_id=%s run=%s jobs=%d", conversation_id, n, raised)
    return {"run": conversations.get_run(conversation_id, n), "raised": raised, "errors": sorted(set(errors))}


def wait_until_idle(timeout: float = 5.0) -> None:
    """For tests: let finished-callback threads run."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(t.name.startswith("run-finished-") for t in threading.enumerate()):
            return
        time.sleep(0.01)
