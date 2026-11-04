"""Replies that failed because Claude Code reached the subscription's usage limit, answered again
by themselves once the limit resets.

A Claude Code turn that fails that way (claude_code.limit_resets) leaves the conversation's
`retry_at` (conversations.set_retry), when the limit resets; a timer here starts the turn
again MARGIN_SECONDS after that, as "Reintentar" would. A reply the limit cut off midway (it
had already run tools, so it was stored, with its error) is carried on instead: the app adds a
`continue` message (chat.CONTINUE_NOTE) and the bot answers it -- unless runs it launched are
still pending, whose results bring it back anyway. The time is stored, so a restart
schedules it again (resume). It doesn't fire if the last message has been answered meanwhile,
or the user wrote (both clear `retry_at`), or the user turned it off (DELETE …/retry); a
conversation still busy then is tried again a minute later.
"""
import logging
import threading
import time

from . import conversations

logger = logging.getLogger(__name__)

MARGIN_SECONDS = 60
BUSY_DELAY_SECONDS = 60

_timers: dict[str, threading.Timer] = {}
_guard = threading.Lock()


def schedule(conversation_id: str, at: float) -> None:
    """Answer the conversation's last message again at `at` (epoch seconds) plus the margin."""
    delay = max(0.0, at + MARGIN_SECONDS - time.time())
    timer = threading.Timer(delay, _fire, args=(conversation_id,))
    timer.daemon = True
    with _guard:
        previous = _timers.pop(conversation_id, None)
        _timers[conversation_id] = timer
    if previous is not None:
        previous.cancel()
    timer.start()
    logger.info("reply retry scheduled conversation_id=%s in=%ds", conversation_id, round(delay))


def cancel(conversation_id: str) -> None:
    with _guard:
        timer = _timers.pop(conversation_id, None)
    if timer is not None:
        timer.cancel()


def pending() -> set[str]:
    with _guard:
        return set(_timers)


def _fire(conversation_id: str) -> None:
    from . import conversation_turn  # it schedules retries: imported here, not at the top

    with _guard:
        _timers.pop(conversation_id, None)
    conversation = conversations.get(conversation_id)
    if conversation is None or not conversation.get("retry_at"):
        return
    messages = conversation["messages"]
    cut_off = bool(messages) and messages[-1]["role"] == "assistant" and bool(messages[-1].get("error"))
    pending = any(run["status"] == "pending" for run in conversation.get("runs") or [])
    if not messages or not (messages[-1]["role"] in ("user", "results") or (cut_off and not pending)):
        conversations.set_retry(conversation_id, None)  # answered, or its runs' results will bring it back
        return
    if conversation_turn.is_busy(conversation_id):
        schedule(conversation_id, time.time() - MARGIN_SECONDS + BUSY_DELAY_SECONDS)
        return
    conversations.set_retry(conversation_id, None)
    if cut_off:
        conversations.append_message(conversation_id, {"role": "user", "content": "", "images": [], "continue": True})
    try:
        conversation_turn.start_turn(conversation_id)
    except conversation_turn.Busy:
        conversations.set_retry(conversation_id, int(time.time()))
        schedule(conversation_id, time.time() - MARGIN_SECONDS + BUSY_DELAY_SECONDS)
        return
    logger.info("reply retried after the usage limit conversation_id=%s", conversation_id)


def resume() -> None:
    """After a restart: schedule again the retries stored in conversations."""
    for summary in conversations.list_summaries():
        if summary.get("retry_at"):
            schedule(summary["id"], summary["retry_at"])
