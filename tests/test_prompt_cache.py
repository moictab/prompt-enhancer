"""Each turn re-sends the conversation; with Anthropic models the provider caches it, and a
cached prefix costs a tenth. That only works if what one turn sends starts exactly like what
the previous turn cached: these tests hold the replay to that."""
from unittest.mock import patch

from app import conversation_turn, families, memory
from app.openrouter_client import StreamEnd

MODEL = "anthropic/claude-sonnet-5"


def _parts(content):
    parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
    return [{k: v for k, v in p.items() if k != "cache_control"} for p in parts]


def _marks(messages):
    """(message index, part index) of every cache breakpoint."""
    return [(i, j) for i, m in enumerate(messages) if isinstance(m.get("content"), list)
            for j, p in enumerate(m["content"]) if "cache_control" in p]


def _prefix(messages, mark):
    """The request up to and including a breakpoint, without the marks themselves."""
    i, j = mark
    return [{**m, "content": _parts(m["content"])} for m in messages[:i]] + [_parts(messages[i]["content"])[:j + 1]]


class Recorder:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, api_key, model, messages, temperature, context=None, **kwargs):
        self.calls.append(messages)
        yield self.replies.pop(0)
        yield StreamEnd(cost=0.01)


def _turns(api_client, auth_headers, tmp_path, *contents, replies):
    family = families.create_family("Krea 2", "rules", False, path=str(tmp_path / "families.json"))
    payload = {"family_id": family["id"], "llm_model": MODEL, "temperature": 0.5}
    conversation_id = api_client.post("/api/conversations", json=payload, auth=auth_headers).json()["id"]
    memory.create_note("likes: cold light")
    recorder = Recorder(*replies)
    with patch("app.conversation_turn.stream_chat", recorder):
        for content in contents:
            api_client.post(f"/api/conversations/{conversation_id}/messages", data={"content": content},
                            auth=auth_headers)
    return recorder.calls


def test_a_turn_starts_exactly_like_the_previous_one_cached(api_client, auth_headers, tmp_path):
    # The bot takes a note in its first reply: the notes the next turn carries are different.
    first, second, third = _turns(
        api_client, auth_headers, tmp_path, "Una bruja", "Más luz", "Ahora de noche",
        replies=["<title>Bruja</title>Hecha. <remember>likes: witches</remember>", "Vale.", "Hecho."],
    )

    for earlier, later in ((first, second), (second, third)):
        # The earlier turn's breakpoint on its latest message, before the notes...
        stable = _marks(earlier)[-2]
        assert earlier[stable[0]]["content"][stable[1] + 1]["text"].startswith("[The app's context")
        # ...is where the later turn starts exactly the same, and marks it again so it is read from cache.
        assert _prefix(later, stable) == _prefix(earlier, stable)
        assert stable in _marks(later)
    # The system prompt carries no notes (they changed): it is cached as it is, every turn.
    assert "likes: witches" not in second[0]["content"][0]["text"]
    assert "[2] likes: witches" in second[-1]["content"][-1]["text"]


def test_four_breakpoints_at_most():
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"}, {"role": "user", "content": [
                    {"type": "text", "text": "c"}, {"type": "text", "text": "context"}]}]

    marked = conversation_turn.with_cache_breakpoints(MODEL, messages, history_length=4, volatile_tail=1)

    assert _marks(marked) == [(0, 0), (1, 0), (3, 0), (3, 1)]
    assert messages[3]["content"][0] == {"type": "text", "text": "c"}  # the original is untouched
