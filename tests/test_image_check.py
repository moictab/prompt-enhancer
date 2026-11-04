"""The independent check of a run's images, and zooming in on one (Claude Code)."""
import base64
import io
import json

from PIL import Image

from app import bot_seat, claude_code, conversation_turn, conversations, image_check

from .conftest import png_bytes


def _conversation_with_a_batch(**settings):
    conversation = conversations.create("fam", "", 0.7, engine="claude-code", claude_model="opus", **settings)
    images = [conversations.save_image(conversation["id"], png_bytes(size=(1200, 1800)), "image/png") for _ in range(3)]
    conversations.append_message(conversation["id"], {"role": "user", "content": "hazla", "images": []})
    conversations.append_message(conversation["id"], {"role": "assistant", "content": "Lanzada."})
    conversations.append_message(conversation["id"], {"role": "results", "run": 1, "content": "", "images": images})
    return conversations.get(conversation["id"])


ANSWER = json.dumps({"images": [
    {"image": 1, "verdict": "flawed", "issues": [
        {"what": "seis dedos en la mano izquierda", "where": "abajo a la izquierda", "severity": "major"}]},
    {"image": 2, "verdict": "ok", "issues": []},
    {"image": 99, "verdict": "flawed", "issues": [{"what": "not asked about"}]},
]})


def test_a_batch_is_inspected_once_and_the_bot_gets_what_was_found(monkeypatch):
    conversation = _conversation_with_a_batch()
    asked = []

    def ask(system, prompt, model, context, **kwargs):
        asked.append((prompt, model, context))
        return "Here it is:\n" + ANSWER, {"estimated_cost": 0.02}

    monkeypatch.setattr(claude_code, "ask", ask)
    load = conversation_turn._image_loader(conversation["id"])

    assert image_check.check(conversation, load) == 3
    prompt, model, context = asked[0]
    assert model == "opus" and context["mode"] == "image_check"
    assert prompt[0]["text"] == "Inspect these images, in this order: image 1, image 2, image 3."
    assert [b["type"] for b in prompt[1:]] == ["image"] * 3
    # At the size the model takes, not reduced further.
    sent = Image.open(io.BytesIO(base64.b64decode(prompt[1]["source"]["data"])))
    assert max(sent.size) == 1568

    stored = conversations.get(conversation["id"])
    checks = [i.get("check") for i in stored["messages"][-1]["images"]]
    assert checks[0]["verdict"] == "flawed" and checks[0]["issues"][0]["severity"] == "major"
    assert checks[1]["verdict"] == "ok" and checks[2]["verdict"] == "unknown"  # not answered: not asked again
    text = image_check.context_text(stored)
    assert text.startswith(image_check.CONTEXT_HEADING)
    assert "- Image 1: seis dedos en la mano izquierda (abajo a la izquierda, major)." in text
    assert "- Image 2: nothing found." in text and "Image 3" not in text

    # A retry (or a redo) doesn't pay for it again.
    assert image_check.check(stored, load) == 0 and len(asked) == 1


def test_no_check_when_turned_off_or_after_the_bot_has_spoken(monkeypatch):
    monkeypatch.setattr(claude_code, "ask", lambda *a, **k: (_ for _ in ()).throw(AssertionError("asked")))
    off = _conversation_with_a_batch(image_check=False)
    assert image_check.check(off, conversation_turn._image_loader(off["id"])) == 0
    answered = _conversation_with_a_batch()
    conversations.append_message(answered["id"], {"role": "assistant", "content": "Las he visto."})
    answered = conversations.get(answered["id"])
    assert image_check.check(answered, conversation_turn._image_loader(answered["id"])) == 0
    assert image_check.context_text(answered) is None


def test_a_failed_check_leaves_the_turn_alone(monkeypatch):
    conversation = _conversation_with_a_batch()

    def fail(*args, **kwargs):
        raise claude_code.AskError("Claude Code: usage limit")

    monkeypatch.setattr(claude_code, "ask", fail)
    assert image_check.check(conversation, conversation_turn._image_loader(conversation["id"])) == 0
    assert not any(i.get("check") for i in conversations.get(conversation["id"])["messages"][-1]["images"])


def test_ask_sends_images_as_stream_json(monkeypatch):
    launched = {}

    class Done:
        returncode = 0

        def communicate(self, sent, timeout=None):
            launched["sent"] = sent
            events = [{"type": "rate_limit_event", "rate_limit_info": {"status": "allowed", "unifiedWindows": {}}},
                      {"type": "result", "subtype": "success", "is_error": False, "result": "{}",
                       "total_cost_usd": 0.01}]
            return "\n".join(json.dumps(e) for e in events) + "\n", ""

    def spawn(args, env, cwd):
        launched["args"] = args
        return Done()

    monkeypatch.setattr(claude_code, "_spawn", spawn)
    blocks = [{"type": "text", "text": "look"},
              {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "eA=="}}]
    text, info = claude_code.ask("system", blocks, "opus", {"mode": "image_check"})
    assert text == "{}" and info["estimated_cost"] == 0.01
    args = launched["args"]
    assert args[args.index("--input-format") + 1] == "stream-json" and "--no-session-persistence" in args
    assert json.loads(launched["sent"])["message"]["content"] == blocks


def test_zooming_in_crops_the_original_at_full_detail(tmp_path):
    path = tmp_path / "big.png"
    image = Image.new("RGB", (1000, 2000), (10, 10, 10))
    image.paste((250, 0, 0), (0, 1500, 200, 1700))  # a red "hand" low on the left
    image.save(path)

    zoomed = bot_seat._zoomed(str(path), [{"x": 0, "y": 75, "width": 20, "height": 10},
                                          {"x": 50, "y": 0, "width": 50, "height": 100}])
    assert [b.type for b in zoomed] == ["text", "image", "text", "image"]
    hand = Image.open(io.BytesIO(base64.b64decode(zoomed[1].data)))
    # 200x200 px of the original, scaled up for a close look; all red.
    assert hand.size == (bot_seat.ZOOM_MIN_SIDE, bot_seat.ZOOM_MIN_SIDE) and hand.getpixel((300, 300))[0] > 200
    assert "200x200 px of the original" in zoomed[0].text
    half = Image.open(io.BytesIO(base64.b64decode(zoomed[3].data)))
    assert max(half.size) == 2000 * bot_seat.ZOOM_MAX_SIDE // 2000  # 500x2000: down to what the model takes
    assert isinstance(bot_seat._zoomed(str(path), [{"x": 1}]), str)
