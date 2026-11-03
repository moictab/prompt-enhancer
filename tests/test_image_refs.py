"""#N mentions image N: the bot gets that image again with the message that mentions it."""
from app import chat


def test_image_refs_reads_hash_numbers_only():
    assert chat.image_refs("mira #4 y #12; no #ff0000, #1a, &#39; ni ##3; otra vez #4") == [4, 12]


def _conversation(last="vuelve a #1, compárala con #18 y con #99"):
    # 20 images: the window attaches 9-20, so #1 isn't attached any more and #18 is.
    images = [{"id": f"i{n}", "content_type": "image/png"} for n in range(1, 21)]
    images[0]["metadata"] = {"format": "PNG", "width": 1, "height": 1, "positive": "a fox"}
    return {"id": "c", "runs": [], "messages": [
        {"role": "user", "content": "primera", "images": images[:3]},
        {"role": "assistant", "content": "Vale."},
        {"role": "user", "content": "como #2", "images": images[3:20]},
        {"role": "assistant", "content": "Hecho."},
        {"role": "user", "content": last, "images": []},
    ]}


def _urls(messages):
    return [p["image_url"]["url"] for m in messages if isinstance(m["content"], list)
            for p in m["content"] if p["type"] == "image_url"]


def test_a_mentioned_image_is_attached_again_to_the_latest_message():
    messages = chat.build_llm_messages(_conversation(), load_image=lambda image: f"data:{image['id']}")

    last = messages[-1]["content"]
    assert last[0] == {"type": "text", "text": "vuelve a #1, compárala con #18 y con #99"}
    assert last[1]["text"].startswith("Image 1 (mentioned as #1, attached again):\nImage 1 metadata")
    assert "- Positive: a fox" in last[1]["text"]
    assert last[2] == {"type": "image_url", "image_url": {"url": "data:i1"}}
    # #18 is attached anyway, where it was shared; #99 doesn't exist.
    assert len(last) == 3
    # They change from turn to turn: the end of the message, after what the prompt cache keeps.
    assert messages.volatile_tail == 2


def test_only_the_message_being_answered_brings_images_back():
    conversation = _conversation(last="otra")
    conversation["messages"][2]["content"] = "como #1"
    load = lambda image: f"data:{image['id']}"  # noqa: E731

    # "como #1" is answered already: #1 doesn't come back with it.
    assert "data:i1" not in _urls(chat.build_llm_messages(conversation, load))
    conversation["messages"] = conversation["messages"][:3]
    assert "data:i1" in _urls(chat.build_llm_messages(conversation, load))
