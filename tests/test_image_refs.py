"""#N mentions image N: the bot gets that image again with the message that mentions it."""
from app import chat


def test_image_refs_reads_hash_numbers_only():
    assert chat.image_refs("mira #4 y #12; no #ff0000, #1a, &#39; ni ##3; otra vez #4") == [4, 12]


def _conversation():
    images = [{"id": f"i{n}", "content_type": "image/png"} for n in range(1, 8)]
    images[0]["metadata"] = {"format": "PNG", "width": 1, "height": 1, "positive": "a fox"}
    return {"id": "c", "runs": [], "messages": [
        {"role": "user", "content": "primera", "images": images[:3]},
        {"role": "assistant", "content": "Vale."},
        {"role": "user", "content": "como #2", "images": images[3:7]},
        {"role": "assistant", "content": "Hecho."},
        {"role": "user", "content": "vuelve a #1, compárala con #7 y con #99", "images": []},
    ]}


def test_a_mentioned_image_is_attached_again_to_the_latest_message():
    messages = chat.build_llm_messages(_conversation(), load_image=lambda image: f"data:{image['id']}")

    last = messages[-1]["content"]
    assert last[0] == {"type": "text", "text": "vuelve a #1, compárala con #7 y con #99"}
    assert last[1]["text"].startswith("Image 1 (mentioned as #1, attached again):\nImage 1 metadata")
    assert "- Positive: a fox" in last[1]["text"]
    assert last[2] == {"type": "image_url", "image_url": {"url": "data:i1"}}
    # #7 is among the recent images, attached where it was shared; #99 doesn't exist.
    assert len(last) == 3


def test_only_the_latest_message_brings_images_back():
    conversation = _conversation()
    conversation["messages"] = conversation["messages"][:4]  # the latest user message is "como #2"

    messages = chat.build_llm_messages(conversation, load_image=lambda image: f"data:{image['id']}")
    urls = [p["image_url"]["url"] for m in messages if isinstance(m["content"], list)
            for p in m["content"] if p["type"] == "image_url"]

    assert "data:i2" in urls  # mentioned in the latest user message
    conversation["messages"].append({"role": "user", "content": "otra", "images": []})
    later = chat.build_llm_messages(conversation, load_image=lambda image: f"data:{image['id']}")
    later_urls = [p["image_url"]["url"] for m in later if isinstance(m["content"], list)
                  for p in m["content"] if p["type"] == "image_url"]
    assert "data:i2" not in later_urls  # an older mention doesn't keep it attached
