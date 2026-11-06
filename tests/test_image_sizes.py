"""The chat gets each image's size, to keep its place before the images load."""
from app import conversations

from .conftest import png_bytes
from .test_api_conversar_comfy import _conversation, _family


def test_a_conversation_comes_with_its_images_sizes_as_shown(api_client, auth_headers, tmp_path):
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]
    small = conversations.save_image(conversation_id, png_bytes(size=(64, 48)), "image/png")
    large = conversations.save_image(conversation_id, png_bytes(size=(1800, 2700)), "image/png")
    conversations.append_message(conversation_id, {"role": "results", "run": 1, "content": "", "images": [small, large]})

    images = api_client.get(f"/api/conversations/{conversation_id}", auth=auth_headers).json()["messages"][0]["images"]
    assert (images[0]["width"], images[0]["height"]) == (64, 48)
    # A large one is shown as its small copy: its size, same proportions.
    assert large["small"] and (images[1]["width"], images[1]["height"]) == (1045, 1568)
    # Worked out for the answer, never stored.
    assert "width" not in conversations.get(conversation_id)["messages"][0]["images"][0]
