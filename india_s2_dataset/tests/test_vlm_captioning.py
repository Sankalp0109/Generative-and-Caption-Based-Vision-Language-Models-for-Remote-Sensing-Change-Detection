import json

import numpy as np
import pytest

from vlm_captioning import array_to_data_url, caption_pair, caption_pair_two_stage, _load_api_key_from_env_file


def make_image(value):
    return np.full((8, 8, 3), value, dtype=np.float32)


class Response:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class RecordingSession:
    def __init__(self, reply_content):
        self.reply_content = reply_content
        self.last_request = None

    def post(self, url, headers=None, json=None, timeout=None):
        self.last_request = {"url": url, "headers": headers, "json": json, "timeout": timeout}
        return Response({"choices": [{"message": {"content": self.reply_content}}]})


class SequentialSession:
    """Returns a different canned reply for each successive call, in order."""

    def __init__(self, reply_contents):
        self.reply_contents = list(reply_contents)
        self.requests = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.requests.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        content = self.reply_contents[len(self.requests) - 1]
        return Response({"choices": [{"message": {"content": content}}]})


def test_array_to_data_url_round_trips_shape():
    url = array_to_data_url(make_image(0.5))
    assert url.startswith("data:image/png;base64,")


def test_caption_pair_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # isolate from this project's real local .env file
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        caption_pair(make_image(0.2), make_image(0.8), session=RecordingSession("{}"))


def test_caption_pair_sends_both_images_and_region_hints():
    reply = json.dumps({
        "overall_caption": "New construction appeared in the lower-right quadrant.",
        "likely_artifact": False,
        "change_type": "construction",
        "regions": [{"hint_index": 0, "description": "new building footprint", "confirmed": True}],
    })
    session = RecordingSession(reply)
    regions = [{"x": 1, "y": 2, "width": 3, "height": 4, "score": 0.9}]

    result = caption_pair(make_image(0.2), make_image(0.8), regions=regions,
                          api_key="test-key", session=session, send_region_crops=False)

    assert result.overall_caption.startswith("New construction")
    assert result.change_type == "construction"
    assert result.regions[0]["confirmed"] is True

    sent = session.last_request["json"]
    image_blocks = [b for b in sent["messages"][1]["content"] if b["type"] == "image_url"]
    assert len(image_blocks) == 2  # before + after, no marked view or crops
    text_blocks = " ".join(b["text"] for b in sent["messages"][1]["content"] if b["type"] == "text")
    assert "hint_index=0" in text_blocks
    assert sent["response_format"] == {"type": "json_object"}
    assert session.last_request["headers"]["Authorization"] == "Bearer test-key"


def test_caption_pair_crops_are_opt_in():
    reply = json.dumps({"overall_caption": "x", "likely_artifact": False, "change_type": "other", "regions": []})
    session = RecordingSession(reply)
    regions = [
        {"x": 1, "y": 2, "width": 3, "height": 4, "score": 0.9},
        {"x": 0, "y": 0, "width": 2, "height": 2, "score": 0.5},
    ]

    caption_pair(make_image(0.2), make_image(0.8), regions=regions, api_key="test-key",
                session=session, send_region_crops=True)

    sent = session.last_request["json"]
    content = sent["messages"][1]["content"]
    image_blocks = [b for b in content if b["type"] == "image_url"]
    # before + after + (before-crop, after-crop) per region
    assert len(image_blocks) == 2 + 2 * len(regions)
    text_blocks = [b["text"] for b in content if b["type"] == "text"]
    assert any("hint_index=0 -- BEFORE crop" in t for t in text_blocks)
    assert any("hint_index=1 -- AFTER crop" in t for t in text_blocks)


def test_caption_pair_crops_clip_to_image_bounds():
    # A region near the edge with generous padding must not raise/crash and must
    # still produce a non-empty crop clipped to the array bounds.
    session = RecordingSession(json.dumps({"overall_caption": "x", "likely_artifact": False,
                                           "change_type": "other", "regions": []}))
    regions = [{"x": 0, "y": 0, "width": 8, "height": 8, "score": 0.9}]
    caption_pair(make_image(0.2), make_image(0.8), regions=regions,
                api_key="test-key", session=session, send_region_crops=True, crop_padding=50)
    assert session.last_request is not None


def test_caption_pair_two_stage_requires_regions():
    with pytest.raises(ValueError, match="requires at least one region"):
        caption_pair_two_stage(make_image(0.2), make_image(0.8), regions=[],
                               api_key="test-key", session=RecordingSession("{}"))


def test_caption_pair_two_stage_uses_neutral_observations_to_confirm_real_change():
    stage1_reply = json.dumps({"observations": [
        {"hint_index": 0, "observation": "A paved road appears where there was a dirt track."},
    ]})
    stage2_reply = json.dumps({
        "overall_caption": "A road was resurfaced.",
        "likely_artifact": False,
        "change_type": "road_infrastructure",
        "regions": [{"hint_index": 0, "description": "Road resurfaced from dirt to paved.", "confirmed": True}],
    })
    session = SequentialSession([stage1_reply, stage2_reply])
    regions = [{"x": 1, "y": 2, "width": 3, "height": 4, "score": 0.9}]

    result = caption_pair_two_stage(make_image(0.2), make_image(0.8), regions=regions,
                                    api_key="test-key", session=session)

    assert len(session.requests) == 2
    assert result.regions[0]["confirmed"] is True
    assert result.change_type == "road_infrastructure"
    assert "stage1" in result.raw_response and "stage2" in result.raw_response

    # Stage 1 must not carry the artifact-judging schema/instructions at all -- it only
    # describes pixels, neutrally -- that separation from stage 2 is the whole point of the fix.
    stage1_sent = session.requests[0]["json"]
    assert "likely_artifact" not in stage1_sent["messages"][0]["content"]
    stage1_images = [b for b in stage1_sent["messages"][1]["content"] if b["type"] == "image_url"]
    assert len(stage1_images) == 2 * len(regions)

    # Stage 2 must receive the neutral observations produced by stage 1.
    stage2_sent = session.requests[1]["json"]
    stage2_text = " ".join(b["text"] for b in stage2_sent["messages"][1]["content"] if b["type"] == "text")
    assert "paved road appears" in stage2_text


def test_caption_pair_raises_on_invalid_json():
    session = RecordingSession("not json")
    with pytest.raises(ValueError, match="did not return valid JSON"):
        caption_pair(make_image(0.2), make_image(0.8), api_key="test-key", session=session)


def test_load_api_key_from_env_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text('OPENROUTER_API_KEY="sk-or-abc123"\n')
    assert _load_api_key_from_env_file() == "sk-or-abc123"


def test_load_api_key_from_env_file_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _load_api_key_from_env_file() is None
