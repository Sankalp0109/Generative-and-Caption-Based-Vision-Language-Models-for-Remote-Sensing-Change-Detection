"""Generate change captions for an aligned before/after pair via an OpenRouter VLM.

Uses OpenRouter's OpenAI-compatible chat completions endpoint
(https://openrouter.ai/api/v1/chat/completions) so no extra SDK is needed beyond
``requests``, which this project already depends on.

Default model is ``qwen/qwen3-vl-8b-instruct`` -- checked against OpenRouter's live
model catalogue while wiring this up: $0.117 / $0.455 per 1M prompt/completion
tokens, i.e. a handful of cents for the whole 43-pair local set. Two other Qwen3-VL
variants were live on OpenRouter at the same price point or cheaper and can be
swapped in via CONFIG without code changes: ``qwen/qwen3-vl-32b-instruct`` (larger,
priced slightly *below* the 8B model at time of writing) and
``qwen/qwen3-vl-30b-a3b-instruct`` (a mixture-of-experts variant, similar cost).
Check https://openrouter.ai/models for current pricing before a large run --
OpenRouter pricing changes over time and is not pinned by this module.
"""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass, field

import numpy as np
import requests
from PIL import Image

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3-vl-8b-instruct"
ENV_FILE = ".env"


def _load_api_key_from_env_file(path: str = ENV_FILE) -> str | None:
    """Read OPENROUTER_API_KEY=... from a local dotenv-style file, if present.

    Never printed or logged -- read directly into memory and returned. Lets the
    key live in a gitignored local file instead of being pasted into a chat
    transcript or exported per-shell-call (tool shells don't persist env vars
    between calls here).
    """
    try:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except FileNotFoundError:
        return None
    return None

SYSTEM_PROMPT = """\
You are analyzing a bi-temporal remote-sensing image pair (a BEFORE image and an \
AFTER image of the exact same ground area, taken at different times) to caption \
land-cover / land-use change for a research dataset.

Ground rules:
- Only describe change that is visible in both images at the same location; do not \
invent detail neither image supports.
- Many apparent differences are NOT real change: cloud cover, cloud shadow, sun \
angle/shadow length, seasonal vegetation color, and water level from rainfall can \
all look like "change" but are not the kind of change this dataset is collecting. \
Explicitly say so when you suspect one of these, and set "likely_artifact": true.
- A set of heuristic candidate regions (pixel boxes from a simple RGB-difference \
detector, not a trained model) may be provided as hints, each as its own small \
BEFORE/AFTER crop pair. Judge each cropped hint strictly on its own two crops --  \
describe only what differs between that specific before/after crop pair -- before \
you reason about the scene as a whole. Do not let a dominant story about the rest \
of the scene (e.g. "this looks like a quiet village with only seasonal change") \
override what a specific crop actually shows; a real, localized change can exist \
even in an otherwise unchanged scene. Only after judging every hint crop \
independently should you write the overall_caption synthesizing everything found. \
Treat hints only as candidates to check, not ground truth -- reject any that show \
no real difference, and feel free to describe real change the hints missed.
- Respond with strict JSON only, matching this schema, no markdown fences, no prose \
outside the JSON:
{
  "overall_caption": "one or two sentences summarizing the dominant real change, or stating there is no clear real change",
  "likely_artifact": true/false,
  "change_type": "one short label, e.g. construction | vegetation_loss | vegetation_gain | water_level | urban_expansion | road_infrastructure | none | other",
  "regions": [
    {"hint_index": <int or null>, "description": "what actually changed here", "confirmed": true/false}
  ]
}
"""


@dataclass
class CaptionResult:
    overall_caption: str
    likely_artifact: bool
    change_type: str
    regions: list[dict] = field(default_factory=list)
    raw_response: dict = field(default_factory=dict)
    captions: list[str] = field(default_factory=list)   # extra phrasings when n_captions > 1


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Accepts a float array in [0, 1] or [0, 255], or an already-uint8 array."""
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 1 if array.max() <= 1.0 else 255)
        if array.max() <= 1.0:
            array = array * 255
        array = array.astype(np.uint8)
    return array[..., :3]


def array_to_data_url(image: np.ndarray, format: str = "PNG") -> str:
    """Encode a display-ready image array as a base64 data URL for the API."""
    buffer = io.BytesIO()
    Image.fromarray(_to_uint8_rgb(image)).save(buffer, format=format)
    encoded = base64.standard_b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{format.lower()};base64,{encoded}"


def _region_hints_text(regions: list[dict] | None) -> str:
    if not regions:
        return "No heuristic candidate regions were provided for this pair."
    lines = ["Heuristic candidate regions (x, y, width, height in pixels; score 0-1):"]
    for index, region in enumerate(regions):
        lines.append(
            f"  hint_index={index}: x={region['x']}, y={region['y']}, "
            f"width={region['width']}, height={region['height']}, score={region['score']:.3f}"
        )
    return "\n".join(lines)


def _crop(image: np.ndarray, region: dict, padding: int) -> np.ndarray:
    height, width = image.shape[0], image.shape[1]
    x0 = max(0, region["x"] - padding)
    y0 = max(0, region["y"] - padding)
    x1 = min(width, region["x"] + region["width"] + padding)
    y1 = min(height, region["y"] + region["height"] + padding)
    return image[y0:y1, x0:x1]


def _region_crop_content(before_view: np.ndarray, after_view: np.ndarray,
                         regions: list[dict], padding: int) -> list[dict]:
    """Build before/after crop image pairs, one per hint, for independent per-hint judgment."""
    blocks = []
    for index, region in enumerate(regions):
        blocks.append({"type": "text", "text": f"hint_index={index} -- BEFORE crop:"})
        blocks.append({"type": "image_url", "image_url": {"url": array_to_data_url(_crop(before_view, region, padding))}})
        blocks.append({"type": "text", "text": f"hint_index={index} -- AFTER crop:"})
        blocks.append({"type": "image_url", "image_url": {"url": array_to_data_url(_crop(after_view, region, padding))}})
    return blocks


CHANGE_TYPE_ENUM_HINT = (
    "construction | vegetation_loss | vegetation_gain | water_level | "
    "urban_expansion | road_infrastructure | none | other"
)

STAGE1_SYSTEM_PROMPT = """\
You are comparing small cropped image patches. Each crop pair shows the exact \
same tiny ground location at two different times (time 1 and time 2).

For EACH crop pair, describe only what is visibly different between time 1 and \
time 2, based strictly on pixels you can see. Do not speculate about the cause \
-- do not mention sun angle, shadow, season, clouds, or "artifact"; just describe \
what is different, or say "no visible difference" if there genuinely is none.

Respond with strict JSON only, no markdown fences, no prose outside the JSON:
{"observations": [{"hint_index": <int>, "observation": "..."}]}
"""

STAGE2_SYSTEM_PROMPT = f"""\
You are analyzing a bi-temporal remote-sensing image pair (a BEFORE image and an \
AFTER image of the exact same ground area, taken at different times) to caption \
land-cover / land-use change for a research dataset.

An earlier, independent pass already inspected small crops of several candidate \
regions and recorded neutral pixel-level observations for each (provided below) \
without judging their cause. Use those observations as your primary evidence for \
each region -- do not override an observation that clearly describes a visible \
difference by re-labeling it as an artifact unless the full-scene images actively \
contradict it. A specific, clearly-described local difference (e.g. a paved road \
appearing, a new building) should be confirmed as real change even inside an \
otherwise quiet scene.

Cloud cover, shadow, sun angle, and seasonal vegetation color are still valid \
reasons to call the OVERALL scene an artifact when that is what actually \
dominates it -- this caution applies at the whole-scene level, not as a default \
override for every specific, well-described local observation.

Respond with strict JSON only, matching this schema, no markdown fences, no prose \
outside the JSON:
{{
  "overall_caption": "one or two sentences summarizing the dominant real change, or stating there is no clear real change",
  "likely_artifact": true/false,
  "change_type": "one short label, e.g. {CHANGE_TYPE_ENUM_HINT}",
  "regions": [
    {{"hint_index": <int or null>, "description": "what actually changed here", "confirmed": true/false}}
  ]
}}
"""


def _resolve_api_key(api_key: str | None) -> str:
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or _load_api_key_from_env_file()
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Put it in a local .env file "
            "(OPENROUTER_API_KEY=sk-or-...) or export it in your shell before "
            "calling caption_pair -- do not hardcode it in a notebook or script."
        )
    return api_key


def _call_vlm(system: str, content: list[dict], *, model: str, api_key: str,
              max_output_tokens: int, temperature: float, request_timeout: float,
              session: requests.Session | None) -> tuple[dict, dict]:
    """POST one chat completion and return (parsed_json, raw_response_body)."""
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional per OpenRouter docs, but recommended so usage is attributable.
        "X-Title": "Indian bi-temporal change dataset captioning",
    }
    http = session or requests
    response = http.post(OPENROUTER_URL, headers=headers, json=payload, timeout=request_timeout)
    response.raise_for_status()
    body = response.json()
    if "error" in body:
        raise RuntimeError(f"OpenRouter error: {body['error']}")
    message_text = body["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(message_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"model did not return valid JSON: {message_text!r}") from error
    return parsed, body


def caption_pair(before_view: np.ndarray, after_view: np.ndarray,
                 marked_view: np.ndarray | None = None, regions: list[dict] | None = None,
                 *, model: str = DEFAULT_MODEL, api_key: str | None = None,
                 max_output_tokens: int = 700, temperature: float = 0.2,
                 request_timeout: float = 120, session: requests.Session | None = None,
                 send_region_crops: bool = False, crop_padding: int = 20) -> CaptionResult:
    """Call an OpenRouter VLM to caption the change between an aligned before/after pair.

    Single-call version: fast and cheap, but empirically prone to a specific
    failure mode -- a real, localized change (e.g. a resurfaced road) sitting
    inside an otherwise-quiet scene gets swept into a dominant "no real change,
    probably seasonal" narrative and dismissed, even when it is one of the
    region hints. See ``caption_pair_two_stage`` for the validated fix.

    ``before_view``/``after_view`` should be the same display-ready arrays the
    notebooks already render (percentile-stretched, HxWx3, values in [0, 1] or
    [0, 255]). ``marked_view`` (optional) is the same after-image with the
    heuristic evidence boxes drawn on it, giving the model a visual anchor for
    the region hints in ``regions`` (the ``regions`` list from
    ``change_visualization.prepare_visual_evidence``).

    ``send_region_crops`` additionally sends each hint as its own small
    before/after crop pair. Tested and found *not* to fix the miss above on its
    own -- adding crops without separating the neutral-observation step from
    the artifact-cautious synthesis step just gave the model more detail to
    rationalize away under the same skeptical framing, at roughly double the
    token cost. Kept as an option, off by default; use ``caption_pair_two_stage``
    instead when accurate per-region localization matters.
    """
    api_key = _resolve_api_key(api_key)

    content = [
        {"type": "text", "text": "BEFORE image:"},
        {"type": "image_url", "image_url": {"url": array_to_data_url(before_view)}},
        {"type": "text", "text": "AFTER image:"},
        {"type": "image_url", "image_url": {"url": array_to_data_url(after_view)}},
    ]
    if marked_view is not None:
        content.append({"type": "text", "text": "AFTER image with heuristic candidate regions marked:"})
        content.append({"type": "image_url", "image_url": {"url": array_to_data_url(marked_view)}})
    content.append({"type": "text", "text": _region_hints_text(regions)})
    if send_region_crops and regions:
        content.extend(_region_crop_content(before_view, after_view, regions, crop_padding))

    parsed, body = _call_vlm(SYSTEM_PROMPT, content, model=model, api_key=api_key,
                             max_output_tokens=max_output_tokens, temperature=temperature,
                             request_timeout=request_timeout, session=session)
    return CaptionResult(
        overall_caption=parsed.get("overall_caption", ""),
        likely_artifact=bool(parsed.get("likely_artifact", False)),
        change_type=parsed.get("change_type", "other"),
        regions=parsed.get("regions", []),
        raw_response=body,
    )


def caption_pair_two_stage(before_view: np.ndarray, after_view: np.ndarray,
                          regions: list[dict], *, model: str = DEFAULT_MODEL,
                          api_key: str | None = None, max_output_tokens: int = 900,
                          temperature: float = 0.2, request_timeout: float = 120,
                          session: requests.Session | None = None,
                          crop_padding: int = 20, n_captions: int = 1) -> CaptionResult:
    """Caption a pair via a validated two-stage call that avoids the single-call miss.

    ``n_captions`` > 1 also asks stage 2 for that many differently worded one-sentence captions
    of the same change (reference variants for training/evaluating change-captioning models).

    Stage 1: each region hint's before/after crop pair is shown in one neutral
    call that describes only pixel differences, with no artifact-caution
    framing at all (no mention of sun angle/shadow/season/cloud).
    Stage 2: the full before/after images plus those neutral observations are
    given to the normal captioning schema, instructed to trust a clearly
    described local observation as real change rather than defaulting to the
    scene's dominant narrative.

    This was validated on a case where the single-call version dismissed a real
    resurfaced-road/new-building region as "sun angle" -- the two-stage version
    correctly confirmed it as real change. Costs roughly 2x a single hinted
    call (two requests instead of one); ``raw_response`` holds both stages'
    response bodies under "stage1"/"stage2" so combined cost can be read back
    via ``raw_response["stage1"]["usage"]`` and ``raw_response["stage2"]["usage"]``.
    Requires at least one region -- use ``caption_pair`` for a pair with none.
    """
    if not regions:
        raise ValueError("caption_pair_two_stage requires at least one region hint")
    api_key = _resolve_api_key(api_key)

    stage1_content = _region_crop_content(before_view, after_view, regions, crop_padding)
    # _region_crop_content labels crops "BEFORE crop"/"AFTER crop"; stage 1 speaks in
    # time-1/time-2 terms to avoid implying which direction is "correct", so relabel here.
    for block in stage1_content:
        if block["type"] == "text":
            block["text"] = block["text"].replace("BEFORE crop", "time 1 crop").replace("AFTER crop", "time 2 crop")
    stage1_parsed, stage1_body = _call_vlm(
        STAGE1_SYSTEM_PROMPT, stage1_content, model=model, api_key=api_key,
        max_output_tokens=max_output_tokens, temperature=temperature,
        request_timeout=request_timeout, session=session,
    )
    observations = stage1_parsed.get("observations", [])

    stage2_content = [
        {"type": "text", "text": "BEFORE image:"},
        {"type": "image_url", "image_url": {"url": array_to_data_url(before_view)}},
        {"type": "text", "text": "AFTER image:"},
        {"type": "image_url", "image_url": {"url": array_to_data_url(after_view)}},
        {"type": "text", "text": "Neutral per-region observations from the earlier independent pass:"},
        {"type": "text", "text": json.dumps(observations)},
    ]
    stage2_system = STAGE2_SYSTEM_PROMPT
    if n_captions > 1:
        stage2_system += (f'\nAlso include the key "captions": a list of exactly {n_captions} different '
                          "one-sentence captions of the same change (same facts, varied wording).\n")
    stage2_parsed, stage2_body = _call_vlm(
        stage2_system, stage2_content, model=model, api_key=api_key,
        max_output_tokens=max_output_tokens, temperature=temperature,
        request_timeout=request_timeout, session=session,
    )

    return CaptionResult(
        overall_caption=stage2_parsed.get("overall_caption", ""),
        likely_artifact=bool(stage2_parsed.get("likely_artifact", False)),
        change_type=stage2_parsed.get("change_type", "other"),
        regions=stage2_parsed.get("regions", []),
        raw_response={"stage1": stage1_body, "stage2": stage2_body, "stage1_observations": observations},
        captions=[c for c in stage2_parsed.get("captions", []) if isinstance(c, str)],
    )
