"""The haze/cloud question asked of the vision model, and how its answer becomes keep/reject.

Shared by the Ada job (local open-weight model) and the API comparison, so both are asked
exactly the same thing and judged by exactly the same rule.

v1 ended with an example answer of all zeros; the model copied it for 100 of 100 pairs. The
answer format therefore carries placeholders only, the model writes one observation sentence
before scoring, and haze is described the way it is actually seen: by comparing the two dates.
"""

from __future__ import annotations

import json
import re

PROMPT_VERSION = "v2"
MODEL_SIDE_PX = 512  # 256 px patches are upscaled so the model sees the texture of haze

PROMPT = """You are checking the image quality of a satellite image pair before it is used in a dataset.
Image 1 is BEFORE and image 2 is AFTER. Both show the same 2.56 km x 2.56 km area in Sentinel-2 true colour (10 m pixels).

Look carefully at the ATMOSPHERE in each image, not at changes on the ground.
- Haze (smog, fog, dust haze, thin or wispy cloud) looks like a milky, whitish or bluish-grey veil: colours look faded,
  dark areas (vegetation, water, shadows) look grey instead of dark, and fine detail looks soft or blurred.
  It is easiest to see by comparing the two images: if one looks washed out or lower in contrast next to the other,
  that image has haze. Haze may cover only part of an image.
- Cloud means opaque bright white patches, or the dark shadows that clouds cast.
Do NOT count these as haze or cloud: bright ground (sand, concrete, roofs, dry fields, riverbeds), snow or glaciers,
water colour or sediment, crops that differ between the dates, or sun and shadow differences.

Score each image from 0 to 3: 0 = none, 1 = trace (barely noticeable), 2 = moderate (clearly visible), 3 = heavy.

Reply with JSON only, using these keys and filling in your own values:
{"observation": "<one sentence describing the atmosphere in each image>",
 "before": {"haze": <0-3>, "cloud": <0-3>},
 "after": {"haze": <0-3>, "cloud": <0-3>}}"""

MAX_NEW_TOKENS = 120
# Haze level at which a pair is rejected. Chosen per model on the separate dev set (30 pairs):
# the 8B model (8-bit) under-rates visible haze as "1", so it needs 1; 8B in 4-bit and the 235B
# model rate haze higher and are right at 2.
REJECT_AT = {"8b": 1, "8b_4bit": 2, "235b": 2}


def parse(text: str) -> dict | None:
    """The model's ratings, or None if the answer is not the expected JSON."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return {role: {k: int(data[role][k]) for k in ("haze", "cloud")} for role in ("before", "after")}
    except (ValueError, KeyError, TypeError):
        return None


def verdict(ratings: dict | None, reject_at: int = 2) -> str:
    """keep / reject / unparsed: reject when haze reaches ``reject_at`` or any cloud reaches 2
    in either image. An unreadable answer is never silently kept."""
    if ratings is None:
        return "unparsed"
    haze = max(ratings["before"]["haze"], ratings["after"]["haze"])
    cloud = max(ratings["before"]["cloud"], ratings["after"]["cloud"])
    return "reject" if haze >= reject_at or cloud >= 2 else "keep"
