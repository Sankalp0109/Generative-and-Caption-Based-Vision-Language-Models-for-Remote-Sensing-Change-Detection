#!/usr/bin/env python3
"""
Bi-temporal satellite change map (every transition coloured) + caption.

Pipeline
  1. Load before/after RGB images (co-registered: same area, same size).
  2. Detect ALL changed pixels by colour difference (Lab distance), whatever they are.
  3. Classify each date into: vegetation, bare soil, water, built-up, shadow.
     Built-up is the default for anything that does not look natural, so coloured
     roofs (blue, red, white...) count too.
  4. Split built-up into BUILDING vs ROAD by shape:
       - strips narrower than --road-width are peeled off with morphological opening
       - rectangle-like region (short, well-filled box)        -> building
       - closed structure (wide region with a hole/courtyard)  -> building
       - long strip, or long thin branching network            -> road
  5. Each changed pixel gets a transition (before-class -> after-class) and its own colour.
  6. Stats (area %, patches, location) per transition -> caption.
  7. prompt.txt: ready-made prompt for a vision-language model, to send together with
     before image, after image and change_map.png (in that order).

Colours: every transition has a base colour (TRANSITION_COLORS), optimised so the
  36 colours are as far apart as possible. In each image, the colours actually used
  are checked again and swapped for high-contrast spares if any two are closer than
  MIN_GAP (CIEDE2000), so everything in one map's legend is clearly distinguishable.
  Change map: dark-grey background. Overlay: unchanged areas dimmed to greyscale,
  changed areas in strong colour with black outlines.

Usage
  pip install opencv-python scikit-image numpy
  python change_caption.py before.png after.png --out results --road-width 12

--road-width is in PIXELS: about road width in metres / image resolution (m/pixel).
Optional: --built-mask-before / --built-mask-after = built-up masks from any
segmentation model (non-zero = building or road); the shape split still applies.
"""
import argparse
import colorsys
import json
import os

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.exposure import match_histograms

CLASSES = ["vegetation", "bare soil", "water", "building", "road", "shadow"]
VEG, SOIL, WATER, BUILDING, ROAD, SHADOW = range(6)
N = len(CLASSES)

# Base colours: optimised so all 36 are as far apart as possible (CIEDE2000) while
# staying bright enough to show on the dark background. In each image, colours of the
# transitions actually present are re-checked and swapped if any two are too similar.
TRANSITION_COLORS = {  # (from, to): RGB
    # building appeared
    (VEG, BUILDING): (255, 57, 111), (SOIL, BUILDING): (255, 90, 77),
    (WATER, BUILDING): (255, 134, 198), (SHADOW, BUILDING): (206, 192, 228),
    (ROAD, BUILDING): (240, 111, 4),
    # building disappeared
    (BUILDING, VEG): (4, 156, 255), (BUILDING, SOIL): (136, 145, 214),
    (BUILDING, WATER): (207, 137, 243), (BUILDING, SHADOW): (249, 0, 184),
    (BUILDING, ROAD): (247, 184, 199),
    # road appeared
    (VEG, ROAD): (189, 255, 43), (SOIL, ROAD): (173, 141, 12),
    (WATER, ROAD): (250, 237, 43), (SHADOW, ROAD): (255, 181, 0),
    # road disappeared
    (ROAD, VEG): (47, 245, 248), (ROAD, SOIL): (32, 157, 82),
    (ROAD, WATER): (62, 207, 255), (ROAD, SHADOW): (28, 164, 154),
    # natural transitions
    (VEG, SOIL): (196, 142, 106), (SOIL, VEG): (153, 255, 176),
    (VEG, WATER): (52, 159, 188), (WATER, VEG): (27, 202, 28),
    (SOIL, WATER): (179, 70, 251), (WATER, SOIL): (253, 255, 179),
    # shadow transitions (often from new/removed tall objects or sun angle)
    (VEG, SHADOW): (172, 185, 0), (SHADOW, VEG): (63, 208, 172),
    (SOIL, SHADOW): (183, 116, 112), (SHADOW, SOIL): (255, 181, 153),
    (WATER, SHADOW): (137, 142, 140), (SHADOW, WATER): (242, 236, 241),
    # same class, appearance changed
    (BUILDING, BUILDING): (255, 218, 162), (ROAD, ROAD): (161, 174, 143),
    (VEG, VEG): (88, 128, 119), (SOIL, SOIL): (183, 163, 163),
    (WATER, WATER): (163, 202, 205), (SHADOW, SHADOW): (135, 119, 136),
}
# High-contrast spare colours used when two transitions in one image look too alike
SPARE_COLORS = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 255, 255),
                (255, 0, 0), (0, 128, 255), (255, 128, 0), (160, 100, 255), (128, 255, 128),
                (255, 150, 150), (0, 200, 150), (200, 200, 0), (150, 200, 255), (255, 200, 255),
                (190, 0, 50), (243, 195, 0), (161, 202, 241), (230, 143, 172), (141, 182, 0)]
BACKGROUND = (40, 40, 40)   # change map background (dark grey, not black)
MIN_GAP = 25                # minimum CIEDE2000 distance between colours in one image
CLASS_COLORS = np.array([(0, 160, 0), (190, 150, 100), (30, 80, 200),
                         (220, 60, 60), (230, 230, 230), (40, 40, 40)], np.uint8)
GRID = [["top-left", "top", "top-right"],
        ["left", "centre", "right"],
        ["bottom-left", "bottom", "bottom-right"]]


def code(f, t):          # label-map value for a transition (0 = no change)
    return 1 + f * N + t


def decode(v):
    return (v - 1) // N, (v - 1) % N


def transition_name(f, t):
    return f"{CLASSES[f]} changed" if f == t else f"{CLASSES[f]} -> {CLASSES[t]}"


def transition_phrase(f, t):
    prev = "a former road" if f == ROAD else f"former {CLASSES[f]}"
    if f == t == BUILDING:
        return "Existing buildings changed (rebuilt or repainted)"
    if f == t == ROAD:
        return "Existing roads changed (resurfaced or widened)"
    if t == BUILDING:
        return f"New buildings appeared on {prev}"
    if t == ROAD:
        return f"New roads were built on {prev}"
    if f == BUILDING:
        return f"Buildings disappeared, leaving {CLASSES[t]}"
    if f == ROAD:
        return f"Roads disappeared, replaced by {CLASSES[t]}"
    if f == t:
        return f"Areas of {CLASSES[f]} changed in appearance"
    if (f, t) == (VEG, SOIL):
        return "Vegetation was cleared to bare soil"
    if (f, t) == (SOIL, VEG):
        return "Bare soil became vegetated"
    return f"{CLASSES[f].capitalize()} changed to {CLASSES[t]}"


# ---------------------------------------------------------------- loading
def load_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
    if img.dtype != np.uint8:  # e.g. 16-bit GeoTIFF -> stretch to 8-bit
        img = img.astype(np.float32)
        lo, hi = np.percentile(img, (1, 99))
        img = (np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    return img


def load_mask(path, w, h):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Could not read {path}")
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0


def save_rgb(path, img):
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------- building vs road (shape)
def has_closed_hole(sub, min_hole_area):
    padded = np.pad(sub.astype(np.uint8), 1)
    contours, hier = cv2.findContours(padded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return False
    return any(hier[0][i][3] >= 0 and cv2.contourArea(contours[i]) >= min_hole_area
               for i in range(len(contours)))


def shape_of(sub):
    """Return (length, aspect ratio, rectangularity) of one region."""
    (_, _), (rw, rh), _ = cv2.minAreaRect(cv2.findNonZero(sub.astype(np.uint8)))
    rw, rh = rw + 1, rh + 1                       # pixel centres -> pixel extents
    length, width = max(rw, rh), min(rw, rh)
    return length, length / width, sub.sum() / (rw * rh)


def split_built(built, a):
    """Split a built-up mask into (building_mask, road_mask) using shape."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (a.road_width + 2, a.road_width + 2))
    b8 = built.astype(np.uint8)
    wide = cv2.morphologyEx(b8, cv2.MORPH_OPEN, k)     # strips narrower than a road vanish
    thin = b8 & (1 - wide)                              # ...and end up here
    building = np.zeros(built.shape, bool)
    road = np.zeros(built.shape, bool)

    hug = None
    for part, is_wide in ((wide, True), (thin, False)):
        if not is_wide:   # region around buildings found so far, to catch edge slivers
            hug = cv2.dilate(building.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        n, lab, st, _ = cv2.connectedComponentsWithStats(part, connectivity=8)
        for i in range(1, n):
            x, y, w, h = st[i, :4]
            sub = lab[y:y + h, x:x + w] == i
            length, aspect, rect = shape_of(sub)
            rect_like = aspect < a.max_aspect and rect >= a.min_rect
            if is_wide:
                # wide region: rectangle-like or closed (courtyard) -> building, else wide road
                is_building = rect_like or has_closed_hole(sub, a.road_width ** 2)
            else:
                # thin region: long strip or long branching network -> road;
                # short compact leftovers (small houses, building corners) -> building
                # thin sliver lying mostly along a building's edge -> part of that building
                strip = aspect >= a.max_aspect or (length >= 3 * a.road_width and rect < a.min_rect)
                hugs_building = hug[y:y + h, x:x + w][sub].mean() > 0.5
                is_building = hugs_building or not strip
            target = building if is_building else road
            target[y:y + h, x:x + w] |= sub
    return building, road


# ---------------------------------------------------------------- per-date classification
def mode_filter(cls, k=5):
    votes = np.stack([cv2.blur((cls == c).astype(np.float32), (k, k)) for c in range(N)])
    return votes.argmax(0).astype(np.uint8)


def classify(rgb, a, built_mask=None):
    f = rgb.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    H, S, V = hsv[..., 0], hsv[..., 1] / 255, hsv[..., 2] / 255   # OpenCV hue: 0-180
    exg = (2 * g - r - b) / (f.sum(2) + 1e-6)                        # Excess Green

    cls = np.full(rgb.shape[:2], BUILDING, np.uint8)   # default: built-up, any colour
    cls[(H >= 8) & (H <= 30) & (S >= 0.12) & (S <= 0.55) & (V >= 0.25) & (V <= 0.85)] = SOIL
    cls[exg > a.exg] = VEG
    water = (b > r) & (b >= 0.9 * g) & (S > 0.15) & (V < a.water_v)
    cls[water] = WATER
    cls[(V < a.shadow_v) & ~water] = SHADOW
    cls = mode_filter(cls)
    if built_mask is not None:
        cls[built_mask] = BUILDING

    building, road = split_built(cls == BUILDING, a)
    cls[building] = BUILDING
    cls[road] = ROAD
    return cls


# ---------------------------------------------------------------- change detection
def change_mask(before, after, min_diff):
    lab_b = cv2.cvtColor(before, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_a = cv2.cvtColor(after, cv2.COLOR_RGB2LAB).astype(np.float32)
    diff = cv2.GaussianBlur(np.linalg.norm(lab_a - lab_b, axis=2), (5, 5), 0)
    return diff > min_diff


def drop_small(mask, min_area):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def clean(mask, min_area):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return drop_small(m, min_area)


# ---------------------------------------------------------------- description
def locate(mask):
    h, w = mask.shape
    n, _, st, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    names = []
    for i in np.argsort(-st[1:, cv2.CC_STAT_AREA])[:3] + 1:
        cx, cy = cent[i]
        name = GRID[min(int(3 * cy / h), 2)][min(int(3 * cx / w), 2)]
        if name not in names:
            names.append(name)
    if n - 1 >= 8 and len(names) >= 3:
        return "across the whole scene"
    joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return "the " + joined + (" part" if len(names) == 1 else " parts") + " of the image"


def make_caption(transitions, min_pct=0.3, max_sentences=5):
    if sum(t["area_pct"] for t in transitions) < 0.5:
        return "There is no significant change between the two images."
    parts = []
    for t in transitions[:max_sentences]:
        if t["area_pct"] < min_pct:
            break
        loc = t["location"] if t["location"].startswith("across") else "in " + t["location"]
        n = t["patches"]
        parts.append(f"{t['phrase']} {loc} (about {t['area_pct']:.1f}% of the area, "
                     f"{n} separate area{'s' if n != 1 else ''}).")
    return " ".join(parts) or "Only minor, scattered changes are visible."


def add_legend(img, entries, min_width=360):
    h, w = img.shape[:2]
    W = max(w, min_width)
    if W > w:
        img = np.hstack([img, np.full((h, W - w, 3), 255, np.uint8)])
    bar = np.full((26 * len(entries) + 10, W, 3), 255, np.uint8)
    for k, (name, color, pct) in enumerate(entries):
        y = 8 + 26 * k
        cv2.rectangle(bar, (10, y), (30, y + 18), color, -1)
        cv2.rectangle(bar, (10, y), (30, y + 18), (0, 0, 0), 1)
        label = f"{name}  ({pct:.1f}%)" if pct is not None else name
        cv2.putText(bar, label, (40, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([img, bar])


def to_lab(rgb):
    return rgb2lab(np.array([[rgb]], np.float64) / 255.0)[0, 0]


def family_variants(rgb, max_hue_shift=40):
    """Shades of the same colour family, ordered from most to least similar to rgb."""
    h, sat, val = colorsys.rgb_to_hsv(*(c / 255 for c in rgb))
    out = []
    for dh in range(-max_hue_shift, max_hue_shift + 1, 5):
        for sv, vv in ((1, 1), (1, 0.75), (0.55, 1), (1, 0.6), (0.4, 1)):
            r, g, b = colorsys.hsv_to_rgb((h + dh / 360) % 1, min(sat * sv, 1), min(val * vv, 1))
            out.append(tuple(int(round(x * 255)) for x in (r, g, b)))
    base = to_lab(rgb)
    return sorted(set(out), key=lambda c: deltaE_ciede2000(to_lab(c), base))


def assign_colors(codes_by_area):
    """Give each present transition a colour at least MIN_GAP away from the others
    and from the background. Largest transitions keep their base colour first."""
    chosen = {}
    taken = [to_lab(BACKGROUND)]
    for v in codes_by_area:
        base = TRANSITION_COLORS[decode(v)]
        gap = min(deltaE_ciede2000(to_lab(base), t) for t in taken)
        color = base
        if gap < MIN_GAP:
            # too similar to a colour already used: first try other shades of the same
            # family (nearby hue / lighter / darker), closest to the base colour first
            ok = [c for c in family_variants(base)
                  if min(deltaE_ciede2000(to_lab(c), t) for t in taken) >= MIN_GAP]
            if ok:
                color = ok[0]
            else:   # nothing in the family works -> most distinct spare colour
                pool = [c for c in SPARE_COLORS if c not in chosen.values()]
                color = max(pool, key=lambda c: min(deltaE_ciede2000(to_lab(c), t) for t in taken))
        chosen[v] = tuple(int(x) for x in color)
        taken.append(to_lab(color))
    return chosen


def draw_overlay(after, label, colors):
    """Unchanged areas: dimmed greyscale. Changed areas: strong colour + black outline."""
    grey = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
    out = (np.stack([grey] * 3, axis=2) * 0.55).astype(np.uint8)
    for v, color in colors.items():
        m = label == v
        out[m] = (0.8 * np.array(color) + 0.2 * after[m]).astype(np.uint8)
    for v in colors:
        contours, _ = cv2.findContours((label == v).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, (0, 0, 0), 1)
    return out


# ---------------------------------------------------------------- prompt for a vision-language model
NAMED_COLORS = {
    "red": (230, 30, 30), "dark red": (139, 0, 0), "orange": (255, 140, 0), "yellow": (255, 230, 0),
    "mustard": (190, 150, 20), "olive": (128, 128, 0), "lime": (190, 255, 40), "green": (0, 200, 0),
    "dark green": (0, 100, 0), "mint": (150, 255, 180), "teal": (0, 150, 140), "cyan": (0, 255, 255),
    "sky blue": (70, 170, 255), "blue": (30, 60, 230), "navy": (0, 0, 128),
    "lavender": (180, 160, 240), "purple": (130, 40, 200), "magenta": (255, 0, 255),
    "pink": (255, 140, 190), "hot pink": (255, 60, 160), "salmon": (250, 128, 114),
    "peach": (255, 200, 160), "brown": (140, 80, 40), "tan": (210, 180, 140),
    "cream": (250, 245, 200), "white": (255, 255, 255), "light grey": (190, 190, 190),
    "grey": (128, 128, 128), "slate grey": (110, 120, 140),
}


def color_names(rgbs):
    """Nearest everyday colour name for each RGB; duplicates get light/dark prefixes."""
    names = []
    for c in rgbs:
        lab = to_lab(c)
        names.append(min(NAMED_COLORS, key=lambda n: deltaE_ciede2000(to_lab(NAMED_COLORS[n]), lab)))
    for n in set(names):
        idx = [i for i, x in enumerate(names) if x == n]
        if len(idx) > 1:
            idx.sort(key=lambda i: to_lab(rgbs[i])[0])
            for rank, i in enumerate(idx):
                prefix = "darker" if rank == 0 else "lighter" if rank == len(idx) - 1 else "medium"
                names[i] = f"{prefix} {n}"
    return names


MEANING = {
    "to_building": "a new building was constructed where there was {f} before",
    "from_building": "a building was demolished/removed; the area is now {t}",
    "to_road": "a new road was built where there was {f} before",
    "from_road": "a road was removed; the area is now {t}",
    "same": "the {f} is still there but its appearance changed (e.g. rebuilt, repainted, resurfaced)",
    "other": "{f} changed to {t}",
}


def meaning_of(t):
    f, to = t["from"], t["to"]
    if f == to:
        key = "same"
    elif to == "building":
        key = "to_building"
    elif f == "building":
        key = "from_building"
    elif to == "road":
        key = "to_road"
    elif f == "road":
        key = "from_road"
    else:
        key = "other"
    return MEANING[key].format(f=f, t=to)


def build_prompt(transitions, caption, w, h):
    names = color_names([tuple(t["color_rgb"]) for t in transitions])
    if transitions:
        legend = "\n".join(
            f"- {n} (RGB {r},{g},{b}): {t['from']} -> {t['to']} = {meaning_of(t)}. "
            f"Covers {t['area_pct']:.1f}% of the image, {t['patches']} separate "
            f"area{'s' if t['patches'] != 1 else ''}, located in {t['location']}."
            for n, t, (r, g, b) in zip(names, transitions, (t["color_rgb"] for t in transitions)))
        total = sum(t["area_pct"] for t in transitions)
    else:
        legend = "- (none: no changed areas were detected)"
        total = 0.0
    return PROMPT_TEMPLATE.format(legend=legend, total=total, w=w, h=h, draft=caption)


PROMPT_TEMPLATE = """You are an expert in interpreting satellite images. You are given three images of the same area:

Image 1 (BEFORE): satellite image at the earlier date.
Image 2 (AFTER): satellite image of the same area at the later date.
Image 3 (CHANGE MAP): computed automatically from Images 1 and 2. Dark grey = no change. Every coloured region is an area that changed; its colour tells what it changed FROM and TO, according to the legend below.

CHANGE MAP LEGEND AND MEASUREMENTS
These were measured automatically from the images. Treat them as the facts for WHAT changed, WHERE, and HOW MUCH.
{legend}
Total changed area: {total:.1f}% of the image. Image size: {w} x {h} pixels.
Locations use a 3x3 grid: top-left, top, top-right, left, centre, right, bottom-left, bottom, bottom-right.

Rule-based draft caption (correct but plain): "{draft}"

YOUR TASK
Write an accurate caption describing what changed between BEFORE and AFTER.

RULES
1. Describe only the changes listed in the legend. Do not invent changes, objects, counts or numbers that the legend and images do not support.
2. For each listed change: find its colour in Image 3, then look at the same location in Image 1 and Image 2, and describe what is actually visible there (for example roof colour, shape and size of buildings, how buildings are arranged, which direction a road runs, what it connects).
3. Describe larger changes first. You may skip changes below 0.3% of the image if larger ones exist.
4. Ignore differences caused only by lighting, season, shadows, clouds or slight misalignment between the two images.
5. The change map is automatic and can occasionally mislabel things (for example a long narrow building labelled as road, or grey soil labelled as building). Only if Images 1 and 2 CLEARLY show a different object, describe what you see and explain it in "notes". Otherwise follow the legend.
6. If the legend lists no changes, both captions must say: "There is no change between the two images."
7. Use plain, factual language. Use "building(s)" and "road(s)", not technical class names.

OUTPUT
Reply with JSON only, no other text:
{{
  "short_caption": "one sentence, at most 20 words, e.g. 'several new buildings with red roofs are constructed on the bare land in the top right'",
  "detailed_caption": "2 to 4 sentences covering every listed change above 0.3%, largest first, with where it is and what it looks like",
  "notes": "any disagreement between the change map and what Images 1 and 2 show, otherwise an empty string"
}}
"""


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="Bi-temporal change map + caption")
    p.add_argument("before")
    p.add_argument("after")
    p.add_argument("--out", default="change_output")
    p.add_argument("--road-width", type=int, default=12, help="typical road width in PIXELS")
    p.add_argument("--max-aspect", type=float, default=4.0, help="length/width at or above this = strip (road)")
    p.add_argument("--min-rect", type=float, default=0.6, help="fill of its box needed to be rectangle-like")
    p.add_argument("--built-mask-before", help="built-up mask for the before image (any model)")
    p.add_argument("--built-mask-after", help="built-up mask for the after image (any model)")
    p.add_argument("--match-hist", action="store_true", help="match after colours to before (lighting)")
    p.add_argument("--min-diff", type=float, default=15, help="min Lab colour change to count as change")
    p.add_argument("--min-area", type=int, default=30, help="drop change patches smaller than this (px)")
    p.add_argument("--exg", type=float, default=0.05, help="vegetation threshold (Excess Green)")
    p.add_argument("--water-v", type=float, default=0.45, help="max brightness for water")
    p.add_argument("--shadow-v", type=float, default=0.18, help="max brightness for shadow")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    before = load_rgb(a.before)
    after_orig = load_rgb(a.after)
    h, w = before.shape[:2]
    if after_orig.shape != before.shape:
        after_orig = cv2.resize(after_orig, (w, h), interpolation=cv2.INTER_AREA)
    after = after_orig
    if a.match_hist:
        after = np.clip(match_histograms(after_orig, before, channel_axis=-1), 0, 255).astype(np.uint8)

    # 1) where did anything change?
    changed = clean(change_mask(before, after, a.min_diff), a.min_area)

    # 2) what was it before / after?
    mb = load_mask(a.built_mask_before, w, h) if a.built_mask_before else None
    ma = load_mask(a.built_mask_after, w, h) if a.built_mask_after else None
    cls_b, cls_a = classify(before, a, mb), classify(after, a, ma)

    # 3) transition label map
    label = np.zeros((h, w), np.uint8)
    label[changed] = (1 + cls_b.astype(np.int32) * N + cls_a)[changed]
    # edge pixels around a real class change often look like "same-class change";
    # absorb them into the neighbouring class-change transition
    same_codes = [code(c, c) for c in range(N)]
    cross = [v for v in np.unique(label[label > 0]) if v not in same_codes]
    cross.sort(key=lambda v: -(label == v).sum())
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    for v in cross:
        near = cv2.dilate((label == v).astype(np.uint8), k) > 0
        label[near & np.isin(label, same_codes)] = v
    for v in np.unique(label[label > 0]):          # remove tiny fragments of a transition
        m = label == v
        label[m & ~drop_small(m, max(a.min_area // 3, 1))] = 0

    # 4) stats per transition (colours chosen so the ones in THIS image are distinct)
    present = sorted((int(v) for v in np.unique(label[label > 0])), key=lambda v: -(label == v).sum())
    colors = assign_colors(present)
    transitions = []
    for v in present:
        f, t = decode(v)
        m = label == v
        transitions.append({
            "from": CLASSES[f], "to": CLASSES[t], "name": transition_name(f, t),
            "phrase": transition_phrase(f, t), "color_rgb": list(colors[v]),
            "area_pct": round(float(m.sum() / label.size * 100), 2),
            "patches": int(cv2.connectedComponents(m.astype(np.uint8))[0] - 1),
            "location": locate(m),
        })
    caption = make_caption(transitions)

    # 5) images
    heat = np.full_like(before, BACKGROUND)
    for v, color in colors.items():
        heat[label == v] = color
    overlay = draw_overlay(after_orig, label, colors)
    legend = [(t["name"], tuple(t["color_rgb"]), t["area_pct"]) for t in transitions]
    class_legend = [(c, tuple(int(x) for x in CLASS_COLORS[i]), None) for i, c in enumerate(CLASSES)]

    save_rgb(os.path.join(a.out, "change_map.png"), add_legend(heat, legend))
    save_rgb(os.path.join(a.out, "comparison.png"),
             add_legend(np.hstack([before, after_orig, overlay]), legend))
    save_rgb(os.path.join(a.out, "classes_before_after.png"),
             add_legend(np.hstack([CLASS_COLORS[cls_b], CLASS_COLORS[cls_a]]), class_legend))
    np.save(os.path.join(a.out, "label_map.npy"), label)
    with open(os.path.join(a.out, "stats.json"), "w") as fh:
        json.dump({"image_size": [w, h], "caption": caption, "transitions": transitions}, fh, indent=2)

    with open(os.path.join(a.out, "prompt.txt"), "w") as fh:
        fh.write(build_prompt(transitions, caption, w, h))

    print("Caption:", caption)
    print(f"Outputs saved in {a.out}/")


if __name__ == "__main__":
    main()
