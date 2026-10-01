"""Per-site preprocessing: verify, align-check, brightness-match, cut 256 px patches, screen, label.

For one downloaded site (``data/sentinel2/<site>/pair_1``) this produces 16 patch folders, each a
self-contained, labelled sample:

    <patch_id>/before.tif, after.tif    raw 8-bit true colour, exact georeference, lossless
               before.png, after.png    display copies (shared colour-preserving stretch) = model input
               meta.json                every label: where, when, quality, scores, status, reason

A patch that fails a check here is written straight to ``rejected/<reason>/``; one that passes goes
to ``_pending/`` to wait for the Ada haze screen. Nothing is ever deleted.

Brightness matching (after -> before, fitted on unchanged pixels) is used only for the change score.
The model sees un-matched images: matching would partly cancel a haze veil and hide it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from change_visualization import joint_stretch
from stac_select import SCL_SNOW, SCL_WATER

PATCH = 256
GRID = 4                       # 1024 px site -> 4 x 4 patches
MAX_SHIFT_PX = 0.5             # before/after misregistration tolerated
SCL_CLOUD_SHADOW = (3, 8, 9, 10)
LIMITS = {"nodata": 0.001, "cloud_shadow": 0.01, "snow": 0.20, "water": 0.70, "whiteout": 0.30}
WHITE_LEVEL = 245
CHANGE_DN = 40                 # |after_matched - before| (0-255) counted as changed; 25 flagged 72% of patches (crops, season), 40 flags 42%
CHANGED_FRACTION = 0.05        # a patch is "changed" if >= 5% of its pixels changed


def _read(path: Path):
    with rasterio.open(path) as src:
        data = src.read()
        return (np.moveaxis(data, 0, -1) if src.count > 1 else data[0]), src.crs, src.transform


def verify(pair_dir: Path) -> tuple[dict | None, str | None]:
    """Load a site and check it is intact and on one grid. Returns (arrays, None) or (None, reason)."""
    needed = ["before.tif", "after.tif", "scl_before.tif", "scl_after.tif", "pair.json"]
    missing = [n for n in needed if not (pair_dir / n).exists()]
    if missing:
        return None, f"missing_files:{','.join(missing)}"
    try:
        before, crs_b, tf_b = _read(pair_dir / "before.tif")
        after, crs_a, tf_a = _read(pair_dir / "after.tif")
        scl_b, crs_sb, tf_sb = _read(pair_dir / "scl_before.tif")
        scl_a, crs_sa, tf_sa = _read(pair_dir / "scl_after.tif")
        meta = json.loads((pair_dir / "pair.json").read_text())
    except (rasterio.errors.RasterioIOError, ValueError, OSError) as error:
        return None, f"unreadable:{type(error).__name__}"
    size = PATCH * GRID
    if before.shape != (size, size, 3) or after.shape != (size, size, 3):
        return None, "wrong_shape"
    if scl_b.shape != (size, size) or scl_a.shape != (size, size):
        return None, "wrong_shape_scl"
    if not (crs_b == crs_a == crs_sb == crs_sa) or not (tf_b == tf_a == tf_sb == tf_sa):
        return None, "grid_mismatch"
    return {"before": before, "after": after, "scl_before": scl_b, "scl_after": scl_a,
            "meta": meta, "crs": crs_b, "transform": tf_b}, None


def alignment_shift(before: np.ndarray, after: np.ndarray) -> tuple[float, float]:
    """Sub-pixel (dy, dx) shift of after relative to before, by phase correlation on edge images.

    Edges (gradient magnitude) instead of raw brightness, so seasonal colour differences don't
    dominate; a Hann window suppresses the image border.
    """
    def edges(img):
        lum = img.astype("float32").mean(axis=2)
        gy, gx = np.gradient(lum)
        e = np.hypot(gx, gy)
        return (e - e.mean()) * np.outer(np.hanning(e.shape[0]), np.hanning(e.shape[1]))

    fa, fb = np.fft.fft2(edges(before)), np.fft.fft2(edges(after))
    cross = fa * np.conj(fb)
    corr = np.fft.ifft2(cross / np.maximum(np.abs(cross), 1e-9)).real
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    shift = []
    for axis, p in enumerate(peak):
        n = corr.shape[axis]
        idx = [peak[0], peak[1]]
        idx[axis] = (p - 1) % n; left = corr[tuple(idx)]
        idx[axis] = (p + 1) % n; right = corr[tuple(idx)]
        centre = corr[peak]
        denom = left - 2 * centre + right
        frac = 0.5 * (left - right) / denom if denom != 0 else 0.0
        s = p + frac
        shift.append(s - n if s > n / 2 else s)
    return float(shift[0]), float(shift[1])


def brightness_match(before: np.ndarray, after: np.ndarray, valid: np.ndarray) -> list[tuple[float, float]]:
    """Per-band gain/offset mapping after -> before, fitted on pixels that look unchanged.

    First fit on all valid pixels, then refit twice on the 40% of pixels closest to that fit
    (the ones least likely to have really changed).
    """
    params = []
    for band in range(3):
        x = after[..., band][valid].astype("float64")
        y = before[..., band][valid].astype("float64")
        keep = np.ones_like(x, dtype=bool)
        gain, offset = 1.0, 0.0
        for _ in range(3):
            if keep.sum() < 100:
                break
            gain, offset = np.polyfit(x[keep], y[keep], 1)
            gain = float(np.clip(gain, 0.5, 2.0))
            residual = np.abs(y - (gain * x + offset))
            keep = residual <= np.percentile(residual, 40)
        params.append((round(float(gain), 4), round(float(offset), 3)))
    return params


def apply_match(after: np.ndarray, params) -> np.ndarray:
    out = np.empty(after.shape, dtype="float32")
    for band, (gain, offset) in enumerate(params):
        out[..., band] = after[..., band] * gain + offset
    return np.clip(out, 0, 255)


def _save_tif(path: Path, array: np.ndarray, crs, transform: Affine) -> None:
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
                       count=array.shape[2], dtype="uint8", crs=crs, transform=transform,
                       compress="deflate", predictor=2, tiled=True, blockxsize=256, blockysize=256) as out:
        out.write(np.moveaxis(array, -1, 0))


def site_region(state: str) -> str:
    groups = {
        "north_plains": ["Punjab", "Haryana", "Delhi", "Chandigarh", "Uttar Pradesh", "Bihar", "West Bengal"],
        "himalaya": ["Jammu and Kashmir", "Ladakh", "Himachal Pradesh", "Uttarakhand", "Sikkim", "Arunachal Pradesh"],
        "northeast": ["Assam", "Meghalaya", "Nagaland", "Manipur", "Mizoram", "Tripura"],
        "west_desert": ["Rajasthan", "Gujarat", "Dadra and Nagar Haveli and Daman and Diu"],
        "central_east": ["Madhya Pradesh", "Chhattisgarh", "Jharkhand", "Odisha"],
        "deccan": ["Maharashtra", "Telangana", "Andhra Pradesh", "Karnataka", "Goa"],
        "south_islands": ["Tamil Nadu", "Kerala", "Puducherry", "Andaman and Nicobar Islands", "Lakshadweep"],
    }
    return next((region for region, states in groups.items() if state in states), "other")


def process_site(site: str, pair_dir: Path, out_root: Path, site_info: dict, site_reject: str | None = None) -> list[dict]:
    """Write the 16 patch folders of one site; return their manifest rows.

    ``site_reject`` (e.g. "overlap_duplicate") rejects every patch of the site for that reason.
    """
    arrays, reason = verify(pair_dir)
    if arrays is None:  # nothing can be cut: record one site-level row so the loss is visible
        return [{"patch_id": f"{site}_site", "site": site, "status": "rejected", "stage": "verify",
                 "reason": reason, "state": site_info.get("state")}]
    before, after, meta = arrays["before"], arrays["after"], arrays["meta"]
    crs, transform = arrays["crs"], arrays["transform"]
    dy, dx = alignment_shift(before, after)
    shift = float(np.hypot(dy, dx))
    valid = np.all(before > 0, axis=2) & np.all(after > 0, axis=2)
    match = brightness_match(before, after, valid)
    matched = apply_match(after, match)
    view_b, view_a = [(v * 255).astype("uint8") for v in joint_stretch(before, after, 1, 99.5)]

    state = site_info.get("state") or (meta.get("site") or {}).get("state") or "?"
    rows = []
    for r in range(GRID):
        for c in range(GRID):
            pid = f"{site}_r{r}c{c}"
            win = np.s_[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH]
            pb, pa = before[win], after[win]
            fractions = {}
            for key, fn in (("nodata", lambda s, img: (s == 0) | np.all(img == 0, axis=2)),
                            ("cloud_shadow", lambda s, img: np.isin(s, SCL_CLOUD_SHADOW)),
                            ("snow", lambda s, img: s == SCL_SNOW),
                            ("water", lambda s, img: s == SCL_WATER),
                            ("whiteout", lambda s, img: img.min(axis=2) >= WHITE_LEVEL)):
                fractions[key] = round(float(max(fn(arrays["scl_before"][win], pb).mean(),
                                                 fn(arrays["scl_after"][win], pa).mean())), 4)
            diff = np.abs(matched[win] - pb.astype("float32")).mean(axis=2)
            change = {"changed_fraction": round(float((diff > CHANGE_DN).mean()), 4),
                      "mean_abs_diff": round(float(diff.mean()), 2)}
            change["changed"] = change["changed_fraction"] >= CHANGED_FRACTION

            reason, stage = None, None
            if site_reject:
                reason, stage = site_reject, "site"
            elif shift > MAX_SHIFT_PX:
                reason, stage = "misaligned", "alignment"
            else:
                for key in ("nodata", "cloud_shadow", "snow", "water", "whiteout"):
                    if fractions[key] > LIMITS[key]:
                        reason, stage = key, "patch_screen"
                        break

            ptransform = transform * Affine.translation(c * PATCH, r * PATCH)
            west, south, east, north = array_bounds(PATCH, PATCH, ptransform)
            bounds_wgs84 = transform_bounds(crs, "EPSG:4326", west, south, east, north, densify_pts=5)
            record = {
                "patch_id": pid, "site": site, "row": r, "col": c,
                "state": state, "region": site_region(state), "category": site_info.get("category"),
                "tile": meta["block"]["tile"], "block": meta["block"]["id"],
                "crs": str(crs), "transform": list(ptransform)[:6],
                "bounds_wgs84": [round(v, 6) for v in bounds_wgs84], "size_px": PATCH, "pixel_m": 10,
                "before": {"item_id": meta["before"]["item_id"], "date": meta["before"]["datetime"][:10],
                           "sun_elevation": meta["before"].get("sun_elevation")},
                "after": {"item_id": meta["after"]["item_id"], "date": meta["after"]["datetime"][:10],
                          "sun_elevation": meta["after"].get("sun_elevation")},
                "season_gap_days": meta.get("gap_days"),
                "alignment": {"dy_px": round(dy, 3), "dx_px": round(dx, 3), "shift_px": round(shift, 3)},
                "screen": fractions,
                "brightness_match": {"after_to_before_gain_offset": match},
                "change": change,
                "model": None,
                "status": "rejected" if reason else "pending",
                "stage": stage, "reason": reason,
            }
            folder = out_root / ("rejected/" + reason if reason else "_pending") / pid
            shutil.rmtree(folder, ignore_errors=True)
            folder.mkdir(parents=True)
            _save_tif(folder / "before.tif", pb, crs, ptransform)
            _save_tif(folder / "after.tif", pa, crs, ptransform)
            Image.fromarray(view_b[win]).save(folder / "before.png")
            Image.fromarray(view_a[win]).save(folder / "after.png")
            (folder / "meta.json").write_text(json.dumps(record, indent=1))
            rows.append({**record, "folder": str(folder.relative_to(out_root))})
    return rows


def finalize(out_root: Path, patch_id: str, model: dict, verdict: str) -> dict:
    """Move a pending patch to accepted/ or rejected/haze_cloud/ with the model's labels."""
    src = out_root / "_pending" / patch_id
    record = json.loads((src / "meta.json").read_text())
    record["model"] = model
    if verdict == "keep":
        record.update(status="accepted", stage=None, reason=None)
        dest = out_root / "accepted" / patch_id
    else:
        reason = "model_unreadable" if verdict == "unparsed" else "haze_cloud"
        record.update(status="rejected", stage="model", reason=reason)
        dest = out_root / "rejected" / reason / patch_id
    (src / "meta.json").write_text(json.dumps(record, indent=1))
    shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    return {**record, "folder": str(dest.relative_to(out_root))}
