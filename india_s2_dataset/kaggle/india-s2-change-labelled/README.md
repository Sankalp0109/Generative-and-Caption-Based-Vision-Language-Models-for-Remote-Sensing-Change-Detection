# India Sentinel-2 Change Pairs — Labelled (1,002 pairs)

Bi-temporal Sentinel-2 image pairs from across India, labelled for **lasting land-use change vs seasonal change vs no change**. Each pair shows the same 2.56 km × 2.56 km area about six years apart (2019–20 vs 2025–26), in the same dry-season window.

## Contents

| File | What it is |
|---|---|
| `labels.csv` | One row per pair: label, label source, location, dates, scene IDs, split, baseline model answers |
| `images.zip` | `images/<patch_id>/before.png`, `after.png` (8-bit RGB, 256×256) and `before.tif`, `after.tif` (same pixels, georeferenced GeoTIFF) |
| `stats.json` | Counts by label, source and split |

## Labels

| Label | Meaning | Pairs |
|---|---|---|
| `change` | Lasting land-use change: new or expanded buildings, roads, construction, quarries or mines, new water bodies, cleared vegetation, solar farms | 143 |
| `seasonal` | Only crop stage, vegetation greenness or water level differ | 400 |
| `no_change` | No meaningful difference | 383 |
| `cant_tell` | Haze or cloud prevents a decision | 76 |

**`label_source`, please read before use:**

- `human` (535 pairs): labelled blind by a human annotator from the full-size before and after images.
- `automatic` (467 pairs): labelled by visual inspection without human review. A random check found about **20%** of these disagree with a human, so treat them as noisy.

For evaluation, the cleanest choice is to use only the `human` rows. Note that the human-reviewed pairs over-represent difficult cases, because they were chosen where automatic labels disagreed.

`caption` gives a one-sentence description for some automatic `change` pairs only.

## Splits

`split` is a 70/15/15 train/val/test split **by site**, so neighbouring patches from the same 10 km site never appear in two splits:

- **train:** 664 pairs
- **val:** 186 pairs
- **test:** 152 pairs

The pairs come from 172 sites in 32 states and union territories. `source_split` is the split of the larger unlabelled release.

## Baseline columns

`qwen3vl_32b_says` and `qwen3vl_235b_says` are the answers of Qwen3-VL-32B and Qwen3-VL-235B to the question *"Has anything lasting changed (like buildings, roads, ponds, cleared land), or are the differences only seasonal?"*. They are provided as baselines and are not labels. On the human-labelled pairs, the best of them was right on only about half of its `change` calls.

## How the imagery was made

- **Source:** Sentinel-2 Level-2A, Collection-1 (reprocessed archive), true-colour product at 10 m.
- **Season matching:** before and after images come from the same region-specific dry-season window, no more than 30 days apart in position within the season.
- **Quality screening:**
  - cloud and shadow ≤ 1% (≤ 3% in cloudy regions)
  - snow ≤ 20%, water ≤ 70%
  - no-data ≤ 0.1%
  - haze and whiteout checks
  - before/after misalignment ≤ 0.5 px
- **Grid:** both dates come from the same Sentinel-2 tile grid and are not resampled.

## Known limitations

- At 10 m resolution a single building covers 1–2 pixels, so small changes are often not visible.
- Some thin haze remains in a minority of pairs.
- Seasonal differences can look like change in RGB alone.

## Licence and attribution

Released under **CC BY 4.0**.

- Contains modified Copernicus Sentinel data (2019–2026).
- State boundaries: geoBoundaries (CC BY 4.0).
