# India Sentinel-2 Bi-Temporal Change Pairs (15,610 pairs)

Quality-screened Sentinel-2 image pairs covering all of India: 15,610 pairs from 1,669 sites in 35 states and union territories. Each pair shows the same 2.56 km × 2.56 km area about six years apart (2019–20 vs 2025–26), in the same dry-season window, on an identical pixel grid.

Most pairs are **unlabelled**. They are intended for change detection, self-supervised pretraining, change captioning and remote-sensing research on India. 1,002 pairs carry change labels. Those labels are released separately with full detail in the companion dataset **India Sentinel-2 Change Pairs (Labelled)**, and are repeated here in the `label` columns.

## Contents

| File | What it is |
|---|---|
| `pairs.csv` | One row per pair: location, dates, scene IDs, split, quality measures, label (where available) |
| `images_train/`, `images_val/`, `images_test/` | `<patch_id>/before.png`, `after.png` (8-bit RGB, 256×256) and `before.tif`, `after.tif` (same pixels, georeferenced GeoTIFF) |
| `stats.json` | Counts by split |

## Columns (`pairs.csv`)

| Column | Meaning |
|---|---|
| `patch_id`, `site` | Pair ID, and the 10.24 km site it was cut from (16 patches per site) |
| `split` | train / val / test by site: 12,303 / 1,706 / 1,601 pairs |
| `state`, `region`, `lat`, `lon`, `bbox_wgs84` | Location |
| `before_date`, `after_date`, `before_scene`, `after_scene` | Acquisition dates and Sentinel-2 scene IDs |
| `crs`, `tile` | UTM projection and MGRS tile |
| `pixel_changed_fraction` | Share of pixels with a large colour change after brightness matching. **Mostly seasonal crop change, not land-use change.** |
| `cloud_shadow_fraction`, `water_fraction`, `snow_fraction` | From the Sentinel-2 scene classification |
| `alignment_shift_px` | Measured before/after misalignment (all ≤ 0.5 px) |
| `haze_rating` | Highest haze rating of the two dates from the automatic screen (0–3; pairs rated ≥ 2 were removed) |
| `label`, `label_source` | For the 1,002 labelled pairs: `change` / `seasonal` / `no_change` / `cant_tell`, and `human` or `automatic` |
| `image_dir` | Folder holding the pair's images |

## Splits

Splits are made **by site**, so neighbouring patches never appear in two splits. Sites in the labelled dataset keep the same split here, so the two releases can be combined without leakage. For example, you can pretrain on `train` and evaluate on the labelled `test` pairs.

## How it was made

- **Sites:** 1,200 sites spread evenly over every state and union territory (farthest-point sampling), plus top-up sites. Sampling is not biased toward change.
- **Imagery:** Sentinel-2 Level-2A, Collection-1 (reprocessed archive), true-colour product at 10 m. Both dates come from the same tile grid, so pixels are not resampled.
- **Season matching:** both dates come from the same region-specific dry-season window, no more than 30 days apart in position within the season.
- **Screening:**
  - cloud and shadow ≤ 1% (≤ 3% in cloudy regions)
  - snow ≤ 20%, water ≤ 70%
  - no-data ≤ 0.1%
  - whiteout ≤ 30%
  - misalignment ≤ 0.5 px
  - an automatic haze screen
- Of 27,888 patch pairs, 15,610 were accepted.

## Known limitations

- **Change is rare.** Only about 15% of evenly sampled pairs contain lasting land-use change. `pixel_changed_fraction` mostly reflects crop cycles.
- **Resolution.** At 10 m a single building covers 1–2 pixels.
- **Residual haze.** The haze screen is imperfect: some thin haze remains, and some clean pairs were removed.
- **Uneven coverage.** Lakshadweep is absent because its sites are open sea, and small territories have few sites.

## Licence and attribution

Released under **CC BY 4.0**.

- Contains modified Copernicus Sentinel data (2019–2026).
- State boundaries: geoBoundaries (CC BY 4.0).
