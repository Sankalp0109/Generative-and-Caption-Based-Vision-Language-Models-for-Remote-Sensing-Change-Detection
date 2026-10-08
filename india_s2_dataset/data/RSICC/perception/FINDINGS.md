# Can the change be seen at 10 m, and can the model see it? (29 Sep 2026)

Test set: 60 pairs, labelled by eye at full size. 40 come from the gold test set, plus 12 from the
curated known-change sites and 8 high-change patches. The labels were 12 land-use change,
24 seasonal only, 17 no change, and 7 with haze that slipped through the filter.
API spend for this investigation: about $0.12.

## 1. Is land-use change visible in 10 m Sentinel-2?

Yes, for area-sized changes: new roads and interchanges, town growth, construction sites,
quarries, ponds, and large buildings or complexes. Individual houses are not visible (1–2 pixels).
Real land-use change is rare in evenly spread sites: 4 of 40 gold pairs (10%), against about
50% of the top-change patches at curated sites. The pixel-difference "changed" flag mostly
catches seasonal crop differences, not land use.

## 2. Can the vision model see it? (4-class task, 60 pairs; results in `results.jsonl`)

| Model, variant | Land-use found | False land-use | Hazy caught |
|---|---|---|---|
| Qwen3-VL-8B, plain | 11/12 | 7 | 0/7 |
| Gemma 3 27B, plain | 11/12 | 15 | 0/7 |
| Qwen3-VL-32B, plain | 12/12 | 16 | 0/7 |
| Qwen3-VL-235B, plain | 5/12 | 0 | 0/7 |
| **32B (confidence ≥ 0.9) AND 8B agree** | **11/12** | **1 (of 48)** | n/a |

- The models can see land-use change. The problem is calibration, not perception: the 32B
  over-calls it, and the 235B under-calls it and labels everything "seasonal".
- The change-outline overlay image did not help.
- No model flags leftover haze in a 4-class task. A dedicated haze prompt doesn't separate it
  either (235B: 4/7 caught with 27/53 false alarms). Pixel statistics don't separate it.
- Because haze does not create land-use change, the strict triage rule stops hazy pairs from
  becoming false change. They end up as "no land-use change", which is correct.

## 3. Can it describe the change? (12 land-use pairs, `landuse_captions.html`)

With a focused land-use prompt, both the 32B and the 235B describe the change correctly in
about 9 of 12 cases: urban growth, construction and new built-up areas. Their weak spots are the
exact location and specific types (ponds, roads). The 32B matches the 235B at about a quarter
of the cost.

## Root causes of the bad captions

1. **Forced change.** The two-stage method always proposed regions and told the model to trust them.
2. **No "no change" answer** was allowed in the prompt.
3. **Seasonal differences were treated as change.**
4. **Leftover haze** was described as change.
5. **Sampling.** Evenly spread sites contain little real land-use change.

## Solution (feasible)

1. **Redefine the label as land-use change** (not pixel change). Keep "seasonal" as a sub-tag.
2. **Triage every accepted pair.** Use Qwen3-VL-32B with the 4-class prompt, and require
   confidence ≥ 0.9 *and* agreement from the 8B (which can run free on Ada) to call land-use change.
   Validated at 11/12 found, 1/48 false. Cost is about $1.7 via the API with the 8B on Ada,
   or about $3.6 fully via the API, for 15,610 pairs.
3. **Caption by class.**
   - **Land-use:** the focused prompt, Qwen3-VL-32B, 3 captions, about $0.0005 per pair.
   - **No land-use change:** LEVIR-CC-style fixed "no change" captions, with a
     "only seasonal crop differences" variant chosen from the triage. Free.
4. **Grow the land-use pairs** by targeted sampling of change hotspots (industrial, construction,
   quarries, new roads, fast-growing town edges, big projects) found through OpenStreetMap.
   Evenly spread sites give about 10–15% land-use change; hotspots about 50%.
5. **Balanced set:** land-use change vs no land-use change at 50/50. Expected size is about
   800–1,300 land-use pairs from the current data, and about 2,500 after roughly 500
   hotspot sites.
6. **Gold set:** 100 pairs reviewed by the user before any large captioning run.
