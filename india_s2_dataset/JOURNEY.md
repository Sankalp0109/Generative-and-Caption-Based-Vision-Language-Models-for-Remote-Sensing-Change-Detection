# Project journey

One record of what was built, tried, broken, and fixed — so no fact needs a second
home. The current architecture and status live in `master.md`; this file is the history
behind it.

## Phase 0 — Prototype (mid-Sep 2026)

Built the first end-to-end pass on 3 hand-picked Sentinel-2 sites (Polavaram AP, Navi
Mumbai airport, Statue of Unity) plus 43 pre-aligned local PNG pairs, to prove the
architecture before scaling it: acquire → align → heuristic change detection → VLM
caption → manual QC.

**What it produced:** 21 Sentinel-2 pairs (2018→2024, `visual`/TCI asset, up to 12
monthly pairs per site, float32 GeoTIFFs, ~204 MB), two review notebooks
(`change_detection_review.ipynb`, `local_pair_review.ipynb`), a bounded synthetic test
(`run_bounded_test.sh`), and a PDF report (`change_captioning_report.pdf`) walking
through 11 newly-captioned pairs plus 4 earlier ones.

**What it found — the finding that shaped everything after:**
- **A real miss (Img27).** A single-call caption (`caption_pair`) can fold a real,
  localized change into the scene's dominant "nothing changed, probably seasonal"
  narrative and dismiss it — even when the change sits inside a region hint the model
  was directly shown. Manual review (Stage 5) is what caught it.
- **The fix, validated:** `caption_pair_two_stage` in `vlm_captioning.py`. Each region
  hint's before/after crop is described in a neutral pass first (no mention of shadow,
  season, or "artifact" allowed), then that description is carried into the normal
  full-scene call, instructed to trust a clearly-described local observation instead of
  defaulting to the scene's dominant story. Confirmed against the PDF's test set.
- **Selection-quality problems in the downloader**, found from the manifests: Navi
  Mumbai lost 6 of 12 months to "image does not fully cover the AOI" (scene choice
  ignored real footprint coverage, only tile-level cloud); Polavaram/Statue of Unity
  accepted months with 22–29% *tile* cloud that could still be locally clouded (no
  AOI-level cloud check); monsoon months (Jul–Sep) were empty everywhere; the 9–12
  monthly pairs per site were largely near-duplicates of the same change.
- Config settled on: `max_dimension` 1024, stretch 2/98 percentiles, `attention_percentile`
  75, `attention_radius` 6px, `max_regions` 8, `crop_padding` 20px, model
  `qwen/qwen3-vl-8b-instruct` via OpenRouter (~$0.117 / $0.455 per 1M prompt/completion
  tokens at the time).

**Cleanup after Phase 0** (once the lessons above were captured): deleted the 3 old
Sentinel-2 site downloads, both review notebooks, `run_bounded_test.sh` and its
`results/`, the old `README.md`/`PROJECT_LOG.md` writeups, and `change_captioning_report.pdf`
— all superseded by this file and by `master.md`. Kept the 43 local PNG pairs
(`data/before/`, `data/after/`) as a standing secondary source, and kept
`change_visualization.py` and `vlm_captioning.py` (Stages 3–4) unchanged, since Phase 0
had already validated them.

## Phase 1 — All-India concurrent download (this session)

Goal: replace the 3-site prototype downloader with one that can reach **~15,000
before/after patch pairs spread across all of India**, running concurrently and
resumable from a crash or Ctrl-C rather than from scratch.

**Rebuilt, replacing `sentinel2_monthly.py` + `download_india_change_dataset.py`:**
- `stac_select.py` — STAC search against Earth Search's reprocessed **Collection-1**
  (`sentinel-2-c1-l2a`, one processing baseline across years, so a baseline change
  can't look like ground change), footprint-containment filtering (not just bbox
  overlap), AOI-level cloud measured from the SCL mask via a windowed COG read (not
  tile-level `eo:cloud_cover`), same-tile before/after pairing, day-of-season gap
  control, and near-duplicate rejection between pairs of the same AOI.
- `state_db.py` — SQLite (WAL mode) state: one row per AOI
  (`pending → running → done/no_pairs/failed`), one row per saved pair, an event log,
  and per-run metadata — the resumability the user asked for.
- `aoi_download.py` — per-AOI worker: search → select → align (reuses
  `pair_alignment.load_aligned_pair`) → validate (no-data fraction) → save as **uint8**
  GeoTIFF (was float32) + SCL masks + enriched `pair.json` (scene ids, dates, tile,
  sun angle, processing baseline, per-image AOI cloud). Writes to a `.tmp` folder and
  renames atomically, so a kill mid-pair leaves no partial output.
- `download_batch.py` — concurrent runner (`seed` / `run` / `status` subcommands),
  `ThreadPoolExecutor`-based, `--retry`, `--limit`, `--category`/`--state` filters,
  `--select-only` dry run, a live `status --watch` view with rate/ETA/by-state/
  by-category breakdowns and recent failures.
- `locations.py` + `locations/india_aois.csv` — registry moved from a hardcoded dict to
  a CSV of `key, label, state, district, category, lon, lat, size_km, season_months, ...`,
  bbox derived from centre + size so hundreds of rows don't mean typing coordinates.
  Seeded with the 3 original sites; the full all-India sampler (~2,400 AOIs, stratified
  by state/district/land-cover/terrain/climate) is the next piece, not yet built.
- New tests: `test_stac_select.py`, `test_state_db.py`, `test_aoi_download.py`,
  `test_download_batch.py` — all run offline against synthetic rasters/fake search
  results. 41 tests passing.

**Bugs found and fixed while checking this against the live catalog** (this is why the
Step-0 feasibility probe from the plan mattered):
1. **Wrong tile field.** `s2:tile_id` is a *per-scene* id, not the MGRS tile — using it
   for same-tile pairing matched nothing. Fixed to use `grid:code` (e.g. `MGRS-43QBB`),
   with an `mgrs:*` fallback. Caught by comparing raw STAC responses, then locked in
   with `test_tile_is_grid_code_not_the_per_scene_id`.
2. **`GDAL_CACHEMAX` as a string threw `TypeError`** inside `rasterio.Env` (needs an
   int, unlike the other GDAL env options). Fixed; added
   `test_gdal_options_are_accepted_by_rasterio` so a future option change can't
   reintroduce a silent per-worker crash.
3. **Per-band `reproject()` calls** in `pair_alignment.warp()` fetched each remote COG
   block once per band instead of once total — collapsed to a single multi-band
   `reproject()` call.
4. **Placeholder-count bug** in `state_db.seed()`'s INSERT (15 columns, 16 `?`s) —
   caught immediately by the first live run, fixed, covered by
   `test_seed_is_idempotent_and_keeps_progress`.

**Measured, not assumed:** a live `--select-only` run against the 3 seed AOIs on the
real archive found before/after candidate pairs for all three (previously Navi Mumbai
alone lost 6/12 months). A real download of one AOI, however, exposed that **this
network's throughput to the Earth Search S3 bucket is very slow** — roughly
15–300 KB/s even fanned out across dozens of parallel range requests — which is the
open item below, not a resolved one.

**Code review pass (after the first real pair):** one full real download succeeded
(Navi Mumbai: 519×518 px, 1.06 MB saved, **484 s**). The review then found and fixed:
1. **No STAC pagination.** `search()` read only the first page, so a busy season window
   (tile-overlap zones, 3 satellites in 2025) could silently drop scenes. It now follows
   `next` links; checked live (paging 25 at a time returns the same 88 scenes as one page).
2. **Off-grid resampling blur.** The aligned grid sat a fraction of a pixel off the source
   grid, so bilinear resampling smoothed every pixel of both images. Output is now snapped
   onto the source pixel grid, so same-tile pairs are exact pixel copies.
3. **Non-consecutive `season_months`** (e.g. `1,2,3,11,12`) silently widened to Jan–Dec.
   Now rejected.
4. **`top_k` was global, not per tile**, so one tile's clearer scenes could take every
   cloud-check slot in an overlap zone.
5. **A per-AOI `max_aoi_cloud` of 0 fell back to the 1% default** (`0 or default`).
6. **`--keys` was applied after `--limit`**, so `--keys X --limit 1` could do nothing.
7. **`--select-only` produced no yield estimate** in `status`, which was its whole point.
   It now reports the hit rate and projected patch-pairs.
8. **The Ctrl-C handler was never restored** after a run.
Each fix has a regression test that fails on the old code (48 tests total).

**Why one pair took 484 s:** Sentinel-2 true-colour COGs store 1024×1024-px blocks of
~1.8 MB, and reads fetch whole blocks. The 518-px AOI straddled 4 blocks per image, so the
pair cost **~14 MB fetched for 1 MB saved**. The SCL cloud checks are cheap (~0.05 MB).

## Phase 1b — Block-aligned sites, all-India registry, pilot (27 Sep 2026)

**Decision: one site = one Sentinel-2 block** (option B of three). Because a read fetches
whole 1024 px blocks, a block-aligned 10.24 km site costs ~3.6 MB per pair for 16 patches,
versus ~8–14 MB for 4 patches with free-floating 5 km sites. The full target drops from
~30–54 GB to fetch to ~3.4 GB. Sites now keep 1 pair each; the block is read directly
from both scenes (same tile grid), so the saved pixels are the source pixels.

**Decision: sites spread evenly over every state and UT**, as the user asked (not biased
toward known change). `build_aoi_registry.py` places 1,200 sites over 36 states/UTs
(geoBoundaries ADM1): a floor of 12 per state (fewer only where it can't fit 12 blocks),
the rest by area, farthest-point spread, rows interleaved by state.

**Found by checking the real catalogue before the pilot:**
1. **Collection-1 is sparse before May 2019** (0–4 scenes/month). Seasons are now July–June
   cycles (2019 → Jul 2019–Jun 2020), so Jan–Apr regions don't land in early 2019.
2. **Blocks on a tile's top row / left column were always rejected** (~20% of all blocks).
   Their edge coincides with the footprint edge, and footprint polygons' straight lon/lat
   chords sag up to 178 m inside the true UTM tile edge. The coverage test is now inset
   250 m. Found via a Telangana site with 192 scenes and 0 "covering".
3. **DNS failures under 12 parallel workers** ("Could not resolve host") failed 13 of 36
   sites outright. GDAL's HTTP retries don't cover DNS. Sites now retry transient
   network errors 3 times with backoff.
4. **Zero-pair sites had no explanation** of gap vs snow rejections. Both are now counted
   in the diagnostics (Arunachal: 20 pairs rejected for snow difference, working as
   intended).

Effect, one site per state, live dry run: pairs found for 18/36 → **33/36**. The remaining
3 are genuine: Andaman (cloud), Lakshadweep (no early-2020 imagery), Arunachal (snow).

**Worker tuning (measured on the live pilot, OS network counters, 3-min windows):**

| Workers | KB/s | Pairs/h |
|---|---|---|
| 12 | 227 | — |
| 20 | 399 | 220 |
| 28 | 436 | 180 |
| 36 | 542 | 180 |
| 48 | 592 | 240 |

Throughput kept rising but flattened (36 → 48: +9%), and an earlier raw test got worse at
64 streams. So **48 is the default**, changeable live with `download_batch.py workers N`
(added so the tuning needed no restarts).

**Two more network bugs**, found in the pilot's failures at higher concurrency:
5. **GDAL caches a DNS failure as "file does not exist"**, so the site retry failed again
   without touching the network. The retry now clears GDAL's HTTP cache
   (`VSICurlClearCache`), and that message counts as transient.
6. **A DNS error while saving was treated as a bad candidate**, so a network blip could
   mark a good site `no_pairs` (Sikkim, Puducherry, Rajasthan). Such errors now retry the
   whole site. The 10 failed and 3 wrongly empty sites were re-queued.

## Phase 1c — Pilot review, quality checks, patch sizes (27–28 Sep 2026)

**Review tool:** `review_gallery.py` (browser page, blink/swipe viewer, 1–4 labels, CSV →
DB). It replaced `make_dataset_previews.py`.

**Colour bug found through the user's review:** every display/model stretch worked per
colour channel, which rebalanced colour (green forest turned grey-violet, bare soil pink).
`change_visualization.joint_stretch` now uses one shared range for all channels and both
dates. It also fixes what the captioning model would have been shown.

**Regional audit (28 random pilot pairs, 4 per region):** alignment perfect everywhere, no
visible cloud. Real change present (a new Kerala highway, jhum clearings in Mizoram,
Haryana/AP construction). Problems: winter haze on the plains, snow/glacier whiteout in
the Himalaya, blocks that are mostly sea. NE and Deccan were clean; Himalaya was worst.

**Quality checks added to the downloader:**
- SCL snow > 20% or water > 70% per scene: not used (free, before download).
- Haze (dark-floor gap > 25) or whiteout (> 30% saturated white): swap for the next
  candidate avoiding the bad date.

Sen2Cor's aerosol (AOT) layer was tested as a haze gate and rejected: it didn't separate
hazy from clear scenes. Re-checking the pilot failed 49 of 245 pairs (20%: haze 25, snow 16,
white 15, water 11), and those sites were re-downloaded.

**Patch size:** pixels are fixed at 10 m (Sentinel-2). Sites are stored as full 1024 px
blocks, lossless, so any patch size (256 / 128 / 64 px = 2.56 km / 1.28 km / 640 m) can be
cut later without re-downloading. Samples at every size: `data/samples/delhi/index.html`
(12 Delhi blocks, kept separate from the dataset). Decision pending.

## Phase 1d — Ada cluster haze-check trial (28 Sep 2026)

Full report: `data/trial/REPORT.md`; code in `cluster/`.

- **Ada:** 1 × RTX 2080 Ti per job, 8-bit Qwen3-VL-8B (9.8 GB), ~12 s/pair. `/share1` isn't
  mounted on GPU nodes and has a ~3,000-file quota, so the env is in home and the model on
  node `/ssd_scratch`. Installs must run on a compute node (login node caps processes at
  512 MB). gnode076 is faulty (CUDA init fails); the job now stops at once on a node like that.
- **Prompt bug:** v1's all-zero example answer was copied for 100 of 100 pairs. v2 fixed it,
  with the threshold tuned on a separate 30-pair dev set.
- **Labels:** made from thumbnails, they missed thin haze. They were re-adjudicated at large
  size using the 235B model as a second opinion.
- **Result (100 test pairs, borderline excluded):** Ada 8B 8-bit: 78% accuracy, 70% of haze
  caught, 12% of clean wrongly rejected. Full-precision 8B: 73% / 56% / 8%, so quantization is
  not the limit. 235B: ~92% / 91% / 8%.
- **200-pair trial sorted:** 126 kept, 74 rejected.
- **Recommendation:** use 235B via the API (~$6 for all patches) for the real screen.
- The OpenRouter key was replaced (the old one was revoked).

## Phase 2 — Full filtered patch dataset (28–29 Sep 2026)

Streaming pipeline (`pipeline/`): download → preprocess → Ada haze screen → sort, batch by batch,
state in `data/FINAL/pipeline.sqlite`. Result: **`data/FINAL/`**, with `REPORT.md` for details.

- **27,888 patches:** 15,610 accepted (target 15,000 met), 12,278 rejected with the reason
  recorded.
- **Balanced set:** 8,808 patches (65% changed / 35% unchanged), split by site: train 7,000,
  val 893, test 915, with 0 site leakage.
- **Checks:** 0 grid problems; manifest = folders exactly; 35 of 36 states/UTs (Lakshadweep
  is sea).
- **Rejections:** model haze/cloud 10,994, misaligned 592, water 408, snow 133, no-data 77,
  cloud/shadow 55, whiteout 19.
- **Plains re-downloaded with Feb–Apr:** 193/195 sites got pairs, a smog-free season.
- **Ada screen: 4-bit, batches of 8, reject at haze ≥ 2** (chosen on the dev set). Same accuracy as
  8-bit (77% vs 78%), catches more haze (81% vs 70%), rejects more clean (28% vs 12%), 8× faster
  (1.4 s/pair). 8-bit batching ran out of GPU memory on the 11 GB 2080 Ti.
- **One long-running Ada worker** loads the model once and screens batches as they arrive. It
  replaced per-batch jobs that each waited ~30 min in the queue plus ~15 min of model download.
- **Top-up started early** from a projection (683 sites), so its downloads overlapped screening.
- **Faulty Ada nodes excluded automatically:** gnode076, gnode066, gnode090, gnode077 (CUDA
  unknown error; gnode077 failed mid-job).
- **Survived a machine reboot/power cut.** State was resumable; the Ada worker kept running;
  an `@reboot` auto-resume was used during the run and removed afterwards.
- **Change label:** at least 5% of pixels differing by more than 40 levels after brightness
  matching. The looser 25 flagged 72% of patches (crops, season); 40 flags about 42%.

## Phase 3a — Why the captions failed, and a feasible fix (29 Sep 2026)

Details in `data/RSICC/perception/FINDINGS.md`. About $0.12 of API was spent on the test.

- **Visibility (60 pairs labelled by eye):** land-use change is visible at 10 m when it is
  area-sized (roads, interchanges, town growth, construction, ponds, quarries), but it is rare in
  evenly spread sites: 10%. The pixel "changed" flag is mostly seasonal crop change.
- **Model test (4 classes, 4 models):** the models can see land-use change. The 32B over-calls
  it and the 235B under-calls it. The rule "32B confidence ≥ 0.9 AND 8B agrees" found 11/12
  with 1/48 false. No method reliably flags the leftover haze, but a strict triage keeps haze
  from becoming false change.
- **Focused land-use captions:** about 9/12 correct for both the 32B and 235B, so the 32B is
  chosen on cost.
- **Fix:** redefine the label as land-use change; triage, then caption by class (fixed
  captions for no change); grow land-use pairs by hotspot sampling.

## Phase 3b — Reference labels and what helps detection (30 Sep – 1 Oct 2026)

**Change gate** (`pipeline/change_gate.py`): colour, new-edge, bright-grey, blob and ResNet-50
features with boosted trees, trained on Claude's visual labels. At 90% land-use recall it keeps
62% of pairs (PR-AUC 0.45, against 0.16–0.20 for colour alone); 9,148 of 15,610 pass. **Decision:**
the gate sets the order of work, never a negative label.

**Triage trial** (`pipeline/vlm_triage.py`, 100 pairs, $0.025): the strict rule (32B ≥ 0.9 and 8B
agree and zone check) found only 7 of 15 land-use pairs. The identical-image controls were all "none".
The 8B's stated confidence was 0.95 for every pair, which makes it useless for routing.

**Haze screen, rechecked from the raw trial files:** 4-bit, haze ≥ 2: 35/43 hazy caught,
11/41 clean wrongly rejected (27%, from a small, Delhi-heavy set). `data/FINAL/REPORT.md` is wrong
twice: it says 8-bit and haze ≥ 1.

**Reference labels:**
- Built a labelling web page (`labeler/`, pushed to branch `man1`) and two 150-pair,
  site-disjoint sets (`labeler_set1_dev_150.zip`, `labeler_set2_150.zip`).
- **The human team is no longer available**, so Claude's visual labels are the reference:
  852 pairs as dev, and 200 more planned as a locked test set.
- Independent spot checks found real errors. `madhya_pradesh_006_r1c0` was a missed land-use
  change, and `rajasthan_088_r1c2` had the right label but a caption describing seasonal water.
  Both were corrected.
- One earlier "100% on 60 pairs" check was circular: it compared a file with a copy of itself.
- All accuracy figures below therefore mean *agreement with Claude*, not ground truth.

**Do change-region boxes help the model?** (`pipeline/prompt_bias_test.py`, Qwen3-VL-8B via API
because Ada was unreachable; 50 land-use, 70 seasonal/none, 15 identical-image controls; $0.14)

| Variant | Land-use found | False land-use | False on controls |
|---|---|---|---|
| A plain, no boxes | **31/50** | 7/70 | 0/15 |
| B pixel-difference boxes | 13/50 | 2/70 | 0/15 |
| C pixel boxes + crops | 16/50 | 6/70 | 0/15 |
| D old "change was detected here" prompt | 40/50 | 43/70 | 12/15 |
| E CNN (ResNet-50) boxes | 12/50 | 5/70 | 0/15 |
| F CNN boxes + crops | 18/50 | 3/70 | 0/15 |

- A forcing prompt invents change: 12 of 15 identical pairs.
- Neutral boxes, pixel or CNN, don't invent change, but they **halve detection**. The model
  looks only inside the boxes, and never reported a change outside them even when asked.
- **Decision:** detect on the plain pair with no boxes. Regions may only help to locate or
  caption a change after it has been confirmed.

**Challenges now:**
- Land-use change is rare at evenly spread sites (about 10%).
- Seasonal crop change looks like change.
- Some haze is left in the accepted set.
- The 8B misses about 4 in 10 land-use pairs even on plain images.
- There is no human ground truth.
- Ada was unreachable on 1 Oct.

## Phase 3c — 1,000-pair validation of the API step (1–2 Oct 2026)

Everything is in **`validation_1k/`**; `REPORT.md` there has the full numbers.

**Setup:**
- **Reference labels:** 1,002 Claude visual labels: the 852 earlier ones plus 150 new pairs viewed at 300 px, 6 per sheet.
- **The question** was asked plainly, with no format and no hints: *"Before and after images of the same place, 6 years apart. Has anything lasting changed (like buildings, roads, ponds, cleared land), or are the differences only seasonal?"*
- **Scoring:** a text-only 32B call reads each free answer and marks it land_use, seasonal, none or unclear.
- **Cost:** $1.00 in total ($0.30 for the 32B, $0.70 for the 235B).

**Provisional result** (truth is Claude's labels; 150 land-use pairs, 852 others):

| Rule | Found | Precision | Pairs flagged |
|---|---|---|---|
| 32B | 90% | 25% | 550 |
| 235B | 65% | 40% | 244 |
| 32B and 235B agree | 65% | 43% | 226 |

**Self-audit of the false alarms:** 24 random pairs where both models said change and Claude didn't:
- the models were wrong on 17
- Claude was wrong on 2
- 5 were unsure

Label errors explain only a small part of the low precision.

**Verdict so far: the API step isn't trustworthy enough to scale.** Both models call seasonal crops,
water level and haze "lasting change". The best rule stays far below the 90% precision target,
even allowing for Claude's label errors.

**Human check: done (535/535).** Verified result: 962 pairs scored, 40 "can't tell" left out.
Truth is the human answer on all 535 checked pairs and Claude's label on the 467 unchecked pairs.

| Rule | Found | Precision | False alarms |
|---|---|---|---|
| 32B | 96% | 26% | 47% |
| 235B | 79% | 48% | 15% |
| 32B and 235B agree | 78% | 51% | 13% |

- Claude's labels matched the human on 84% of checked pairs.
- In the random sample of pairs where Claude and both models agreed, the human disagreed on 9 of 45 (20%):
  4 were changes everyone missed, and 5 were changes everyone wrongly called.
- **Final verdict: don't scale the API step.** The best rule is about 50% precise, far below the 90% target.
- The change-map method from a teammate (`changemap_test/`) was also tested and was worse: it called change on all 50 pairs.

**Human check setup:**
- 535 pairs: all 485 disagreements plus 50 random agreements.
- The page is `validation_1k/check/`, with one click per pair.
- `score.py` then rewrites `REPORT.md` with verified numbers.

## Open items

- **Full run** awaits the user's go-ahead (pilot reviewed; quality checks in).
- **Patch size** (256 / 128 / 64 px) to choose; it doesn't affect the download.
- **Himalaya yield** will be low (snow/glacier sites give no pair); top up evenly if the
  projection falls under 15k (approved).
- **Two-stage captioning** (Stage 4) is unit-tested against mocked responses only; a
  fresh live end-to-end check is still open once real Phase 1 imagery exists.
- **Stage 6 batch assembly** (turning saved pairs + captions into final dataset
  records) is designed but not built.
