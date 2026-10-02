# India Sentinel-2 Bi-Temporal Change Dataset — creation pipeline

Code used to build the **India Sentinel-2 Change Pairs** dataset (15,610 quality-screened before/after pairs across all Indian states and union territories) and its **labelled subset** (1,002 pairs), and to run the labelling experiments described in the final report.

Data is not stored here (it is large). It is released on Kaggle:

- 15,610 pairs: https://www.kaggle.com/datasets/kspsvlnsiddardha/india-sentinel2-change-pairs
- Labelled subset (1,002 pairs): https://www.kaggle.com/datasets/kspsvlnsiddardha/india-sentinel2-change-labelled
- Model run reports: https://www.kaggle.com/datasets/kspsvlnsiddardha/rsicc-model-run-reports

## Layout

| Path | What it does |
|---|---|
| `build_aoi_registry.py`, `locations.py`, `locations/india_aois.csv` | Places 1,200 sites evenly over all states/UTs (one 10.24 km Sentinel-2 block each) |
| `stac_select.py` | Scene search and pairing on Sentinel-2 Collection-1 (footprint coverage, block-level cloud/snow/water from the SCL mask, season matching) |
| `aoi_download.py`, `download_batch.py`, `state_db.py` | Concurrent, resumable download of before/after blocks (SQLite state) |
| `pair_alignment.py`, `change_visualization.py` | Grid alignment, display stretch, pixel-change evidence |
| `pipeline/` | Streaming pipeline: patching and screening (`patches.py`, `orchestrator.py`, `topup.py`), dataset assembly (`assemble.py`), change gate (`change_gate.py`) and model-labelling experiments (`vlm_triage.py`, `perception_test.py`, `prompt_bias_test.py`) |
| `cluster/` | Haze/cloud screen with Qwen3-VL-8B on a SLURM GPU cluster |
| `labeler/`, `validation_1k/check/`, `validation_1k/review2/` | Web review tools (one-click blind labelling; two-step review with caption check) |
| `validation_1k/` | 1,000-pair validation of automatic labelling (`run_api.py`, `make_check.py`, `score.py`) |
| `changemap_test/` | Test of the rule-based change-map captioning method |
| `kaggle/` | Packaging scripts and dataset cards for the Kaggle releases |
| `tests/` | Unit tests (offline) |
| `submissions/final/` | Final report (Markdown + PDF) and the scripts that build the report figures |
| `master.md`, `JOURNEY.md` | Design description and full experiment history |

## Rebuild

```bash
pip install -r requirements.txt
python build_aoi_registry.py          # site registry (needs geoBoundaries ADM1 for India in locations/layers/)
python download_batch.py run          # download (resumable)
python -m pipeline.orchestrator run   # patch, screen, sort
python -m pipeline.assemble           # manifest + splits
python kaggle/build_full.py           # package for Kaggle
pytest tests/
```

API-based steps read `OPENROUTER_API_KEY` from a local `.env` file (never committed).

Contains modified Copernicus Sentinel data (2019–2026). Released under CC BY 4.0.
