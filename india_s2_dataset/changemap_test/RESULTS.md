# Change-map test (50 pairs)

| Method | Land-use found (of 25) | False change on seasonal/none (of 25) |
|---|---|---|
| Change map alone: any building/road appeared or vanished >= 0.3% | 25 | 25 |
| Change map alone: any change at all (draft caption not 'no change') | 25 | 25 |
| Map + prompt, Qwen3-VL-32B (API) | 25 | 25 |
| Reference: plain question, 32B, no map (validation_1k) | 24 | 11 |

Changed area marked by the map: median 82% of the image, max 98%. Truth = Claude's labels (provisional).

## Verdict

The change map does not help at 10 m. It marks most of every image as changed (median 82%), and
calls dry fields, desert and crops "new buildings" or "roads". The prompt tells the model to treat
the map as fact, so the 32B describes construction on every no-change pair (25/25 false), e.g.
*"New buildings with rectangular shapes and light-colored roofs are constructed on former vegetation"*
for an unchanged Rajasthan desert patch. Same failure as the forced-region prompt tested on 1 Oct
(variant D, 12/15 identical pairs reported as changed). The plain question without a map is far better (11/25).

## Ada (8B) half

Not run: Ada (ada.iiit.ac.in:22) timed out from this machine all day. Ready to go, independent of the API:
`bash changemap_test/upload_to_ada.sh` (upload + sbatch), later `bash changemap_test/upload_to_ada.sh fetch`
and `python changemap_test/run_test.py score`.
