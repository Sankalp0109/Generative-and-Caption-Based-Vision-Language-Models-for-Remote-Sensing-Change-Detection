#!/bin/bash
# Copy the change-map test to Ada, start the job, and later fetch the results.
#   bash changemap_test/upload_to_ada.sh          # upload + sbatch
#   bash changemap_test/upload_to_ada.sh fetch    # copy results_ada_8b.jsonl back, then: python changemap_test/run_test.py score
set -euo pipefail
cd "$(dirname "$0")/.."
SSH="ssh -i $HOME/.ssh/ada_ed25519 -o BatchMode=yes"
REMOTE=sankalp0109@ada.iiit.ac.in
if [ "${1:-}" = fetch ]; then
  rsync -e "$SSH" "$REMOTE:changemap/results_ada_8b.jsonl" changemap_test/ && exit 0
fi
STAGE=$(mktemp -d)
cp changemap_test/{ada_check.py,ada_check.sbatch,pairs.json} "$STAGE"/
cp -r changemap_test/maps "$STAGE"/
python3 - "$STAGE" <<'PY'
import json, shutil, sys
from pathlib import Path
stage = Path(sys.argv[1])
for p in json.loads((stage / "pairs.json").read_text()):
    d = stage / "pairs" / p["patch_id"]; d.mkdir(parents=True)
    for n in ("before", "after"):
        shutil.copy(Path("data/FINAL") / p["folder"] / f"{n}.png", d / f"{n}.png")
PY
rsync -a -e "$SSH" "$STAGE"/ "$REMOTE:changemap/"
$SSH "$REMOTE" 'cd ~/changemap && sbatch ada_check.sbatch'
rm -rf "$STAGE"
