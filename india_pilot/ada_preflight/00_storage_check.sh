#!/bin/bash
# Stage 0 preflight: storage allocation check.
# Run this directly on an Ada LOGIN node (no sbatch needed — /archive/home
# and /share1 are both visible there; /scratch is checked separately by
# 01_gpu_and_scratch.sbatch since it's node-local and only exists inside a job).
#
# Usage: bash 00_storage_check.sh

set -u
echo "=== /archive/home ==="
echo "-- quota --"
quota -s 2>/dev/null || echo "(quota command not available/not applicable on this account)"
echo "-- du -sh \$HOME --"
du -sh "$HOME" 2>/dev/null
echo

echo "=== /share1/\$USER (root /share1/ confirmed NOT writable -- must use the per-user subdir) ==="
if [ -d "/share1/${USER}" ] || [ -d /share1 ]; then
  echo "/share1 exists and is visible."
  TESTDIR="/share1/${USER}/preflight_test"
  mkdir -p "$TESTDIR" && echo "mkdir under /share1/\$USER: OK ($TESTDIR)" || echo "mkdir under /share1/\$USER: FAILED"
  if [ -d "$TESTDIR" ]; then
    dd if=/dev/urandom of="$TESTDIR/testfile.bin" bs=1M count=50 status=none && echo "wrote 50MB test file: OK" || echo "wrote 50MB test file: FAILED"
    ls -la "$TESTDIR"
    du -sh "$TESTDIR"
    echo "-- cleaning up test file --"
    rm -rf "$TESTDIR" && echo "cleanup: OK" || echo "cleanup: FAILED (remove manually: $TESTDIR)"
  fi
else
  echo "/share1 does NOT exist or is not visible from this node — flag this, quota may need to be provisioned/mounted before Stage 1."
fi
echo

echo "=== Other /share2../share6 visibility (informational only, not part of this account's quota per the spec doc) ==="
for i in 2 3 4 5 6; do
  if [ -d "/share$i" ]; then
    echo "/share$i: visible"
  else
    echo "/share$i: not visible from this node"
  fi
done
