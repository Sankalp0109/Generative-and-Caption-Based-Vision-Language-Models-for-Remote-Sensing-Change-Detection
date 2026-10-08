#!/bin/bash
# Event-driven waiter for the pipeline: exits on a newly sorted batch, any failure, or after 60 min.
cd "$(dirname "$0")/.."
PY=~/miniforge3/envs/rscc/bin/python
LOG=data/FINAL/logs/orchestrator.log
q() { $PY -c "import sqlite3;c=sqlite3.connect('data/FINAL/pipeline.sqlite');print(c.execute(\"select count(*) from batch where stage='$1'\").fetchone()[0])"; }
s0=$(q sorted); f0=$(q failed); e0=$(grep -c "error in loop" $LOG)
for i in $(seq 1 60); do
  sleep 60
  ps -eo cmd | grep -q "[p]ipeline.orchestrator run" || { reason="orchestrator stopped"; break; }
  grep -q "DONE: final dataset assembled" $LOG && { reason="pipeline DONE"; break; }
  [ "$(q failed)" -gt "$f0" ] && { reason="batch FAILED"; break; }
  [ "$(grep -c 'error in loop' $LOG)" -gt "$e0" ] && { reason="loop error"; break; }
  [ "$(q sorted)" -gt "$s0" ] && { reason="batch sorted"; break; }
done
echo "WAKE: ${reason:-60 min elapsed}"
$PY -m pipeline.orchestrator status | head -1
tail -3 $LOG | cut -c1-170
