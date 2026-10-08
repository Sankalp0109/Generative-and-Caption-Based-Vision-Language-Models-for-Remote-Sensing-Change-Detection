#!/bin/bash
# Relaunch the pipeline after a reboot/power cut, unless it is already running or already finished.
cd "$(dirname "$0")/.."
sleep 60   # let the network come up
grep -q "DONE: final dataset assembled" data/FINAL/logs/orchestrator.log 2>/dev/null && exit 0
pgrep -f "pipeline.orchestrator run" >/dev/null && exit 0
setsid nohup ~/miniforge3/envs/rscc/bin/python -m pipeline.orchestrator run >> data/FINAL/logs/orchestrator.out 2>&1 < /dev/null &
