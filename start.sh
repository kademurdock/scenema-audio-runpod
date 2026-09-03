#!/usr/bin/env bash
# SeedVC checkpoints (1.6 GB) live on the volume, not the container disk, so a cold
# start does not re-download them. Upstream hardcodes SEEDVC_PATH/checkpoints.
set -e
VOL="${MODEL_DIR:-/runpod-volume/scenema}"
mkdir -p "$VOL/seedvc-checkpoints" "$VOL/hf_cache"
rm -rf /app/seed-vc/checkpoints
ln -s "$VOL/seedvc-checkpoints" /app/seed-vc/checkpoints
exec python3 -u /app/handler.py
