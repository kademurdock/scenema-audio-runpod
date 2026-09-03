#!/usr/bin/env bash
# SeedVC checkpoints (1.6 GB) live on the volume, not the container disk, so a cold
# start does not re-download them. Upstream hardcodes SEEDVC_PATH/checkpoints.
set -e
VOL="${MODEL_DIR:-/runpod-volume/scenema}"
mkdir -p "$VOL/seedvc-checkpoints" "$VOL/hf_cache"
rm -rf /app/seed-vc/checkpoints
ln -s "$VOL/seedvc-checkpoints" /app/seed-vc/checkpoints

# Precision follows the card (Part 119.9, measured live): bf16 Gemma needs ~24 GB
# on its own, so on a 24 GB card it streams from CPU RAM and a 31-second clip took
# 188 s (0.17x real-time). On <40 GB, quantise ONLY the text encoder (NF4) and keep
# the audio transformer — the part that makes the sound — at bf16, all resident.
# A 48 GB card runs everything bf16. GEMMA_QUANTIZE set explicitly wins.
if [ -z "${GEMMA_QUANTIZE_EXPLICIT:-}" ]; then
  VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  if [ -n "$VRAM_MIB" ] && [ "$VRAM_MIB" -lt 40000 ]; then
    export GEMMA_QUANTIZE=nf4
    echo "[start] ${VRAM_MIB} MiB VRAM: Gemma NF4, audio transformer bf16"
  else
    export GEMMA_QUANTIZE=
    echo "[start] ${VRAM_MIB:-?} MiB VRAM: everything bf16"
  fi
fi
exec python3 -u /app/handler.py
