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
  if [ -n "$VRAM_MIB" ] && [ "$VRAM_MIB" -ge 30000 ] && [ "$VRAM_MIB" -lt 40000 ]; then
    # Part 127 (Sep 4 2026): a 32 GB card (RTX 5090) holds Gemma in 8-bit (~13 GB)
    # beside the INT8 audio model, all resident — closer to bf16 quality at NF4
    # speed. patch_gemma_int8.py adds the mode; it falls back to NF4 if the load
    # fails. 48 GB cards are Low stock in every datacentre with a volume.
    export GEMMA_QUANTIZE=int8
    export AUDIO_CKPT="$VOL/scenema-audio-transformer-int8.safetensors"
    echo "[start] ${VRAM_MIB} MiB VRAM: INT8 audio + INT8 Gemma (32 GB config)"
  elif [ -n "$VRAM_MIB" ] && [ "$VRAM_MIB" -lt 40000 ]; then
    # README's 24 GB row: INT8 audio (identical quality per upstream) + NF4 Gemma,
    # everything resident. bf16 audio + NF4 Gemma OOM'd a 4090 at 23.5 GB (05:05Z).
    export GEMMA_QUANTIZE=nf4
    export AUDIO_CKPT="$VOL/scenema-audio-transformer-int8.safetensors"
    echo "[start] ${VRAM_MIB} MiB VRAM: INT8 audio + NF4 Gemma (24 GB config)"
  else
    export GEMMA_QUANTIZE=
    echo "[start] ${VRAM_MIB:-?} MiB VRAM: everything bf16"
  fi
fi
exec python3 -u /app/handler.py
