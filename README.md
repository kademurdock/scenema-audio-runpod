# scenema-audio-runpod

Kade-AI's RunPod Serverless worker for [Scenema Audio](https://github.com/ScenemaAI/scenema-audio)
(upstream pinned at `3e3d403a`, MIT code; model weights under the LTX-2 Community License;
Gemma 3 12B under Google's Gemma terms, pulled from the ungated `unsloth/gemma-3-12b-it` mirror).

- Queue-based worker (`handler.py`): Scenema `<speak>` XML in, MP3 on Backblaze B2 out (URL + duration).
- No weights in the image; they download once onto the endpoint's network volume (`MODEL_DIR`).
- Full precision by default (bf16 audio transformer + bf16 Gemma) — needs a 48 GB card.
- Built by GitHub Actions to `ghcr.io/kademurdock/scenema-audio-runpod:latest`.

Plan and design: `SCENEMA_SERVERLESS_PLAN_2026-09-03.md` in the Kade-AI project folder (Part 119).

## Chatterbox worker

`chatterbox_handler.py` / `Dockerfile.chatterbox` serve Resemble AI Chatterbox (MIT; code pinned at
`ab55923c`) as image `ghcr.io/kademurdock/scenema-audio-runpod:chatterbox-<sha>`: `model` picks
`turbo` (ResembleAI/chatterbox-turbo@749d1c1a), `original` or `multilingual` (V3; both
ResembleAI/chatterbox@5bb1f6ee). Models load on first use and stay resident (all three fit a 24 GB card).
`mode` is `design` (built-in voice) or `clone` (`reference_url`). Every take carries upstream's Perth watermark.
