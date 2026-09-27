# Voice worker ("Sing it in my voice")

A RunPod serverless worker that re-sings a recording with a trained RVC v2 voice. Its own image (`Dockerfile.voice`), its own
workflow (`.github/workflows/voice.yml`, branch `voice-rvc` only), its own endpoint. The YuE2 and AuK images are not touched.

## Endpoint settings

- Image: `ghcr.io/kademurdock/scenema-audio-runpod:voice-rvc-<commit sha>` (pin the sha tag; `voice-rvc-latest` only feeds the
  build cache).
- GPUs: cards from before the RTX 50 series (RTX 4090, 3090, A5000, L4, L40S, A40, RTX A6000). torch 2.7.1+cu118 has no Blackwell
  kernels, and the worker refuses such a card with a plain error. 24 GB is plenty; a 3-minute song peaks well under 10 GB.
- Container disk 20 GB or more (the image is about 12 GB). A network volume is optional: when one is mounted at `/runpod-volume`,
  voice models are cached there across workers.
- Environment: `AWS_ENDPOINT_URL`, `AWS_REGION`, `AWS_BUCKET_NAME`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (the private
  bucket, the same values the music worker uses) and `VOICE_AUDIO_HOSTS` (the bucket's host, for signed `audio_url` links).
  Optional: `VOICE_AUDIO_PREFIXES` (default `yue2/,voice/,audios/`), `VOICE_CACHE_FILES` (default 8).
- Execution timeout 15 minutes is ample; a job is roughly 1.5 to 3 minutes of GPU time for a 3-minute song.

## Request

```json
{"input": {
  "mode": "song",
  "audio_key": "yue2/<id>/master.wav",
  "model_key": "voice-models/<owner>/<random>/model.pth", "model_sha256": "<hex>",
  "index_key": "voice-models/<owner>/<random>/model.index", "index_sha256": "<hex>",
  "pitch": "auto",
  "voice_range": {"p05": 55.0, "p50": 63.0, "p95": 70.0},
  "options": {"extractor": "hyperace", "fallback": "bs_roformer", "lead_split": true, "lead_model": "frazer",
              "dereverb": false, "room": true, "soft_s": true,
              "index_rate": 0.5, "protect": 0.33, "rms_mix_rate": 0.25, "f0_method": "rmvpe"}
}}
```

`mode` is `song` (split, re-sing the lead, put it back) or `vocal` (a dry vocal, converted as it is). Send `audio_key` or a signed
`audio_url`. `pitch` is `auto` (keep the melody; move one octave only when the song's middle note is outside `voice_range`) or whole
semitones. Every option is optional; the defaults above are the round 2 chain (voice-persona RUNBOOK section 17, measured with
Whisper against the lyrics): HyperACE v2 vocals, the frazer & becruily lead split, no dereverb, and `soft_s` (the input's own S
hiss above about 4 kHz in its unvoiced frames, so RVC's rebuilt S sounds lose their robotic edge). Round 1's chain is
`{"extractor": "bs_roformer", "fallback": "demucs", "lead_model": "aufr33", "dereverb": true, "soft_s": false}`.
Extractors: `hyperace`, `bs_roformer`, `melband_kim`, `demucs` (baked) and `melband_becruily` (downloaded once per worker from the
pinned model commit). Lead models: `frazer` (names its lead stem) and `aufr33` (the lead is the louder, steadier-pitched part), both
baked. `room` only matters with `dereverb` on: the room taken off is put back after. The HyperACE v2 and frazer & becruily weights state no licence
(community weights, fine for this private worker; RUNBOOK section 17); Kim's Mel-band RoFormer is MIT if that ever matters.

## Response

`url`/`key` (the song, MP3), `wav_url`/`wav_key` (24-bit WAV), `vocal_url`/`vocal_key` (the dry converted voice, MP3),
`vocal_wav_url`/`vocal_wav_key`, `report_key`, `duration_s`, `gpu` (card name, for pricing), `pitch`, `separation`, `settings`,
`worker_notes`, `timing`, `processing_ms`, `features`. Links are signed for seven days. A refusal or failure is `{"error": sentence}`;
nothing is retried automatically.

## Tests

`cd voice && python -m unittest -v test_voice.py` (numpy + ffmpeg; stand-in separators and RVC). The image build also runs
`selftest.py`: every baked separator and RVC's own inference (with a tiny random model) on the CPU, before anything is pushed.
