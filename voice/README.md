# Voice worker ("Sing it in my voice")

A RunPod serverless worker that re-sings a recording with a trained RVC v2 voice, and since Nov 2026 also makes such a voice
from uploaded recordings (`mode: "train"`). Its own image (`Dockerfile.voice`), its own workflow
(`.github/workflows/voice.yml`, branch `voice-rvc` only), its own endpoint. The YuE2 and AuK images are not touched.

## Endpoint settings

- Image: `ghcr.io/kademurdock/scenema-audio-runpod:voice-rvc-<commit sha>` (pin the sha tag; `voice-rvc-latest` only feeds the
  build cache).
- GPUs: cards from before the RTX 50 series (RTX 4090, 3090, A5000, L4, L40S, A40, RTX A6000). torch 2.7.1+cu118 has no Blackwell
  kernels, and the worker refuses such a card with a plain error. 24 GB is plenty; a 3-minute song peaks well under 10 GB, and
  training runs at batch 8 on 24 GB (smaller cards get a smaller batch).
- Container disk 20 GB or more for songs (the image is about 13 GB); 30 GB if train jobs will use `isolate`, `dereverb` or
  `denoise` on close to an hour of recordings. A network volume is optional: when one is mounted at `/runpod-volume`, voice
  models are cached there across workers.
- Environment: `AWS_ENDPOINT_URL`, `AWS_REGION`, `AWS_BUCKET_NAME`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (the private
  bucket, the same values the music worker uses) and `VOICE_AUDIO_HOSTS` (the bucket's host, for signed `audio_url` links).
  Optional: `VOICE_AUDIO_PREFIXES` (default `yue2/,voice/,audios/`), `VOICE_CACHE_FILES` (default 8), `VOICE_TRAIN_SECONDS`
  (default 6600: how long a train job gives itself in all), `VOICE_TRAIN_BATCH` (overrides the batch size).
- Execution timeout: 15 minutes is ample for a song (roughly 1.5 to 3 minutes of GPU time for a 3-minute song; a 10-minute song
  with every option on stays well inside it). **A train job needs 2 hours**: the caller sends that timeout with the job
  (`policy.executionTimeout`), and the worker plans to be done in 110 minutes.

## Request: a song or a vocal

```json
{"input": {
  "mode": "song",
  "audio_key": "yue2/<id>/master.wav",
  "model_key": "voice-models/<owner>/<random>/model.pth", "model_sha256": "<hex>",
  "index_key": "voice-models/<owner>/<random>/model.index", "index_sha256": "<hex>",
  "pitch": "auto",
  "transpose_song": false,
  "voice_range": {"p05": 55.0, "p50": 63.0, "p95": 70.0},
  "options": {"extractor": "hyperace", "fallback": "bs_roformer", "lead_split": true, "lead_model": "frazer",
              "dereverb": false, "room": true, "soft_s": true,
              "index_rate": 0.5, "protect": 0.33, "rms_mix_rate": 0.25, "f0_method": "rmvpe",
              "dereverb_strength": 1, "denoise": 0, "backing": "keep", "formant": 0},
  "vocal_fx": "none"
}}
```

`mode` is `song` (split, re-sing the lead, put it back) or `vocal` (a dry vocal, converted as it is). Send `audio_key` or a signed
`audio_url`. A recording may be up to ten minutes long. Every option is optional; the defaults above are the round 2 chain
(voice-persona RUNBOOK section 17, measured with Whisper against the lyrics): HyperACE v2 vocals, the frazer & becruily lead
split, no dereverb, and `soft_s` (the input's own S hiss above about 4 kHz in its unvoiced frames, so RVC's rebuilt S sounds lose
their robotic edge). Round 1's chain is
`{"extractor": "bs_roformer", "fallback": "demucs", "lead_model": "aufr33", "dereverb": true, "soft_s": false}`.
Extractors: `hyperace`, `bs_roformer`, `melband_kim`, `demucs` (baked) and `melband_becruily` (downloaded once per worker from the
pinned model commit). Lead models: `frazer` (names its lead stem) and `aufr33` (the lead is the louder, steadier-pitched part), both
baked. `room` only matters with `dereverb` on: the room taken off is put back after. The HyperACE v2 and frazer & becruily weights state no licence
(community weights, fine for this private worker; RUNBOOK section 17); Kim's Mel-band RoFormer is MIT if that ever matters.

### Pitch and key

`pitch` is one of:

- `"auto"`: keep the melody; move it one octave only when the song's middle note is outside `voice_range`.
- a whole number of semitones from -24 to 24.
- `"fit"`: the whole-semitone move that leaves the fewest of the melody's notes outside `voice_range` (its p05 to p95), counted
  to the nearest percent; ties go to the smallest move, so a melody that already fits is not moved. Without a `voice_range` it
  behaves as `auto` and `pitch.why` says so.

`transpose_song` (off by default, always on with `"fit"`): in song mode, when the voice moved by anything but whole octaves, the
music and the kept backing vocals are moved to the same key with ffmpeg's rubberband filter (formants kept for the backing
vocals). The octaves are dropped first, so the band never moves more than six semitones: a voice moved 14 semitones gets a band
moved 2. With it off, a move that is not whole octaves adds a worker note that the voice may be out of key with the music. The
answer's `pitch.transposed_song` says whether the music moved and `pitch.song_shift` by how many semitones.

### The Nov 2026 options

All four are off unless set; with none of them set every audio file is byte for byte what the worker made before they existed.

- `dereverb_strength` (0 to 1, default 1): only with `dereverb` on. How much of the room comes off the lead before it is re-sung:
  the lead that goes to RVC is `lead*(1-s) + dry*s`, and the room that is put back shrinks with it. At 0 the dereverb model is
  not run.
- `denoise` (0 to 1, default 0): hiss and background noise taken off the voice before it is re-sung, by aufr33's Mel-band
  RoFormer denoise model (baked, from the same pinned catalog as the separators), blended in by the amount given. Works in
  `vocal` mode too.
- `backing`: `"keep"` (the default: backing vocals left as sung), `"convert"` (the backing vocals get their own RVC pass at the
  same pitch shift, as loud as the backing they replace, then join the mix) or `"remove"` (left out of the mix). It needs the
  lead split to have found backing vocals; otherwise a worker note says why it had no effect.
- `formant` (-6 to 6 semitones, default 0): the converted voice's timbre with its pitch left alone. Negative is bigger and
  darker, positive smaller and brighter. Pure numpy (`vpaudio.formant_shift`), deterministic. With `backing: "convert"` the same
  move goes on the converted backing vocals.

`f0_method` is `rmvpe` (the default) or `pm`.

### Vocal effect

`vocal_fx` (optional) puts a studio effect on the converted lead before the remix (`vocalfx.py`): `none` (the default; every audio
file is then byte for byte what the worker made before effects existed), `studio` (gentle EQ, compression, de-essing and a short
quiet plate), `plate` (a lush 2 s plate), `hall` (a 2.8 s hall), `slapback` (one short repeat), `echo` (a dotted-eighth ping-pong
echo at the song's detected tempo, darker with each repeat and ducked while the voice sings) or `dreamy` (a slight chorus, a
quarter-note echo and a 3.8 s hall). Every preset starts from the `studio` polish. The effected lead is matched to the dry lead's
loudness, so it sits in the band exactly as loud; it replaces the put-back room (`dereverb` + `room`) rather than doubling it. The
dry vocal files are never touched: the effected voice is its own pair, `vocal_fx.mp3` and `vocal_fx.wav` (24-bit stereo, with the
effect's tail); in `vocal` mode it is also the result (`url`). Pure numpy + ffmpeg: no new wheel in the image, nothing copyleft,
and the impulse responses come from fixed seeds, so the same song always gives the same file.

## Response: a song or a vocal

`url`/`key` (the song, MP3), `wav_url`/`wav_key` (24-bit WAV), `vocal_url`/`vocal_key` (the dry converted voice, MP3),
`vocal_wav_url`/`vocal_wav_key`, `report_key`, `duration_s`, `gpu` (card name, for pricing), `pitch`, `separation`, `settings`,
`worker_notes`, `timing`, `processing_ms`, `features`, `vocal_fx` (null, or the preset, its label, the tempo and delay it used, the
reverb length and the level match) and, with an effect, `vocal_fx_url`/`vocal_fx_key` and `vocal_fx_wav_url`/`vocal_fx_wav_key`.

In `song` mode the stems always come too, each as an MP3 and a 24-bit WAV with the mix's own master gain, so they sit at the
level they have in the song:

- `instrumental_url`/`_key`, `instrumental_wav_url`/`_key`: the band as extracted, after any transposition.
- `lead_original_url`/`_key`, `lead_original_wav_url`/`_key`: the original singer's lead exactly as it went into RVC (after any
  dereverb and denoise), mono.
- `backing_url`/`_key`, `backing_wav_url`/`_key`: only when the lead split found backing vocals. With `keep` they are the backing
  vocals as they are in the mix; with `convert` the re-sung ones (mono); with `remove` the backing vocals as they were sung,
  which the mix leaves out. When the song was transposed they are in the new key.

A field is present only when its file exists; `vocal` mode has no stems. `settings` now also carries the four Nov 2026 options,
and `pitch` carries `transposed_song` and `song_shift`. Links are signed for seven days. A refusal or failure is
`{"error": sentence}`; nothing is retried automatically.

## Train a voice (`mode: "train"`)

```json
{"input": {
  "mode": "train",
  "owner": "<1-64 of A-Z a-z 0-9 _ ->",
  "voice_id": "<8-64 of a-z 0-9 _ ->",
  "sample_keys": ["voice-samples/<owner>/<voice_id>/<file>", "..."],
  "options": {"epochs": "auto", "isolate": false, "denoise": 0, "dereverb": false}
}}
```

Makes an RVC v2 48 kHz voice with pitch from 1 to 60 recordings (256 MB each, 2 GB in all). Every key must be under exactly
`voice-samples/<owner>/<voice_id>/` (letters, digits, `.`, `_`, `=`, `-` and `/` after it), or the job is refused before
anything is read. The job writes three files and nothing else: `voice-models/<owner>/<voice_id>/model.pth`, `model.index` and
`train_report.json`. The recordings, the slices cut from them and every checkpoint stay in the job's temp folder and are deleted
with it.

- `isolate` (default off: the recordings are a voice on its own): on runs the vocal extractor (and its fallback) and the lead
  split on every recording first, for recordings with music under the voice.
- `dereverb`, `denoise` (0 to 1): the same models as in a song job, on every recording.
- `epochs`: a whole number from 20 to 600, or `"auto"`, by minutes of voice, so a run stays roughly 15 to 40 minutes on a 24 GB
  card: under 5 minutes 300 rounds, 5 to under 15 minutes 200, 15 to under 30 minutes 120, 30 to under 45 minutes 80, 45 minutes
  and over 60. Rounds that cannot finish inside the job are trimmed before training starts (a worker note says so when the
  number was asked for). The plan rests on an estimate of the card's pace, so the first three rounds are timed: when they show
  the plan cannot finish, training is started once more with the rounds that do fit (a worker note says how many), or, when
  fewer than 20 would fit, the job ends there with a plain error instead of running out the clock.

The steps are the ones RVC's own WebUI runs at the pinned commit (`voice_train.py` has the details): listen (and isolate,
dereverb, denoise) -> measure the seconds that are not silence -> `train/preprocess.py` (48 kHz) -> `extract_f0.py` (RMVPE on
the GPU) -> `extract_hubert_feature.py` (v2) -> the file list and config as `run_train_model` writes them -> `train/train.py`
from the pretrained `f0G48k.pth` / `f0D48k.pth` -> `train/train_index.py` -> the new voice sings a four-second test line.
Under a minute of voice is refused ("That is under a minute of clear voice. Add more recordings: ten minutes or more makes a
much better voice."); over an hour, only the first hour is used.

Response: `engine` (`rvc`), `mode` (`train`), `model_key`, `model_sha256`, `index_key`, `index_sha256`, `report_key`, `range`
(`{"p05", "p50", "p95"}`: the voice's low, middle and high notes as MIDI numbers, ready to send back as `voice_range`),
`minutes` (of voice used), `epochs` (rounds really trained), `sample_count` (recordings used), `gpu`, `timing`, `worker_notes`,
`processing_ms`, `features`. Or `{"error": sentence}`. Progress sentences while it runs: "Fetching the recordings", "Listening
to the recordings" (then, when asked for, "Taking the music off the recordings", "Taking the room off the recordings",
"Cleaning the noise off the recordings"), "Getting the recordings ready", "Learning the voice", "Learning the voice, epoch 40
of 200" (every 30 seconds at most), "Building the voice's index", "Saving the voice".

Safety nets a first GPU job may show in `worker_notes`: RMVPE in its GPU form making nothing falls back to its CPU form;
HuBERT in half precision making nothing is run again in full precision; a `/dev/shm` too small for `train.py`'s four loader
workers (under about 1.8 GB free at batch 8) switches the workers off (`train_site/sitecustomize.py`), which `train_report.json`
records as `loader: 0` beside `shm_free_gb`.

## Tests

`cd voice && python -m unittest -v test_voice.py` (numpy + ffmpeg; stand-in separators, RVC and training scripts). The image
build also runs `selftest.py` (every baked separator and RVC's own inference with a tiny random model, on the CPU) and
`selftest_train.py` (the worker's own train flow through the real RVC scripts and pretrained weights: one round on 24 seconds
of synthetic singing, then the new model and its index sing through `infer/cli.py`), before anything is pushed. What they
proved is in `/opt/voice/selftest.json`; an image whose training proof failed refuses train jobs in plain words. A CPU build
cannot prove the CUDA forms of the training scripts (`train.py` in half precision at batch 8, RMVPE and HuBERT on the GPU):
those are first proven by a real train job.
