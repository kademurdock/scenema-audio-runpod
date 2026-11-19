"""Build step (RVC venv, CPU only), after selftest.py: prove the training chain before the image is pushed, because nobody can
try training on a GPU for free.

The worker's own voice_train.run() makes a voice from 24 seconds of a synthetic sung vowel, through the real RVC scripts and the
real pretrained weights, exactly as a train job does except for the device and the size:
  listen and measure -> train/preprocess.py -> train/dataset/extract_f0.py (RMVPE, in its CPU form) ->
  train/dataset/extract_hubert_feature.py (v2) -> the file list and config -> train/train.py for one round at batch 1 from
  f0G48k / f0D48k -> train/train_index.py -> the new model and its index sing four seconds through infer/cli.py
Then train.py runs once more with its data loader untouched (the first run holds it to no workers when /dev/shm is as small as
a build container's: voice_train.loader_plan), so both ways of loading are proven. "rounds_counted" is the worker's round
counter read from train.py's real end-of-round line: 1 after the one round.

What a CPU build cannot prove is written into the result too ("unproven"): the CUDA forms of the same scripts.
The result is merged into /opt/voice/selftest.json as "train". This step never fails the build, because song and vocal jobs do
not need it: when "ok" is false the handler refuses train jobs with a plain sentence instead (voice_handler.train)."""
import json
import os
import sys
import tempfile
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import voice_pipeline  # noqa: E402
import voice_train  # noqa: E402
import vpaudio as va  # noqa: E402

OUT = os.environ.get("VOICE_SELFTEST", "/opt/voice/selftest.json")
SR = voice_pipeline.SR
UNPROVEN = [
    "train/train.py on a CUDA card: the one-process path, half precision, batch 8 (the CPU run goes through the same run() "
    "in a child process, in full precision, at batch 1)",
    "train/dataset/extract_f0.py in its cuda form (RMVPE on the GPU, half precision); its CPU form is proven and is what the "
    "worker falls back to",
    "train/dataset/extract_hubert_feature.py on a CUDA card in half precision",
    "how long a real run takes, how much GPU memory it uses, and how good the voice is",
]


def synthetic_voice(seconds=24.0, sr=SR):
    """Four sung phrases of an "ah" (a harmonic source with a falling tilt under three formants, a slow glide and vibrato, a
    little breath noise) with short rests between them, so the slicer, RMVPE and HuBERT all get something voice-like."""
    rng = np.random.default_rng(2)
    t = np.arange(int(seconds * sr)) / sr
    f0 = 196 * 2 ** ((4 * np.sin(2 * np.pi * 0.11 * t) + 0.3 * np.sin(2 * np.pi * 5.5 * t)) / 12)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    x = np.zeros(len(t))
    for k in range(1, 40):
        f = k * f0
        x += sum(1 / np.sqrt(1 + ((f - fc) / (bw / 2)) ** 2) for fc, bw in ((730, 90), (1090, 110), (2440, 170))) / k * np.sin(k * phase)
    phrase = ((t % 6.0) < 5.4).astype(np.float64)
    edge = np.convolve(phrase, np.hanning(int(0.08 * sr)) / np.hanning(int(0.08 * sr)).sum(), mode="same")
    x = (x / np.abs(x).max() + 0.01 * rng.standard_normal(len(t))) * edge
    return (0.6 * x / np.abs(x).max()).astype(np.float32)


def main():
    result = {"ok": False, "steps": {}, "seconds": {}, "unproven": UNPROVEN}
    began = time.monotonic()
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ, VOICE_DEVICE="cpu", VOICE_RVC_PYTHON=sys.executable)
        log = os.path.join(td, "tools.log")
        runner = voice_pipeline.Runner(log, env=env)
        trainer = voice_train.TrainRunner(log, os.path.join(td, "rvc"), env=env)
        sample = os.path.join(td, "sample_000.wav")
        va.encode(sample, synthetic_voice(), SR, bits=32)
        work = os.path.join(td, "work")
        os.makedirs(work)
        req = {"mode": "train", "voice_id": "selftest", "options": {"epochs": 1, "isolate": False, "denoise": 0.0, "dereverb": False}}
        trace, said = {}, []
        try:
            out = voice_train.run(req, [sample], work, runner, trainer, progress=said.append, trace=trace, min_seconds=10, batch=1)
            report = out["report"]
            result["steps"] = dict(trace)
            result["seconds"] = report["timing"]
            result.update({"range": report["range"], "slices": report["slices"], "loader": report["loader"],
                           "shm_free_gb": report["shm_free_gb"], "model_bytes": report["model_bytes"],
                           "index_bytes": report["index_bytes"], "notes": report["notes"]})
            result["ok"] = all(bool(trace.get(k)) for k in ("preprocess", "f0", "features", "filelist", "train", "index", "check"))
        except Exception as error:  # the proof failed: say where, keep the build going
            result["steps"] = dict(trace)
            result["error"] = f"{type(error).__name__}: {error}"
            traceback.print_exc()
            print(voice_pipeline.log_tail(log, 6000), flush=True)
        print("train self-test said:", said, flush=True)
        if trace.get("train"):
            # the round counter that feeds "Learning the voice, epoch 40 of 200" and the pace check, on train.py's real line
            result["rounds_counted"] = trainer.epochs_done()
            # once more with train.py's own loader, whatever the first run used
            trainer.loader = "upstream"
            ok, secs, _ = trainer.train(1, 1, 2700)
            result["upstream_loader"], result["seconds"]["train_upstream_s"] = bool(ok), secs
            if not ok:
                print(voice_pipeline.log_tail(log, 6000), flush=True)
        try:
            with open(log, encoding="utf8", errors="replace") as f:
                lines = [line for line in f.read().splitlines() if voice_train.EPOCH_LINE.search(line)]
            print("train.py end-of-round lines:", lines[-3:], flush=True)
        except OSError:
            pass
    result["seconds"]["all_s"] = round(time.monotonic() - began, 1)
    merged = va.read_json(OUT, {}) or {}
    merged["train"] = result
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    va.write_json(OUT, merged)
    print(json.dumps(result), flush=True)
    print("TRAIN SELFTEST OK" if result["ok"] else "TRAIN SELFTEST FAILED (train jobs will be refused by this image)", flush=True)


if __name__ == "__main__":
    main()
