"""Build step for engine "seedvc" (Seed-VC venv, CPU only, run from the pinned Seed-VC folder): fetch every weight file the 44 kHz
singing path loads (the DiT checkpoint and its config from Plachta/Seed-VC, RMVPE, CAM++, BigVGAN v2 44 kHz, Whisper small) into
the folder's own ./checkpoints cache, then prove the whole path runs: seedvc_run.py re-sings four seconds of a synthetic voice in
the "voice" of four seconds of another synthetic voice, at four diffusion steps, on the CPU, offline. Exits 1 when anything fails,
so an image whose talking-voice engine cannot run is never pushed."""
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 44100


def voice(seconds, f0, seed):
    t = np.arange(int(seconds * SR)) / SR
    phase = 2 * np.pi * np.cumsum(f0 * 2 ** (np.sin(2 * np.pi * 0.4 * t) * 3 / 12)) / SR
    x = sum(np.sin(k * phase) / k for k in range(1, 10)) * (0.5 + 0.5 * np.sin(2 * np.pi * 1.7 * t) ** 2)
    x = x + 0.01 * np.random.default_rng(seed).standard_normal(len(t))
    return (0.3 * x / np.abs(x).max()).astype(np.float32)


def main():
    sys.path.insert(0, os.getcwd())
    import argparse
    import torch
    import inference as sv
    sv.device = torch.device("cpu")
    began = time.monotonic()
    sv.load_models(argparse.Namespace(f0_condition=True, checkpoint=None, config=None, fp16=False))
    print(f"seedvc weights fetched and loaded in {time.monotonic() - began:.0f} s", flush=True)
    with tempfile.TemporaryDirectory() as td:
        src, ref, out = (os.path.join(td, n) for n in ("src.wav", "ref.wav", "out.wav"))
        sf.write(src, voice(4.0, 220.0, 1), SR, subtype="FLOAT")
        sf.write(ref, voice(4.0, 130.0, 2), SR, subtype="FLOAT")
        began = time.monotonic()
        r = subprocess.run([sys.executable, os.path.join(HERE, "seedvc_run.py"), "--ref", ref, "--source", src, "--output", out,
                            "--shift", "0", "--steps", "4", "--device", "cpu"],
                           env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONPATH=os.getcwd()))
        if r.returncode != 0 or not os.path.exists(out):
            sys.exit("Seed-VC self-test failed: no conversion")
        y, sr = sf.read(out)
        if sr != SR or len(y) < 3 * SR or not np.isfinite(y).all() or np.abs(y).max() < 1e-4:
            sys.exit(f"Seed-VC self-test failed: output {len(y)} samples at {sr} Hz, peak {np.abs(y).max() if len(y) else 0}")
        print(f"seedvc self-test ok: {len(y) / sr:.2f} s at {sr} Hz in {time.monotonic() - began:.0f} s on the CPU", flush=True)


if __name__ == "__main__":
    main()
