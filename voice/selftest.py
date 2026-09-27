"""Build step (RVC venv, CPU only): prove the baked tools run before the image is pushed, so no GPU money finds out instead.

  - every baked separator (the vocal extractors, the karaoke lead model, the dereverb model) runs through RVC's own pymss CLI on
    six seconds of synthetic singing over a synthetic band and writes the stem the worker reads
  - RVC's own infer/cli.py converts three seconds of a synthetic voice with a tiny RVC v2 48 kHz model made here from random
    weights (the same checkpoint layout RVC's trainer saves), through HuBERT and RMVPE, so the whole inference path is exercised
    without anybody's voice
Writes /opt/voice/selftest.json; the worker skips (and says so) any extractor that failed here. Exits 1 when RVC or any part of
the default chain (voice_request.DEFAULTS: the vocal extractor and its fallback, the lead split model, and dereverb when it is on)
fails; a failing optional model is only recorded."""
import glob
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from voice_request import DEFAULTS, DEREVERB_MODEL, EXTRACTORS, LEAD_MODELS  # noqa: E402

RVC = os.environ.get("VOICE_RVC_DIR", "/opt/rvc")
MODELS = os.environ.get("PYMSS_MODEL_DIR", "/opt/pymss_models")
OUT = os.environ.get("VOICE_SELFTEST", "/opt/voice/selftest.json")
SR = 44100


def write_wav(path, x, sr):
    import soundfile as sf
    sf.write(path, x, sr, subtype="FLOAT")


def synthetic_song(seconds=6.0, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    f0 = 220 * 2 ** (np.sin(2 * np.pi * 0.25 * t) * 5 / 12)  # a slow sung glide around A3
    phase = 2 * np.pi * np.cumsum(f0) / sr
    voice = sum(np.sin(k * phase) / k for k in range(1, 12)) * (0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * t) ** 2)
    band = 0.3 * (np.sin(2 * np.pi * 110 * t) + np.sin(2 * np.pi * 165 * t)) + 0.05 * np.random.default_rng(1).standard_normal(len(t))
    mix = np.stack([0.25 * voice + band, 0.25 * voice + 0.9 * band], 1)
    return (mix / np.abs(mix).max() * 0.8).astype(np.float32), (voice / np.abs(voice).max() * 0.8).astype(np.float32)


def pymss(model, in_dir, out_dir):
    began = time.monotonic()
    r = subprocess.run([sys.executable, "-m", "tools.pymss.cli", "infer", model, "--model-dir", MODELS, "-i", in_dir, "-o", out_dir,
                        "--device", "cpu", "--format", "wav", "--wav-bit-depth", "FLOAT"],
                       cwd=RVC, capture_output=True, text=True, timeout=1800, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        print((r.stdout + r.stderr)[-3000:], flush=True)
    return r.returncode == 0, round(time.monotonic() - began, 1)


def has_stem(out_dir, stem, instr):
    return any(os.path.splitext(os.path.basename(p))[0].lower() == f"{stem}_{instr}"
               for p in glob.glob(os.path.join(out_dir, "**", f"{stem}_*"), recursive=True))


def tiny_rvc_model(path):
    import torch
    sys.path.insert(0, RVC)
    from infer.module.models import SynthesizerTrnMs768NSFsid
    hps = json.load(open(os.path.join(RVC, "configs", "v2", "48k.json")))
    d, m = hps["data"], hps["model"]
    config = [d["filter_length"] // 2 + 1, 32, m["inter_channels"], m["hidden_channels"], m["filter_channels"], m["n_heads"],
              m["n_layers"], m["kernel_size"], m["p_dropout"], m["resblock"], m["resblock_kernel_sizes"], m["resblock_dilation_sizes"],
              m["upsample_rates"], m["upsample_initial_channel"], m["upsample_kernel_sizes"], 1, m["gin_channels"], d["sampling_rate"]]
    torch.manual_seed(0)
    net = SynthesizerTrnMs768NSFsid(*config, is_half=False)
    weight = {k: v.half() for k, v in net.state_dict().items() if "enc_q" not in k}
    torch.save({"weight": weight, "config": config, "info": "selftest", "sr": "48k", "f0": 1, "version": "v2"}, path)


def main():
    report = {"extractors": {}, "seconds": {}}
    fatal = []
    with tempfile.TemporaryDirectory() as td:
        mix, voice = synthetic_song()
        src = os.path.join(td, "in")
        os.makedirs(src)
        write_wav(os.path.join(src, "mix.wav"), mix, SR)
        for name, entry in EXTRACTORS.items():
            if not entry["baked"]:
                continue
            ok, secs = pymss(entry["model"], src, os.path.join(td, name))
            ok = ok and has_stem(os.path.join(td, name), "mix", "vocals")
            report["extractors"][name], report["seconds"][name] = ok, secs
            print(f"extractor {name}: {'ok' if ok else 'FAILED'} in {secs} s", flush=True)
            if not ok and name in (DEFAULTS["extractor"], DEFAULTS["fallback"]):
                fatal.append(name)
        kin = os.path.join(td, "kin")
        os.makedirs(kin)
        write_wav(os.path.join(kin, "vocals.wav"), np.stack([voice, voice], 1), SR)
        report["lead_models"] = {}
        for name, entry in LEAD_MODELS.items():
            if not entry["baked"]:
                continue
            kout = os.path.join(td, f"kout_{name}")
            ok, secs = pymss(entry["model"], kin, kout)
            parts = glob.glob(os.path.join(kout, "**", "vocals_*.wav"), recursive=True)
            # a model that names its lead must write that stem (the worker reads exactly it); the other kind needs both parts
            ok = ok and (has_stem(kout, "vocals", entry["lead_stem"]) if entry["lead_stem"] else len(parts) >= 2)
            report["lead_models"][name], report["seconds"][f"lead_{name}"] = ok, secs
            print(f"lead split {name}: {'ok' if ok else 'FAILED'} in {secs} s {[os.path.basename(p) for p in parts]}", flush=True)
            if not ok and name == DEFAULTS["lead_model"]:
                fatal.append(f"lead_{name}")
        report["lead_split"] = report["lead_models"].get(DEFAULTS["lead_model"], False)
        din = os.path.join(td, "din")
        os.makedirs(din)
        write_wav(os.path.join(din, "lead.wav"), np.stack([voice, voice], 1), SR)
        ok, secs = pymss(DEREVERB_MODEL, din, os.path.join(td, "dout"))
        ok = ok and has_stem(os.path.join(td, "dout"), "lead", "noreverb")
        report["dereverb"], report["seconds"]["dereverb"] = ok, secs
        print(f"dereverb: {'ok' if ok else 'FAILED'} in {secs} s", flush=True)
        if not ok and DEFAULTS["dereverb"]:
            fatal.append("dereverb")
        model = os.path.join(td, "selftest.pth")
        tiny_rvc_model(model)
        vin, vout = os.path.join(td, "voice.wav"), os.path.join(td, "voice_out.wav")
        write_wav(vin, voice[: 3 * SR], SR)
        began = time.monotonic()
        r = subprocess.run([sys.executable, "infer/cli.py", "--model", model, "--input", vin, "--output", vout, "--pitch", "0",
                            "--f0-method", "rmvpe", "--index-rate", "0", "--protect", "0.33", "--rms-mix-rate", "0.25", "--overwrite"],
                           cwd=RVC, env=dict(os.environ, PYTHONSAFEPATH="1", PYTHONPATH=RVC), capture_output=True, text=True,
                           timeout=1800, stdin=subprocess.DEVNULL)
        ok = r.returncode == 0 and os.path.exists(vout)
        if ok:
            import soundfile as sf
            info = sf.info(vout)
            ok = abs(info.duration - 3.0) < 0.25
            report["rvc_out"] = {"sr": info.samplerate, "seconds": round(info.duration, 3)}
        else:
            print((r.stdout + r.stderr)[-3000:], flush=True)
        report["rvc"], report["seconds"]["rvc"] = ok, round(time.monotonic() - began, 1)
        print(f"rvc: {'ok' if ok else 'FAILED'} {report.get('rvc_out')}", flush=True)
        if not ok:
            fatal.append("rvc")
    report["fatal"] = fatal
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report), flush=True)
    if fatal:
        raise SystemExit(f"SELFTEST FAILED: {fatal}")
    print("SELFTEST OK", flush=True)


if __name__ == "__main__":
    main()
