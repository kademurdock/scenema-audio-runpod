"""One voice job, start to finish, on files already on the worker. The separation and remix rules are the ones the Sep 27 voice test
ran on a pod (separate.py, convert.py, mix.py there), moved into functions with the tools passed in, so the CPU tests can run every
step with stand-in separators and a stand-in RVC.

song mode:
  1. vocals: the chosen pymss extractor on the mix; if it fails or finds nothing, the fallback extractor. band = mix - vocals.
  2. lead split (optional): the chosen karaoke model (voice_request.LEAD_MODELS). The frazer & becruily model names its lead
     stem; for aufr33's the louder, steadier-pitched part is the lead. Kept only when it holds at least a quarter of the vocal
     energy. backing = vocals - lead, left as it was sung.
  3. dereverb (optional, off by default since round 2): the gentle dereverb model; the dry lead is converted, and the
     reverb-to-dry ratio is remembered.
  4. pitch: the octave rule (vpaudio.choose_shift) against the singer's range, or the semitones asked for.
  5. RVC (infer/cli.py of the pinned RVC) re-sings the (dry) lead.
  6. softer S (optional, on by default): the input's own S hiss above ~4 kHz in its unvoiced frames (vpaudio.unvoiced_blend).
  7. remix: aligned to the lead it replaces (+-300 ms), exactly as loud as that lead, a synthetic room at the measured reverb ratio
     (-30..-6 dB) when dereverb ran, + backing + band; then the whole song back to the original's loudness through a -1 dBFS
     limiter.
vocal mode: the recording is converted as it is (no separation, no room; softer S when asked) and matched to its own loudness.
Outputs (in the work folder): mix.mp3 + mix.wav (24-bit) and vocal.mp3 + vocal.wav (24-bit mono, the dry converted voice)."""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vpaudio as va  # noqa: E402
from voice_request import DEREVERB_MODEL, EXTRACTORS, LEAD_MODELS, MAX_SECONDS  # noqa: E402

SR = 44100
ROOM_RT60 = 1.2
CEILING_DB = -1.0


class Runner:
    """The real tools: RVC's bundled pymss CLI and RVC's infer/cli.py, both in the RVC venv, run as the pod ran them."""

    def __init__(self, log_path, env=os.environ):
        self.rvc = env.get("VOICE_RVC_DIR", "/opt/rvc")
        self.python = env.get("VOICE_RVC_PYTHON", "/opt/rvc-venv/bin/python")
        self.models = env.get("PYMSS_MODEL_DIR", "/opt/pymss_models")
        self.endpoint = env.get("PYMSS_ENDPOINT", "")
        self.device = env.get("VOICE_DEVICE", "cuda")
        self.log_path = log_path

    def _run(self, cmd, env, timeout):
        began = time.monotonic()
        with open(self.log_path, "a", encoding="utf-8", errors="replace") as log:
            log.write(f"\n=== {time.strftime('%H:%M:%S')} {' '.join(map(str, cmd[:6]))}\n")
            log.flush()
            try:
                r = subprocess.run(list(map(str, cmd)), cwd=self.rvc, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, timeout=timeout)
                code = r.returncode
            except subprocess.TimeoutExpired:
                code = -9
                log.write("TIMED OUT\n")
        return code == 0, round(time.monotonic() - began, 1)

    def pymss(self, model, in_dir, out_dir, timeout=900):
        os.makedirs(out_dir, exist_ok=True)
        cmd = [self.python, "-m", "tools.pymss.cli", "infer", model, "--model-dir", self.models, "-i", in_dir, "-o", out_dir,
               "--device", self.device, "--format", "wav", "--wav-bit-depth", "FLOAT"]
        if self.endpoint:
            cmd[5:5] = ["--endpoint", self.endpoint]
        return self._run(cmd, dict(os.environ, PYMSS_MODEL_DIR=self.models), timeout)

    def rvc_convert(self, model_path, index_path, src, dst, shift, opts, timeout=900):
        rate = opts["index_rate"] if index_path else 0.0
        cmd = [self.python, "infer/cli.py", "--model", model_path, "--input", src, "--output", dst, "--pitch", int(shift),
               "--f0-method", opts["f0_method"], "--index-rate", rate, "--protect", opts["protect"],
               "--rms-mix-rate", opts["rms_mix_rate"], "--overwrite"]
        if index_path and rate > 0:
            cmd += ["--index", index_path]
        # PYTHONSAFEPATH + PYTHONPATH: the fix the pod needed (a system Python puts infer/ first on sys.path otherwise)
        env = dict(os.environ, PYTHONSAFEPATH="1", PYTHONPATH=self.rvc)
        return self._run(cmd, env, timeout)


def log_tail(path, n=1500):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")[-n:]
    except OSError:
        return ""


def selftest():
    """What the image's build self-test proved runs (VOICE_SELFTEST, written by selftest.py); {} when absent (tests, older images)."""
    return va.read_json(os.environ.get("VOICE_SELFTEST", "/opt/voice/selftest.json"), {}) or {}


def extractor_ready(name):
    proved = selftest().get("extractors", {})
    return proved.get(name, True) is not False


def stem_file(out_dir, stem, instr):
    """pymss writes <input stem>_<instrument>.<ext>; the instrument's case follows the model's config ("Vocals", "vocals")."""
    for p in glob.glob(os.path.join(out_dir, "**", f"{stem}_*"), recursive=True):
        if os.path.splitext(os.path.basename(p))[0].lower() == f"{stem}_{instr}".lower():
            return p
    return None


def rms_db(x):
    return 10 * np.log10(float((np.asarray(x, np.float64) ** 2).mean()) + 1e-12)


def fit(x, n):
    x = va.as2d(x)
    return x[:n] if len(x) >= n else np.concatenate([x, np.zeros((n - len(x), x.shape[1]), np.float32)])


def lufs(x):
    m = va.measure(x, SR)["I"]
    return m if np.isfinite(m) and m > -69 else None


def separate(mix, work, opts, runner, progress, notes):
    """-> dict(band [n,2], backing [n,2] or None, lead_dry [n] mono, reverb_ratio_db or None, info)."""
    n = len(mix)
    info = {"extractor": opts["extractor"], "vocals_by": None, "lead_split": "off", "dereverb": "off", "reverb_ratio_db": None}
    src = os.path.join(work, "sep_in")
    os.makedirs(src, exist_ok=True)
    va.encode(os.path.join(src, "mix.wav"), mix, SR, bits=32)
    progress("Splitting the voice from the music")
    vocals = None
    tried = [opts["extractor"]] + ([opts["fallback"]] if opts.get("fallback") else [])
    for name in tried:
        label = EXTRACTORS[name]["label"]
        if not extractor_ready(name):
            notes.append(f"{label} failed the worker's own check when it was built, so it was skipped.")
            continue
        out = os.path.join(work, f"sep_{name}")
        ok, secs = runner.pymss(EXTRACTORS[name]["model"], src, out)
        found = stem_file(out, "mix", "vocals") if ok else None
        if found:
            y, _ = va.decode(found, sr=SR, channels=2)
            y = fit(y, n)
            if rms_db(y) > rms_db(mix) - 35:
                vocals, info["vocals_by"], info["vocals_s"] = y, name, secs
                if name != opts["extractor"]:
                    info["fallback_used"] = name
                    notes.append(f"{EXTRACTORS[opts['extractor']]['label']} could not split this recording, so {label} did.")
                break
            notes.append(f"{label} found almost no voice in this recording.")
        else:
            print(f"separation failed: {name}\n{log_tail(runner.log_path)}", flush=True)
            notes.append(f"{label} could not split this recording.")
    if vocals is None:
        raise ValueError("No singing could be split from the music in this recording. Try another vocal extractor, or choose "
                         "Just a vocal if the recording has no music.")
    band = mix - vocals
    lead, backing = vocals, None
    if opts["lead_split"]:
        progress("Finding the lead singer")
        lead_model = LEAD_MODELS[opts["lead_model"]]
        info["lead_model"] = opts["lead_model"]
        kin, kout = os.path.join(work, "lead_in"), os.path.join(work, "lead_out")
        os.makedirs(kin, exist_ok=True)
        va.encode(os.path.join(kin, "vocals.wav"), vocals, SR, bits=32)
        ok, secs = runner.pymss(lead_model["model"], kin, kout)
        info["lead_split"] = "failed: all vocals re-sung"
        parts = [p for p in glob.glob(os.path.join(kout, "**", "vocals_*"), recursive=True) if p.lower().endswith(".wav")] if ok else []
        named, cands = lead_model["lead_stem"], []
        if named:  # the model says which output is the lead: exactly that one
            hit = [p for p in parts if os.path.splitext(os.path.basename(p))[0].lower() == f"vocals_{named}"]
            if hit:
                y = fit(va.decode(hit[0], sr=SR, channels=2)[0], n)
                cands.append({"y": y, "rms_db": rms_db(y), "file": os.path.basename(hit[0])})
            elif ok:
                print(f"lead split: no {named} stem in {[os.path.basename(p) for p in parts]}", flush=True)
        elif len(parts) >= 2:  # otherwise the louder, steadier-pitched part is the lead
            for p in parts:
                y = fit(va.decode(p, sr=SR, channels=2)[0], n)
                tr = va.pitch_track(y.mean(1), SR)
                act = tr["rms_db"] > np.percentile(tr["rms_db"], 95) - 30 if len(tr["rms_db"]) else np.zeros(0, bool)
                vfrac = float(tr["voiced"][act].mean()) if act.any() else 0.0
                cands.append({"y": y, "rms_db": rms_db(y), "voiced_frac": vfrac, "file": os.path.basename(p)})
            cands.sort(key=lambda c: -(c["rms_db"] + 30 * c["voiced_frac"]))
        if cands:
            share = 10 ** ((cands[0]["rms_db"] - rms_db(vocals)) / 10)
            info.update({"lead_energy_share": round(share, 3), "lead_s": secs})
            if share >= 0.25:
                lead = cands[0]["y"]
                backing = vocals - lead  # exact complement, so lead + backing = vocals
                info["lead_split"] = "used"
            else:
                info["lead_split"] = "misfired: all vocals re-sung"
        if info["lead_split"] != "used":
            notes.append("The lead singer could not be told apart from the backing vocals, so every voice was re-sung together.")
    dry, ratio = lead, None
    if opts["dereverb"]:
        progress("Taking the room off the voice")
        din, dout = os.path.join(work, "dry_in"), os.path.join(work, "dry_out")
        os.makedirs(din, exist_ok=True)
        va.encode(os.path.join(din, "lead.wav"), lead, SR, bits=32)
        ok, secs = runner.pymss(DEREVERB_MODEL, din, dout)
        found = stem_file(dout, "lead", "noreverb") if ok else None
        if found:
            dry = fit(va.decode(found, sr=SR, channels=2)[0], n)
            ratio = round(rms_db(lead - dry) - rms_db(dry), 2)
            info.update({"dereverb": "used", "reverb_ratio_db": ratio, "dereverb_s": secs})
        else:
            info["dereverb"] = "failed: the voice was re-sung with its room"
            notes.append("The room could not be taken off the voice, so it was re-sung with its echo and no room was added back.")
    return {"band": band, "backing": backing, "lead_dry": dry.mean(1).astype(np.float32), "reverb_ratio_db": ratio, "info": info}


def choose_pitch(req, voice):
    if req["pitch"] != "auto":
        return {"shift": int(req["pitch"]), "source": "set", "why": f"moved {req['pitch']:+d} semitones, as asked"}
    if not req.get("voice_range"):
        return {"shift": 0, "source": "auto", "why": "no voice range is on record, so the melody stays where it is"}
    tr = va.pitch_track(voice, SR)
    stats = va.pitch_stats(tr)
    if stats.get("voiced_frames", 0) < 50:
        return {"shift": 0, "source": "auto", "why": "too little clear pitch to choose an octave, so the melody stays where it is"}
    st = va.hz_to_st(tr["f0"][tr["voiced"]])
    sh = va.choose_shift(req["voice_range"], stats, st)
    out = {"shift": int(sh["shift"]), "source": "auto", "why": sh["why"]}
    if "share_above_top" in sh:
        out["share_above_top"] = sh["share_above_top"]
    return out


def finish(x, mp3, wav, target_lufs):
    """Gain to target_lufs (None keeps the level), the -1 dBFS limiter, then the same samples as a 320 kbps MP3 and a 24-bit WAV."""
    x = va.as2d(x)
    before = lufs(x)
    gain = float(np.clip(target_lufs - before, -24, 24)) if (target_lufs is not None and before is not None) else 0.0
    y, limited = va.limiter(x * np.float32(10 ** (gain / 20)), SR, CEILING_DB)
    va.encode(mp3, y, SR)
    va.encode(wav, y, SR, bits=24)
    return {"in_lufs": None if before is None else round(before, 2), "gain_db": round(gain, 2), "limit_db": round(limited, 2)}


def load_input(audio_path):
    """The recording as float32 [n, 2] at SR, at its own level. ffmpeg's mono-to-stereo upmix puts each side
    3 dB down, so a mono file is decoded as mono and copied to both sides instead (Sep 27 2026: a mono upload
    came back 3 dB quieter); more than two channels are folded to stereo by ffmpeg as before."""
    x, _ = va.decode(audio_path, sr=SR)
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    if x.shape[1] == 2:
        return x
    return va.decode(audio_path, sr=SR, channels=2)[0]


def run(req, audio_path, model_path, index_path, work, runner, progress=lambda text: None):
    """-> {"files": {name: path}, "report": {...}}. ValueError carries a sentence for the person."""
    notes, timing = [], {}
    began = time.monotonic()
    mix = load_input(audio_path)
    seconds = len(mix) / SR
    if seconds > MAX_SECONDS:
        raise ValueError(f"This recording is {int(seconds // 60)} minutes {int(seconds % 60)} seconds long. Use one up to six minutes.")
    if seconds < 2 or rms_db(mix) < -70:
        raise ValueError("The recording is silent or too short to sing.")
    timing["decode_s"] = round(time.monotonic() - began, 1)
    opts = req["options"]
    song = req["mode"] == "song"
    began = time.monotonic()
    sep = separate(mix, work, opts, runner, progress, notes) if song else None
    voice = sep["lead_dry"] if song else mix.mean(1).astype(np.float32)
    timing["separate_s"] = round(time.monotonic() - began, 1)
    if rms_db(voice) < -60:
        raise ValueError("No singing was found in this recording.")
    pitch = choose_pitch(req, voice)
    progress("Singing it in your voice")
    began = time.monotonic()
    src, conv_path = os.path.join(work, "voice_in.wav"), os.path.join(work, "voice_out.wav")
    va.encode(src, voice, SR, bits=32)
    ok, secs = runner.rvc_convert(model_path, index_path, src, conv_path, pitch["shift"], opts)
    if not ok or not os.path.exists(conv_path):
        print(f"RVC failed\n{log_tail(runner.log_path)}", flush=True)
        raise RuntimeError("RVC conversion failed")
    timing["convert_s"] = round(time.monotonic() - began, 1)
    progress("Putting the song back together" if song else "Finishing your vocal")
    began = time.monotonic()
    n = len(voice)
    conv = va.decode(conv_path, sr=SR, channels=1)[0][:, 0]
    lag, peak = va.align_lag(voice, conv, SR, max_ms=300)
    conv = va.shift_fit(conv, lag, n)[:, 0]
    l_ref, l_conv = lufs(voice), lufs(conv)
    vgain = float(np.clip(l_ref - l_conv, -20, 20)) if (l_ref is not None and l_conv is not None) else 0.0
    conv = (conv * 10 ** (vgain / 20)).astype(np.float32)
    placed = {"lag_ms": round(1000 * lag / SR, 1), "align_peak": round(peak, 3), "vocal_gain_db": round(vgain, 2)}
    if opts["soft_s"]:
        began_s = time.monotonic()
        conv, placed["soft_s"] = va.unvoiced_blend(conv, voice, SR)
        timing["soft_s_s"] = round(time.monotonic() - began_s, 1)
    files = {name: os.path.join(work, name) for name in ("vocal.mp3", "vocal.wav", "mix.mp3", "mix.wav")}
    vocal_master = finish(conv, files["vocal.mp3"], files["vocal.wav"], None)
    if song:
        vocal = np.repeat(conv[:, None], 2, axis=1)
        ratio = sep["reverb_ratio_db"]
        if ratio is not None and opts["room"]:
            wet_db = float(np.clip(ratio, -30, -6))
            wet = va.convolve(conv, va.room_ir(SR, rt60=ROOM_RT60, seed=7))[:n]
            e_c, e_w = float((conv ** 2).mean()), float((wet ** 2).mean())
            if e_w > 0 and e_c > 0:
                vocal = vocal + wet * np.float32(np.sqrt(e_c / e_w * 10 ** (wet_db / 10)))
                placed["room_db"] = wet_db
        if sep["backing"] is not None:
            vocal = vocal + sep["backing"]
        target = lufs(mix)
        target = None if target is None else float(np.clip(target, -23, -9))
        mix_master = finish(sep["band"] + vocal, files["mix.mp3"], files["mix.wav"], target)
    else:
        files["mix.mp3"], files["mix.wav"] = files["vocal.mp3"], files["vocal.wav"]
        mix_master = vocal_master
    timing["mix_s"] = round(time.monotonic() - began, 1)
    report = {
        "mode": req["mode"], "seconds": round(seconds, 2), "pitch": pitch, "placed": placed,
        "separation": sep["info"] if song else None, "master": mix_master, "vocal_master": vocal_master,
        "settings": {**opts, "index_used": bool(index_path and opts["index_rate"] > 0), "rvc_s": secs},
        "notes": notes, "timing": timing,
    }
    with open(os.path.join(work, "report.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1)
    files["report.json"] = os.path.join(work, "report.json")
    return {"files": files, "report": report}
