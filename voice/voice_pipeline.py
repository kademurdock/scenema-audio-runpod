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
vocal effect (optional, request vocal_fx, "none" by default; vocalfx.py): a studio preset on the converted lead after step 6,
matched to the dry lead's loudness; the effected lead replaces the dry one in the remix (and replaces step 7's synthetic room,
which would double the space). The dry vocal files are never touched.
Outputs (in the work folder): mix.mp3 + mix.wav (24-bit) and vocal.mp3 + vocal.wav (24-bit mono, the dry converted voice); with a
vocal effect also vocal_fx.mp3 + vocal_fx.wav (24-bit stereo, the effected voice with its tail; in vocal mode it is the mix).
With vocal_fx "none" every audio file is byte for byte what the worker made before vocal effects existed.

Nov 2026, each off unless asked for (voice_request.EXTRA_DEFAULTS), so a request without them (and with auto_mix off) makes the
same mix and vocal files:
  auto_mix           ON by default (Nov 19 2026): step 6b rides the converted lead to the original singer phrase by phrase
                     (va.ride), quiets it between phrases and tilts its tone toward the original (va.tone_match, one octave);
                     step 8 EQs the finished song toward the original song's tonal balance (a third of an octave, +-4 dB)
  dereverb_strength  step 3 takes only that share of the room off: the lead that is re-sung is lead*(1-s) + dry*s, and the
                     remembered reverb ratio (so the room put back in step 7) shrinks with it
  denoise            after step 3 (and in vocal mode): the denoise model's clean voice, blended in by the amount asked for
  pitch "fit"        step 4 picks the whole-semitone move that puts the most of the melody in the singer's range (va.fit_shift)
  formant            after step 5: the converted voice's timbre moved with its pitch left alone (va.formant_shift); the same
                     move goes on converted backing vocals, so the lead and the backing stay one voice
  backing            "convert": the backing vocals get their own RVC pass at the same shift, as loud as the backing they
                     replace; "remove": they are left out of the mix
  transpose_song     when the voice moved by anything but whole octaves, the band and the kept backing vocals move to the same
                     key (va.transpose; the octaves are dropped, so the band never moves more than six semitones). Without it a
                     worker note says the voice may be out of key with the music.
  stems              song mode always also writes instrumental.mp3/.wav (the band), backing.mp3/.wav (when the lead split found
                     backing vocals: as they are in the mix, or as they were sung when the mix leaves them out) and
                     lead_original.mp3/.wav (the original singer's lead exactly as it went into RVC), each with the mix's own
                     master gain, so the stems sit at the level they have in the song."""
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
import vocalfx  # noqa: E402
import vpaudio as va  # noqa: E402
from voice_request import DENOISE_MODEL, DEREVERB_MODEL, EXTRA_DEFAULTS, EXTRACTORS, LEAD_MODELS, MAX_SECONDS, SEEDVC  # noqa: E402

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
        self.seedvc = env.get("VOICE_SEEDVC_DIR", "/opt/seedvc")
        self.seedvc_python = env.get("VOICE_SEEDVC_PYTHON", "/opt/seedvc-venv/bin/python")
        self.seedvc_script = env.get("VOICE_SEEDVC_SCRIPT", os.path.join(HERE, "seedvc_run.py"))
        self.log_path = log_path

    def _run(self, cmd, env, timeout, cwd=None):
        began = time.monotonic()
        with open(self.log_path, "a", encoding="utf-8", errors="replace") as log:
            log.write(f"\n=== {time.strftime('%H:%M:%S')} {' '.join(map(str, cmd[:6]))}\n")
            log.flush()
            try:
                r = subprocess.run(list(map(str, cmd)), cwd=cwd or self.rvc, env=env, stdout=log, stderr=subprocess.STDOUT,
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

    def seedvc_convert(self, reference, src, dst, shift, timeout=1500):
        """Seed-VC (seedvc_run.py in its own venv, run from the pinned seed-vc folder, weights baked in: no network)."""
        cmd = [self.seedvc_python, self.seedvc_script, "--ref", reference, "--source", src, "--output", dst, "--shift", int(shift),
               "--steps", SEEDVC["diffusion_steps"], "--cfg", SEEDVC["cfg_rate"], "--device", self.device]
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONPATH=self.seedvc)
        return self._run(cmd, env, timeout, cwd=self.seedvc)


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


def denoise_ready():
    return selftest().get("denoise", True) is not False


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


def separate(mix, work, opts, runner, progress, notes, dereverb_strength=1.0):
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
    if opts["dereverb"] and dereverb_strength > 0:  # strength 0 asks for none of the room to come off: nothing to run
        progress("Taking the room off the voice")
        din, dout = os.path.join(work, "dry_in"), os.path.join(work, "dry_out")
        os.makedirs(din, exist_ok=True)
        va.encode(os.path.join(din, "lead.wav"), lead, SR, bits=32)
        ok, secs = runner.pymss(DEREVERB_MODEL, din, dout)
        found = stem_file(dout, "lead", "noreverb") if ok else None
        if found:
            dry = fit(va.decode(found, sr=SR, channels=2)[0], n)
            if dereverb_strength < 1:  # part of the room stays on the voice; what came off (lead - dry) shrinks with it
                dry = (lead * np.float32(1 - dereverb_strength) + dry * np.float32(dereverb_strength)).astype(np.float32)
                info["dereverb_strength"] = dereverb_strength
            ratio = round(rms_db(lead - dry) - rms_db(dry), 2)
            info.update({"dereverb": "used", "reverb_ratio_db": ratio, "dereverb_s": secs})
        else:
            info["dereverb"] = "failed: the voice was re-sung with its room"
            notes.append("The room could not be taken off the voice, so it was re-sung with its echo and no room was added back.")
    return {"band": band, "backing": backing, "lead_dry": dry.mean(1).astype(np.float32), "reverb_ratio_db": ratio, "info": info}


def denoise(voice, work, runner, amount, progress, notes):
    """The mono voice with the hiss and background noise taken off by the denoise model, blended in by amount (0..1):
    voice*(1-amount) + clean*amount. -> (mono float32, info). A model that cannot run leaves the voice as it was, with a note."""
    if not denoise_ready():
        notes.append("The noise cleanup failed the worker's own check when it was built, so the voice was used as it was.")
        return voice, {"denoise": "skipped"}
    progress("Cleaning the noise off the voice")
    nin, nout = os.path.join(work, "clean_in"), os.path.join(work, "clean_out")
    os.makedirs(nin, exist_ok=True)
    va.encode(os.path.join(nin, "voice.wav"), np.repeat(va.as2d(voice), 2, axis=1), SR, bits=32)
    ok, secs = runner.pymss(DENOISE_MODEL, nin, nout)
    found = stem_file(nout, "voice", "dry") if ok else None
    if not found:
        print(f"denoise failed\n{log_tail(runner.log_path)}", flush=True)
        notes.append("The noise could not be cleaned off the voice, so it was used as it was.")
        return voice, {"denoise": "failed"}
    clean = fit(va.decode(found, sr=SR, channels=2)[0], len(voice)).mean(1)
    out = (voice * np.float32(1 - amount) + clean * np.float32(amount)).astype(np.float32)
    return out, {"denoise": "used", "amount": amount, "removed_db": round(rms_db(voice - out) - rms_db(voice), 1), "denoise_s": secs}


def choose_pitch(req, voice):
    if req["pitch"] not in ("auto", "fit"):
        return {"shift": int(req["pitch"]), "source": "set", "why": f"moved {req['pitch']:+d} semitones, as asked"}
    source = req["pitch"]
    if not req.get("voice_range"):
        return {"shift": 0, "source": source, "why": "no voice range is on record, so the melody stays where it is"}
    tr = va.pitch_track(voice, SR)
    stats = va.pitch_stats(tr)
    if stats.get("voiced_frames", 0) < 50:
        return {"shift": 0, "source": source, "why": "too little clear pitch to choose "
                + ("an octave" if source == "auto" else "a key") + ", so the melody stays where it is"}
    st = va.hz_to_st(tr["f0"][tr["voiced"]])
    if source == "fit":
        sh = va.fit_shift(req["voice_range"], st)
        return {"shift": int(sh["shift"]), "source": "fit", "why": sh["why"], "share_above_top": sh["share_above_top"],
                "outside_before": sh["outside_before"], "outside_after": sh["outside_after"]}
    sh = va.choose_shift(req["voice_range"], stats, st)
    out = {"shift": int(sh["shift"]), "source": "auto", "why": sh["why"]}
    if "share_above_top" in sh:
        out["share_above_top"] = sh["share_above_top"]
    return out


def key_shift(shift):
    """The semitones the music moves when the voice moved by shift: the same key with the octaves dropped (-6..6), because a
    band moved an octave and a bit sounds far worse than one moved just the bit, and it is the same key either way."""
    r = int(shift) % 12
    if r > 6 or (r == 6 and shift < 0):
        r -= 12
    return r


def finish(x, mp3, wav, target_lufs):
    """Gain to target_lufs (None keeps the level), the -1 dBFS limiter, then the same samples as a 320 kbps MP3 and a 24-bit WAV."""
    x = va.as2d(x)
    before = lufs(x)
    gain = float(np.clip(target_lufs - before, -24, 24)) if (target_lufs is not None and before is not None) else 0.0
    y, limited = va.limiter(x * np.float32(10 ** (gain / 20)), SR, CEILING_DB)
    va.encode(mp3, y, SR)
    va.encode(wav, y, SR, bits=24)
    return {"in_lufs": None if before is None else round(before, 2), "gain_db": round(gain, 2), "limit_db": round(limited, 2)}


def load_input(audio_path, limit_s=None):
    """The recording as float32 [n, 2] at SR, at its own level. ffmpeg's mono-to-stereo upmix puts each side
    3 dB down, so a mono file is decoded as mono and copied to both sides instead (Sep 27 2026: a mono upload
    came back 3 dB quieter); more than two channels are folded to stereo by ffmpeg as before.
    limit_s (training) reads no more than that many seconds of it."""
    x, _ = va.decode(audio_path, sr=SR, limit_s=limit_s)
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    if x.shape[1] == 2:
        return x
    return va.decode(audio_path, sr=SR, channels=2, limit_s=limit_s)[0]


def vocal_effect(conv, name, beat_source):
    """The converted lead (mono) -> (stereo effected lead, report), at the dry lead's loudness as it sits in the mix (the same
    voice on both sides), so the singer is exactly as loud against the band as without the effect. The delays follow the tempo
    found in beat_source (the whole recording in song mode, the voice itself in vocal mode), or 120 BPM with no clear beat."""
    tempo = vocalfx.detect_tempo(beat_source, SR) if "delay" in vocalfx.PRESETS[name] else None
    fx, report = vocalfx.apply(conv, SR, name, tempo)
    ref, got = lufs(np.repeat(conv[:, None], 2, axis=1)), lufs(fx)
    gain = float(np.clip(ref - got, -20, 20)) if (ref is not None and got is not None) else 0.0
    report["level_match_db"] = round(gain, 2)
    return (fx * np.float32(10 ** (gain / 20))).astype(np.float32), report


def talk_range(reference):
    """Engine seedvc with no voice_range on record: a singing range guessed from how the reference voice SPEAKS. Speech sits low
    and narrow; the same person sings from about two semitones under their speaking low to about seven over their speaking high,
    centred about three over their speaking middle. Only the octave rule reads it (a whole octave or nothing), so a rough guess
    is enough to keep a deep voice from being handed a soprano line. -> {"p05", "p50", "p95"} or None."""
    stats = va.pitch_stats(va.pitch_track(reference, SR))
    if stats.get("voiced_frames", 0) < 50:
        return None
    low, mid, high = stats["p05"] - 2, stats["p50"] + 3, stats["p95"] + 7
    low, high = max(30.0, low), min(100.0, high)
    mid = min(max(mid, low), high)
    return {"p05": round(low, 2), "p50": round(mid, 2), "p95": round(high, 2)}


def resing(runner, model_path, index_path, voice, work, name, shift, opts, reference=None):
    """One RVC pass (or with reference, one Seed-VC pass in that voice): the mono voice written to <name>_in.wav, re-sung, read
    back and lined up with the voice it replaces (+-300 ms). -> (mono float32 the length of voice, {"lag_ms", "align_peak"},
    seconds the conversion took)."""
    src, conv_path = os.path.join(work, f"{name}_in.wav"), os.path.join(work, f"{name}_out.wav")
    va.encode(src, voice, SR, bits=32)
    if reference:
        ok, secs = runner.seedvc_convert(reference, src, conv_path, shift)
    else:
        ok, secs = runner.rvc_convert(model_path, index_path, src, conv_path, shift, opts)
    if not ok or not os.path.exists(conv_path):
        tool = "Seed-VC" if reference else "RVC"
        print(f"{tool} failed\n{log_tail(runner.log_path)}", flush=True)
        raise RuntimeError(f"{tool} conversion failed")
    conv = va.decode(conv_path, sr=SR, channels=1)[0][:, 0]
    lag, peak = va.align_lag(voice, conv, SR, max_ms=300)
    conv = va.shift_fit(conv, lag, len(voice))[:, 0]
    return conv, {"lag_ms": round(1000 * lag / SR, 1), "align_peak": round(peak, 3)}, secs


def run(req, audio_path, model_path, index_path, work, runner, progress=lambda text: None, reference_path=None):
    """-> {"files": {name: path}, "report": {...}}. ValueError carries a sentence for the person.
    Engine "seedvc": model_path and index_path are None and reference_path is the voice's reference recording."""
    notes, timing = [], {}
    engine = req.get("engine") or "rvc"
    reference, range_source = None, "record" if req.get("voice_range") else None
    if engine == "seedvc":
        if not reference_path:
            raise RuntimeError("seedvc job without a reference recording")
        reference = os.path.join(work, "reference.wav")
        ref_audio = load_input(reference_path, limit_s=25)
        if len(ref_audio) < 3 * SR or rms_db(ref_audio) < -60:
            raise ValueError("The voice's reference recording is silent or shorter than three seconds.")
        va.encode(reference, ref_audio.mean(1).astype(np.float32), SR, bits=32)
        if not req.get("voice_range"):
            guessed = talk_range(ref_audio.mean(1).astype(np.float32))
            if guessed:
                req = dict(req, voice_range=guessed)
                range_source = "guessed from the reference"
    began = time.monotonic()
    mix = load_input(audio_path)
    seconds = len(mix) / SR
    if seconds > MAX_SECONDS:
        raise ValueError(f"This recording is {int(seconds // 60)} minutes {int(seconds % 60)} seconds long. Use one up to ten minutes.")
    if seconds < 2 or rms_db(mix) < -70:
        raise ValueError("The recording is silent or too short to sing.")
    timing["decode_s"] = round(time.monotonic() - began, 1)
    opts = req["options"]
    extra = {**EXTRA_DEFAULTS, **(req.get("extra") or {})}
    song = req["mode"] == "song"
    began = time.monotonic()
    sep = separate(mix, work, opts, runner, progress, notes, extra["dereverb_strength"]) if song else None
    voice = sep["lead_dry"] if song else mix.mean(1).astype(np.float32)
    timing["separate_s"] = round(time.monotonic() - began, 1)
    if rms_db(voice) < -60:
        raise ValueError("No singing was found in this recording.")
    cleaned = None
    if extra["denoise"] > 0:
        began = time.monotonic()
        voice, cleaned = denoise(voice, work, runner, extra["denoise"], progress, notes)
        timing["denoise_s"] = round(time.monotonic() - began, 1)
    pitch = choose_pitch(req, voice)
    # the key: a voice moved by anything but whole octaves is in a new key, and the music either follows it or is left behind
    move = key_shift(pitch["shift"]) if song else 0
    pitch["transposed_song"], pitch["song_shift"] = False, 0
    if move and not req.get("transpose_song"):
        notes.append(f"The voice was moved {abs(pitch['shift'])} semitone{'' if abs(pitch['shift']) == 1 else 's'} "
                     f"{'up' if pitch['shift'] > 0 else 'down'} and the music was not, so the voice may be out of key with the music.")
        move = 0
    elif move and not va.has_rubberband():
        notes.append("This worker could not move the music to the new key, so the voice may be out of key with the music.")
        move = 0
    progress("Singing it in your voice")
    began = time.monotonic()
    conv, placed, secs = resing(runner, model_path, index_path, voice, work, "voice", pitch["shift"], opts, reference)
    timing["convert_s"] = round(time.monotonic() - began, 1)
    backing_mode = extra["backing"]
    backing = sep["backing"] if song else None
    backing_info = {"mode": backing_mode, "status": "none" if backing is None else {"keep": "kept", "remove": "left out"}.get(backing_mode)}
    backing_conv = None
    if backing_mode != "keep" and backing is None:
        notes.append("The backing vocals setting had no effect, because the lead singer was not split from the backing vocals."
                     if song else "The backing vocals setting only applies to a song with music.")
    elif backing_mode == "convert" and rms_db(backing) < -60:
        backing_info["status"] = "kept"
        notes.append("There were no backing vocals to speak of, so none were re-sung.")
    elif backing_mode == "convert":
        progress("Singing the backing vocals in your voice")
        began_b = time.monotonic()
        back_in = backing.mean(1).astype(np.float32)
        backing_conv, back_placed, back_secs = resing(runner, model_path, index_path, back_in, work, "backing", pitch["shift"],
                                                      opts, reference)
        if extra["formant"]:
            backing_conv = va.formant_shift(backing_conv, SR, extra["formant"])
        # exactly as loud as the backing it replaces (the same voice on both sides against the stereo original)
        l_ref, l_conv = lufs(backing), lufs(np.repeat(backing_conv[:, None], 2, axis=1))
        bgain = float(np.clip(l_ref - l_conv, -20, 20)) if (l_ref is not None and l_conv is not None) else 0.0
        backing_conv = (backing_conv * 10 ** (bgain / 20)).astype(np.float32)
        backing_info.update({"status": "converted", "gain_db": round(bgain, 2), "rvc_s": back_secs, **back_placed})
        timing["convert_backing_s"] = round(time.monotonic() - began_b, 1)
    want_fx = req.get("vocal_fx", "none") != "none"
    if not want_fx:
        progress("Putting the song back together" if song else "Finishing your vocal")
    began = time.monotonic()
    n = len(voice)
    if extra["formant"]:
        began_f = time.monotonic()
        conv = va.formant_shift(conv, SR, extra["formant"])
        placed["formant_st"] = extra["formant"]
        timing["formant_s"] = round(time.monotonic() - began_f, 1)
    l_ref, l_conv = lufs(voice), lufs(conv)
    vgain = float(np.clip(l_ref - l_conv, -20, 20)) if (l_ref is not None and l_conv is not None) else 0.0
    conv = (conv * 10 ** (vgain / 20)).astype(np.float32)
    placed["vocal_gain_db"] = round(vgain, 2)
    auto_mix = None
    if extra["auto_mix"]:
        # 6b (Nov 19 2026): the voice rides the original singer phrase by phrase and goes quiet in the gaps, its tone is
        # tilted toward the original's (a whole octave of smoothing: brightness and warmth, never the singer's formants),
        # and its song-wide loudness is set again so the level matched above stays
        began_a = time.monotonic()
        conv, auto_mix = va.ride(conv, voice, SR)
        conv, auto_mix["vocal_eq"] = va.tone_match(conv, voice, SR, strength=0.6, max_db=6.0, octave=1.0)
        l_conv = lufs(conv)
        regain = float(np.clip(l_ref - l_conv, -12, 12)) if (l_ref is not None and l_conv is not None) else 0.0
        conv = (conv * 10 ** (regain / 20)).astype(np.float32)
        timing["auto_mix_s"] = round(time.monotonic() - began_a, 1)
    if opts["soft_s"]:
        began_s = time.monotonic()
        conv, placed["soft_s"] = va.unvoiced_blend(conv, voice, SR)
        timing["soft_s_s"] = round(time.monotonic() - began_s, 1)
    files = {name: os.path.join(work, name) for name in ("vocal.mp3", "vocal.wav", "mix.mp3", "mix.wav")}
    vocal_master = finish(conv, files["vocal.mp3"], files["vocal.wav"], None)
    fx, fx_report = None, None
    if want_fx:
        progress("Adding the vocal effect")
        began_fx = time.monotonic()
        fx, fx_report = vocal_effect(conv, req["vocal_fx"], mix if song else voice)
        files["vocal_fx.mp3"], files["vocal_fx.wav"] = os.path.join(work, "vocal_fx.mp3"), os.path.join(work, "vocal_fx.wav")
        fx_report["master"] = finish(fx, files["vocal_fx.mp3"], files["vocal_fx.wav"], None)
        timing["vocal_fx_s"] = round(time.monotonic() - began_fx, 1)
        progress("Putting the song back together" if song else "Finishing your vocal")
    stems = None
    if song:
        vocal = np.repeat(conv[:, None], 2, axis=1) if fx is None else fx[:n]
        ratio = sep["reverb_ratio_db"]
        if ratio is not None and opts["room"] and fx is not None:
            placed["room"] = "replaced by the vocal effect"
        elif ratio is not None and opts["room"]:
            wet_db = float(np.clip(ratio, -30, -6))
            wet = va.convolve(conv, va.room_ir(SR, rt60=ROOM_RT60, seed=7))[:n]
            e_c, e_w = float((conv ** 2).mean()), float((wet ** 2).mean())
            if e_w > 0 and e_c > 0:
                vocal = vocal + wet * np.float32(np.sqrt(e_c / e_w * 10 ** (wet_db / 10)))
                placed["room_db"] = wet_db
        band = sep["band"]
        if move:
            began_k = time.monotonic()
            band = va.transpose(band, SR, move)
            if backing is not None and backing_conv is None:  # kept or left out: either way the stem matches the new key
                backing = va.transpose(backing, SR, move, voice=True)
            pitch["transposed_song"], pitch["song_shift"] = True, move
            timing["transpose_s"] = round(time.monotonic() - began_k, 1)
        if backing_conv is not None:
            vocal = vocal + backing_conv[:, None]
        elif backing is not None and backing_mode != "remove":
            vocal = vocal + backing
        target = lufs(mix)
        target = None if target is None else float(np.clip(target, -23, -9))
        song_out = band + vocal
        if auto_mix is not None:
            # 8 (auto mix): the finished song EQ'd toward the original song's tonal balance (a third of an octave of
            # smoothing, at most 4 dB either way), so the mud and dullness a rebuilt mix gains are taken back out
            began_m = time.monotonic()
            song_out, auto_mix["mix_eq"] = va.tone_match(song_out, mix, SR, strength=0.7, max_db=4.0, octave=1 / 3)
            timing["auto_mix_s"] = round(timing.get("auto_mix_s", 0) + time.monotonic() - began_m, 1)
        mix_master = finish(song_out, files["mix.mp3"], files["mix.wav"], target)
        # the stems, each with the mix's own master gain, so they sit at the level they have in the song
        began_st = time.monotonic()
        gain = np.float32(10 ** (mix_master["gain_db"] / 20))
        parts = {"instrumental": band, "lead_original": voice}
        if backing is not None:
            parts["backing"] = backing if backing_conv is None else backing_conv
        stems = {"gain_db": mix_master["gain_db"]}
        for name, x in parts.items():
            files[f"{name}.mp3"], files[f"{name}.wav"] = os.path.join(work, f"{name}.mp3"), os.path.join(work, f"{name}.wav")
            stems[name] = finish(x * gain, files[f"{name}.mp3"], files[f"{name}.wav"], None)
        timing["stems_s"] = round(time.monotonic() - began_st, 1)
    elif fx is not None:  # vocal mode with an effect: the effected voice is the result, the dry one stays beside it
        files["mix.mp3"], files["mix.wav"] = files["vocal_fx.mp3"], files["vocal_fx.wav"]
        mix_master = fx_report["master"]
    else:
        files["mix.mp3"], files["mix.wav"] = files["vocal.mp3"], files["vocal.wav"]
        mix_master = vocal_master
    timing["mix_s"] = round(time.monotonic() - began, 1)
    report = {
        "mode": req["mode"], "seconds": round(seconds, 2), "pitch": pitch, "placed": placed,
        "separation": sep["info"] if song else None, "master": mix_master, "vocal_master": vocal_master, "vocal_fx": fx_report,
        "settings": {**opts, **extra, "index_used": bool(index_path and opts["index_rate"] > 0), "rvc_s": secs, "engine": engine,
                     **({"seedvc": dict(SEEDVC), "voice_range": req.get("voice_range"), "range_source": range_source}
                        if engine == "seedvc" else {})},
        "backing": backing_info, "denoise": cleaned, "stems": stems, "auto_mix": auto_mix,
        "notes": notes, "timing": timing,
    }
    with open(os.path.join(work, "report.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1)
    files["report.json"] = os.path.join(work, "report.json")
    return {"files": files, "report": report}
