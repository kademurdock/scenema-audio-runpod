"""Shared audio helpers for the voice worker (taken from the voice-persona test, where they ran on the PC and on the pod).
numpy + ffmpeg only, so the CPU unit tests run the same code the worker runs. Python 3.10+.

  probe / decode / encode      ffmpeg pipes (float32, [n, ch]); encode writes .wav/.flac/.mp3 atomically (tmp + rename)
  measure                      EBU R128 via ffmpeg ebur128: integrated loudness I (LUFS), true peak TP (dBTP), LRA
  yin / pitch_track            numpy YIN (the house voice_set.py implementation, singing range 65-1100 Hz, 16 kHz, 16 ms hop)
  pitch_stats / choose_shift   semitone statistics and the octave rule (0 or +/-12) used for every song
  align / reverb / limiter     envelope cross-correlation, a synthetic stereo room (FFT overlap-add) and a look-ahead peak limiter
  master                       gain to a loudness target + limiter + MP3, the one path every listening file goes through
"""
import json
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np

YIN_SR, YIN_FRAME, YIN_HOP, YIN_FMIN, YIN_FMAX = 16000, 1024, 256, 65.0, 1100.0
NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ---------------------------------------------------------------- ffmpeg
def ffmpeg_bin():
    for p in (os.environ.get("VP_FFMPEG"), shutil.which("ffmpeg")):
        if p and os.path.exists(p):
            return p
    # RuntimeError, not SystemExit: inside the RunPod handler a SystemExit would end the worker instead of the job.
    raise RuntimeError("ffmpeg not found: install it or set VP_FFMPEG")


def _run(args, data=None):
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error"] + args
    r = subprocess.run(cmd, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       stdin=None if data is not None else subprocess.DEVNULL)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + " ".join(args[:12]) + "\n" + r.stderr.decode("utf-8", "replace")[-1500:])
    return r


def probe(path):
    """(sample_rate, channels, duration_s) from ffmpeg's own header dump (ffmpeg-static ships no ffprobe)."""
    r = subprocess.run([ffmpeg_bin(), "-hide_banner", "-i", str(path)], stdin=subprocess.DEVNULL,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    t = r.stderr.decode("utf-8", "replace")
    m = re.search(r"Audio: [^,]+, (\d+) Hz, ([^,]+),", t)
    if not m:
        raise RuntimeError(f"no audio stream in {path}")
    lay = m.group(2).strip()
    ch = {"mono": 1, "stereo": 2}.get(lay)
    if ch is None:
        n = re.match(r"(\d+) channels", lay)
        ch = int(n.group(1)) if n else 2
    d = re.search(r"Duration: (\d+):(\d+):([\d.]+)", t)
    dur = int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3)) if d else float("nan")
    return int(m.group(1)), ch, dur


def decode(path, sr=None, channels=None, af=None):
    """-> (float32 [n, ch], sr). sr/channels None keep the file's own."""
    fsr, fch, _ = probe(path)
    sr, channels = sr or fsr, channels or fch
    args = ["-i", str(path), "-vn"]
    if af:
        args += ["-af", af]
    args += ["-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-c:a", "pcm_f32le", "pipe:1"]
    x = np.frombuffer(_run(args).stdout, dtype="<f4").reshape(-1, channels)
    return np.ascontiguousarray(x, dtype=np.float32), sr


def as2d(x):
    x = np.asarray(x, dtype=np.float32)
    return x[:, None] if x.ndim == 1 else x


def encode(path, x, sr, af=None, bits=16, mp3_rate="320k"):
    """Write x ([n] or [n, ch] float) to .wav (bits 16/24/32 float) / .flac (16/24) / .mp3 (CBR). Atomic: tmp file then rename."""
    x = as2d(x)
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".part" + ext
    args = ["-y", "-f", "f32le", "-ar", str(sr), "-ac", str(x.shape[1]), "-i", "pipe:0"]
    if af:
        args += ["-af", af]
    if ext == ".wav":
        args += ["-c:a", {16: "pcm_s16le", 24: "pcm_s24le", 32: "pcm_f32le"}[bits]]
    elif ext == ".flac":
        args += ["-c:a", "flac"] + (["-sample_fmt", "s32", "-bits_per_raw_sample", "24"] if bits == 24 else ["-sample_fmt", "s16"])
    elif ext == ".mp3":
        args += ["-c:a", "libmp3lame", "-b:a", mp3_rate]
    else:
        raise ValueError(f"unsupported output type {ext}")
    args += [tmp]
    _run(args, np.ascontiguousarray(x, dtype="<f4").tobytes())
    os.replace(tmp, path)
    return path


def measure(x, sr):
    """EBU R128 of an array: {'I': LUFS, 'TP': dBTP, 'LRA': LU}. Silence gives I = -70."""
    x = as2d(x)
    r = subprocess.run([ffmpeg_bin(), "-hide_banner", "-nostats", "-f", "f32le", "-ar", str(sr), "-ac", str(x.shape[1]), "-i", "pipe:0",
                        "-af", "ebur128=peak=true", "-f", "null", "-"], input=np.ascontiguousarray(x, dtype="<f4").tobytes(),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return _parse_r128(r.stderr.decode("utf-8", "replace"))


def measure_file(path):
    r = subprocess.run([ffmpeg_bin(), "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"],
                       stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return _parse_r128(r.stderr.decode("utf-8", "replace"))


def _parse_r128(t):
    s = t[t.rfind("Summary:"):]
    def num(pat):
        m = re.search(pat, s)
        if not m:
            return float("nan")
        return -float("inf") if m.group(1) == "-inf" else float(m.group(1))
    return {"I": num(r"I:\s+(-?[\d.]+|-inf) LUFS"), "TP": num(r"Peak:\s+(-?[\d.]+|-inf) dBFS"), "LRA": num(r"LRA:\s+(-?[\d.]+) LU")}


def resample(x, sr_in, sr_out):
    """High-quality resample through ffmpeg's swr (filter_size 64)."""
    if sr_in == sr_out:
        return as2d(x)
    x = as2d(x)
    r = _run(["-f", "f32le", "-ar", str(sr_in), "-ac", str(x.shape[1]), "-i", "pipe:0", "-af",
              f"aresample={sr_out}:filter_size=64:phase_shift=10:cutoff=0.97", "-f", "f32le", "-c:a", "pcm_f32le", "pipe:1"],
             np.ascontiguousarray(x, dtype="<f4").tobytes())
    return np.frombuffer(r.stdout, dtype="<f4").reshape(-1, x.shape[1]).copy()


# ---------------------------------------------------------------- pitch
def yin(x, sr=YIN_SR, frame=YIN_FRAME, hop=YIN_HOP, fmin=YIN_FMIN, fmax=YIN_FMAX, thr=0.15, block=2048):
    """f0 [n_frames] (nan = unvoiced), cmnd minimum [n_frames], rms [n_frames]. Vectorised YIN (de Cheveigne & Kawahara 2002);
    the house implementation from yue2-lora/soulreal/scripts/pod/voice_set.py with a singing range."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    tmin, tmax = int(sr // fmax), int(np.ceil(sr / fmin))
    Wn = frame - tmax
    if len(x) < frame:
        return np.array([]), np.array([]), np.array([])
    n = 1 + (len(x) - frame) // hop
    f0 = np.full(n, np.nan); cmin = np.ones(n); rms = np.zeros(n)
    nfft = 1 << int(np.ceil(np.log2(2 * frame)))
    taus = np.arange(tmax + 1)
    for b0 in range(0, n, block):
        idx = np.arange(b0, min(n, b0 + block))
        fr = np.stack([x[i * hop:i * hop + frame] for i in idx])
        rms[idx] = np.sqrt((fr ** 2).mean(1))
        A = np.fft.rfft(fr[:, :Wn], nfft, axis=1)
        B = np.fft.rfft(fr, nfft, axis=1)
        cc = np.fft.irfft(np.conj(A) * B, nfft, axis=1)[:, :tmax + 1]
        cs = np.concatenate([np.zeros((len(idx), 1)), np.cumsum(fr ** 2, axis=1)], axis=1)
        e0 = cs[:, Wn:Wn + 1]
        et = cs[:, taus + Wn] - cs[:, taus]
        d = np.maximum(e0 + et - 2 * cc, 0)
        cm = np.ones_like(d)
        run = np.cumsum(d[:, 1:], axis=1)
        cm[:, 1:] = d[:, 1:] * np.arange(1, tmax + 1) / np.maximum(run, 1e-12)
        seg = cm[:, tmin:tmax]
        below = seg < thr
        has = below.any(1)
        first = np.where(has, below.argmax(1), seg.argmin(1))
        rows = np.arange(len(idx))
        for _ in range(12):  # walk from the first dip below the threshold down to its local minimum
            nxt = np.minimum(first + 1, seg.shape[1] - 1)
            step = seg[rows, nxt] < seg[rows, first]
            first = np.where(step, nxt, first)
        t = first + tmin
        cmin[idx] = cm[rows, t]
        tl = np.clip(t - 1, 1, tmax); tr = np.clip(t + 1, 1, tmax)
        yl, y0, yr = cm[rows, tl], cm[rows, t], cm[rows, tr]
        den = yl - 2 * y0 + yr
        ok = np.abs(den) > 1e-12
        shift = np.zeros_like(den)
        shift[ok] = 0.5 * (yl - yr)[ok] / den[ok]
        tt = t + np.clip(shift, -1, 1)
        f0[idx] = np.where(has, sr / tt, np.nan)
    return f0, cmin, rms


def pitch_track(x, sr):
    """Mono vocal at any rate -> dict(f0, conf_voiced mask, rms_db, hop_s) at 16 kHz / 16 ms frames."""
    m = as2d(x).mean(1)
    if sr != YIN_SR:
        m = resample(m, sr, YIN_SR)[:, 0]
    f0, cmin, rms = yin(m)
    rms_db = 20 * np.log10(rms + 1e-9)
    ref = np.percentile(rms_db, 95) if len(rms_db) else -100
    voiced = np.isfinite(f0) & (cmin < 0.2) & (rms_db > max(-60.0, ref - 30.0))
    return {"f0": f0, "voiced": voiced, "cmin": cmin, "rms_db": rms_db, "hop_s": YIN_HOP / YIN_SR}


def hz_to_st(f):
    return 69 + 12 * np.log2(np.asarray(f, dtype=np.float64) / 440.0)


def note(st):
    st = int(round(float(st)))
    return f"{NOTE[st % 12]}{st // 12 - 1}"


def pitch_stats(track):
    """Semitone (MIDI) percentiles of confident voiced frames."""
    f = track["f0"][track["voiced"]]
    if len(f) < 50:
        return {"voiced_frames": int(len(f))}
    st = hz_to_st(f)
    p = {k: float(np.percentile(st, q)) for k, q in (("p05", 5), ("p10", 10), ("p25", 25), ("p50", 50), ("p75", 75), ("p90", 90), ("p95", 95))}
    out = {"voiced_frames": int(len(f)), "voiced_s": round(len(f) * track["hop_s"], 1)}
    out.update({k: round(v, 2) for k, v in p.items()})
    out["notes"] = {k: note(v) for k, v in p.items()}
    out["hz"] = {k: round(float(440 * 2 ** ((v - 69) / 12)), 1) for k, v in p.items()}
    return out


def choose_shift(her, song, song_st=None):
    """Octave rule (the voice models copy the song's melody, so the only choice is which octave to sing it in).
    `her` is the singer's range: the 5th, 50th and 95th percentile notes (MIDI) of their own training recordings.
    0 unless the song's middle note (median of the lead vocal's voiced frames) lies OUTSIDE that range; then one octave toward
    the singer's middle (-12 when the song sits above the top, +12 below the bottom), kept only if it really puts more of the
    melody inside the range. A song whose middle is inside the range is never moved for its high notes (that would turn it into a
    low, different-sounding performance); how far its top goes past the singer's top is reported instead, so those notes can be
    checked by ear.
    song_st: the song's voiced-frame semitones (MIDI numbers), for the in-range shares; optional."""
    med = song["p50"]
    s = -12 if med > her["p95"] else (12 if med < her["p05"] else 0)
    shares = {}
    if song_st is not None and len(song_st):
        st = np.asarray(song_st, dtype=np.float64)
        for k in (0, -12, 12):
            shares[k] = round(float(((st + k < her["p05"]) | (st + k > her["p95"])).mean()), 3)
        if s != 0 and shares[s] >= shares[0]:
            s = 0
    over = song["p95"] + s - her["p95"]
    under = her["p05"] - (song["p05"] + s)
    where = ("inside your range" if her["p05"] <= med <= her["p95"] else ("above your top" if med > her["p95"] else "below your bottom"))
    why = (f"song middle {note(med)} is {where} ({note(her['p05'])}-{note(her['p95'])}, middle {note(her['p50'])}): "
           + ("sung in the original octave" if s == 0 else f"moved {'up' if s > 0 else 'down'} one octave"))
    out = {"shift": s, "median_gap_st": round(her["p50"] - med, 2), "top_over_st": round(over, 2), "bottom_under_st": round(under, 2),
           "why": why}
    if shares:
        out["outside_share"] = {str(k): v for k, v in shares.items()}
        st = np.asarray(song_st, dtype=np.float64) + s
        out["share_above_top"] = round(float((st > her["p95"]).mean()), 3)
        out["share_above_top_plus3"] = round(float((st > her["p95"] + 3).mean()), 3)
    return out


# ---------------------------------------------------------------- alignment, room, limiter
def envelope(x, sr, ms=1.0):
    m = as2d(x).mean(1)
    b = max(1, int(round(sr * ms / 1000)))
    n = len(m) // b
    e = np.sqrt((m[:n * b].reshape(n, b) ** 2).mean(1) + 1e-10)
    return np.log(e), b


def align_lag(ref, x, sr, max_ms=300):
    """Samples by which x lags ref (positive: x is late), from log-envelope cross-correlation at 1 ms resolution."""
    a, b = envelope(ref, sr)
    c, _ = envelope(x, sr)
    n = min(len(a), len(c))
    if n < 1000:
        return 0, 0.0
    a = a[:n] - a[:n].mean(); c = c[:n] - c[:n].mean()
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    cc = np.fft.irfft(np.conj(np.fft.rfft(a, nfft)) * np.fft.rfft(c, nfft), nfft)
    lags = np.concatenate([np.arange(0, max_ms + 1), np.arange(-max_ms, 0)])
    vals = np.concatenate([cc[:max_ms + 1], cc[-max_ms:]])
    k = int(np.argmax(vals))
    peak = float(vals[k] / (np.sqrt((a ** 2).sum() * (c ** 2).sum()) + 1e-12))
    return int(lags[k] * b), peak


def shift_fit(x, lag, n):
    """Remove lag (x late by lag samples) and fit to n samples."""
    x = as2d(x)
    if lag > 0:
        x = x[lag:]
    elif lag < 0:
        x = np.concatenate([np.zeros((-lag, x.shape[1]), np.float32), x])
    if len(x) < n:
        x = np.concatenate([x, np.zeros((n - len(x), x.shape[1]), np.float32)])
    return x[:n]


def room_ir(sr, rt60=1.4, predelay_ms=22, lp_hz=5500, seed=7):
    """Synthetic stereo room: decorrelated noise with an exponential decay (-60 dB at rt60), a one-pole low-pass that darkens the
    tail, a few early reflections; unit energy per channel."""
    rng = np.random.default_rng(seed)
    n = int(sr * rt60 * 1.1)
    t = np.arange(n) / sr
    env = 10 ** (-3 * t / rt60)
    ir = rng.standard_normal((n, 2)) * env[:, None]
    a = np.exp(-2 * np.pi * lp_hz / sr)
    for ch in range(2):  # one-pole low-pass, block-wise to keep it vectorised enough
        y = ir[:, ch].copy()
        for i in range(1, n):
            y[i] = (1 - a) * y[i] + a * y[i - 1]
        ir[:, ch] = y
    pre = int(sr * predelay_ms / 1000)
    ir = np.concatenate([np.zeros((pre, 2)), ir])
    for ms, g, side in ((11, 0.5, 0), (17, 0.45, 1), (29, 0.35, 0), (37, 0.3, 1)):
        k = int(sr * ms / 1000)
        if k < len(ir):
            ir[k, side] += g * np.abs(ir).max()
    ir /= np.sqrt((ir ** 2).sum(0, keepdims=True)) + 1e-12
    return ir.astype(np.float32)


def convolve(x, ir, block=1 << 17):
    """Mono x [n] * stereo ir [m, 2] -> [n + m - 1, 2], FFT overlap-add."""
    x = as2d(x).mean(1)
    m = len(ir)
    nfft = 1 << int(np.ceil(np.log2(block + m - 1)))
    H = np.fft.rfft(ir, nfft, axis=0)
    out = np.zeros((len(x) + m - 1, ir.shape[1]), np.float64)
    for s in range(0, len(x), block):
        seg = x[s:s + block]
        Y = np.fft.irfft(np.fft.rfft(seg, nfft)[:, None] * H, nfft, axis=0)[:len(seg) + m - 1]
        out[s:s + len(Y)] += Y
    return out.astype(np.float32)


def _running_min(v, L):
    """Minimum over [i, i + L) for every i (van Herk / Gil-Werman, O(n))."""
    n = len(v)
    pad = (-(n + L)) % L
    w = np.concatenate([v, np.ones(L + pad, v.dtype)])
    blocks = w.reshape(-1, L)
    g = np.minimum.accumulate(blocks, axis=1).reshape(-1)
    h = np.minimum.accumulate(blocks[:, ::-1], axis=1)[:, ::-1].reshape(-1)
    return np.minimum(h[:n], g[L - 1:L - 1 + n])


def limiter(x, sr, ceiling_db=-1.0, lookahead_ms=5.0, release_ms=80.0, blk=32):
    """Look-ahead peak limiter (sample peak) -> (y, max gain reduction dB). Transparent when the peak is already under the ceiling."""
    x = as2d(x)
    c = 10 ** (ceiling_db / 20)
    pk = np.abs(x).max(1)
    if pk.max() <= c:
        return x, 0.0
    need = np.minimum(1.0, c / np.maximum(pk, 1e-9))
    L = max(1, int(sr * lookahead_ms / 1000))
    g = _running_min(need, L)
    nb = int(np.ceil(len(g) / blk))
    gb = np.concatenate([g, np.ones(nb * blk - len(g))]).reshape(nb, blk).min(1)
    rel = 1 - np.exp(-blk / (sr * release_ms / 1000))
    sm = np.empty(nb); cur = 1.0
    for i in range(nb):
        cur = gb[i] if gb[i] < cur else cur + (1 - cur) * rel
        sm[i] = cur
    gs = np.interp(np.arange(len(g)), np.arange(nb) * blk + blk / 2, sm)
    gs = np.minimum(gs, g)  # never let interpolation overshoot the needed reduction
    return (x * gs[:, None]).astype(np.float32), float(-20 * np.log10(gs.min()))


# ---------------------------------------------------------------- S sounds (round 2, voice-persona improve/pod/ana_i.py)
def stft(x, n_fft=2048, hop=512):
    """Hann-window STFT, frames centred (n_fft // 2 of zeros in front) -> complex64 [frames, n_fft // 2 + 1]."""
    x = np.asarray(x, np.float32)
    pad = n_fft // 2
    xp = np.concatenate([np.zeros(pad, np.float32), x, np.zeros(pad + n_fft, np.float32)])
    n = 1 + (len(xp) - n_fft) // hop
    win = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    out = np.empty((n, n_fft // 2 + 1), np.complex64)
    for b in range(0, n, 4096):
        idx = np.arange(b, min(n, b + 4096))
        fr = np.stack([xp[i * hop:i * hop + n_fft] for i in idx]) * win
        out[idx] = np.fft.rfft(fr, axis=1).astype(np.complex64)
    return out


def istft(S, length, n_fft=2048, hop=512):
    """Weighted overlap-add inverse of stft(); exact for an unchanged spectrogram."""
    win = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    n = S.shape[0]
    total = (n - 1) * hop + n_fft
    y = np.zeros(total, np.float64)
    wsum = np.zeros(total, np.float64)
    for b in range(0, n, 4096):
        idx = np.arange(b, min(n, b + 4096))
        fr = np.fft.irfft(S[idx], n_fft, axis=1) * win
        for k, i in enumerate(idx):
            y[i * hop:i * hop + n_fft] += fr[k]
            wsum[i * hop:i * hop + n_fft] += win ** 2
    y = y / np.maximum(wsum, 1e-8)
    pad = n_fft // 2
    return y[pad:pad + length].astype(np.float32)


def _band(sr, n_fft, lo, hi):
    f = np.arange(n_fft // 2 + 1) * sr / n_fft
    return (f >= lo) & (f < hi)


def _smooth(v, k):
    return v if k <= 1 else np.convolve(v, np.ones(k) / k, mode="same")


def _movmax(v, k):
    if k <= 1 or not len(v):
        return v
    pad = k // 2
    vp = np.concatenate([np.full(pad, v[0]), v, np.full(pad, v[-1])])
    return np.max(np.stack([vp[i:i + len(v)] for i in range(k)]), axis=0)


def _db(p):
    return 10 * np.log10(np.maximum(p, 1e-12))


def sib_frames(src, sr, n_fft=2048, hop=512):
    """STFT frames where a voice is hissy and unvoiced (an S, SH, T or F): 4-11 kHz holds more than 40% of the 80 Hz-11 kHz energy,
    no confident pitch nearby, and the frame is not silence. -> (bool mask, the voice's STFT)."""
    S = stft(src, n_fft, hop)
    pw = np.abs(S) ** 2
    hf = pw[:, _band(sr, n_fft, 4000, 11000)].sum(1)
    tot = pw[:, _band(sr, n_fft, 80, 11000)].sum(1)
    del pw
    ratio = hf / np.maximum(tot, 1e-12)
    lvl = _db(tot)
    tr = pitch_track(src, sr)
    tt = np.arange(len(tr["voiced"])) * tr["hop_s"]
    ft = np.arange(S.shape[0]) * hop / sr
    voiced = np.interp(ft, tt, tr["voiced"].astype(np.float64)) > 0.25 if len(tt) else np.zeros(len(ft), bool)
    return (ratio > 0.4) & ~voiced & (lvl > np.percentile(lvl, 95) - 45), S


def unvoiced_blend(conv, src, sr, lo=3500, hi_x=4500, n_fft=2048, hop=512):
    """Softer S sounds: in the input voice's S frames (sib_frames), the converted voice above ~4 kHz is replaced by the input's own
    hiss (3.5-4.5 kHz crossover, ~30 ms fades), the input first scaled so its loud level matches the converted voice's, so each S
    keeps the input's natural S-to-vowel balance. S hiss carries little of who is singing; RVC rebuilds every S, which is the
    slightly robotic edge. conv and src: mono, lined up, same length. -> (mono float32, info)."""
    conv = np.asarray(conv, np.float32)
    m, Ss = sib_frames(src, sr, n_fft, hop)
    Sc = stft(conv, n_fft, hop)
    n = min(len(m), len(Sc))
    m, Ss, Sc = m[:n].astype(np.float64), Ss[:n], Sc[:n]
    if m.sum() < 1:
        return conv, {"blend_frames": 0, "blend_s": 0.0, "src_gain_db": 0.0}
    k = max(1, int(round(0.03 * sr / hop)))
    w = np.clip(_smooth(_movmax(m, 3), 2 * k + 1) * 1.5, 0, 1)
    ls = np.percentile(_db((np.abs(Ss) ** 2).sum(1)), 95)
    lc = np.percentile(_db((np.abs(Sc) ** 2).sum(1)), 95)
    g = 10 ** ((lc - ls) / 20)
    f = np.arange(n_fft // 2 + 1) * sr / n_fft
    fx = np.clip((f - lo) / (hi_x - lo), 0, 1)
    W = (w[:, None] * fx[None, :]).astype(np.float32)
    y = istft(Sc * (1 - W) + Ss * np.float32(g) * W, len(conv), n_fft, hop)
    return y, {"blend_frames": int((w > 0.5).sum()), "blend_s": round(float((w > 0.5).sum()) * hop / sr, 1),
               "src_gain_db": round(float(lc - ls), 2)}


def master(path, x, sr, target_lufs=-16.0, ceiling_db=-1.0):
    """The one listening path: gain to target_lufs, peak-limit to ceiling_db, write (mp3/flac/wav). Returns the measurements."""
    before = measure(x, sr)
    gain = target_lufs - before["I"] if np.isfinite(before["I"]) and before["I"] > -60 else 0.0
    y = as2d(x) * np.float32(10 ** (gain / 20))
    y, gr = limiter(y, sr, ceiling_db)
    encode(path, y, sr)
    after = measure_file(path)
    return {"in_lufs": round(before["I"], 2), "gain_db": round(gain, 2), "limit_db": round(gr, 2), "out_lufs": round(after["I"], 2),
            "out_tp": round(after["TP"], 2)}


def write_json(path, obj):
    tmp = str(path) + ".part"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default
