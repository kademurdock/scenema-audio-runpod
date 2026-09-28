"""Studio vocal effects for "Sing it in my voice" (the request's vocal_fx, Sep 27 2026).

Her words: "adding some reverb/delay/echo type effects to put on my voice if I want it, like optional ... I can always take dry
vocals into a daw". So the effect is a choice, "none" by default (the worker's output is then exactly what it was), it goes on
the converted lead only, before the remix, and the plain dry vocal file is never touched: the effected vocal is its own file.

numpy + the house helpers only (vpaudio), no new wheel in the image and nothing with a copyleft licence: every filter is the
RBJ cookbook biquad turned into a short FIR through its exact frequency response, every reverb and delay is an FFT convolution
with an impulse response generated here from a fixed seed (so the same song gives the same file every time), and the dynamics
(compressor, de-esser, ducking) run on 1.5 ms blocks with attack and release, like vpaudio.limiter.

Every preset is the same polish plus its own space:
  polish    high-pass 80 Hz, -2.5 dB at 320 Hz (the mud), +1.5 dB at 3.2 kHz (presence), +2 dB shelf from 10 kHz (air), a 3:1
            soft-knee compressor that takes the loud phrases about 4 dB down, and a split-band de-esser above 5.5 kHz (at most
            5 dB, only on the loudest S sounds)
  studio    polish + a short, quiet plate (0.9 s) that glues the voice to the band without sounding like an effect
  plate     polish + a lush plate (2 s, 25 ms pre-delay), dense and bright, the classic pop vocal plate
  hall      polish + a big hall (2.8 s, 40 ms pre-delay, early reflections, darker tail)
  slapback  polish + one short repeat (a sixteenth note between 85 and 135 ms at the song's tempo, else 110 ms) + a touch of plate
  echo      polish + a dotted-eighth ping-pong echo at the song's tempo, each repeat darker than the last, ducked while the voice
            sings so the words stay clear and the repeats bloom in the gaps + a touch of plate
  dreamy    polish + a slight stereo chorus + a quarter-note echo + a 3.8 s hall fed by the voice and its echo
Wet levels are set against the polished voice's own energy (reverbs) or as the first repeat's level (delays); the result is
matched to the dry voice's loudness afterwards, so the voice sits in the band exactly as loud as the dry version does."""
import numpy as np

import vpaudio as va

PRESETS = {
    "studio": {"label": "Studio polish",
               "reverb": {"rt60": 0.9, "low_x": 0.9, "high_x": 0.6, "predelay_ms": 12, "onset_ms": 1.0, "hp_hz": 250,
                          "lp_hz": 9000, "width": 0.9, "seed": 11, "wet_db": -21.0, "duck_db": 0.0}},
    "plate": {"label": "Plate reverb",
              "reverb": {"rt60": 2.0, "low_x": 0.85, "high_x": 0.7, "predelay_ms": 25, "onset_ms": 1.5, "hp_hz": 220,
                         "lp_hz": 10000, "width": 1.0, "seed": 23, "wet_db": -14.0, "duck_db": 2.0}},
    "hall": {"label": "Hall reverb",
             "reverb": {"rt60": 2.8, "low_x": 1.2, "high_x": 0.5, "predelay_ms": 40, "onset_ms": 25.0, "hp_hz": 160,
                        "lp_hz": 6500, "width": 0.9, "seed": 37, "early": True, "wet_db": -14.0, "duck_db": 3.0}},
    "slapback": {"label": "Slapback",
                 "delay": {"note": "sixteenth", "feedback": 0.0, "first_db": -10.0, "lp_hz": 4500, "hp_hz": 250,
                           "pingpong": False, "duck_db": 0.0},
                 "reverb": {"rt60": 0.9, "low_x": 0.9, "high_x": 0.6, "predelay_ms": 12, "onset_ms": 1.0, "hp_hz": 250,
                            "lp_hz": 9000, "width": 0.9, "seed": 11, "wet_db": -25.0, "duck_db": 0.0}},
    "echo": {"label": "Echo",
             "delay": {"note": "dotted_eighth", "feedback": 0.38, "first_db": -9.0, "lp_hz": 4200, "hp_hz": 180,
                       "pingpong": True, "duck_db": 6.0},
             "reverb": {"rt60": 1.6, "low_x": 0.85, "high_x": 0.7, "predelay_ms": 20, "onset_ms": 1.5, "hp_hz": 220,
                        "lp_hz": 9000, "width": 1.0, "seed": 23, "wet_db": -22.0, "duck_db": 2.0, "echo_send": 0.6}},
    "dreamy": {"label": "Dreamy",
               "chorus": {"wet_db": -15.0},
               "delay": {"note": "quarter", "feedback": 0.45, "first_db": -11.0, "lp_hz": 3500, "hp_hz": 220,
                         "pingpong": True, "duck_db": 5.0},
               "reverb": {"rt60": 3.8, "low_x": 1.1, "high_x": 0.5, "predelay_ms": 55, "onset_ms": 45.0, "hp_hz": 200,
                          "lp_hz": 5500, "width": 1.0, "seed": 41, "early": True, "wet_db": -11.0, "duck_db": 2.0,
                          "echo_send": 0.8}},
}
NAMES = ("none",) + tuple(PRESETS)
DEFAULT_BPM = 120.0
BLK = 64  # dynamics block: 1.45 ms at 44.1 kHz
ONSET_FLOOR = 3.0  # onset strength a beat needs (drum hits and sung syllables reach 15-60; noise, hum and vibrato stay under 2)


# ---------------------------------------------------------------- filters
def rbj(kind, f0, sr, q=0.7071, gain_db=0.0):
    """One RBJ cookbook biquad -> (b, a), a[0] == 1. kind: hp, lp, peak, lowshelf, highshelf (shelves at slope 1)."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    c, s = np.cos(w0), np.sin(w0)
    alpha = s / (2 * q)
    if kind == "lp":
        b, a = [(1 - c) / 2, 1 - c, (1 - c) / 2], [1 + alpha, -2 * c, 1 - alpha]
    elif kind == "hp":
        b, a = [(1 + c) / 2, -(1 + c), (1 + c) / 2], [1 + alpha, -2 * c, 1 - alpha]
    elif kind == "peak":
        b, a = [1 + alpha * A, -2 * c, 1 - alpha * A], [1 + alpha / A, -2 * c, 1 - alpha / A]
    elif kind in ("lowshelf", "highshelf"):
        sq = 2 * np.sqrt(A) * s / np.sqrt(2)
        if kind == "lowshelf":
            b = [A * ((A + 1) - (A - 1) * c + sq), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sq)]
            a = [(A + 1) + (A - 1) * c + sq, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sq]
        else:
            b = [A * ((A + 1) + (A - 1) * c + sq), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sq)]
            a = [(A + 1) - (A - 1) * c + sq, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sq]
    else:
        raise ValueError(kind)
    b, a = np.array(b, np.float64), np.array(a, np.float64)
    return b / a[0], a / a[0]


def response(sections, n_fft, sr):
    """The cascade's exact complex frequency response on the rfft grid of n_fft points."""
    z = np.exp(-1j * np.pi * np.arange(n_fft // 2 + 1) / (n_fft // 2))
    H = np.ones(n_fft // 2 + 1, np.complex128)
    for kind, f0, q, gain in sections:
        b, a = rbj(kind, f0, sr, q, gain)
        H *= (b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)
    return H


def fir(sections, sr, n=8192, n_fft=1 << 16):
    """The cascade's impulse response (causal, minimum phase like the IIR it comes from), cut to n samples with a short fade.
    Every section here decays in well under n samples, so the cut changes nothing audible."""
    h = np.fft.irfft(response(sections, n_fft, sr), n_fft)[:n]
    fade = max(1, n // 8)
    h[-fade:] *= np.cos(np.linspace(0, np.pi / 2, fade)) ** 2
    return h


def conv(x, ir, out_len=None):
    """Mono x [n] * ir [m] or [m, ch] -> float32 [out_len, ch] (default n + m - 1), FFT overlap-add in float64."""
    x = np.asarray(x, np.float64).reshape(-1)
    ir = np.asarray(ir, np.float64)
    ir = ir[:, None] if ir.ndim == 1 else ir
    n, m = len(x), len(ir)
    out_len = n + m - 1 if out_len is None else int(out_len)
    block = max(1 << 16, 1 << int(np.ceil(np.log2(max(m, 2)))))
    nfft = 1 << int(np.ceil(np.log2(block + m - 1)))
    H = np.fft.rfft(ir, nfft, axis=0)
    out = np.zeros((out_len, ir.shape[1]), np.float64)
    for s in range(0, min(n, out_len), block):
        seg = x[s:s + block]
        Y = np.fft.irfft(np.fft.rfft(seg, nfft)[:, None] * H, nfft, axis=0)[:len(seg) + m - 1]
        e = min(out_len, s + len(Y))
        out[s:e] += Y[:e - s]
    return out.astype(np.float32)


def linear_highpass(sr, f_hz, taps=255):
    """Linear-phase windowed-sinc high-pass (odd length; delay (taps - 1) / 2), exactly complementary to its low-pass."""
    m = (taps - 1) // 2
    k = np.arange(taps) - m
    lp = 2 * f_hz / sr * np.sinc(2 * f_hz / sr * k) * np.blackman(taps)
    lp /= lp.sum()
    hp = -lp
    hp[m] += 1.0
    return hp


# ---------------------------------------------------------------- dynamics
def block_power(x, blk=BLK, smooth_blocks=1):
    x = np.asarray(x, np.float64).reshape(-1)
    nb = int(np.ceil(len(x) / blk))
    p = np.concatenate([x, np.zeros(nb * blk - len(x))]).reshape(nb, blk)
    p = (p * p).mean(1)
    if smooth_blocks > 1:
        p = np.convolve(p, np.ones(smooth_blocks) / smooth_blocks, mode="same")
    return p


def smooth_db(v, attack, release):
    """One-pole attack/release on a per-block gain in dB: attack when the gain falls (more reduction), release when it rises.
    attack/release are per-block coefficients (0 = instant)."""
    out = np.empty(len(v))
    cur = 0.0
    for i, g in enumerate(v.tolist()):
        k = attack if g < cur else release
        cur = g + k * (cur - g)
        out[i] = cur
    return out


def coef(ms, sr, blk=BLK):
    return float(np.exp(-blk / (sr * ms / 1000.0))) if ms > 0 else 0.0


def to_samples(g_db, n, blk=BLK):
    """Per-block gains in dB -> a per-sample linear gain (interpolated between block centres)."""
    centres = np.arange(len(g_db)) * blk + blk / 2
    return np.interp(np.arange(n), centres, 10 ** (np.asarray(g_db) / 20)).astype(np.float32)


def loud_ref(level_db):
    """The loud level of a voice: the 95th percentile of its active blocks (within 45 dB of its peak)."""
    active = level_db > np.max(level_db) - 45
    return float(np.percentile(level_db[active], 95)) if active.sum() >= 10 else None


def compress(x, sr, ratio=3.0, gr_db=4.0, knee_db=6.0, attack_ms=8.0, release_ms=120.0):
    """Soft-knee feed-forward compressor, threshold set from the voice itself so its loud phrases come down about gr_db."""
    x = np.asarray(x, np.float32)
    lv = 10 * np.log10(block_power(x, BLK, smooth_blocks=7) + 1e-12)  # ~10 ms RMS
    ref = loud_ref(lv)
    if ref is None:
        return x, {"applied": False}
    thr = ref - gr_db * ratio / (ratio - 1)
    over = lv - thr
    slope = 1 - 1 / ratio
    gr = np.where(over <= -knee_db / 2, 0.0,
                  np.where(over >= knee_db / 2, -slope * over, -slope * (over + knee_db / 2) ** 2 / (2 * knee_db)))
    gr = smooth_db(gr, coef(attack_ms, sr), coef(release_ms, sr))
    loud = lv > ref - 6  # the loud phrases: how far they came down (the middle of it)
    info = {"applied": True, "ratio": ratio, "threshold_db": round(thr, 1),
            "loud_reduction_db": round(float(-np.median(gr[loud])), 2) if loud.any() else 0.0}
    return x * to_samples(gr, len(x)), info


def deess(x, sr, f_hz=5500.0, over_db=8.0, ratio=4.0, max_db=5.0, attack_ms=1.0, release_ms=45.0):
    """Split-band de-esser: the band above f_hz comes down (at most max_db) only while it is louder than over_db under the
    voice's own loud level, which only a loud S, SH, T or F reaches; everything below f_hz is untouched."""
    x = np.asarray(x, np.float32)
    h = linear_highpass(sr, f_hz)
    d = len(h) // 2
    hf = conv(x, h, len(x) + d)[d:, 0]
    lf = x - hf
    lv = 10 * np.log10(block_power(x, BLK, smooth_blocks=2) + 1e-12)
    ref = loud_ref(lv)
    if ref is None:
        return x, {"applied": False}
    lh = 10 * np.log10(block_power(hf, BLK, smooth_blocks=2) + 1e-12)
    gr = -np.clip((lh - (ref - over_db)) * (1 - 1 / ratio), 0, max_db)
    gr = smooth_db(gr, coef(attack_ms, sr), coef(release_ms, sr))
    return lf + hf * to_samples(gr, len(x)), {"applied": True, "max_db": round(float(-gr.min()), 2),
                                             "active_s": round(float((gr < -1).sum()) * BLK / sr, 1)}


def duck(wet, dry, sr, depth_db, attack_ms=15.0, release_ms=320.0):
    """Sidechain ducking: the effect comes down by up to depth_db while the dry voice sings and comes back up in the gaps."""
    if depth_db <= 0:
        return wet
    dry = np.asarray(dry, np.float32).reshape(-1)
    if len(dry) < len(wet):  # the tail after the voice ends: silence, so the effect comes back up there
        dry = np.concatenate([dry, np.zeros(len(wet) - len(dry), np.float32)])
    lv = 10 * np.log10(block_power(dry, BLK, smooth_blocks=7) + 1e-12)
    ref = loud_ref(lv)
    if ref is None:
        return wet
    g = -depth_db * np.clip((lv - (ref - 24)) / 18, 0, 1)
    g = smooth_db(g, coef(attack_ms, sr), coef(release_ms, sr))
    gs = to_samples(g, min(len(wet), len(g) * BLK))
    out = np.array(wet, np.float32, copy=True)
    out[:len(gs)] *= gs[:, None]
    return out


# ---------------------------------------------------------------- spaces
def reverb_ir(sr, rt60, low_x=1.0, high_x=0.6, predelay_ms=20, onset_ms=2.0, hp_hz=200, lp_hz=8000, width=1.0, seed=7,
              early=False, **_):
    """Stereo reverb impulse response from decorrelated noise, decaying at its own rate in each octave band (rt60 at 1 kHz,
    rt60 * low_x at 125 Hz, rt60 * high_x at 8 kHz, log-interpolated), with a soft onset, optional early reflections (a hall),
    the send's high-pass and low-pass, pre-delay and a stereo width. Unit energy per channel. Same seed, same response."""
    rng = np.random.default_rng(seed)
    longest = rt60 * max(1.0, low_x)
    n = int(sr * longest * 1.05)
    t = np.arange(n) / sr
    N = np.fft.rfft(rng.standard_normal((n, 2)), axis=0)
    f = np.fft.rfftfreq(n, 1 / sr)
    octv = np.clip(np.log2(np.maximum(f, 1.0) / 1000.0), -3.0, 4.0)
    ir = np.zeros((n, 2))
    for centre in range(-3, 5):
        dist = np.abs(octv - centre)
        w = np.where(dist < 1, np.cos(np.pi / 2 * dist) ** 2, 0.0)
        if not w.any():
            continue
        if centre <= 0:
            r = rt60 * low_x ** (-centre / 3)
        else:
            r = rt60 * high_x ** (min(centre, 3) / 3) * (0.8 if centre > 3 else 1.0)
        band = np.fft.irfft(N * w[:, None], n, axis=0)
        ir += band * np.exp(-6.9078 * t / r)[:, None]
    ir *= (1 - np.exp(-t / max(onset_ms / 1000.0, 1e-4)))[:, None]
    if early:  # a hall's first reflections, alternating sides, before the dense tail builds up
        scale = float(np.sqrt((ir[int(0.05 * sr):int(0.15 * sr)] ** 2).mean())) * 25
        for ms, g, side in ((9, 0.8, 0), (14, 0.7, 1), (21, 0.6, 0), (27, 0.55, 1), (34, 0.45, 0), (43, 0.4, 1),
                            (52, 0.3, 0), (61, 0.28, 1), (73, 0.2, 0), (84, 0.18, 1)):
            ir[int(sr * ms / 1000), side] += g * scale
    nf = 1 << int(np.ceil(np.log2(2 * n)))
    H = response([("hp", hp_hz, 0.7071, 0.0), ("lp", lp_hz, 0.7071, 0.0)], nf, sr)
    ir = np.fft.irfft(np.fft.rfft(ir, nf, axis=0) * H[:, None], nf, axis=0)[:n]
    mid, side = (ir[:, 0] + ir[:, 1]) / 2, (ir[:, 0] - ir[:, 1]) / 2 * width
    ir = np.stack([mid + side, mid - side], 1)
    ir = np.concatenate([np.zeros((int(sr * predelay_ms / 1000), 2)), ir])
    ir /= np.sqrt((ir ** 2).sum(0, keepdims=True)) + 1e-12
    return ir.astype(np.float32)


def delay_seconds(note, bpm):
    """A note length at the tempo, kept in a range that sounds like that effect (whole-beat multiples, so it stays in time)."""
    beat = 60.0 / (bpm or DEFAULT_BPM)
    if note == "sixteenth":
        d = beat / 4
        return d if 0.085 <= d <= 0.135 else 0.110
    d = beat * (0.75 if note == "dotted_eighth" else 1.0)
    lo, hi = (0.25, 0.6) if note == "dotted_eighth" else (0.3, 0.8)
    while d < lo:
        d *= 2
    while d > hi:
        d /= 2
    return d


def delay_ir(sr, d_s, feedback, first_db, lp_hz, hp_hz, pingpong=True, cross=0.25, floor_db=-50.0, **_):
    """Stereo echo impulse response: repeats every d_s, the first at first_db, each next one feedback times as loud and passed
    through the low-pass once more (so every repeat is darker, like tape), all high-passed once. Ping-pong puts the repeats
    left, right, left..., each with a little of itself on the other side. feedback 0 is one repeat (a slapback), a touch
    wider on the right."""
    tap = 4096
    nf = 8192
    hp = response([("hp", hp_hz, 0.7071, 0.0)], nf, sr)
    lp = response([("lp", lp_hz, 0.7071, 0.0)], nf, sr)
    gains = [10 ** (first_db / 20)]
    while feedback > 0 and len(gains) < 16 and gains[-1] * feedback >= 10 ** (floor_db / 20):
        gains.append(gains[-1] * feedback)
    d = int(round(d_s * sr))
    ir = np.zeros((len(gains) * d + tap + int(0.01 * sr), 2))
    for k, g in enumerate(gains, start=1):
        h = np.fft.irfft(hp * lp ** k, nf)[:tap]
        at = k * d
        if pingpong:
            gl, gr = (1.0, cross) if k % 2 else (cross, 1.0)
            ir[at:at + tap, 0] += g * gl * h
            ir[at:at + tap, 1] += g * gr * h
        else:
            late = at + int(0.007 * sr)
            ir[at:at + tap, 0] += g * 0.95 * h
            ir[late:late + tap, 1] += g * 0.95 * h
    return ir.astype(np.float32), {"delay_ms": round(1000 * d_s, 1), "repeats": len(gains)}


def chorus(x, sr, voices=((0.014, 0.0022, 0.33, 0.0, 0), (0.019, 0.0028, 0.47, 1.9, 1)), cross=0.3, chunk=1 << 18):
    """A slight two-voice chorus, wet only: each voice is the input at a slowly swinging delay (linear interpolation), one
    mostly left, one mostly right."""
    x = np.asarray(x, np.float64).reshape(-1)
    n = len(x)
    out = np.zeros((n, 2), np.float64)
    for base, depth, rate, phase, side in voices:
        for s in range(0, n, chunk):
            idx = np.arange(s, min(n, s + chunk))
            pos = idx - (base + depth * np.sin(2 * np.pi * rate * idx / sr + phase)) * sr
            i0 = np.floor(pos).astype(np.int64)
            frac = pos - i0
            ok = i0 >= 0
            i0 = np.clip(i0, 0, n - 2)
            y = (x[i0] * (1 - frac) + x[i0 + 1] * frac) * ok
            out[idx, side] += y
            out[idx, 1 - side] += cross * y
    return out.astype(np.float32)


def energy(x):
    return float((np.asarray(x, np.float64) ** 2).sum())


def at_level(wet, ref, db):
    """wet scaled so its energy sits db under (or over) ref's."""
    e_w, e_r = energy(wet), energy(ref)
    if e_w <= 0 or e_r <= 0:
        return wet
    return (wet * np.float32(np.sqrt(e_r / e_w * 10 ** (db / 10)))).astype(np.float32)


# ---------------------------------------------------------------- tempo
def onset_envelope(x, sr):
    """Spectral flux in 32 log bands (60 Hz-5 kHz) of the recording at 11 kHz, one value per 11.6 ms -> (flux, hop seconds)."""
    m = va.as2d(x).mean(1)
    rs = 11025
    y = va.resample(m, sr, rs)[:, 0].astype(np.float64)
    frame, hop = 1024, 128
    nfr = 1 + (len(y) - frame) // hop
    if nfr < 2:
        return np.zeros(0), hop / rs
    win = np.hanning(frame)
    band = np.digitize(np.fft.rfftfreq(frame, 1 / rs), np.geomspace(60, 5000, 33)) - 1
    pool = np.zeros((frame // 2 + 1, 32))
    ok = (band >= 0) & (band < 32)
    pool[np.nonzero(ok)[0], band[ok]] = 1.0
    ref = float(np.sqrt((y ** 2).mean())) * win.sum() + 1e-12  # one scale for the whole song, so blocks join cleanly
    flux = np.zeros(nfr)
    prev = None
    for b0 in range(0, nfr, 2048):
        idx = np.arange(b0, min(nfr, b0 + 2048))
        fr = np.stack([y[i * hop:i * hop + frame] for i in idx]) * win
        lg = np.log1p(100 * (np.abs(np.fft.rfft(fr, axis=1)) @ pool) / ref)
        before = np.vstack([(lg[:1] if prev is None else prev[None, :]), lg[:-1]])
        flux[idx] = np.maximum(lg - before, 0).sum(1)
        prev = lg[-1]
    return flux, hop / rs


def detect_tempo(x, sr, lo=60.0, hi=200.0):
    """The tempo of a recording from its onsets: the onset envelope's autocorrelation, the beat and its double summed, a gentle
    preference for 70-160 BPM. -> {"bpm", "confidence"} or None when there is no steady beat (too short, too quiet, no real
    onsets, or too weak a pulse). A beat found at double or half speed still gives delays in time with the song."""
    m = va.as2d(x).mean(1)
    if len(m) < 8 * sr or float(np.sqrt((m.astype(np.float64) ** 2).mean())) < 1e-4:
        return None
    flux, hop_s = onset_envelope(m, sr)
    k = max(1, int(round(0.4 / hop_s)))
    env = np.maximum(flux - np.convolve(flux, np.ones(k) / k, mode="same"), 0)
    if len(env) < 10 or float(np.percentile(env, 99)) < ONSET_FLOOR:  # a steady tone or a hum: nothing to follow
        return None
    env -= env.mean()
    nf = 1 << int(np.ceil(np.log2(2 * len(env))))
    E = np.fft.rfft(env, nf)
    ac = np.fft.irfft(E * np.conj(E), nf)[:len(env)]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]
    lmin, lmax = int(np.floor(60 / (hi * hop_s))), int(np.ceil(60 / (lo * hop_s)))
    if 2 * lmax + 2 >= len(ac):
        return None
    lags = np.arange(max(lmin, 2), lmax + 1)
    bpm = 60 / (lags * hop_s)
    prior = np.exp(-0.5 * (np.log2(bpm / 115.0) / 0.9) ** 2)
    score = (ac[lags] + 0.5 * ac[2 * lags]) * prior
    L = int(lags[int(np.argmax(score))])
    conf = float(ac[L])
    if conf < 0.1:
        return None
    a, b, c = ac[L - 1], ac[L], ac[L + 1]
    den = a - 2 * b + c
    shift = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
    return {"bpm": round(float(60 / ((L + float(np.clip(shift, -0.5, 0.5))) * hop_s)), 1), "confidence": round(conf, 3)}


# ---------------------------------------------------------------- the chain
POLISH_EQ = [("hp", 80.0, 0.7071, 0.0), ("peak", 320.0, 1.2, -2.5), ("peak", 3200.0, 0.9, 1.5),
             ("highshelf", 10000.0, 0.7071, 2.0)]


def polish(x, sr):
    """The shared studio polish: EQ, compression, de-essing. Mono in, mono out, same length."""
    x = np.asarray(x, np.float32)
    y = conv(x, fir(POLISH_EQ, sr), len(x))[:, 0]
    y, comp = compress(y, sr)
    y, ds = deess(y, sr)
    return y.astype(np.float32), {"eq": "high-pass 80 Hz, -2.5 dB at 320 Hz, +1.5 dB at 3.2 kHz, +2 dB air from 10 kHz",
                                  "compressor": comp, "de_esser": ds}


def apply(voice, sr, name, tempo=None):
    """The converted lead (mono, float32) -> (stereo float32 [n + tail, 2], info). name is a PRESETS key. The level is NOT
    matched here; voice_pipeline matches it to the dry voice. tempo: detect_tempo()'s answer, or None (120 BPM is assumed)."""
    spec = PRESETS[name]
    voice = np.asarray(voice, np.float32).reshape(-1)
    n = len(voice)
    dry, info_polish = polish(voice, sr)
    bpm = tempo["bpm"] if tempo else None
    info = {"preset": name, "label": spec["label"], "polish": info_polish}
    if "delay" in spec:
        info["tempo"] = ({"bpm": bpm, "confidence": tempo["confidence"], "source": "detected"} if tempo else
                         {"bpm": DEFAULT_BPM, "source": "assumed"})
    parts, tail = [np.repeat(dry[:, None], 2, axis=1)], 0
    echo = None
    if "chorus" in spec:
        ch = at_level(chorus(dry, sr), dry, spec["chorus"]["wet_db"])
        parts.append(ch)
        info["chorus_db"] = spec["chorus"]["wet_db"]
    if "delay" in spec:
        dl = spec["delay"]
        d_s = delay_seconds(dl["note"], bpm)
        ir, dinfo = delay_ir(sr, d_s, **dl)
        echo = conv(dry, ir)
        echo = duck(echo, dry, sr, dl["duck_db"])
        parts.append(echo)
        tail = max(tail, len(ir))
        info["delay"] = {**dinfo, "note": dl["note"].replace("_", " "), "first_repeat_db": dl["first_db"],
                         "feedback": dl["feedback"], "duck_db": dl["duck_db"]}
    if "reverb" in spec:
        rv = spec["reverb"]
        ir = reverb_ir(sr, **rv)
        send = dry
        if echo is not None and rv.get("echo_send"):
            send = np.concatenate([dry, np.zeros(len(echo) - n, np.float32)]) + rv["echo_send"] * echo.mean(1)
        wet = conv(send, ir)
        wet = at_level(wet, dry, rv["wet_db"])
        wet = duck(wet, dry, sr, rv["duck_db"])
        parts.append(wet)
        tail = max(tail, len(ir) + (len(send) - n))
        info["reverb"] = {"seconds": rv["rt60"], "predelay_ms": rv["predelay_ms"], "wet_db": rv["wet_db"],
                          "duck_db": rv["duck_db"]}
    total = n + min(tail, int(6 * sr))
    out = np.zeros((total, 2), np.float32)
    for p in parts:
        k = min(total, len(p))
        out[:k] += p[:k]
    loud = np.abs(out[n:]).max(1) > 1e-5 if total > n else np.zeros(0, bool)  # the tail, cut where it has died away
    out = out[:min(total, n + int(np.nonzero(loud)[0][-1]) + 1 + int(0.05 * sr))] if loud.any() else out[:n]
    info["tail_s"] = round((len(out) - n) / sr, 2)
    return out, info
