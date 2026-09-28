"""The voice worker's request contract: what a job may ask for, checked before any GPU work, with every refusal in plain words.

Input (RunPod job "input"):
  mode          "song"  a mix with music: split the voice from the band, re-sing the lead, put it back (the default)
                "vocal" a dry vocal with no music: re-sing it as it is
  audio_key     a key in the private bucket (a Sound Booth take or an earlier voice output), or
  audio_url     an https link on an allowed host (VOICE_AUDIO_HOSTS), such as a signed link to an imported recording
  model_key     the RVC voice model (.pth) in the private bucket, under voice-models/<owner>/...
  model_sha256  optional; the worker caches models by content hash and refuses a download that does not match
  index_key / index_sha256   optional retrieval index (.index) from the same voice-models/<owner>/ folder
  pitch         "auto" (the octave rule, needs voice_range) or whole semitones -24..24
  voice_range   optional {"p05", "p50", "p95"}: the singer's own low, middle and high notes as MIDI numbers
  options       extractor, fallback, lead_split, lead_model, dereverb, room, soft_s, index_rate, protect, rms_mix_rate, f0_method
                (DEFAULTS below)
  vocal_fx      optional studio effect on the converted lead (vocalfx.py): "none" (the default: output exactly as before),
                "studio", "plate", "hall", "slapback", "echo" or "dreamy"; the dry vocal files stay dry either way
  output_prefix optional "voice/<8-64 letters, digits, - or _>"; default voice/<random>
Nothing here names a person: the owner is whatever the caller's registry put in the model key."""
import ipaddress
import os
import re
import socket
import urllib.parse
import uuid

# Vocal extractors: pymss models (RVC's bundled separator). Baked into the image unless marked on demand, which download once per
# worker from the pinned model endpoint. "demucs" is the vocals model of htdemucs_ft.
EXTRACTORS = {
    "hyperace": {"model": "bs_roformer_voc_hyperacev2.ckpt", "label": "BS-RoFormer HyperACE v2", "baked": True},
    "bs_roformer": {"model": "model_bs_roformer_ep_317_sdr_12.9755.ckpt", "label": "BS-RoFormer ep 317", "baked": True},
    "melband_kim": {"model": "Kim_MelBandRoformer.ckpt", "label": "Mel-band RoFormer (Kim)", "baked": True},
    "melband_becruily": {"model": "mel_band_roformer_vocals_becruily.ckpt", "label": "Mel-band RoFormer (becruily)", "baked": False},
    "demucs": {"model": "HTDemucs4_FT_vocals_official.th", "label": "HTDemucs ft (vocals)", "baked": True},
}
# Lead vs backing (karaoke) models. lead_stem names the output that IS the lead (pymss writes <input>_<instrument>.wav); None means
# the lead is found by listening (the louder, steadier-pitched part), as round 1 did with aufr33's model.
LEAD_MODELS = {
    "frazer": {"model": "bs_roformer_karaoke_frazer_becruily.ckpt", "label": "BS-RoFormer karaoke (frazer & becruily)",
               "lead_stem": "vocals", "baked": True},
    "aufr33": {"model": "model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
               "label": "Mel-band RoFormer karaoke (aufr33 & viperx)", "lead_stem": None, "baked": True},
}
DEREVERB_MODEL = "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt"
# The round 2 chain (Sep 27 2026, voice-persona RUNBOOK section 17, measured with Whisper against the lyrics on the three songs
# with the most backing singers): HyperACE v2 vocals (BS-RoFormer ep 317 if it fails), the frazer & becruily lead split, NO
# dereverb (so no room is added back), RVC index rate 0.5, protect 0.33, RMS mix 0.25, RMVPE pitch, then the input's own S hiss
# above ~4 kHz in its unvoiced frames (soft_s). The lead that reached RVC lost 4 of 591 sung words instead of 105, her voice sang
# 494 of them right instead of 322, and the S sounds matched the original singer's level (within 0.1 dB) and texture.
# Round 1's chain is still one request away: extractor bs_roformer, fallback demucs, lead_model aufr33, dereverb on, soft_s off.
DEFAULTS = {
    "extractor": "hyperace",
    "fallback": "bs_roformer",
    "lead_split": True,
    "lead_model": "frazer",
    "dereverb": False,
    "room": True,
    "soft_s": True,
    "index_rate": 0.5,
    "protect": 0.33,
    "rms_mix_rate": 0.25,
    "f0_method": "rmvpe",
}
# Vocal effects (vocalfx.PRESETS; listed here too because this file is copied into the image before the audio code is).
VOCAL_FX = ("none", "studio", "plate", "hall", "slapback", "echo", "dreamy")
MAX_SECONDS = 6 * 60 + 5
MAX_BYTES = 256 * 1024 * 1024
KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/=-]{0,1023}$")
MODEL_KEY_RE = re.compile(r"^voice-models/([A-Za-z0-9_-]{1,64})/[A-Za-z0-9._/-]{1,512}\.pth$")
INDEX_KEY_RE = re.compile(r"^voice-models/([A-Za-z0-9_-]{1,64})/[A-Za-z0-9._/-]{1,512}\.index$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
PREFIX_RE = re.compile(r"^voice/[A-Za-z0-9_-]{8,64}$")


def _audio_prefixes():
    raw = os.environ.get("VOICE_AUDIO_PREFIXES", "yue2/,voice/,audios/")
    return tuple(p.strip() for p in raw.split(",") if p.strip())


def _audio_hosts():
    raw = os.environ.get("VOICE_AUDIO_HOSTS") or os.environ.get("AUK_AUDIO_HOSTS", "")
    return {h.strip().lower() for h in raw.split(",") if h.strip()}


def _flag(options, key):
    value = options.get(key, DEFAULTS[key])
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("on", "off", "true", "false", "yes", "no", "1", "0"):
        return value.strip().lower() in ("on", "true", "yes", "1")
    raise ValueError(f"The {key.replace('_', ' ')} setting must be on or off.")


def _number(options, key, low, high, label):
    value = options.get(key, DEFAULTS[key])
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value or not low <= value <= high:
        raise ValueError(f"{label} must be a number from {low} to {high}.")
    return float(value)


def _key(value, pattern, what):
    if not isinstance(value, str) or ".." in value or "//" in value or not pattern.match(value):
        raise ValueError(f"The {what} is not a file this worker can read.")
    return value


def _sha(value, what):
    if value is None:
        return None
    if not isinstance(value, str) or not SHA_RE.match(value.lower()):
        raise ValueError(f"The {what} record has a broken checksum.")
    return value.lower()


def check_url(url, resolve=True):
    """A signed https link on an allowed host that resolves only to public addresses (no metadata service, no LAN)."""
    if not isinstance(url, str) or len(url) > 12000:
        raise ValueError("Import the recording again.")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in (None, 443) \
            or (parsed.hostname or "").lower() not in _audio_hosts():
        raise ValueError("Import this recording into the Sound Booth first.")
    if resolve:
        for address in socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM):
            if not ipaddress.ip_address(address[4][0]).is_global:
                raise ValueError("The recording's host is not public.")
    return url


def parse(raw, resolve=True):
    """-> a clean request dict, or ValueError with a sentence the Sound Booth can say as it is."""
    if not isinstance(raw, dict):
        raise ValueError("The voice request was empty.")
    mode = raw.get("mode", "song")
    if mode not in ("song", "vocal"):
        raise ValueError('Choose "song" for a recording with music or "vocal" for a voice on its own.')
    has_key, has_url = raw.get("audio_key") is not None, raw.get("audio_url") is not None
    if has_key == has_url:
        raise ValueError("Send one recording: an audio_key or an audio_url.")
    audio_key = audio_url = None
    if has_key:
        audio_key = _key(raw["audio_key"], KEY_RE, "recording")
        if not audio_key.startswith(_audio_prefixes()):
            raise ValueError("The recording is not a file this worker can read.")
    else:
        audio_url = check_url(raw["audio_url"], resolve=resolve)
    model_key = _key(raw.get("model_key"), MODEL_KEY_RE, "voice model")
    owner = MODEL_KEY_RE.match(model_key).group(1)
    index_key = None
    if raw.get("index_key") is not None:
        index_key = _key(raw["index_key"], INDEX_KEY_RE, "voice index")
        if INDEX_KEY_RE.match(index_key).group(1) != owner:
            raise ValueError("The voice index belongs to a different voice model.")
    pitch = raw.get("pitch", "auto")
    if pitch != "auto":
        if isinstance(pitch, bool) or not isinstance(pitch, (int, float)) or pitch != int(pitch) or not -24 <= pitch <= 24:
            raise ValueError("Pitch must be automatic or a whole number of semitones from -24 to 24.")
        pitch = int(pitch)
    voice_range = raw.get("voice_range")
    if voice_range is not None:
        try:
            voice_range = {k: float(voice_range[k]) for k in ("p05", "p50", "p95")}
        except (TypeError, KeyError, ValueError):
            raise ValueError("The voice range needs a low, middle and high note.") from None
        if not (30 <= voice_range["p05"] <= voice_range["p50"] <= voice_range["p95"] <= 100):
            raise ValueError("The voice range notes are out of order.")
    options = raw.get("options") or {}
    if not isinstance(options, dict):
        raise ValueError("The voice options were not readable.")
    extractor = options.get("extractor", DEFAULTS["extractor"])
    if extractor not in EXTRACTORS:
        raise ValueError("Choose one of the listed vocal extractors.")
    fallback = options.get("fallback", DEFAULTS["fallback"])
    if fallback not in EXTRACTORS and fallback != "none":
        raise ValueError("Choose one of the listed vocal extractors as the fallback, or none.")
    f0_method = options.get("f0_method", DEFAULTS["f0_method"])
    if f0_method not in ("rmvpe", "pm"):
        raise ValueError("Pitch tracking must be rmvpe or pm.")
    lead_model = options.get("lead_model", DEFAULTS["lead_model"])
    if lead_model not in LEAD_MODELS:
        raise ValueError("Choose one of the listed lead split models.")
    clean = {
        "extractor": extractor,
        "fallback": None if fallback in ("none", extractor) else fallback,
        "lead_split": _flag(options, "lead_split"),
        "lead_model": lead_model,
        "dereverb": _flag(options, "dereverb"),
        "room": _flag(options, "room"),
        "soft_s": _flag(options, "soft_s"),
        "index_rate": _number(options, "index_rate", 0, 1, "Voice likeness (index rate)"),
        "protect": _number(options, "protect", 0, 0.5, "Protect"),
        "rms_mix_rate": _number(options, "rms_mix_rate", 0, 1, "Loudness follow (RMS mix rate)"),
        "f0_method": f0_method,
    }
    vocal_fx = raw.get("vocal_fx")
    vocal_fx = "none" if vocal_fx is None else (vocal_fx.strip().lower() if isinstance(vocal_fx, str) else vocal_fx)
    if not isinstance(vocal_fx, str) or vocal_fx not in VOCAL_FX:
        raise ValueError("Choose one of the listed vocal effects, or none.")
    prefix = raw.get("output_prefix")
    if prefix is not None and (not isinstance(prefix, str) or not PREFIX_RE.match(prefix)):
        raise ValueError("The output folder name is not allowed.")
    return {
        "mode": mode,
        "audio_key": audio_key,
        "audio_url": audio_url,
        "model_key": model_key,
        "model_sha256": _sha(raw.get("model_sha256"), "voice model"),
        "index_key": index_key,
        "index_sha256": _sha(raw.get("index_sha256"), "voice index") if index_key else None,
        "pitch": pitch,
        "voice_range": voice_range,
        "options": clean,
        "vocal_fx": vocal_fx,
        "output_prefix": prefix or f"voice/{uuid.uuid4().hex}",
    }
