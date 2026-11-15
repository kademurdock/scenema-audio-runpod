"""VoxCPM2 (openbmb/VoxCPM2) voice design and cloning, private B2 outputs.

Queue worker in the same style as auk_handler.py: the model loads once (at
boot, behind a lock) and stays warm; each job writes a 48 kHz WAV and an MP3
to B2 and returns seven-day signed links.
"""
import gc
import ipaddress
import logging
import os
import random
import re
import socket
import subprocess
import tempfile
import threading
import time
import urllib.parse
import uuid
from pathlib import Path

from voxcpm_contract import parse, with_prefix

log = logging.getLogger("voxcpm-worker")
logging.basicConfig(level=logging.INFO)

MODEL_SHA = "7936080b41163874f38610f46f438e2c0757f9fd"
DEFAULT_MODEL_DIR = ("/runpod-volume/huggingface-cache/hub/models--openbmb--VoxCPM2/snapshots/" + MODEL_SHA)
JOIN_GAP_S = 0.15
REF_MAX_S = 20.0          # plain or styled cloning: the clip's first 20 s
PROMPT_REF_MAX_S = 30.0   # "ultimate" cloning: the transcript must cover the whole clip

ENGINE = None
ENGINE_LOCK = threading.Lock()
NORMALIZER = None
LOAD_SECONDS = None


def _env_flag(name, default):
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "off")


def engine():
    global ENGINE
    with ENGINE_LOCK:
        if ENGINE is None:
            ENGINE = load_engine()
    return ENGINE


def load_engine():
    global LOAD_SECONDS
    model_dir = Path(os.environ.get("VOXCPM_MODEL_DIR", DEFAULT_MODEL_DIR))
    if not (model_dir / "config.json").is_file() or not (model_dir / "model.safetensors").is_file():
        raise ValueError("The VoxCPM2 model cache is not ready. No audio was generated.")
    from voxcpm import VoxCPM
    started = time.monotonic()
    # No denoiser: it would fetch a ModelScope model at load. torch.compile is
    # off by default (slow first call, recompiles per length); VOXCPM_OPTIMIZE=1
    # turns it on.
    model = VoxCPM(voxcpm_model_path=str(model_dir), zipenhancer_model_path=None,
                   enable_denoiser=False, optimize=_env_flag("VOXCPM_OPTIMIZE", "0"), device="cuda")
    LOAD_SECONDS = round(time.monotonic() - started, 1)
    torch = _cuda()
    log.info("VoxCPM2 loaded in %.1f s; %.2f GiB on the GPU; output %d Hz", LOAD_SECONDS,
             torch.cuda.memory_allocated() / 1024**3 if torch else 0.0, model.tts_model.sample_rate)
    return model


def _cuda():
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


def reset_gpu_peak():
    torch = _cuda()
    if torch is not None:
        torch.cuda.reset_peak_memory_stats()


def gpu_memory(label):
    torch = _cuda()
    if torch is None:
        return None
    try:
        free, total = torch.cuda.mem_get_info()
    except Exception as error:  # A reading must never fail a take.
        log.warning("GPU memory reading failed (%s)", type(error).__name__)
        return None
    note = {"peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
            "free_gib": round(free / 1024**3, 2), "total_gib": round(total / 1024**3, 2),
            "device": torch.cuda.get_device_name(0)}
    log.info("VoxCPM2 GPU memory after %s: peak %.2f GiB allocated, %.2f GiB reserved, %.2f of %.2f GiB free (%s)",
             label, note["peak_allocated_gib"], note["peak_reserved_gib"], note["free_gib"], note["total_gib"],
             note["device"])
    return note


def seed_everything(seed):
    """voxcpm 2.0.3's generate() takes no seed, so every RNG is set before each call."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % 2**32)
    except ImportError:
        pass
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(text):
    """voxcpm's own normalizer, applied to the spoken words only.

    generate(normalize=True) would also run it over the bracketed lead.
    """
    global NORMALIZER
    if NORMALIZER is None:
        from voxcpm.utils.text_normalize import TextNormalizer
        NORMALIZER = TextNormalizer()
    return NORMALIZER.normalize(text)


def allowed_hosts():
    hosts = set(h.strip() for h in os.environ.get("VOXCPM_AUDIO_HOSTS", "").split(",")) - {""}
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    bucket = os.environ.get("AWS_BUCKET_NAME")
    if endpoint:
        host = urllib.parse.urlsplit(endpoint).hostname
        if host:
            hosts.add(host)
            if bucket:
                hosts.add(f"{bucket}.{host}")
    return {h.lower() for h in hosts}


def download(url, dest):
    # Only our own storage hosts: a clip URL cannot reach the metadata service
    # or an internal network, and redirects are not followed.
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host not in allowed_hosts() or parsed.username or parsed.port not in (None, 443):
        raise ValueError("The reference clip must be stored with the platform first.")
    for address in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM):
        if not ipaddress.ip_address(address[4][0]).is_global:
            raise ValueError("The reference clip host is not public.")
    import requests
    with requests.get(url, stream=True, timeout=(15, 90), allow_redirects=False) as r:
        if r.status_code != 200:
            raise ValueError("The reference clip could not be fetched.")
        size = 0
        with open(dest, "wb") as f:
            for data in r.iter_content(1024 * 1024):
                size += len(data)
                if size > 64 * 1024 * 1024:
                    raise ValueError("The reference clip is too large.")
                f.write(data)


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", *map(str, args)],
                   check=True, timeout=180, capture_output=True)


def prepare_reference(source, work, keep_whole):
    """Decode any mp3/wav to 16 kHz mono PCM, the rate VoxCPM2 encodes at."""
    import soundfile as sf
    import numpy as np
    full = work / "reference-full.wav"
    try:
        ffmpeg("-i", source, "-vn", "-ac", 1, "-ar", 16000, "-c:a", "pcm_s16le", full)
    except subprocess.CalledProcessError:
        raise ValueError("The reference clip is not readable audio.") from None
    info = sf.info(full)
    if info.frames == 0:
        raise ValueError("The reference clip has no audio.")
    if keep_whole:
        if info.duration > PROMPT_REF_MAX_S:
            raise ValueError(f"With prompt_text the clip must be at most {PROMPT_REF_MAX_S:.0f} s "
                             "and the transcript must cover all of it.")
        reference = full
    else:
        reference = work / "reference.wav"
        ffmpeg("-i", full, "-t", REF_MAX_S, "-c:a", "pcm_s16le", reference)
    data, _ = sf.read(reference)
    if data.size == 0 or not np.isfinite(data).all() or float(np.abs(data).max()) < 1e-4:
        raise ValueError("The reference clip is silent or damaged.")
    return reference


def synthesize(spec, work, progress=lambda message: None):
    """Run every chunk in order with one voice and return (audio, sample_rate)."""
    import numpy as np
    import soundfile as sf
    model = engine()
    rate = model.tts_model.sample_rate
    knobs = {"cfg_value": spec["cfg_value"], "inference_timesteps": spec["inference_timesteps"],
             "normalize": False, "denoise": False}
    reference = prompt_wav = prompt_text = None
    lead = spec.get("description") if spec["mode"] == "design" else spec.get("style")
    if spec["mode"] == "clone":
        reference = prepare_reference(work / "import", work, keep_whole=bool(spec["prompt_text"]))
        if spec["prompt_text"]:
            # Ultimate cloning: the same clip as reference and as continuation prompt.
            prompt_wav, prompt_text = reference, spec["prompt_text"]
    pieces = []
    total = len(spec["chunks"])
    for n, chunk in enumerate(spec["chunks"]):
        progress(f"VoxCPM2: part {n + 1} of {total}")
        words = normalize_text(chunk) if spec["normalize"] else chunk
        args = dict(knobs)
        if spec["mode"] == "design" and n > 0:
            # A designed voice changes from run to run. Later parts clone the
            # first part, so one long text keeps one voice.
            args["reference_wav_path"] = str(work / "part0.wav")
            text = words
        else:
            text = with_prefix(lead, words)
            if reference is not None:
                args["reference_wav_path"] = str(reference)
            if prompt_wav is not None:
                args.update(prompt_wav_path=str(prompt_wav), prompt_text=prompt_text)
        seed_everything(spec["seed"])
        audio = np.asarray(model.generate(text=text, **args), dtype=np.float32).reshape(-1)
        if audio.size == 0 or not np.isfinite(audio).all():
            raise RuntimeError("empty or non-finite audio")
        if n == 0 and spec["mode"] == "design":
            sf.write(work / "part0.wav", audio, rate, subtype="PCM_16")
        pieces.append(audio)
    gap = np.zeros(int(round(JOIN_GAP_S * rate)), dtype=np.float32)
    joined = []
    for n, audio in enumerate(pieces):
        if n:
            joined.append(gap)
        joined.append(audio)
    return np.clip(np.concatenate(joined), -1.0, 1.0), rate


class Store:
    """Private B2 outputs with seven-day signed links, as the AuK worker does."""

    def __init__(self):
        import boto3
        self.s3 = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"],
                               region_name=os.environ.get("AWS_REGION", "us-east-005"))
        self.bucket = os.environ["AWS_BUCKET_NAME"]

    def sign(self, key):
        return self.s3.generate_presigned_url("get_object", Params={"Bucket": self.bucket, "Key": key},
                                              ExpiresIn=604800)

    def put(self, wav, mp3):
        key = f"voxcpm/{uuid.uuid4()}"  # never overwrites or escapes the prefix
        for path, suffix, mime in ((wav, ".wav", "audio/wav"), (mp3, ".mp3", "audio/mpeg")):
            self.s3.upload_file(str(path), self.bucket, key + suffix, ExtraArgs={"ContentType": mime})
        return {"key": key + ".mp3", "url": self.sign(key + ".mp3"),
                "wav_key": key + ".wav", "wav_url": self.sign(key + ".wav"), "bytes": mp3.stat().st_size}


def handler(job):
    started = time.monotonic()
    try:
        spec = parse(job.get("input") or {})
        import soundfile as sf
        import runpod
        reset_gpu_peak()
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            if spec["mode"] == "clone":
                download(spec["reference_voice_url"], work / "import")
            audio, rate = synthesize(spec, work, lambda m: runpod.serverless.progress_update(job, m))
            memory = gpu_memory("job")
            wav, mp3 = work / "voxcpm.wav", work / "voxcpm.mp3"
            sf.write(wav, audio, rate, subtype="PCM_16")
            # Keep the model's 48 kHz in the MP3 too (no resample).
            ffmpeg("-i", wav, "-ar", rate, "-c:a", "libmp3lame", "-b:a", "192k", mp3)
            stored = Store().put(wav, mp3)
            return {"url": stored["url"], "duration_seconds": round(len(audio) / rate, 2),
                    "sample_rate": rate, "mode": spec["mode"], "seed": spec["seed"],
                    "engine": "voxcpm2", "model_revision": MODEL_SHA,
                    "key": stored["key"], "wav_key": stored["wav_key"], "wav_url": stored["wav_url"],
                    "bytes": stored["bytes"], "parts": len(spec["chunks"]),
                    "cfg_value": spec["cfg_value"], "inference_timesteps": spec["inference_timesteps"],
                    "processing_ms": int((time.monotonic() - started) * 1000),
                    "model_load_seconds": LOAD_SECONDS, "gpu_memory": memory}
    except Exception as error:
        # Never return URLs, signed queries or storage credentials in errors.
        missing = getattr(error, "name", None) if isinstance(error, ModuleNotFoundError) else None
        if missing and re.fullmatch(r"[a-zA-Z0-9_.]+", missing):
            log.error("VoxCPM2 missing dependency: %s", missing)
            return {"error": "The VoxCPM2 worker is missing a required component. No audio was generated.",
                    "error_code": "missing_dependency", "missing_module": missing}
        log.error("VoxCPM2 failed (%s)", type(error).__name__)
        return {"error": str(error) if isinstance(error, ValueError) else
                f"VoxCPM2 could not finish ({type(error).__name__}). No audio was generated."}


def warm():
    """Load while the worker already listens; a first job waits on the same lock."""
    time.sleep(float(os.environ.get("VOXCPM_BOOT_DELAY", "1")))
    try:
        engine()
    except Exception as error:
        log.warning("VoxCPM2 boot load failed (%s); the first job will retry", type(error).__name__)
        gc.collect()
        torch = _cuda()
        if torch is not None:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import runpod
    if _env_flag("VOXCPM_BOOT_LOAD", "1"):
        threading.Thread(target=warm, name="voxcpm-boot-load", daemon=True).start()
    runpod.serverless.start({"handler": handler})
