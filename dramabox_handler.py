"""DramaBox (ResembleAI/Dramabox) expressive speech from one prose prompt, private B2 outputs.

Same queue-worker shape as voxcpm_handler.py: the model loads once (at boot,
behind a lock) and stays warm; each job writes a WAV and an MP3 to B2 and
returns seven-day signed links. The model is upstream's own warm server
(src/inference_server.py TTSServer, repo pinned at CODE_SHA): Gemma 3 12B
(pre-quantized bnb 4-bit, as upstream ships it), the audio-only DiT with the
DramaBox IC-LoRA merged in, and the LTX-2.3 audio VAE and vocoder.
"""
import gc
import ipaddress
import logging
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import uuid
from pathlib import Path

from dramabox_contract import MAX_SINGLE_TAKE_S, parse

log = logging.getLogger("dramabox-worker")
logging.basicConfig(level=logging.INFO)

MODEL_REPO = "ResembleAI/Dramabox"
MODEL_SHA = "404f967f653fa1170dc15a9d1ddd3fdb9a0a842d"
GEMMA_REPO = "unsloth/gemma-3-12b-it-bnb-4bit"
GEMMA_SHA = "826e729dbaeea4ecb143738eed2bcf3539ebf7bf"
CODE_SHA = "b6120855f60c22c4a795a3e6e1f3abc4f13c0d55"
HF_HUB = "/runpod-volume/huggingface-cache/hub"
DIT_FILE = "dramabox-dit-v1.safetensors"
COMPONENTS_FILE = "dramabox-audio-components.safetensors"
MAX_TOTAL_S = 300.0        # a whole job: at most five minutes of audio
TARGET_CHUNK_S = 37.0      # upstream defaults for long scenes
CROSSFADE_MS = 50.0
ENGINE_NAME = "dramabox"

ENGINE = None
ENGINE_LOCK = threading.Lock()
WATERMARKER = None
LOAD_SECONDS = None


def _env_flag(name, default):
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "off")


def cached_snapshot(repo, sha):
    return Path(HF_HUB) / f"models--{repo.replace('/', '--')}" / "snapshots" / sha


def model_dirs():
    """The RunPod cached-model folders (endpoint modelReferences), else a pinned download."""
    dramabox = Path(os.environ.get("DRAMABOX_MODEL_DIR", cached_snapshot(MODEL_REPO, MODEL_SHA)))
    gemma = Path(os.environ.get("DRAMABOX_GEMMA_DIR", cached_snapshot(GEMMA_REPO, GEMMA_SHA)))
    need = {dramabox: (DIT_FILE, COMPONENTS_FILE), gemma: ("config.json", "tokenizer.model")}
    missing = [d for d, files in need.items() if not all((d / f).is_file() for f in files)]
    if missing:
        if not _env_flag("DRAMABOX_ALLOW_DOWNLOAD", "1"):
            raise ValueError("The DramaBox model cache is not ready. No audio was generated.")
        from huggingface_hub import snapshot_download
        log.warning("DramaBox: model cache missing (%s); downloading the pinned revisions",
                    ", ".join(str(d) for d in missing))
        if dramabox in missing:
            dramabox = Path(snapshot_download(MODEL_REPO, revision=MODEL_SHA,
                                              allow_patterns=[DIT_FILE, COMPONENTS_FILE, "LICENSE"]))
        if gemma in missing:
            gemma = Path(snapshot_download(GEMMA_REPO, revision=GEMMA_SHA,
                                           allow_patterns=["*.json", "*.safetensors", "tokenizer.model"]))
    return dramabox, gemma


class DramaBox:
    """Upstream's TTSServer plus its own duration estimate."""

    def __init__(self, server, estimate):
        self.server = server
        self.estimate = estimate

    def generate(self, prompt, **kwargs):
        return self.server.generate(prompt, **kwargs)

    def generate_long(self, prompt, **kwargs):
        return self.server.generate_long(prompt, **kwargs)


def engine():
    global ENGINE
    with ENGINE_LOCK:
        if ENGINE is None:
            ENGINE = load_engine()
    return ENGINE


def load_engine():
    global LOAD_SECONDS
    dramabox, gemma = model_dirs()
    upstream = os.environ.get("DRAMABOX_CODE_DIR", "/app/upstream")
    for sub in ("src", "ltx2"):
        if str(Path(upstream) / sub) not in sys.path:
            sys.path.insert(0, str(Path(upstream) / sub))
    from inference_server import TTSServer, estimate_duration
    started = time.monotonic()
    # bf16 as upstream's app runs it; torch.compile off by default (slow first
    # call, recompiles per length); DRAMABOX_COMPILE=1 turns it on.
    server = TTSServer(checkpoint=str(dramabox / DIT_FILE), full_checkpoint=str(dramabox / COMPONENTS_FILE),
                       gemma_root=str(gemma), device="cuda", dtype="bf16",
                       compile_model=_env_flag("DRAMABOX_COMPILE", "0"), bnb_4bit=True)
    LOAD_SECONDS = round(time.monotonic() - started, 1)
    torch = _cuda()
    log.info("DramaBox loaded in %.1f s; %.2f GiB on the GPU", LOAD_SECONDS,
             torch.cuda.memory_allocated() / 1024**3 if torch else 0.0)
    return DramaBox(server, estimate_duration)


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
    log.info("DramaBox GPU memory after %s: peak %.2f GiB allocated, %.2f GiB reserved, %.2f of %.2f GiB free (%s)",
             label, note["peak_allocated_gib"], note["peak_reserved_gib"], note["free_gib"], note["total_gib"],
             note["device"])
    return note


def allowed_hosts():
    hosts = set(h.strip() for h in os.environ.get("DRAMABOX_AUDIO_HOSTS", "").split(",")) - {""}
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


def prepare_reference(source, work, seconds):
    """Decode any mp3/wav to a 48 kHz stereo PCM WAV of the clip's first `seconds`.

    Upstream reads the clip with PyAV, loops a short clip up to the reference
    length and resamples to the VAE's rate itself. A clip shorter than the
    asked length is used whole instead of looped. Returns (path, seconds used).
    """
    import numpy as np
    import soundfile as sf
    reference = work / "reference.wav"
    try:
        ffmpeg("-i", source, "-vn", "-t", seconds, "-ac", 2, "-ar", 48000, "-c:a", "pcm_s16le", reference)
    except subprocess.CalledProcessError:
        raise ValueError("The reference clip is not readable audio.") from None
    info = sf.info(reference)
    if info.frames == 0:
        raise ValueError("The reference clip has no audio.")
    if info.duration < 3.0:
        raise ValueError("The reference clip must be at least 3 seconds long.")
    data, _ = sf.read(reference)
    if data.size == 0 or not np.isfinite(data).all() or float(np.abs(data).max()) < 1e-4:
        raise ValueError("The reference clip is silent or damaged.")
    return reference, round(min(float(seconds), info.duration), 2)


def watermark(audio, rate):
    """Resemble Perth, the way upstream's generate_to_file applies it: on the mono mix."""
    global WATERMARKER
    import numpy as np
    import perth
    if WATERMARKER is None:
        if getattr(perth, "PerthImplicitWatermarker", None) is None:  # upstream would skip silently
            raise RuntimeError("Perth watermarker unavailable")
        WATERMARKER = perth.PerthImplicitWatermarker()
    mono = audio.mean(axis=0) if audio.shape[0] > 1 else audio[0]
    marked = np.asarray(WATERMARKER.apply_watermark(mono, sample_rate=rate), dtype=np.float32).reshape(1, -1)
    return np.repeat(marked, audio.shape[0], axis=0)


def _numpy(waveform):
    import numpy as np
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().float().cpu().numpy()
    audio = np.asarray(waveform, dtype=np.float32)
    while audio.ndim > 2:
        audio = audio[0]
    if audio.ndim == 1:
        audio = audio[None, :]
    return audio  # (channels, samples)


def synthesize(spec, work, progress=lambda message: None):
    """Returns (audio as (channels, samples), sample_rate, parts, target_seconds, reference_seconds)."""
    import numpy as np
    model = engine()
    reference, ref_seconds = None, None
    if spec["mode"] == "clone":
        reference, ref_seconds = prepare_reference(work / "import", work, spec["reference_seconds"])
    estimate = spec["duration"] or model.estimate(spec["prompt"], spec["duration_multiplier"])
    if estimate > MAX_TOTAL_S:
        raise ValueError(f"The scene is too long: about {estimate:.0f} s of audio, at most {MAX_TOTAL_S:.0f} s.")
    args = {"voice_ref": str(reference) if reference else None, "cfg_scale": spec["cfg_scale"],
            "stg_scale": spec["stg_scale"], "duration_multiplier": spec["duration_multiplier"],
            "seed": spec["seed"], "ref_duration": ref_seconds or spec["reference_seconds"],
            "rescale_scale": spec["rescale_scale"], "denoise_ref": False}
    parts = [1]
    if estimate > MAX_SINGLE_TAKE_S:
        def on_chunk(idx, total, est):
            parts[0] = total
            progress(f"DramaBox: part {idx + 1} of {total}")
        waveform, rate = model.generate_long(spec["prompt"], max_chunk_duration=MAX_SINGLE_TAKE_S,
                                             target_chunk_duration=TARGET_CHUNK_S, crossfade_ms=CROSSFADE_MS,
                                             progress_callback=on_chunk, **args)
    else:
        progress("DramaBox: rendering")
        waveform, rate = model.generate(spec["prompt"], gen_duration=spec["duration"], **args)
    audio = _numpy(waveform)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise RuntimeError("empty or non-finite audio")
    if spec["watermark"]:
        audio = watermark(audio, int(rate))
    return np.clip(audio, -1.0, 1.0), int(rate), parts[0], round(float(estimate), 1), ref_seconds


class Store:
    """Private B2 outputs with seven-day signed links, as the VoxCPM2 worker does."""

    def __init__(self):
        import boto3
        self.s3 = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"],
                               region_name=os.environ.get("AWS_REGION", "us-east-005"))
        self.bucket = os.environ["AWS_BUCKET_NAME"]

    def sign(self, key):
        return self.s3.generate_presigned_url("get_object", Params={"Bucket": self.bucket, "Key": key},
                                              ExpiresIn=604800)

    def put(self, wav, mp3):
        key = f"dramabox/{uuid.uuid4()}"  # never overwrites or escapes the prefix
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
                download(spec["reference_url"], work / "import")
            render_started = time.monotonic()
            audio, rate, parts, target, ref_seconds = synthesize(
                spec, work, lambda m: runpod.serverless.progress_update(job, m))
            render_ms = int((time.monotonic() - render_started) * 1000)
            memory = gpu_memory("job")
            wav, mp3 = work / "dramabox.wav", work / "dramabox.mp3"
            sf.write(wav, audio.T, rate, subtype="PCM_16")
            # Keep the model's own rate in the MP3 too (no resample).
            ffmpeg("-i", wav, "-ar", rate, "-c:a", "libmp3lame", "-b:a", "192k", mp3)
            stored = Store().put(wav, mp3)
            return {"url": stored["url"], "duration_seconds": round(audio.shape[1] / rate, 2),
                    "sample_rate": rate, "channels": int(audio.shape[0]), "mode": spec["mode"],
                    "seed": spec["seed"], "engine": ENGINE_NAME,
                    "model_revision": MODEL_SHA, "gemma_revision": GEMMA_SHA, "code_revision": CODE_SHA,
                    "key": stored["key"], "wav_key": stored["wav_key"], "wav_url": stored["wav_url"],
                    "bytes": stored["bytes"], "parts": parts, "target_seconds": target,
                    "reference_seconds": ref_seconds, "watermark": spec["watermark"],
                    "cfg_scale": spec["cfg_scale"], "stg_scale": spec["stg_scale"],
                    "rescale_scale": spec["rescale_scale"], "duration_multiplier": spec["duration_multiplier"],
                    "steps": 30, "render_ms": render_ms,
                    "processing_ms": int((time.monotonic() - started) * 1000),
                    "model_load_seconds": LOAD_SECONDS, "gpu_memory": memory}
    except Exception as error:
        # Never return URLs, signed queries or storage credentials in errors.
        missing = getattr(error, "name", None) if isinstance(error, ModuleNotFoundError) else None
        if missing and re.fullmatch(r"[a-zA-Z0-9_.]+", missing):
            log.error("DramaBox missing dependency: %s", missing)
            return {"error": "The DramaBox worker is missing a required component. No audio was generated.",
                    "error_code": "missing_dependency", "missing_module": missing}
        log.error("DramaBox failed (%s: %s)", type(error).__name__,
                  re.sub(r"https?://\S+", "<url>", str(error))[:300])
        return {"error": str(error) if isinstance(error, ValueError) else
                f"DramaBox could not finish ({type(error).__name__}). No audio was generated."}


def warm():
    """Load while the worker already listens; a first job waits on the same lock."""
    time.sleep(float(os.environ.get("DRAMABOX_BOOT_DELAY", "1")))
    try:
        engine()
    except Exception as error:
        log.warning("DramaBox boot load failed (%s: %s); the first job will retry",
                    type(error).__name__, str(error)[:300])
        gc.collect()
        torch = _cuda()
        if torch is not None:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import runpod
    if _env_flag("DRAMABOX_BOOT_LOAD", "1"):
        threading.Thread(target=warm, name="dramabox-boot-load", daemon=True).start()
    runpod.serverless.start({"handler": handler})
