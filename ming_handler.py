"""Ming-3.0-tts-tiny (inclusionAI) voice design and cloning, private B2 outputs.

Same queue-worker shape as voxcpm_handler.py: the model loads once (at boot,
behind a lock) and stays warm; each job writes a 44.1 kHz WAV and an MP3 to
B2 and returns seven-day signed links. Ming's own generate() splits long text
at clause punctuation and, for designed voices, carries the first fragment's
voice into the rest, so one call covers the whole text.
"""
import gc
import ipaddress
import logging
import os
import re
import socket
import subprocess
import tempfile
import threading
import time
import urllib.parse
import uuid
from pathlib import Path

from ming_contract import parse

log = logging.getLogger("ming-worker")
logging.basicConfig(level=logging.INFO)

MODEL_SHA = "4715ef5a2f02901f75687d538b6920fd0dd11e6b"
CODE_SHA = "a39b80660bdc5a4394d5e1d41af6dbfd9c443a54"
DEFAULT_MODEL_DIR = ("/runpod-volume/huggingface-cache/hub/models--inclusionAI--Ming-3.0-tts-tiny/snapshots/"
                     + MODEL_SHA)
REF_MAX_S = 20.0          # clip without a transcript: its first 20 s
PROMPT_REF_MAX_S = 30.0   # with prompt_text the transcript must cover the whole clip
ENGINE_NAME = "ming-3.0-tts-tiny"

ENGINE = None
ENGINE_LOCK = threading.Lock()
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
    model_dir = Path(os.environ.get("MING_MODEL_DIR", DEFAULT_MODEL_DIR))
    if not (model_dir / "config.json").is_file() or not (model_dir / "model.safetensors.index.json").is_file():
        raise ValueError("The Ming model cache is not ready. No audio was generated.")
    from ming_tiny import MingTinyTTS
    started = time.monotonic()
    # Release runtime: bf16, portable MoE loop, fp32 recurrent state. The
    # loader checks the checkpoint against that contract and refuses otherwise.
    model = MingTinyTTS.from_pretrained(str(model_dir), device="cuda:0")
    # Load the FST text normalizer now, not on the first job.
    model.normalizer.normalize("Warm up 3 times at 5:30 pm.", language="en")
    LOAD_SECONDS = round(time.monotonic() - started, 1)
    torch = _cuda()
    log.info("Ming loaded in %.1f s; %.2f GiB on the GPU; output %d Hz", LOAD_SECONDS,
             torch.cuda.memory_allocated() / 1024**3 if torch else 0.0, model.sample_rate)
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
    log.info("Ming GPU memory after %s: peak %.2f GiB allocated, %.2f GiB reserved, %.2f of %.2f GiB free (%s)",
             label, note["peak_allocated_gib"], note["peak_reserved_gib"], note["free_gib"], note["total_gib"],
             note["device"])
    return note


def allowed_hosts():
    hosts = set(h.strip() for h in os.environ.get("MING_AUDIO_HOSTS", "").split(",")) - {""}
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


def prepare_reference(source, work, keep_whole, rate=44100):
    """Decode any mp3/wav to mono PCM WAV at Ming's 44.1 kHz.

    Ming reads references with soundfile and resamples itself; a WAV at its
    own rate skips that step.
    """
    import numpy as np
    import soundfile as sf
    full = work / "reference-full.wav"
    try:
        ffmpeg("-i", source, "-vn", "-ac", 1, "-ar", rate, "-c:a", "pcm_s16le", full)
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
    """One Ming generate() call for the whole text; returns (audio, rate, parts)."""
    import numpy as np
    model = engine()
    args = dict(spec["args"])
    if spec["mode"] == "clone":
        args["prompt_wav"] = str(prepare_reference(work / "import", work, keep_whole=spec["uses_prompt_text"]))
    progress(f"Ming: {spec['task']}")
    model.text_normalization = spec["normalize"]
    output = model.generate(task=spec["task"], text=spec["text"], seed=spec["seed"],
                            cfg_scale=spec["cfg_scale"], flow_steps=spec["flow_steps"], **args)
    waveform = output.waveform
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().float().cpu().numpy()
    audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise RuntimeError("empty or non-finite audio")
    parts = len(getattr(output, "segments", None) or []) or 1
    return np.clip(audio, -1.0, 1.0), int(model.sample_rate), parts


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
        key = f"ming/{uuid.uuid4()}"  # never overwrites or escapes the prefix
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
            audio, rate, parts = synthesize(spec, work, lambda m: runpod.serverless.progress_update(job, m))
            memory = gpu_memory("job")
            wav, mp3 = work / "ming.wav", work / "ming.mp3"
            sf.write(wav, audio, rate, subtype="PCM_16")
            # Keep the model's 44.1 kHz in the MP3 too (no resample).
            ffmpeg("-i", wav, "-ar", rate, "-c:a", "libmp3lame", "-b:a", "192k", mp3)
            stored = Store().put(wav, mp3)
            return {"url": stored["url"], "duration_seconds": round(len(audio) / rate, 2),
                    "sample_rate": rate, "mode": spec["mode"], "seed": spec["seed"],
                    "engine": ENGINE_NAME, "task": spec["task"], "sound_tags": spec["sound_tags"],
                    "uses_prompt_text": spec.get("uses_prompt_text", False),
                    "model_revision": MODEL_SHA, "code_revision": CODE_SHA,
                    "key": stored["key"], "wav_key": stored["wav_key"], "wav_url": stored["wav_url"],
                    "bytes": stored["bytes"], "parts": parts,
                    "cfg_scale": spec["cfg_scale"], "flow_steps": spec["flow_steps"],
                    "normalize": spec["normalize"],
                    "processing_ms": int((time.monotonic() - started) * 1000),
                    "model_load_seconds": LOAD_SECONDS, "gpu_memory": memory}
    except Exception as error:
        # Never return URLs, signed queries or storage credentials in errors.
        missing = getattr(error, "name", None) if isinstance(error, ModuleNotFoundError) else None
        if missing and re.fullmatch(r"[a-zA-Z0-9_.]+", missing):
            log.error("Ming missing dependency: %s", missing)
            return {"error": "The Ming worker is missing a required component. No audio was generated.",
                    "error_code": "missing_dependency", "missing_module": missing}
        log.error("Ming failed (%s: %s)", type(error).__name__, re.sub(r"https?://\S+", "<url>", str(error))[:300])
        return {"error": str(error) if isinstance(error, ValueError) else
                f"Ming could not finish ({type(error).__name__}). No audio was generated."}


def warm():
    """Load while the worker already listens; a first job waits on the same lock."""
    time.sleep(float(os.environ.get("MING_BOOT_DELAY", "1")))
    try:
        engine()
    except Exception as error:
        log.warning("Ming boot load failed (%s: %s); the first job will retry",
                    type(error).__name__, str(error)[:300])
        gc.collect()
        torch = _cuda()
        if torch is not None:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import runpod
    if _env_flag("MING_BOOT_LOAD", "1"):
        threading.Thread(target=warm, name="ming-boot-load", daemon=True).start()
    runpod.serverless.start({"handler": handler})
