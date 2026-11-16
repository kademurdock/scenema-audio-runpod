"""Resemble AI Chatterbox (Turbo, original, Multilingual V3) TTS, private B2 outputs.

Queue worker in the same style as voxcpm_handler.py. Each model loads the
first time a job asks for it (CHATTERBOX_BOOT_MODELS are loaded at boot) and
stays resident; all three fit together on a 24 GB card. Weights come from the
RunPod model cache (endpoint modelReferences), pinned to one revision per
repo; only when a cached snapshot is missing are its files fetched from
Hugging Face at that same revision. Each job writes a 24 kHz WAV and an MP3 to
B2 and returns seven-day signed links.

Every take carries upstream's Perth watermark: generate() always applies it.
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

from chatterbox_contract import JobError, parse

log = logging.getLogger("chatterbox-worker")
logging.basicConfig(level=logging.INFO)

CODE_SHA = "ab55923c27757c90cfcc407baa57dc25a7d25217"  # resemble-ai/chatterbox
TURBO_REPO, TURBO_SHA = "ResembleAI/chatterbox-turbo", "749d1c1a46eb10492095d68fbcf55691ccf137cd"
BASE_REPO, BASE_SHA = "ResembleAI/chatterbox", "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"
MULTILINGUAL_T3 = "t3_mtl23ls_v3.safetensors"
# The files each model reads (from_local in upstream's tts_turbo.py, tts.py, mtl_tts.py).
VARIANTS = {
    "turbo": {"repo": TURBO_REPO, "sha": TURBO_SHA, "env": "CHATTERBOX_TURBO_DIR",
              "files": ("ve.safetensors", "t3_turbo_v1.safetensors", "s3gen_meanflow.safetensors", "conds.pt",
                        "vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json",
                        "added_tokens.json")},
    "original": {"repo": BASE_REPO, "sha": BASE_SHA, "env": "CHATTERBOX_DIR",
                 "files": ("ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json",
                           "conds.pt")},
    "multilingual": {"repo": BASE_REPO, "sha": BASE_SHA, "env": "CHATTERBOX_DIR",
                     "files": ("ve.pt", MULTILINGUAL_T3, "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json",
                               "conds.pt", "Cangjie5_TC.json")},
}
JOIN_GAP_S = 0.15
REF_MAX_S = 20.0          # upstream reads at most the first 15 s of a clip
TURBO_REF_MIN_S = 5.0     # Turbo's prepare_conditionals asserts a clip longer than 5 s
REF_MIN_S = 1.0
ENGINE_NAME = "chatterbox"

ENGINES = {}
BUILTIN = {}
LOAD_SECONDS = {}
LOAD_LOCK = threading.Lock()
GEN_LOCK = threading.Lock()


def _env_flag(name, default):
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "off")


def device():
    return os.environ.get("CHATTERBOX_DEVICE", "cuda")


def default_dir(variant):
    info = VARIANTS[variant]
    org, name = info["repo"].split("/")
    return f"{CACHE_ROOT}/models--{org}--{name}/snapshots/{info['sha']}"


def model_dir(variant):
    """The cached snapshot, or the same revision fetched from Hugging Face if it is missing."""
    info = VARIANTS[variant]
    path = Path(os.environ.get(info["env"]) or default_dir(variant))
    if all((path / name).is_file() for name in info["files"]):
        return path
    if not _env_flag("CHATTERBOX_HF_FALLBACK", "1"):
        raise JobError(f"The Chatterbox {variant} model cache is not ready. No audio was generated.")
    log.warning("Chatterbox %s: cached snapshot missing; downloading %s@%s", variant, info["repo"], info["sha"][:8])
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(repo_id=info["repo"], revision=info["sha"], allow_patterns=list(info["files"]),
                                  token=os.environ.get("HF_TOKEN") or None))
    missing = [name for name in info["files"] if not (path / name).is_file()]
    if missing:
        raise JobError(f"The Chatterbox {variant} model files are incomplete. No audio was generated.")
    return path


def _local_cangjie(path):
    """Upstream's Chinese tokenizer calls hf_hub_download(cache_dir=<snapshot>),
    which looks for a cache layout inside the snapshot and fails (or downloads
    the latest revision). Serve the pinned snapshot's own file instead."""
    import chatterbox.models.tokenizers.tokenizer as tok
    original = tok.hf_hub_download

    def local(repo_id=None, filename=None, cache_dir=None, **kwargs):
        if filename == "Cangjie5_TC.json" and (path / filename).is_file():
            return str(path / filename)
        return original(repo_id=repo_id, filename=filename, cache_dir=cache_dir, **kwargs)

    tok.hf_hub_download = local


def load_variant(variant):
    path = model_dir(variant)
    started = time.monotonic()
    if variant == "turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        model = ChatterboxTurboTTS.from_local(path, device())
    elif variant == "original":
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_local(path, device())
    else:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        _local_cangjie(path)
        model = ChatterboxMultilingualTTS.from_local(path, device(), t3_model=MULTILINGUAL_T3)
    if model.conds is None:
        raise RuntimeError("built-in voice missing")
    LOAD_SECONDS[variant] = round(time.monotonic() - started, 1)
    torch = _cuda()
    log.info("Chatterbox %s loaded in %.1f s; %.2f GiB on the GPU; output %d Hz", variant, LOAD_SECONDS[variant],
             torch.cuda.memory_allocated() / 1024**3 if torch else 0.0, model.sr)
    return model


def engine(variant):
    with LOAD_LOCK:
        if variant not in ENGINES:
            model = load_variant(variant)
            # generate() and prepare_conditionals() replace model.conds; the
            # built-in voice is kept here so design mode can return to it.
            BUILTIN[variant] = model.conds
            ENGINES[variant] = model
    return ENGINES[variant]


def builtin_voice(variant):
    conds = BUILTIN[variant]
    # A new Conditionals object: generate() reassigns .t3 on the one it holds.
    return type(conds)(conds.t3, conds.gen)


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
            "allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "free_gib": round(free / 1024**3, 2), "total_gib": round(total / 1024**3, 2),
            "device": torch.cuda.get_device_name(0)}
    log.info("Chatterbox GPU memory after %s: peak %.2f GiB allocated, %.2f GiB reserved, %.2f of %.2f GiB free (%s)",
             label, note["peak_allocated_gib"], note["peak_reserved_gib"], note["free_gib"], note["total_gib"],
             note["device"])
    return note


def seed_everything(seed):
    """Chatterbox's generate() takes no seed, so every RNG is set before each call."""
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


def allowed_hosts():
    hosts = set(h.strip() for h in os.environ.get("CHATTERBOX_AUDIO_HOSTS", "").split(",")) - {""}
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
        raise JobError("The reference clip must be stored with the platform first.")
    for address in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM):
        if not ipaddress.ip_address(address[4][0]).is_global:
            raise JobError("The reference clip host is not public.")
    import requests
    with requests.get(url, stream=True, timeout=(15, 90), allow_redirects=False) as r:
        if r.status_code != 200:
            raise JobError("The reference clip could not be fetched.")
        size = 0
        with open(dest, "wb") as f:
            for data in r.iter_content(1024 * 1024):
                size += len(data)
                if size > 64 * 1024 * 1024:
                    raise JobError("The reference clip is too large.")
                f.write(data)


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", *map(str, args)],
                   check=True, timeout=180, capture_output=True)


def prepare_reference(source, work, variant, rate=24000):
    """Decode any mp3/wav to mono 24 kHz PCM (the rate Chatterbox loads clips at), first 20 s."""
    import numpy as np
    import soundfile as sf
    reference = work / "reference.wav"
    try:
        ffmpeg("-i", source, "-vn", "-ac", 1, "-ar", rate, "-t", REF_MAX_S, "-c:a", "pcm_s16le", reference)
    except subprocess.CalledProcessError:
        raise JobError("The reference clip is not readable audio.") from None
    info = sf.info(reference)
    if info.frames == 0:
        raise JobError("The reference clip has no audio.")
    shortest = TURBO_REF_MIN_S if variant == "turbo" else REF_MIN_S
    if info.duration <= shortest:
        raise JobError(f"The reference clip must be longer than {shortest:.0f} seconds for model {variant} "
                         "(5 to 10 seconds of clear speech is best).")
    data, _ = sf.read(reference)
    if data.size == 0 or not np.isfinite(data).all() or float(np.abs(data).max()) < 1e-4:
        raise JobError("The reference clip is silent or damaged.")
    return reference


def generate_args(spec):
    if spec["model"] == "turbo":
        return {"temperature": spec["temperature"]}
    args = {"exaggeration": spec["exaggeration"], "cfg_weight": spec["cfg_weight"],
            "temperature": spec["temperature"]}
    if spec["model"] == "multilingual":
        args["language_id"] = spec["language_id"]
    return args


def synthesize(spec, work, progress=lambda message: None):
    """Run every chunk in order with one voice and return (audio, sample_rate)."""
    import numpy as np
    variant = spec["model"]
    model = engine(variant)
    rate = int(model.sr)
    args = generate_args(spec)
    reference = prepare_reference(work / "import", work, variant) if spec["mode"] == "clone" else None
    pieces = []
    total = len(spec["chunks"])
    with GEN_LOCK:
        # The voice is set once: every part reuses the same conditionals.
        if reference is not None:
            if variant == "turbo":
                model.prepare_conditionals(str(reference))
            else:
                model.prepare_conditionals(str(reference), exaggeration=spec["exaggeration"])
        else:
            model.conds = builtin_voice(variant)
        for n, chunk in enumerate(spec["chunks"]):
            progress(f"Chatterbox {variant}: part {n + 1} of {total}")
            seed_everything(spec["seed"])
            wav = model.generate(chunk, **args)
            if hasattr(wav, "detach"):
                wav = wav.detach().float().cpu().numpy()
            audio = np.asarray(wav, dtype=np.float32).reshape(-1)
            if audio.size == 0 or not np.isfinite(audio).all():
                raise RuntimeError("empty or non-finite audio")
            pieces.append(audio)
    gap = np.zeros(int(round(JOIN_GAP_S * rate)), dtype=np.float32)
    joined = []
    for n, audio in enumerate(pieces):
        if n:
            joined.append(gap)
        joined.append(audio)
    return np.clip(np.concatenate(joined), -1.0, 1.0), rate


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
        key = f"chatterbox/{uuid.uuid4()}"  # never overwrites or escapes the prefix
        for path, suffix, mime in ((wav, ".wav", "audio/wav"), (mp3, ".mp3", "audio/mpeg")):
            self.s3.upload_file(str(path), self.bucket, key + suffix, ExtraArgs={"ContentType": mime})
        return {"key": key + ".mp3", "url": self.sign(key + ".mp3"),
                "wav_key": key + ".wav", "wav_url": self.sign(key + ".wav"), "bytes": mp3.stat().st_size}


def scrub(text):
    return re.sub(r"https?://\S+", "<url>", text)


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
            audio, rate = synthesize(spec, work, lambda m: runpod.serverless.progress_update(job, m))
            memory = gpu_memory("job")
            wav, mp3 = work / "chatterbox.wav", work / "chatterbox.mp3"
            sf.write(wav, audio, rate, subtype="PCM_16")
            # Keep the model's 24 kHz in the MP3 too (no resample); 160k is
            # the top MPEG-2 Layer III rate at 24 kHz.
            ffmpeg("-i", wav, "-ar", rate, "-c:a", "libmp3lame", "-b:a", "160k", mp3)
            stored = Store().put(wav, mp3)
            info = VARIANTS[spec["model"]]
            return {"url": stored["url"], "duration_seconds": round(len(audio) / rate, 2),
                    "sample_rate": rate, "mode": spec["mode"], "seed": spec["seed"],
                    "engine": ENGINE_NAME, "model": spec["model"],
                    "model_repo": info["repo"], "model_revision": info["sha"], "code_revision": CODE_SHA,
                    "key": stored["key"], "wav_key": stored["wav_key"], "wav_url": stored["wav_url"],
                    "bytes": stored["bytes"], "parts": len(spec["chunks"]),
                    "temperature": spec["temperature"], "exaggeration": spec["exaggeration"],
                    "cfg_weight": spec["cfg_weight"], "language_id": spec["language_id"],
                    "sound_tags": spec["sound_tags"], "ignored_fields": spec["ignored"], "watermark": "perth",
                    "processing_ms": int((time.monotonic() - started) * 1000),
                    "model_load_seconds": LOAD_SECONDS.get(spec["model"]),
                    "loaded_models": sorted(ENGINES), "gpu_memory": memory}
    except Exception as error:
        # Never return URLs, signed queries or storage credentials in errors.
        missing = getattr(error, "name", None) if isinstance(error, ModuleNotFoundError) else None
        if missing and re.fullmatch(r"[a-zA-Z0-9_.]+", missing):
            log.error("Chatterbox missing dependency: %s", missing)
            return {"error": "The Chatterbox worker is missing a required component. No audio was generated.",
                    "error_code": "missing_dependency", "missing_module": missing}
        log.error("Chatterbox failed (%s: %s)", type(error).__name__, scrub(str(error))[:300])
        # Only the worker's own plain-words errors go back as written. Library
        # errors (requests' InvalidURL is a ValueError too) may quote a URL or a path.
        return {"error": scrub(str(error)) if isinstance(error, JobError) else
                f"Chatterbox could not finish ({type(error).__name__}). No audio was generated."}


def warm():
    """Load the boot models while the worker already listens; a first job waits on the same lock."""
    time.sleep(float(os.environ.get("CHATTERBOX_BOOT_DELAY", "1")))
    names = os.environ.get("CHATTERBOX_BOOT_MODELS", "turbo").strip().lower()
    names = list(VARIANTS) if names == "all" else [n.strip() for n in names.split(",") if n.strip() in VARIANTS]
    for variant in names:
        try:
            engine(variant)
        except Exception as error:
            log.warning("Chatterbox %s boot load failed (%s: %s); the first job will retry",
                        variant, type(error).__name__, str(error)[:300])
            gc.collect()
            torch = _cuda()
            if torch is not None:
                torch.cuda.empty_cache()
    gpu_memory("boot load")


if __name__ == "__main__":
    import runpod
    threading.Thread(target=warm, name="chatterbox-boot-load", daemon=True).start()
    runpod.serverless.start({"handler": handler})
