# Scenema Audio — RunPod Serverless handler (Kade-AI, Part 119.9, Sep 3 2026)
#
# Queue-based worker. Loads the upstream AudioProcessor once per worker, then
# each job: {prompt (Scenema <speak> XML), mode?, reference_voice_url?,
# background_sfx?, validate?, seed?, pace?, min_match_ratio?, skip_vc?,
# vc_steps?, vc_cfg_rate?, keep_wav?, out_prefix?}
# -> WAV from the model -> MP3 192k (ffmpeg) -> B2 (S3 API) -> {url, key,
# duration_s, seed, processing_ms, bytes}. Audio never rides RunPod's
# response payload (30 MB cap) and errors come back as {"error": ...} so a
# bad prompt is never retried for twenty minutes.
import asyncio, base64, json, logging, os, subprocess, tempfile, time, uuid
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("scenema-runpod")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402
import boto3  # noqa: E402
import runpod  # noqa: E402

HF_REPO = "ScenemaAI/scenema-audio"
GEMMA_REPO = os.environ.get("GEMMA_REPO", "unsloth/gemma-3-12b-it")   # ungated mirror of google/gemma-3-12b-it
SEEDVC_REPO = "Plachta/Seed-VC"
BIGVGAN_REPO = "nvidia/bigvgan_v2_22khz_80band_256x"
WHISPER_REPO = "openai/whisper-small"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/runpod-volume/scenema"))
TOKEN = os.environ.get("HF_TOKEN") or None   # None everywhere: every repo below is ungated


def download_models():
    """Upstream server.py's _download_models, with two changes: the audio
    checkpoint is whichever file AUDIO_CKPT names (bf16 or INT8), and Gemma
    comes from GEMMA_REPO."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    audio_ckpt = Path(os.environ.get("AUDIO_CKPT", str(MODEL_DIR / "scenema-audio-transformer.safetensors")))
    if not audio_ckpt.exists():
        log.info("downloading audio transformer %s", audio_ckpt.name)
        hf_hub_download(HF_REPO, audio_ckpt.name, local_dir=str(audio_ckpt.parent), token=TOKEN)
    pipeline_ckpt = Path(os.environ.get("PIPELINE_CKPT", str(MODEL_DIR / "scenema-audio-pipeline.safetensors")))
    if not pipeline_ckpt.exists():
        log.info("downloading pipeline checkpoint (~6.7 GB)")
        hf_hub_download(HF_REPO, "scenema-audio-pipeline.safetensors", local_dir=str(pipeline_ckpt.parent), token=TOKEN)
    vae_ckpt = Path(os.environ.get("VAE_ENCODER_CKPT", str(MODEL_DIR / "scenema-audio-vae-encoder.safetensors")))
    if not vae_ckpt.exists():
        hf_hub_download(HF_REPO, "scenema-audio-vae-encoder.safetensors", local_dir=str(vae_ckpt.parent), token=TOKEN)
    gemma_root = Path(os.environ.get("GEMMA_ROOT", str(MODEL_DIR / "gemma-3-12b-it")))
    if not gemma_root.exists() or not any(gemma_root.glob("*.safetensors")):
        log.info("downloading Gemma 3 12B IT from %s (~24 GB, ungated)", GEMMA_REPO)
        snapshot_download(GEMMA_REPO, local_dir=str(gemma_root), ignore_patterns=["*.gguf"], token=TOKEN)
    seedvc_path = Path(os.environ.get("SEEDVC_PATH", "/app/seed-vc"))
    seedvc_cache = seedvc_path / "checkpoints"
    if not seedvc_cache.exists() or not any(seedvc_cache.glob("*.pth")):
        log.info("downloading SeedVC checkpoints (~1.6 GB)")
        hf_cache = seedvc_cache / "hf_cache"
        hf_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HUB_CACHE"] = str(hf_cache)
        hf_hub_download(SEEDVC_REPO, "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth", local_dir=str(seedvc_cache), token=TOKEN)
        hf_hub_download(SEEDVC_REPO, "config_dit_mel_seed_uvit_whisper_small_wavenet.yml", local_dir=str(seedvc_cache), token=TOKEN)
        snapshot_download(BIGVGAN_REPO, local_dir=str(hf_cache / "bigvgan"))
        snapshot_download(WHISPER_REPO, local_dir=str(hf_cache / "whisper-small"))


t0 = time.time()
download_models()
log.info("models present after %.0fs", time.time() - t0)

from audio_core.processor import AudioProcessor  # noqa: E402
from common.handlers.base import ProcessJob  # noqa: E402

processor = AudioProcessor()
processor.startup()
log.info("processor ready after %.0fs total", time.time() - t0)

# B2 via its S3 endpoint. Bucket is private; the bridge presigns for playback.
S3 = boto3.client(
    "s3",
    endpoint_url=os.environ["AWS_ENDPOINT_URL"],
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ.get("AWS_REGION", "us-east-005"),
)
BUCKET = os.environ["AWS_BUCKET_NAME"]
PREFIX = os.environ.get("B2_PREFIX", "scenema/")

ALLOWED = {"prompt", "mode", "reference_voice_url", "background_sfx", "validate", "seed",
           "pace", "min_match_ratio", "skip_vc", "vc_steps", "vc_cfg_rate"}


def probe_duration(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "json", path], capture_output=True, text=True, timeout=60)
        return round(float(json.loads(out.stdout)["format"]["duration"]), 2)
    except Exception:
        return None


async def handler(job):
    # async: RunPod runs handlers inside its own event loop (asyncio.run() is illegal there —
    # the first live job said so, Part 119.9, 04:18Z).
    inp = job.get("input") or {}
    prompt = inp.get("prompt")
    if not isinstance(prompt, str) or "<speak" not in prompt:
        return {"error": "input.prompt must be Scenema <speak ...> XML"}
    model_input = {k: v for k, v in inp.items() if k in ALLOWED}
    model_input.setdefault("validate", True)
    model_input.setdefault("vc_steps", 35)
    keep_wav = bool(inp.get("keep_wav"))
    out_prefix = str(inp.get("out_prefix") or "").strip("/")
    job_id = str(job.get("id") or uuid.uuid4())

    t1 = time.time()
    try:
        result = await processor.process(ProcessJob(job_id=job_id, input=model_input))
    except Exception as e:  # never raise: RunPod would retry a long job
        log.exception("processor failed")
        return {"error": f"processor: {e}"}
    if not result.success or not result.output or not result.output.data:
        return {"error": result.error or "generation failed"}
    processing_ms = int((time.time() - t1) * 1000)

    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "out.wav")
        mp3 = os.path.join(td, "out.mp3")
        with open(wav, "wb") as f:
            f.write(result.output.data)
        try:
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav, "-codec:a", "libmp3lame",
                            "-b:a", "192k", mp3], check=True, timeout=600)
        except Exception as e:
            return {"error": f"ffmpeg: {e}"}
        duration = probe_duration(mp3) or (result.output.metadata or {}).get("duration_s")
        key_base = f"{PREFIX}{out_prefix + '/' if out_prefix else ''}{job_id}"
        S3.upload_file(mp3, BUCKET, key_base + ".mp3", ExtraArgs={"ContentType": "audio/mpeg"})
        size = os.path.getsize(mp3)
        wav_key = None
        if keep_wav:
            wav_key = key_base + ".wav"
            S3.upload_file(wav, BUCKET, wav_key, ExtraArgs={"ContentType": "audio/wav"})
    url = S3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": key_base + ".mp3"},
                                    ExpiresIn=7 * 24 * 3600)
    meta = result.output.metadata or {}
    return {
        "key": key_base + ".mp3", "wav_key": wav_key, "url": url,
        "duration_s": duration, "seed": meta.get("seed"), "processing_ms": processing_ms,
        "model_processing_ms": meta.get("processing_ms"), "bytes": size,
        "mode": meta.get("mode"), "has_reference_voice": meta.get("has_reference_voice"),
    }


runpod.serverless.start({"handler": handler})
