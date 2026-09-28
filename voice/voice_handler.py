"""RunPod serverless handler: "Sing it in my voice". A recording in, the same recording sung by a trained RVC voice out.

The request contract is voice_request.py; the audio work is voice_pipeline.py. Everything lands in the private bucket under
voice/<random>/ (mix.mp3, mix.wav, vocal.mp3, vocal.wav, report.json, and vocal_fx.mp3 + vocal_fx.wav when a vocal effect was
asked for) and comes back as signed links good for seven days.
Voice models come from the private bucket and are cached on the worker by content hash (voice_models.py).
A failure answers {"error": sentence}; nothing is retried automatically.

Environment: AWS_ENDPOINT_URL, AWS_REGION, AWS_BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (the private bucket),
VOICE_AUDIO_HOSTS (hosts an audio_url may use), optional VOICE_AUDIO_PREFIXES, VOICE_CACHE_DIR, VOICE_CACHE_FILES."""
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import voice_models  # noqa: E402
import voice_pipeline  # noqa: E402
import voice_request  # noqa: E402

FEATURES = ["song", "vocal", "auto-octave", "extractors", "lead-split", "lead-models", "dereverb", "room", "soft-s", "model-cache",
            "vocal-fx"]
PRESIGN_S = 7 * 24 * 3600
GPU = {}

GPU_PROBE = r"""
import json, torch
out = {"cuda": torch.cuda.is_available()}
if out["cuda"]:
    major, minor = torch.cuda.get_device_capability(0)
    out["name"] = torch.cuda.get_device_name(0)
    out["capability"] = f"sm_{major}{minor}"
    out["supported"] = out["capability"] in torch.cuda.get_arch_list()
print(json.dumps(out))
"""


def gpu_check():
    """Once per worker: the card's name (for pricing) and whether the pinned torch 2.7.1+cu118 has kernels for it.
    An RTX 50-series / Blackwell card has none, and would fail every job part-way through."""
    if not GPU:
        try:
            r = subprocess.run([os.environ.get("VOICE_RVC_PYTHON", "/opt/rvc-venv/bin/python"), "-c", GPU_PROBE],
                               capture_output=True, text=True, timeout=120)
            GPU.update(json.loads(r.stdout.strip().splitlines()[-1]))
        except Exception as error:
            GPU.update({"cuda": None, "error": type(error).__name__})
        print("voice worker GPU:", {k: GPU.get(k) for k in ("name", "capability", "supported", "cuda")}, flush=True)
    return GPU


def storage():
    import boto3
    client = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"], region_name=os.environ.get("AWS_REGION", "us-east-005"))
    return client, os.environ["AWS_BUCKET_NAME"]


def fetch_audio(req, client, bucket, dest):
    if req["audio_key"]:
        try:
            head = client.head_object(Bucket=bucket, Key=req["audio_key"])
        except Exception as error:
            if type(error).__name__ in ("NoSuchKey", "ClientError"):
                raise ValueError("The recording is no longer in storage. Import it again.") from None
            raise
        if int(head.get("ContentLength") or 0) > voice_request.MAX_BYTES:
            raise ValueError("The recording is too large. Use one up to six minutes.")
        client.download_file(bucket, req["audio_key"], dest)
        return
    import requests
    with requests.get(req["audio_url"], stream=True, timeout=(15, 120), allow_redirects=False) as r:
        if r.status_code != 200:
            raise ValueError("The recording could not be fetched. Import it again.")
        size = 0
        with open(dest, "wb") as f:
            for block in r.iter_content(1 << 20):
                size += len(block)
                if size > voice_request.MAX_BYTES:
                    raise ValueError("The recording is too large. Use one up to six minutes.")
                f.write(block)


def fx_summary(fx):
    """The vocal effect in a few fields for the Sound Booth (the whole account is in report.json); None when there was none."""
    if not fx:
        return None
    tempo = fx.get("tempo") or {}
    return {"preset": fx["preset"], "label": fx["label"], "tempo_bpm": tempo.get("bpm"), "tempo_source": tempo.get("source"),
            "delay_ms": (fx.get("delay") or {}).get("delay_ms"), "reverb_s": (fx.get("reverb") or {}).get("seconds"),
            "level_match_db": fx.get("level_match_db"), "tail_s": fx.get("tail_s")}


CONTENT_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".json": "application/json"}


def handler(job):
    start = time.monotonic()
    try:
        req = voice_request.parse(job.get("input") or {})
        gpu = gpu_check()
        if gpu.get("cuda") and gpu.get("supported") is False:
            raise ValueError(f"The voice worker cannot run on this graphics card ({gpu.get('name')}). Nothing was changed; "
                             "the endpoint needs a card from before the RTX 50 series.")
        import runpod
        timing = {}
        with tempfile.TemporaryDirectory() as td:
            client, bucket = storage()
            runpod.serverless.progress_update(job, "Fetching the recording and your voice")
            began = time.monotonic()
            audio = os.path.join(td, "source")
            fetch_audio(req, client, bucket, audio)
            timing["download_s"] = round(time.monotonic() - began, 1)
            began = time.monotonic()
            model_path, model_info = voice_models.fetch(client, bucket, req["model_key"], ".pth", req["model_sha256"])
            index_path, index_info = None, None
            if req["index_key"]:
                index_path, index_info = voice_models.fetch(client, bucket, req["index_key"], ".index", req["index_sha256"])
            timing["models_s"] = round(time.monotonic() - began, 1)
            work = os.path.join(td, "work")
            os.makedirs(work)
            runner = voice_pipeline.Runner(os.path.join(td, "tools.log"))
            result = voice_pipeline.run(req, audio, model_path, index_path, work, runner,
                                        progress=lambda text: runpod.serverless.progress_update(job, text))
            began = time.monotonic()
            prefix = req["output_prefix"]
            uploaded, by_path = {}, {}
            # the mix last: in vocal mode it is the (effected) vocal itself, stored once under the vocal's own name
            for name, path in sorted(result["files"].items(), key=lambda item: item[0].startswith("mix.")):
                if path in by_path:
                    uploaded[name] = by_path[path]
                    continue
                key = f"{prefix}/{name}"
                client.upload_file(path, bucket, key, ExtraArgs={"ContentType": CONTENT_TYPES[os.path.splitext(name)[1]]})
                uploaded[name] = by_path[path] = key
            timing["upload_s"] = round(time.monotonic() - began, 1)

            def url(name):
                return client.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": uploaded[name]},
                                                     ExpiresIn=PRESIGN_S)
            report = result["report"]
            out = {
                "engine": "rvc", "mode": req["mode"],
                "key": uploaded["mix.mp3"], "url": url("mix.mp3"),
                "wav_key": uploaded["mix.wav"], "wav_url": url("mix.wav"),
                "vocal_key": uploaded["vocal.mp3"], "vocal_url": url("vocal.mp3"),
                "vocal_wav_key": uploaded["vocal.wav"], "vocal_wav_url": url("vocal.wav"),
                "report_key": uploaded["report.json"],
                "duration_s": report["seconds"], "bytes": os.path.getsize(result["files"]["mix.mp3"]),
                "gpu": gpu.get("name"), "pitch": report["pitch"], "separation": report["separation"],
                "settings": report["settings"], "worker_notes": report["notes"],
                "models": {"model_cached": model_info["cached"], "index_cached": index_info["cached"] if index_info else None},
                "timing": {**timing, **report["timing"]},
                "processing_ms": int((time.monotonic() - start) * 1000), "features": FEATURES,
                "vocal_fx": fx_summary(report.get("vocal_fx")),
            }
            if "vocal_fx.mp3" in uploaded:
                out.update({"vocal_fx_key": uploaded["vocal_fx.mp3"], "vocal_fx_url": url("vocal_fx.mp3"),
                            "vocal_fx_wav_key": uploaded["vocal_fx.wav"], "vocal_fx_wav_url": url("vocal_fx.wav")})
            return out
    except ValueError as error:
        return {"error": str(error)}
    except Exception as error:
        print("voice job failed:", type(error).__name__, flush=True)  # the type only: a message can carry a signed link
        return {"error": f"Your voice version could not finish ({type(error).__name__}). The recording is kept; nothing was retried."}


if __name__ == "__main__":
    import runpod
    gpu_check()  # at boot, so FlashBoot keeps the answer and the first job does not pay for it
    runpod.serverless.start({"handler": handler})
