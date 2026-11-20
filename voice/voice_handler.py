"""RunPod serverless handler: "Sing it in my voice". A recording in, the same recording sung by a trained RVC voice out; and
since Nov 2026 "train": recordings of a voice in, a new RVC voice out.

The request contract is voice_request.py; the audio work is voice_pipeline.py; training is voice_train.py. A song or vocal job
lands in the private bucket under voice/<random>/ (mix.mp3, mix.wav, vocal.mp3, vocal.wav, report.json, vocal_fx.mp3 +
vocal_fx.wav when a vocal effect was asked for, and in song mode the stems: instrumental, lead_original and, when the lead split
found any, backing, each as .mp3 and .wav) and comes back as signed links good for seven days.
Voice models come from the private bucket and are cached on the worker by content hash (voice_models.py).
A train job reads its recordings from voice-samples/<owner>/<voice_id>/ and writes exactly three files, model.pth, model.index
and train_report.json, to voice-models/<owner>/<voice_id>/. The recordings and everything made from them on the way stay in the
job's temp folder and go with it.
Since Nov 20 2026 a song or vocal job may ask for engine "seedvc" (talking voices): Seed-VC re-sings the lead in the voice of a
reference recording (reference_key, cached on the worker like a model) instead of an RVC model; everything around it (separation,
pitch, mix, stems) is the same job.
A failure answers {"error": sentence}; nothing is retried automatically.

Environment: AWS_ENDPOINT_URL, AWS_REGION, AWS_BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (the private bucket),
VOICE_AUDIO_HOSTS (hosts an audio_url may use), optional VOICE_AUDIO_PREFIXES, VOICE_CACHE_DIR, VOICE_CACHE_FILES,
VOICE_TRAIN_SECONDS (how long a train job may take in all, default 6600: the caller sends a two hour execution timeout)."""
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
import voice_train  # noqa: E402

FEATURES = ["song", "vocal", "auto-octave", "extractors", "lead-split", "lead-models", "dereverb", "room", "soft-s", "model-cache",
            "vocal-fx", "stems", "backing-modes", "formant", "dereverb-strength", "denoise", "pitch-fit", "transpose", "train", "auto-mix",
            "seedvc"]
PRESIGN_S = 7 * 24 * 3600
GPU = {}
STEMS = ("instrumental", "backing", "lead_original")
TRAIN_FILES = ("model.pth", "model.index", "train_report.json")
SAMPLE_TYPES = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".aiff", ".aif", ".wma", ".webm", ".mp4")

GPU_PROBE = r"""
import json, sys, torch
sys.path.insert(0, sys.argv[1])
from hardware import supports_device
out = {"cuda": torch.cuda.is_available()}
if out["cuda"]:
    major, minor = torch.cuda.get_device_capability(0)
    out["name"] = torch.cuda.get_device_name(0)
    out["capability"] = f"sm_{major}{minor}"
    out["compiled_archs"] = torch.cuda.get_arch_list()
    out["supported"] = supports_device(major, minor, out["compiled_archs"])
    try:
        out["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1)
    except Exception:
        out["vram_gb"] = None
print(json.dumps(out))
"""


def gpu_check():
    """Once per worker: the card's name (for pricing), its memory (training picks its batch size by it) and compatible compiled
    CUDA cubins. Same-major cubins run on equal or higher minor capabilities, including sm_86 on Ada sm_89."""
    if not GPU:
        try:
            r = subprocess.run([os.environ.get("VOICE_RVC_PYTHON", "/opt/rvc-venv/bin/python"), "-c", GPU_PROBE, HERE],
                               capture_output=True, text=True, timeout=120)
            GPU.update(json.loads(r.stdout.strip().splitlines()[-1]))
        except Exception as error:
            GPU.update({"cuda": None, "error": type(error).__name__})
        print("voice worker GPU:", {k: GPU.get(k) for k in ("name", "capability", "compiled_archs", "supported", "cuda", "vram_gb")},
              flush=True)
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
            raise ValueError("The recording is too large. Use one up to ten minutes.")
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
                    raise ValueError("The recording is too large. Use one up to ten minutes.")
                f.write(block)


def fetch_samples(req, client, bucket, folder):
    """The training recordings into folder as sample_000.wav, sample_001.mp3 ... (never the key's own name: only its file type
    is carried over, for ffmpeg). Sizes are checked before anything is downloaded: 256 MB a file, 2 GB in all."""
    sizes = []
    for key in req["sample_keys"]:
        try:
            head = client.head_object(Bucket=bucket, Key=key)
        except Exception as error:
            if type(error).__name__ in ("NoSuchKey", "ClientError"):
                raise ValueError("One of the recordings is no longer in storage. Upload it again.") from None
            raise
        sizes.append(int(head.get("ContentLength") or 0))
        if sizes[-1] > voice_request.MAX_SAMPLE_BYTES:
            raise ValueError("One of the recordings is too large. Use files up to 256 megabytes each.")
    if sum(sizes) > voice_request.MAX_SAMPLES_BYTES:
        raise ValueError("The recordings are too large together. Use up to 2 gigabytes in all.")
    paths = []
    for i, key in enumerate(req["sample_keys"]):
        ext = os.path.splitext(key)[1].lower()
        paths.append(os.path.join(folder, f"sample_{i:03d}{ext if ext in SAMPLE_TYPES else '.audio'}"))
        client.download_file(bucket, key, paths[-1])
    return paths


def fx_summary(fx):
    """The vocal effect in a few fields for the Sound Booth (the whole account is in report.json); None when there was none."""
    if not fx:
        return None
    tempo = fx.get("tempo") or {}
    return {"preset": fx["preset"], "label": fx["label"], "tempo_bpm": tempo.get("bpm"), "tempo_source": tempo.get("source"),
            "delay_ms": (fx.get("delay") or {}).get("delay_ms"), "reverb_s": (fx.get("reverb") or {}).get("seconds"),
            "level_match_db": fx.get("level_match_db"), "tail_s": fx.get("tail_s")}


CONTENT_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".json": "application/json", ".pth": "application/octet-stream",
                 ".index": "application/octet-stream"}


def train(job, req, gpu, start):
    """One train job. The only keys written are <voice-models/owner/voice_id>/model.pth, model.index and train_report.json."""
    import runpod
    proof = voice_pipeline.selftest().get("train")
    if isinstance(proof, dict) and proof.get("ok") is False:
        raise ValueError("This worker cannot train voices. Its own training check failed when it was built.")

    def say(text):
        runpod.serverless.progress_update(job, text)
    timing = {}
    with tempfile.TemporaryDirectory() as td:
        client, bucket = storage()
        say("Fetching the recordings")
        began = time.monotonic()
        samples_dir = os.path.join(td, "samples")
        os.makedirs(samples_dir)
        samples = fetch_samples(req, client, bucket, samples_dir)
        timing["download_s"] = round(time.monotonic() - began, 1)
        work = os.path.join(td, "work")
        os.makedirs(work)
        runner = voice_pipeline.Runner(os.path.join(td, "tools.log"))
        trainer = voice_train.TrainRunner(os.path.join(td, "tools.log"), os.path.join(td, "rvc"))
        deadline = start + float(os.environ.get("VOICE_TRAIN_SECONDS", "6600"))
        result = voice_train.run(req, samples, work, runner, trainer, progress=say, vram_gb=gpu.get("vram_gb"), deadline=deadline)
        say("Saving the voice")
        began = time.monotonic()
        prefix = req["output_prefix"]
        if prefix != f"voice-models/{req['owner']}/{req['voice_id']}" or set(result["files"]) != set(TRAIN_FILES):
            raise RuntimeError("train output outside the voice's folder")
        keys = {}
        for name in TRAIN_FILES:
            keys[name] = f"{prefix}/{name}"
            client.upload_file(result["files"][name], bucket, keys[name],
                               ExtraArgs={"ContentType": CONTENT_TYPES[os.path.splitext(name)[1]]})
        timing["upload_s"] = round(time.monotonic() - began, 1)
        report = result["report"]
        return {
            "engine": "rvc", "mode": "train",
            "model_key": keys["model.pth"], "model_sha256": report["model_sha256"],
            "index_key": keys["model.index"], "index_sha256": report["index_sha256"],
            "report_key": keys["train_report.json"],
            "range": report["range"], "minutes": report["minutes"], "epochs": report["epochs"],
            "sample_count": report["sample_count"], "gpu": gpu.get("name"),
            "timing": {**timing, **report["timing"]}, "worker_notes": report["notes"],
            "processing_ms": int((time.monotonic() - start) * 1000), "features": FEATURES,
        }


def handler(job):
    start = time.monotonic()
    training = False
    try:
        req = voice_request.parse(job.get("input") or {})
        training = req["mode"] == "train"
        gpu = gpu_check()
        if gpu.get("cuda") is not True:
            raise ValueError("The voice worker could not confirm CUDA support on its graphics card. "
                             "Use a worker image and graphics card that support CUDA.")
        if gpu.get("supported") is not True:
            card = gpu.get("name") or "unknown card"
            capability = gpu.get("capability") or "unknown capability"
            raise ValueError(f"The voice worker could not confirm compatible compiled CUDA support for this graphics card "
                             f"({card}, {capability}). Update the worker image or choose a graphics card its CUDA build supports.")
        if training:
            return train(job, req, gpu, start)
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
            model_path = model_info = index_path = index_info = reference_path = reference_info = None
            if req["engine"] == "seedvc":
                suffix = os.path.splitext(req["reference_key"])[1].lower()
                reference_path, reference_info = voice_models.fetch(client, bucket, req["reference_key"], suffix,
                                                                    req["reference_sha256"])
            else:
                model_path, model_info = voice_models.fetch(client, bucket, req["model_key"], ".pth", req["model_sha256"])
            if req["index_key"]:
                index_path, index_info = voice_models.fetch(client, bucket, req["index_key"], ".index", req["index_sha256"])
            timing["models_s"] = round(time.monotonic() - began, 1)
            work = os.path.join(td, "work")
            os.makedirs(work)
            runner = voice_pipeline.Runner(os.path.join(td, "tools.log"))
            result = voice_pipeline.run(req, audio, model_path, index_path, work, runner,
                                        progress=lambda text: runpod.serverless.progress_update(job, text),
                                        reference_path=reference_path)
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
                "engine": req["engine"], "mode": req["mode"],
                "key": uploaded["mix.mp3"], "url": url("mix.mp3"),
                "wav_key": uploaded["mix.wav"], "wav_url": url("mix.wav"),
                "vocal_key": uploaded["vocal.mp3"], "vocal_url": url("vocal.mp3"),
                "vocal_wav_key": uploaded["vocal.wav"], "vocal_wav_url": url("vocal.wav"),
                "report_key": uploaded["report.json"],
                "duration_s": report["seconds"], "bytes": os.path.getsize(result["files"]["mix.mp3"]),
                "gpu": gpu.get("name"), "pitch": report["pitch"], "separation": report["separation"],
                "settings": report["settings"], "worker_notes": report["notes"],
                "models": {"model_cached": model_info["cached"] if model_info else None,
                           "index_cached": index_info["cached"] if index_info else None,
                           **({"reference_cached": reference_info["cached"]} if reference_info else {})},
                "timing": {**timing, **report["timing"]},
                "processing_ms": int((time.monotonic() - start) * 1000), "features": FEATURES,
                "vocal_fx": fx_summary(report.get("vocal_fx")),
            }
            if "vocal_fx.mp3" in uploaded:
                out.update({"vocal_fx_key": uploaded["vocal_fx.mp3"], "vocal_fx_url": url("vocal_fx.mp3"),
                            "vocal_fx_wav_key": uploaded["vocal_fx.wav"], "vocal_fx_wav_url": url("vocal_fx.wav")})
            for stem in STEMS:  # song mode: the band, the backing vocals when there are any, the original lead
                if f"{stem}.mp3" in uploaded:
                    out.update({f"{stem}_key": uploaded[f"{stem}.mp3"], f"{stem}_url": url(f"{stem}.mp3"),
                                f"{stem}_wav_key": uploaded[f"{stem}.wav"], f"{stem}_wav_url": url(f"{stem}.wav")})
            return out
    except ValueError as error:
        return {"error": str(error)}
    except Exception as error:
        print("voice job failed:", type(error).__name__, flush=True)  # the type only: a message can carry a signed link
        if training:
            return {"error": f"The voice could not be trained ({type(error).__name__}). The recordings are kept; nothing was retried."}
        return {"error": f"Your voice version could not finish ({type(error).__name__}). The recording is kept; nothing was retried."}


if __name__ == "__main__":
    import runpod
    gpu_check()  # at boot, so FlashBoot keeps the answer and the first job does not pay for it
    runpod.serverless.start({"handler": handler})
