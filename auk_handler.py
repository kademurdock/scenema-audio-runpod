"""Full-quality AuK, private B2 outputs, zero automatic inference retries."""
import ipaddress
import hashlib
import json
import logging
import os
import re
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.parse
import uuid

from auk_contract import plan, model_instruction, edit_windows, retain_bootstrap_sample

log = logging.getLogger("auk-worker")
logging.basicConfig(level=logging.INFO)
ENGINE = None


def engine():
    global ENGINE
    if ENGINE is None:
        model_dir = Path(os.environ.get('AUK_MODEL_DIR', '/models/AuK'))
        qwen_dir = Path(os.environ.get('AUK_QWEN_DIR', '/models/Qwen'))
        if not (model_dir / 'auk_base.safetensors').is_file():
            raise ValueError('The AuK model cache is not ready. No audio was generated.')
        if not (qwen_dir / 'config.json').is_file():
            raise ValueError('The Qwen model cache is not ready. No audio was generated.')
        from auk.infer.infer_auk import AukInfer
        import torch
        offload_setting = os.environ.get('AUK_CPU_OFFLOAD', 'auto').lower()
        offload = (torch.cuda.get_device_properties(0).total_memory < 40 * 1024**3
                   if offload_setting == 'auto' else offload_setting not in ('0', 'false'))
        log.info('AuK CPU offload: %s', offload)
        ENGINE = AukInfer(str(model_dir / 'config.yaml'), str(model_dir / 'auk_base.safetensors'),
                          qwen_path=str(qwen_dir), dtype="bf16", device="cuda",
                          cpu_offload=offload)
    return ENGINE


def generate(messages, **kwargs):
    runtime = engine()
    # Pinned AuK clears projected text only after successful sampling when
    # CPU offload is disabled. A failed ODE step must not condition the next
    # request with a previous user's instruction on the reused engine.
    runtime.model.transformer.clear_cache()
    try:
        return runtime.generate(messages, **kwargs)
    finally:
        runtime.model.transformer.clear_cache()


def download(url, dest):
    import requests
    # Only the configured private asset host is allowed. Redirects cannot turn
    # a user-supplied clip into a metadata-service or internal-network request.
    parsed = urllib.parse.urlsplit(url)
    hosts = set(os.environ.get("AUK_AUDIO_HOSTS", "").split(",")) - {""}
    if parsed.scheme != "https" or parsed.hostname not in hosts or parsed.username or parsed.port not in (None, 443):
        raise ValueError("Import this recording into the Sound Booth before editing it.")
    for address in socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM):
        if not ipaddress.ip_address(address[4][0]).is_global:
            raise ValueError("Recording host is not public.")
    with requests.get(url, stream=True, timeout=(15, 90), allow_redirects=False) as r:
        if r.status_code != 200:
            raise ValueError("Recording could not be fetched; import it again.")
        size = 0
        with open(dest, "wb") as f:
            for data in r.iter_content(1024 * 1024):
                size += len(data)
                if size > 256 * 1024 * 1024:
                    raise ValueError("Recording exceeds the import size limit.")
                f.write(data)


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", *map(str, args)],
                   check=True, timeout=180, capture_output=True)


def handler(job):
    start = time.monotonic()
    try:
        inp = job.get("input") or {}
        pieces = plan(inp)
        from auk.infer.infer_auk import save_audio
        import soundfile as sf
        import boto3
        import runpod
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            source = None
            editing = inp.get("auk_task") == "edit"
            if inp.get("reference_voice_url"):
                download(inp["reference_voice_url"], work / "import")
                source = work / "source.wav"
                ffmpeg("-i", work / "import", "-vn", "-c:a", "pcm_s24le", source)
            before = None
            after = None
            if editing:
                source_info = sf.info(source)
                # Apply the instruction to every source window, retaining all
                # of the recording. Content edits across a join need review.
                total = source_info.duration
                windows = edit_windows(total, inp)
                steps = []
                if windows[0]['start'] > 0 and inp.get('preserve_before', True):
                    before = work / 'before.wav'
                    ffmpeg('-i', source, '-t', windows[0]['start'], '-c:a', 'pcm_s24le', before)
                if windows[-1]['end'] < total and inp.get('preserve_after', True):
                    after = work / 'after.wav'
                    ffmpeg('-i', source, '-ss', windows[-1]['end'], '-c:a', 'pcm_s24le', after)
                for n, window in enumerate(windows):
                    ref = work / f"ref{n}.wav"
                    ffmpeg("-i", source, "-ss", window['start'],
                           "-t", window['end'] - window['start'], '-ar', 24000, '-ac', 1, ref)
                    steps.append({**pieces[0], "seconds": window['seconds'], "source": ref})
            else:
                if source:
                    # A voice reference is a sample; edits above retain it all.
                    ref = work / "voice.wav"
                    ffmpeg("-i", source, "-t", 8, '-ar', 24000, '-ac', 1, ref)
                    source = ref
                steps = pieces
            paths = []
            diagnostic_sample = None
            heard = sum(1 for s in steps if not s.get("sample"))
            for n, piece in enumerate(steps):
                runpod.serverless.progress_update(job, "AuK HQ: setting the voice" if piece.get("sample")
                                                  else f"AuK HQ: part {len(paths) + 1} of {heard}")
                reference = piece.get("source", source)
                instruction = model_instruction(piece, has_reference=bool(reference))
                content = [{"type": "text", "text": instruction}]
                if reference:
                    content.append({"type": "audio", "audio": str(reference)})
                audio, sr = generate([{"role": "user", "content": content}],
                    gen_seconds=piece["seconds"], nfe=32, cfg_strength=2.0, seed=piece["seed"])
                output = work / f"part{n}.wav"
                save_audio(audio, sr, str(output))
                if editing:
                    normalized = work / f'edited{n}.wav'
                    ffmpeg('-i', output, '-ar', source_info.samplerate, '-ac', source_info.channels,
                           '-c:a', 'pcm_s24le', normalized)
                    output = normalized
                if not piece.get("sample"):
                    paths.append(output)
                if not editing and source is None:
                    source = work / "voice.wav"
                    ffmpeg("-i", output, "-t", 8, source)
                    if piece.get("sample") and retain_bootstrap_sample(inp, os.environ):
                        # Retain the exact cropped reference the audible parts
                        # received, separately from the performance and gallery.
                        diagnostic_sample = source
            if before:
                paths.insert(0, before)
            if after:
                paths.append(after)
            listing = work / "concat.txt"
            listing.write_text("".join(f"file '{p.name}'\n" for p in paths))
            wav = work / "master.wav"; mp3 = work / "master.mp3"
            ffmpeg("-f", "concat", "-safe", 1, "-i", listing, "-c:a", "pcm_s24le" if editing else "pcm_s16le", wav)
            ffmpeg("-i", wav, "-c:a", "libmp3lame", "-b:a", "192k", mp3)
            duration = sf.info(wav).duration
            s3 = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"],
                region_name=os.environ.get("AWS_REGION", "us-east-005"))
            bucket = os.environ["AWS_BUCKET_NAME"]
            # UUID prevents overwriting an existing take or escaping its prefix.
            key = f"auk/{uuid.uuid4()}"
            for path, suffix, mime in [(wav, ".wav", "audio/wav"), (mp3, ".mp3", "audio/mpeg")]:
                s3.upload_file(str(path), bucket, key + suffix, ExtraArgs={"ContentType": mime})
            url = s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key + ".mp3"}, ExpiresIn=604800)
            result = {"engine": "auk", "quality": "base-bf16-32", "key": key + ".mp3",
                    "wav_key": key + ".wav", "wav_url": s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key + ".wav"}, ExpiresIn=604800), "url": url, "duration_s": round(duration, 2),
                    "processing_ms": int((time.monotonic() - start) * 1000),
                    "bytes": mp3.stat().st_size, "seed": pieces[0]["seed"],
                    "has_reference_voice": bool(inp.get("reference_voice_url")), "parts": heard,
                    "voice_sample": any(s.get("sample") for s in steps)}
            if diagnostic_sample is not None:
                try:
                    diagnostic_key = key + '.conditioning.wav'
                    s3.upload_file(str(diagnostic_sample), bucket, diagnostic_key,
                                   ExtraArgs={"ContentType": "audio/wav"})
                    result['diagnostic_sample'] = {
                        'owner_id': inp['out_prefix'], 'key': diagnostic_key,
                        'url': s3.generate_presigned_url('get_object',
                            Params={'Bucket': bucket, 'Key': diagnostic_key}, ExpiresIn=604800),
                        'duration_s': sf.info(diagnostic_sample).duration,
                        'sha256': hashlib.sha256(diagnostic_sample.read_bytes()).hexdigest(),
                        'kind': 'bootstrap_reference',
                    }
                except Exception as error:
                    # An optional capture cannot turn finished audio into a
                    # render failure and invite an unnecessary paid retry.
                    log.warning('AuK diagnostic retention failed (%s)', type(error).__name__)
                    result['diagnostic_sample_error'] = 'retention_failed'
            return result
    except Exception as error:
        # Do not return URLs, signed queries, or storage credentials in errors.
        missing = getattr(error, "name", None) if isinstance(error, ModuleNotFoundError) else None
        if missing and re.fullmatch(r"[a-zA-Z0-9_.]+", missing):
            log.error("AuK missing dependency: %s", missing)
            return {"error": "The AuK worker is missing a required audio component. Generation stopped; your script and source recording are unchanged.",
                    "error_code": "missing_dependency", "missing_module": missing}
        log.error("AuK failed (%s)", type(error).__name__)
        return {"error": str(error) if isinstance(error, ValueError) else
                f"AuK could not finish ({type(error).__name__}). Your source recording is unchanged."}


# NO BOOT-TIME LOAD FOR AuK YET (Part 215, Sep 19 2026). YuE2 loads at boot and it
# works (14.7 s queue, 0 s in-job load). The same warm() was tried here twice on
# image 02224ba and never served a job -- but the test was CONFOUNDED: the same
# night RunPod kept renting A40 hosts in CA whose containers never started
# (runtime null, pods cycling), and the old known-good image b6b823e sat in the
# queue the same way. So it is NOT established that the boot load was at fault.
# It is left out because it could not be verified and AuK must stay simple while
# its hosts are flaky. To retry: first confirm the current image serves a job,
# then change one thing, and prefer loading AFTER runpod.serverless.start() is
# listening, behind a lock shared with engine().


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
