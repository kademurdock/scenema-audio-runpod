"""Full-quality AuK, private B2 outputs, zero automatic inference retries."""
import gc
import ipaddress
import json
import logging
import math
import os
import re
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
import urllib.parse
import uuid

from auk_contract import plan, model_instruction, edit_windows
from auk_edit import (LEVEL_WINDOW_S, crop_bounds, level_gain, rank_candidates, score_transcript,
                      splice, splice_points, target_seconds)

log = logging.getLogger("auk-worker")
logging.basicConfig(level=logging.INFO)
ENGINE = None


ENGINE_LOCK = threading.Lock()


def engine():
    global ENGINE
    # One load only: the boot thread and a first job may both ask for it.
    with ENGINE_LOCK:
        if ENGINE is None:
            ENGINE = load_engine()
    return ENGINE


def load_engine():
    model_dir = Path(os.environ.get('AUK_MODEL_DIR', '/models/AuK'))
    qwen_dir = Path(os.environ.get('AUK_QWEN_DIR', '/models/Qwen'))
    if not (model_dir / 'auk_base.safetensors').is_file():
        raise ValueError('The AuK model cache is not ready. No audio was generated.')
    if not (qwen_dir / 'config.json').is_file():
        raise ValueError('The Qwen model cache is not ready. No audio was generated.')
    from auk.infer.infer_auk import AukInfer
    import torch
    offload = use_cpu_offload(torch.cuda.get_device_properties(0).total_memory)
    started = time.monotonic()
    log.info('AuK CPU offload: %s', offload)
    runtime = AukInfer(str(model_dir / 'config.yaml'), str(model_dir / 'auk_base.safetensors'),
                       qwen_path=str(qwen_dir), dtype="bf16", device="cuda",
                       cpu_offload=offload)
    log.info('AuK loaded in %.1f s; %.1f GiB of weights on the GPU', time.monotonic() - started,
             torch.cuda.memory_allocated() / 1024**3)
    return runtime


def _cuda():
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


def reset_gpu_peak():
    """Start a job's peak-memory count (the number that decides if 24 GB fits)."""
    torch = _cuda()
    if torch is not None:
        torch.cuda.reset_peak_memory_stats()


def gpu_memory(label):
    """Log and return the job's peak so far, what is reserved and what is free."""
    torch = _cuda()
    if torch is None:
        return None
    try:
        free, total = torch.cuda.mem_get_info()
    except Exception as error:  # A reading must never fail a take.
        log.warning('AuK GPU memory reading failed (%s)', type(error).__name__)
        return None
    note = {'peak_allocated_gib': round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            'peak_reserved_gib': round(torch.cuda.max_memory_reserved() / 1024**3, 2),
            'reserved_gib': round(torch.cuda.memory_reserved() / 1024**3, 2),
            'free_gib': round(free / 1024**3, 2), 'total_gib': round(total / 1024**3, 2)}
    log.info('AuK GPU memory after %s: peak %.2f GiB allocated, %.2f GiB reserved (peak %.2f), '
             '%.2f of %.2f GiB free', label, note['peak_allocated_gib'], note['reserved_gib'],
             note['peak_reserved_gib'], note['free_gib'], note['total_gib'])
    return note


def _flag(name, default):
    return os.environ.get(name, default).strip().lower() not in ('0', 'false', 'no', 'off')


def use_cpu_offload(total_bytes, setting=None):
    """Upstream 3253462 keeps Qwen in bf16: full AuK Base is ~14 GiB of weights.

    A 24 GB card (about 22 GiB usable) runs it on the GPU. Only smaller cards
    offload. AUK_CPU_OFFLOAD=1/0 forces it either way.
    """
    setting = (setting if setting is not None else os.environ.get('AUK_CPU_OFFLOAD', 'auto')).strip().lower()
    if setting != 'auto':
        return setting not in ('0', 'false', 'no', 'off')
    return total_bytes < float(os.environ.get('AUK_OFFLOAD_BELOW_GIB', '20')) * 1024**3


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


def transcribe(path, kind):
    """Optional ranking ASR with the Qwen2.5-Omni Thinker AuK already holds.

    No extra model, download or GPU memory beyond a short decode. Off unless a
    job asks (rank: true); the bridge can rank instead.
    """
    runtime = engine()
    if getattr(runtime, 'cpu_offload', False):
        raise RuntimeError('ranking is off while the model is offloaded')
    import torch
    from qwen_omni_utils import process_mm_info
    thinker = runtime.model.text_encoder
    processor = runtime.model.text_processor
    ask = ('Transcribe the lyrics sung in this audio. Output only the words.' if kind == 'vocal'
           else 'Transcribe the speech in this audio. Output only the words.')
    conversation = [
        {'role': 'system', 'content': [{'type': 'text', 'text': 'You are a speech recognition model.'}]},
        {'role': 'user', 'content': [{'type': 'audio', 'audio': str(path)}, {'type': 'text', 'text': ask}]},
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, _, _ = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(text=text, audio=audios, return_tensors='pt', padding=True, use_audio_in_video=False)
    inputs = inputs.to(thinker.device).to(thinker.dtype)
    with torch.inference_mode():
        out = thinker.generate(**inputs, max_new_tokens=160, do_sample=False)
    return processor.batch_decode(out[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()


class Store:
    """Private B2 outputs with seven-day signed links."""

    def __init__(self):
        import boto3
        self.s3 = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"],
                               region_name=os.environ.get("AWS_REGION", "us-east-005"))
        self.bucket = os.environ["AWS_BUCKET_NAME"]

    def put(self, wav, mp3, clip=None):
        # UUID prevents overwriting an existing take or escaping its prefix.
        key = f"auk/{uuid.uuid4()}"
        files = [(wav, ".wav", "audio/wav"), (mp3, ".mp3", "audio/mpeg")]
        if clip is not None:
            files.append((clip, "-clip.mp3", "audio/mpeg"))
        for path, suffix, mime in files:
            self.s3.upload_file(str(path), self.bucket, key + suffix, ExtraArgs={"ContentType": mime})

        def sign(k):
            return self.s3.generate_presigned_url(
                "get_object", Params={"Bucket": self.bucket, "Key": k}, ExpiresIn=604800)
        out = {"key": key + ".mp3", "url": sign(key + ".mp3"),
               "wav_key": key + ".wav", "wav_url": sign(key + ".wav"), "bytes": mp3.stat().st_size}
        if clip is not None:
            out.update(clip_key=key + "-clip.mp3", clip_url=sign(key + "-clip.mp3"))
        return out


def words_edit(job, inp, spec, started):
    """One word or lyric change: crop the marked range, N seeds, splice each back."""
    import numpy as np
    import soundfile as sf
    import runpod
    from auk.infer.infer_auk import save_audio
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        download(inp["reference_voice_url"], work / "import")
        source = work / "source.wav"
        ffmpeg("-i", work / "import", "-vn", "-c:a", "pcm_s24le", source)
        original, rate = sf.read(source, always_2d=True)
        frames, channels = original.shape
        # Every input check runs before the first take: upstream's own checks
        # raise ValueError mid-take.
        if frames == 0 or not np.isfinite(original).all():
            raise ValueError('The recording has no usable audio. Import it again.')
        bounds = crop_bounds(frames / rate, inp)
        crop_start = int(round(bounds['crop_start'] * rate))
        crop_end = min(frames, int(round(bounds['crop_end'] * rate)))
        crop_seconds = (crop_end - crop_start) / rate
        seconds = spec['target_override'] or target_seconds(
            crop_seconds, bounds['range_end'] - bounds['range_start'],
            spec['old'], spec['new'], spec['source_text'])
        native_crop = work / "crop-native.wav"
        sf.write(native_crop, original[crop_start:crop_end], rate, subtype='PCM_24')
        reference = work / "crop.wav"
        ffmpeg("-i", native_crop, "-ar", 24000, "-ac", 1, reference)
        reference_audio, _ = sf.read(reference)
        if reference_audio.size == 0 or not np.isfinite(reference_audio).all():
            raise ValueError('The marked stretch has no usable audio. Choose another range.')
        store = Store()
        fade = int(round(spec['crossfade_ms'] * rate / 1000))
        head_pad = int(round(bounds['range_start'] * rate)) - crop_start
        tail_pad = crop_end - int(round(bounds['range_end'] * rate))
        candidates, failures, memory = [], [], None
        # Load before the takes, so a model cache that is not ready is said plainly.
        engine()
        reset_gpu_peak()
        for n, seed in enumerate(spec['seeds']):
            runpod.serverless.progress_update(job, f"AuK HQ: take {n + 1} of {len(spec['seeds'])}")
            try:
                content = [{"type": "text", "text": spec['instruction']},
                           {"type": "audio", "audio": str(reference)}]
                audio, sr = generate([{"role": "user", "content": content}],
                                     gen_seconds=seconds, nfe=32, cfg_strength=2.0, seed=seed)
                memory = gpu_memory(f'take {n + 1}') or memory
                take = work / f"take{n}.wav"
                save_audio(audio, sr, str(take))
                native = work / f"take{n}-native.wav"
                # Resample only. FFmpeg's mono-to-stereo upmix drops 3 dB, so
                # the take is copied to each channel here at full level.
                ffmpeg("-i", take, "-ar", rate, "-ac", 1, "-c:a", "pcm_s24le", native)
                generated, _ = sf.read(native, always_2d=True)
                # Match the take to the original's level in the padding, so
                # there is no level step at the joins.
                gain = level_gain(original, generated[:, 0], crop_start, crop_end, head_pad, tail_pad,
                                  int(LEVEL_WINDOW_S * rate))
                generated = np.repeat(generated[:, :1] * gain, channels, axis=1)
                # Keep the original until 0.15 s from the range on each side;
                # the take replaces only the range and those margins.
                head, tail = splice_points(rate, bounds, crop_start, crop_end, generated.shape[0], fade)
                kept = generated[head:generated.shape[0] - tail]
                master = np.clip(splice(original, kept, crop_start + head, crop_end - tail, fade), -1.0, 1.0)
                wav, mp3 = work / f"master{n}.wav", work / f"master{n}.mp3"
                sf.write(wav, master, rate, subtype='PCM_24')
                ffmpeg("-i", wav, "-c:a", "libmp3lame", "-b:a", "192k", mp3)
                # The edited stretch with its padding, as it sits in the master,
                # so the bridge can check that the old words are gone.
                clip_wav, clip = work / f"clip{n}.wav", work / f"clip{n}.mp3"
                sf.write(clip_wav, master[crop_start:crop_start + generated.shape[0]], rate, subtype='PCM_24')
                ffmpeg("-i", clip_wav, "-c:a", "libmp3lame", "-b:a", "192k", clip)
                candidate = {"seed": seed, **store.put(wav, mp3, clip),
                             "duration_s": round(master.shape[0] / rate, 2),
                             "edited_seconds": round(generated.shape[0] / rate, 2),
                             "clip_start": round(crop_start / rate, 3),
                             "clip_seconds": round(generated.shape[0] / rate, 3),
                             "level_db": round(20 * math.log10(gain), 2)}
                if spec['rank']:
                    try:
                        candidate['transcript'] = transcribe(take, spec['kind'])
                        candidate.update(score_transcript(candidate['transcript'], spec['old'], spec['new']))
                    except Exception as error:
                        log.warning('AuK ranking skipped (%s)', type(error).__name__)
                        candidate['rank_error'] = 'transcription_failed'
                candidates.append(candidate)
            except Exception as error:
                # No automatic re-roll: a failed seed is reported, the others still run.
                # A ValueError too (upstream checks each take): finished takes
                # are kept, and its raw text never reaches the user.
                log.error('AuK take failed (%s)', type(error).__name__)
                failures.append({"seed": seed, "error": type(error).__name__})
        if not candidates:
            raise RuntimeError('no take finished')
        ranked = bool(spec['rank']) and any(c.get('score') is not None for c in candidates)
        if ranked:
            candidates = rank_candidates(candidates)
        best = candidates[0]
        return {
            "engine": "auk", "quality": "base-bf16-32", "task": "words_edit",
            "kind": spec['kind'], "instruction": spec['instruction'],
            "old_words": spec['old'], "new_words": spec['new'],
            "range_start": round(bounds['range_start'], 3), "range_end": round(bounds['range_end'], 3),
            "crop_start": round(crop_start / rate, 3), "crop_end": round(crop_end / rate, 3),
            "target_seconds": seconds, "crossfade_ms": spec['crossfade_ms'],
            "candidates": candidates, "failed_seeds": failures,
            "ranked": ranked, "any_matched": any(bool(c.get('matched')) for c in candidates) if ranked else None,
            # The first (or best ranked) take, for callers that read one result.
            "key": best["key"], "url": best["url"], "wav_key": best["wav_key"],
            "wav_url": best["wav_url"], "bytes": best["bytes"], "duration_s": best["duration_s"],
            "seed": best["seed"], "processing_ms": int((time.monotonic() - started) * 1000),
            "has_reference_voice": True, "parts": 1, "voice_sample": False,
            "clip_url": best["clip_url"], "clip_key": best["clip_key"],
            **({"gpu_memory": memory} if memory else {}),
        }


def handler(job):
    start = time.monotonic()
    try:
        inp = job.get("input") or {}
        pieces = plan(inp)
        if pieces[0].get("words_edit"):
            return words_edit(job, inp, pieces[0]["words_edit"], start)
        from auk.infer.infer_auk import save_audio
        import soundfile as sf
        import boto3
        import runpod
        reset_gpu_peak()
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
                gpu_memory(f'part {n + 1}')
                output = work / f"part{n}.wav"
                save_audio(audio, sr, str(output))
                if editing:
                    # Resample to mono, then copy to each channel at full level:
                    # FFmpeg's own mono-to-stereo upmix drops 3 dB.
                    mono = work / f'edited{n}-mono.wav'
                    ffmpeg('-i', output, '-ar', source_info.samplerate, '-ac', 1, '-c:a', 'pcm_s24le', mono)
                    data, _ = sf.read(mono, always_2d=True)
                    normalized = work / f'edited{n}.wav'
                    sf.write(normalized, data[:, :1].repeat(source_info.channels, axis=1),
                             source_info.samplerate, subtype='PCM_24')
                    output = normalized
                if not piece.get("sample"):
                    paths.append(output)
                if not editing and source is None:
                    source = work / "voice.wav"
                    ffmpeg("-i", output, "-t", 8, source)
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
            return {"engine": "auk", "quality": "base-bf16-32", "key": key + ".mp3",
                    "wav_key": key + ".wav", "wav_url": s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key + ".wav"}, ExpiresIn=604800), "url": url, "duration_s": round(duration, 2),
                    "processing_ms": int((time.monotonic() - start) * 1000),
                    "bytes": mp3.stat().st_size, "seed": pieces[0]["seed"],
                    "has_reference_voice": bool(inp.get("reference_voice_url")), "parts": heard,
                    "voice_sample": any(s.get("sample") for s in steps)}
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


def warm():
    """Load AuK while the worker already listens (Nov 14 2026).

    Part 215 (Sep 19) left boot loading out after a confounded trial on hosts
    whose containers never started. Here the load runs after
    runpod.serverless.start() is up, behind the same lock as engine(), so a
    job that arrives first waits for this load instead of starting another.
    A failed boot load is retried by the first job and reported there.
    """
    time.sleep(float(os.environ.get('AUK_BOOT_DELAY', '2')))
    try:
        engine()
    except Exception as error:
        log.warning('AuK boot load failed (%s); the first job will retry', type(error).__name__)
        # Free what the failed load left on the card before the first job retries.
        gc.collect()
        torch = _cuda()
        if torch is not None:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import runpod
    if _flag('AUK_BOOT_LOAD', '1'):
        threading.Thread(target=warm, name='auk-boot-load', daemon=True).start()
    runpod.serverless.start({"handler": handler})
