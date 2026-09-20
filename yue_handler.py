"""One full-quality YuE2 candidate per request; private audio and score outputs."""
import os
import json
import subprocess
import tempfile
import time
import uuid
import math
import re
from dataclasses import replace
from pathlib import Path

PIPE = None
STYLE = {'applied': None, 'plain': None}
STYLE_KEY = re.compile(r'^yue2-loras/[A-Za-z0-9._-]{1,80}\.pt$')


def sound_controls(inp):
    def number(key, default, low, high, integer=False):
        value = inp.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high or (integer and type(value) is not int):
            raise ValueError(f'{key} must be between {low} and {high}' + (' in whole numbers.' if integer else '.'))
        return value
    return dict(weirdness=number('weirdness', 50, 0, 100, True),
                steps=number('steps', 32, 16, 64, True),
                guidance=number('guidance', 1.0, 1.0, 3.0))


def render_song(pipe, inp, controls):
    # Reset per request: a previous song's advanced settings must not leak.
    original = pipe.generation_config
    pipe.generation_config = replace(original, ode_steps=controls['steps'])
    variation = controls['weirdness']
    # Score-free songs use YuE2's own 1.01 default; at exactly 1.0 that mode loses its guidance pass.
    guidance = 1.01 if inp.get('cot') == 'off' and controls['guidance'] == 1.0 else controls['guidance']
    try:
        return pipe(**inp, cfg_scale=guidance,
                    abc_sampling={'temperature': round(.4 + .006 * variation, 3)},
                    semantic_sampling={'temperature': round(.7 + .006 * variation, 3)})
    finally:
        pipe.generation_config = original


def request_input(inp):
    style = inp.get('style', '')
    lyrics = inp.get('lyrics', '')
    abc = inp.get('abc') or None
    if not isinstance(style, str) or not 3 <= len(style.strip()) <= 3000:
        raise ValueError('Describe the music in 3 to 3000 characters.')
    if not isinstance(lyrics, str) or len(lyrics) > 8000:
        raise ValueError('Lyrics must be text, up to 8000 characters.')
    if abc is not None and (not isinstance(abc, str) or len(abc) > 40000):
        raise ValueError('The composition must be ABC text, up to 40000 characters.')
    reference = inp.get('reference_voice_url')
    if reference and (not isinstance(reference, str) or len(reference) > 12000):
        raise ValueError('Import the source recording again.')
    if reference and abc:
        raise ValueError('Use a source recording or a composition score, not both.')
    seed = inp.get('seed', 42)
    if type(seed) is not int or not 0 <= seed <= 2147483647:
        raise ValueError('Seed must be a whole number from 0 to 2147483647.')
    cot = inp.get('cot', 'melody' if abc else 'full')
    if cot not in ('full', 'melody', 'off'):
        raise ValueError('Choose full composition, melody planning or no score.')
    if cot == 'off' and (abc or reference):
        raise ValueError('A cover or a score needs melody or full planning.')
    return dict(style=style.strip(), lyrics=lyrics.strip(), abc=abc, cot=cot, seed=seed)


def style_request(inp):
    """A trained style is one AR LoRA file in the private bucket plus a strength."""
    key = inp.get('lora_key')
    if key is None:
        return None
    if not isinstance(key, str) or not STYLE_KEY.match(key):
        raise ValueError('That trained style is not available.')
    scale = inp.get('lora_scale', 1.0)
    if type(scale) not in (int, float) or not math.isfinite(scale) or not 0.1 <= scale <= 1.5:
        raise ValueError('Style strength must be between 0.1 and 1.5.')
    return key, float(scale)


def style_linears(model):
    # Same order the trainer saved: per layer, attention q,k,v,o then mlp gate,up,down.
    for layer in model.model.layers:
        for mod, names in ((layer.self_attn, ('q_proj', 'k_proj', 'v_proj', 'o_proj')),
                           (layer.mlp, ('gate_proj', 'up_proj', 'down_proj'))):
            for name in names:
                yield getattr(mod, name)


def apply_style(pipe, wanted, fetch):
    """Fold the chosen style into the composer weights, or put the plain weights back.

    The LoRA is folded in place (W = plain + scale * B @ A) because the sampler
    runs on CUDA graphs that hold these tensors. Plain weights are kept on the
    CPU and copied back, so switching styles never drifts."""
    if STYLE['applied'] == wanted:
        return
    import torch
    model = pipe._load_model()
    linears = list(style_linears(model))
    with torch.no_grad():
        if STYLE['plain'] is None:
            STYLE['plain'] = [lin.weight.detach().to('cpu', copy=True) for lin in linears]
        tensors = None
        if wanted:
            tensors = torch.load(fetch(wanted[0]), map_location='cpu', weights_only=True)['lora']
            if len(tensors) != 2 * len(linears):
                raise ValueError('That trained style does not fit this model.')
        STYLE['applied'] = 'changing'
        for index, (lin, plain) in enumerate(zip(linears, STYLE['plain'])):
            lin.weight.copy_(plain)
            if tensors:
                a = tensors[2 * index].to(lin.weight.device, torch.float32)
                b = tensors[2 * index + 1].to(lin.weight.device, torch.float32)
                lin.weight.add_((wanted[1] * (b @ a)).to(lin.weight.dtype))
    STYLE['applied'] = wanted


def engine():
    global PIPE
    if PIPE is None:
        from yue2 import YuE2Pipeline
        PIPE = YuE2Pipeline.from_pretrained(
            os.environ['YUE_MODEL_DIR'], vae=os.environ['YUE_VAE_DIR'],
            local_files_only=True, device='cuda')
        STYLE['applied'] = None  # a fresh pipeline reads plain weights from disk
    return PIPE


def warm():
    # FlashBoot snapshots a booted worker. Loading here, not inside the first
    # job, puts a ready model in that snapshot and keeps loading out of the
    # billed song. A failure only falls back to loading on first use.
    began = time.monotonic()
    try:
        engine()
        print(f'YuE2 ready at boot in {time.monotonic() - began:.1f}s', flush=True)
    except Exception as error:
        print('YuE2 boot load skipped:', type(error).__name__, flush=True)


def handler(job):
    start = time.monotonic()
    try:
        inp = request_input(job.get('input') or {})
        controls = sound_controls(job.get('input') or {})
        wanted = style_request(job.get('input') or {})
        import boto3
        import runpod
        import soundfile as sf
        from auk_handler import ffmpeg, download
        with tempfile.TemporaryDirectory() as td:
            reference = (job.get('input') or {}).get('reference_voice_url')
            warnings = []
            cover_began = time.monotonic()
            if reference:
                global PIPE
                if PIPE is not None:
                    PIPE.close()
                    PIPE = None
                runpod.serverless.progress_update(job, 'Transcribing the source melody for your cover')
                source = Path(td) / 'source'
                download(reference, source)
                decoded = Path(td) / 'source.wav'
                ffmpeg('-i', source, '-ar', 24000, '-ac', 1, decoded)
                if sf.info(decoded).duration > 360:
                    raise ValueError('Use a source recording up to six minutes long.')
                score_dir = Path(td) / 'transcription'
                env = {**os.environ, 'LD_LIBRARY_PATH': '/opt/cover/lib', 'PATH': '/opt/cover/bin:'+os.environ['PATH']}
                subprocess.run(['/opt/cover/bin/python', '/app/cover.py', str(decoded), str(score_dir)],
                               env=env, check=True, timeout=600, capture_output=True)
                inp['abc'] = (score_dir/'score.abc').read_text()
                inp['cot'] = 'melody'
                warnings = json.loads((score_dir/'warnings.json').read_text())
            cover_s = round(time.monotonic() - cover_began, 1) if reference else 0
            runpod.serverless.progress_update(job, 'Loading YuE2 and recording your song')
            load_began = time.monotonic()
            pipe = engine()

            def fetch(key):
                local = Path('/tmp') / key.replace('/', '_')
                if not local.exists():
                    part = local.with_suffix('.part')
                    boto3.client('s3', endpoint_url=os.environ['AWS_ENDPOINT_URL'],
                                 region_name=os.environ.get('AWS_REGION', 'us-east-005')
                                 ).download_file(os.environ['AWS_BUCKET_NAME'], key, str(part))
                    part.rename(local)
                return str(local)
            apply_style(pipe, wanted, fetch)
            load_s = round(time.monotonic() - load_began, 1)
            render_began = time.monotonic()
            result = render_song(pipe, inp, controls)
            render_s = round(time.monotonic() - render_began, 1)
            work = Path(td) / 'song'
            result.save_artifacts(work)
            wav = work / 'master.wav'
            mp3 = work / 'master.mp3'
            ffmpeg('-i', work / 'audio.flac', '-c:a', 'pcm_s24le', wav)
            ffmpeg('-i', wav, '-c:a', 'libmp3lame', '-b:a', '320k', mp3)
            duration = sf.info(wav).duration
            client = boto3.client('s3', endpoint_url=os.environ['AWS_ENDPOINT_URL'],
                                  region_name=os.environ.get('AWS_REGION', 'us-east-005'))
            bucket = os.environ['AWS_BUCKET_NAME']
            prefix = f'yue2/{uuid.uuid4()}'
            files = [(mp3, 'audio/mpeg'), (wav, 'audio/wav')]
            files += [(p, 'text/plain' if p.suffix == '.abc' else 'application/json')
                      for p in work.iterdir() if p.suffix in ('.abc', '.json')]
            for p, mime in files:
                client.upload_file(str(p), bucket, prefix+'/'+p.name, ExtraArgs={'ContentType': mime})
            def url(name):
                return client.generate_presigned_url('get_object', Params={'Bucket':bucket,'Key':prefix+'/'+name}, ExpiresIn=604800)
            return {'engine':'yue2', 'quality':'bf16-default', 'key':prefix+'/master.mp3',
                    'wav_key':prefix+'/master.wav', 'url':url('master.mp3'), 'wav_url':url('master.wav'),
                    'score_key':prefix+'/score.abc' if (work/'score.abc').exists() else None,
                    'duration_s':round(duration,2), 'bytes':mp3.stat().st_size,
                    'processing_ms':int((time.monotonic()-start)*1000), 'seed':inp['seed'],
                    'truncated':any(result.truncated.values()), 'truncation_flags':result.truncated,
                    'cover':bool(reference), 'transcription_warnings':warnings,
                    'controls': controls, 'lora_key': wanted[0] if wanted else None,
                    'timing': {'cover_s': cover_s, 'model_load_s': load_s, 'render_s': render_s}}
    except ValueError as error:
        return {'error':str(error)}
    except Exception as error:
        print('YuE2 failed:',type(error).__name__,flush=True)
        return {'error':f'YuE2 could not finish ({type(error).__name__}). Your writing is saved.'}


if __name__ == '__main__':
    import runpod
    warm()
    runpod.serverless.start({'handler': handler})
