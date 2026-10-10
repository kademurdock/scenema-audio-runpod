"""ACE-Step 1.5 XL music worker for the Sound Booth (engine "ace"), beside YuE2.

One RunPod job renders one request: a caption, optional lyrics and a length become 1 to 4 takes, each
stored as a 24-bit 48 kHz WAV and a 320k MP3 in B2 with 7-day presigned links. The shape of the request and
of the answer is the worker contract in ACE_CONTRACT.md; the findings behind the upstream calls are in
ACE_NOTES.md (every upstream name used here was checked against the pinned commit, see UPSTREAM_COMMIT).

Boot loads xl-turbo, the 1.7B planner and the VAE once (warm(), before runpod starts, so FlashBoot snapshots
a ready worker). xl-sft loads on its first use; on a card under 40 GB, turbo is released first.

This file imports with the standard library alone. torch, acestep, soundfile, boto3 and runpod are imported
inside the functions that need them, so the unit tests run on a plain Python with a fake pipeline.

Environment (all optional; the Dockerfile pins the snapshot folders):
  ACE_TURBO_DIR, ACE_SFT_DIR, ACE_MAIN_DIR   pinned RunPod cache snapshot folder of each repo
  ACE_CHECKPOINTS_DIR                         where the upstream checkpoints folder is built (default /tmp/ace-checkpoints)
  ACE_LM_BACKEND                              pt (default) or vllm
  ACE_PLANNER_REWRITES_CAPTION                1 lets the planner rewrite the caption (default: the caller's words are kept)
  ACE_STEP_DIR, ACE_STEP_COMMIT, ACE_WORKER_VERSION   where upstream is cloned and what the answer reports
  AWS_ENDPOINT_URL, AWS_BUCKET_NAME, AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY   B2 (boto3), as for YuE2
"""
import gc
import json
import math
import os
import random
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

UPSTREAM_COMMIT = 'ca1e85fe9430179831e6bc6be790c332190a3866'
WORKER_VERSION = 'ace-worker-1'

# Which Hugging Face repo each folder of the RunPod model cache comes from. The env var holds the pinned snapshot
# folder; a missing or half-filled pin falls back to the newest complete cached snapshot, then to one download.
MODELS = (('ACE_TURBO_DIR', 'ACE-Step/acestep-v15-xl-turbo', 'turbo'),
          ('ACE_SFT_DIR', 'ACE-Step/acestep-v15-xl-sft', 'sft'),
          ('ACE_MAIN_DIR', 'ACE-Step/Ace-Step1.5', 'main'))
RESOLVED = {}      # key -> {'folder': Path or None, 'how': str}
CACHE_ROOT = '/runpod-volume/huggingface-cache/hub'
DOWNLOAD_CACHE = '/tmp/hf-hub'
# Files that never matter here are not downloaded (the DiT code comes from the pinned upstream clone; the 2B
# "acestep-v15-turbo" in the main repo is never loaded).
DOWNLOAD_SKIP = {'turbo': ('*.py', '*.md', '.gitattributes'), 'sft': ('*.py', '*.md', '.gitattributes'),
                 'main': ('acestep-v15-turbo/*', '*.md', '.gitattributes')}

# Upstream finds a DiT by the NAME of a folder under one flat checkpoints directory (ACE_NOTES 1.2, 3.1).
DIT_MODELS = {
    'xl-turbo': {'folder': 'acestep-v15-xl-turbo', 'variant': 'xl_turbo', 'key': 'turbo', 'feature': 'ace-xl-turbo',
                 'label': 'fast', 'steps': 8, 'min_steps': 1, 'max_steps': 8, 'guidance': 1.0, 'shift': 3.0,
                 'dcw': True},
    'xl-sft': {'folder': 'acestep-v15-xl-sft', 'variant': 'xl_sft', 'key': 'sft', 'feature': 'ace-xl-sft',
               'label': 'best-quality', 'steps': 50, 'min_steps': 4, 'max_steps': 100, 'guidance': 7.0, 'shift': 3.0,
               'dcw': False},   # DCW off for sft: known garbled-output trap (upstream issue 1299)
}
MAIN_PARTS = ('vae', 'Qwen3-Embedding-0.6B', 'acestep-5Hz-lm-1.7B')
LM_FOLDER = 'acestep-5Hz-lm-1.7B'
PLACEHOLDER_DIT = 'acestep-v15-turbo'   # upstream insists that this 2B DiT holds weights; it is never loaded
MAIN_FILES = ('vae/config.json', 'vae/diffusion_pytorch_model.safetensors',
              'Qwen3-Embedding-0.6B/config.json', 'Qwen3-Embedding-0.6B/model.safetensors',
              'acestep-5Hz-lm-1.7B/config.json', 'acestep-5Hz-lm-1.7B/model.safetensors')
LINK_SUFFIXES = ('.json', '.safetensors', '.pt', '.txt')
BOTH_MODELS_GIB = 40   # a card with less memory holds one DiT at a time

# DCW strengths (low band, high band) by whether the planner is used. Upstream's own Think-aware defaults:
# acestep/ui/gradio/events/dcw_defaults.py THINK_DCW_DEFAULTS and NON_THINK_DCW_DEFAULTS (docs/en/DCW.md "Think").
# GenerationParams itself only carries the non-Think pair, so the planner case has to be passed explicitly.
DCW_STRENGTHS = {True: (0.02, 0.06), False: (0.05, 0.02)}

FEATURES = ('ace-xl-turbo', 'ace-xl-sft', 'batch', 'instrumental', 'probe')
VALID_LANGUAGES = frozenset((
    'ar az bg bn ca cs da de el en es fa fi fr he hi hr ht hu id is it ja ko la lt ms ne nl no pa pl pt ro ru sa sk '
    'sr sv sw ta te th tl tr uk ur vi yue zh unknown').split())
CAPTION_MAX = 512
LYRICS_MAX = 4096
DURATION_MIN, DURATION_MAX = 10, 600
BPM_MIN, BPM_MAX = 30, 300
SEED_LIMIT = 2 ** 31   # seeds stay in 0..2147483647 like YuE2's, and seed + i wraps inside it
BATCH_MAX = 4
SAMPLE_RATE = 48000
URL_SECONDS = 604800
INSTRUMENTAL = '[Instrumental]'
ADAPTER_SENTENCE = 'Custom ACE styles are not available yet.'
TOO_BIG = 'This song is too long for the memory that is free right now. Try a shorter length.'
FAILED = 'ACE-Step could not finish this song. Your writing is saved.'

STATE = {'ckpt': None, 'built': set(), 'dit': {}, 'llm': None, 'planner_tries': 0, 'api': None, 'env': False}


class RequestError(Exception):
    """A request the caller can fix; the message is the plain sentence returned in {'error': ...}."""


class WorkerError(Exception):
    """The worker could not do the job (models, card); the message is a plain sentence too."""


# ---------------------------------------------------------------- the model cache (same fallback as yue_handler)

def _filled(folder):
    try:
        return folder.is_dir() and any(folder.iterdir())
    except OSError:
        return False


def dit_ready(folder):
    """A DiT snapshot is usable when config.json, silence_latent.pt and every weight shard are there."""
    try:
        folder = Path(folder)
        if not (folder / 'config.json').exists() or not (folder / 'silence_latent.pt').exists():
            return False
        index = folder / 'model.safetensors.index.json'
        if not index.exists():
            return (folder / 'model.safetensors').exists()
        shards = set(json.loads(index.read_text(encoding='utf-8')).get('weight_map', {}).values())
        return bool(shards) and all((folder / shard).exists() for shard in shards)
    except (OSError, ValueError, AttributeError):
        return False


def main_ready(folder):
    """The main repo snapshot is usable when the VAE, the text encoder and the planner each hold config and weights."""
    try:
        return all((Path(folder) / name).exists() for name in MAIN_FILES)
    except OSError:
        return False


READY = {'turbo': dit_ready, 'sft': dit_ready, 'main': main_ready}


def model_folder(pinned, repo, roots=None, check=None):
    """(folder, how) for one repo: the pinned snapshot, else the newest complete cached snapshot, else (None, 'missing')."""
    check = check or _filled
    if pinned:
        pinned = Path(pinned)
        if check(pinned):
            return pinned, 'pinned'
    name = 'models--' + repo.replace('/', '--')
    if roots is None:
        roots = []
        if pinned and len(pinned.parents) > 2:
            roots.append(pinned.parents[2])
        roots += [os.environ.get('HF_HUB_CACHE', CACHE_ROOT), CACHE_ROOT, DOWNLOAD_CACHE]
    seen = set()
    for root in roots:
        if str(root) in seen:
            continue
        seen.add(str(root))
        snaps = Path(root) / name / 'snapshots'
        try:
            found = sorted((s for s in snaps.iterdir() if check(s)), key=lambda s: s.stat().st_mtime, reverse=True)
        except OSError:
            continue
        if found:
            return found[0], 'cached ' + found[0].name[:12]
    return None, 'missing'


def fetch_model(repo, revision, ignore=()):
    """The pinned revision from Hugging Face, once, for a machine whose cache lacks it. Restores offline mode after."""
    constants = None
    previous = os.environ.get('HF_HUB_OFFLINE')
    try:
        os.environ['HF_HUB_OFFLINE'] = '0'
        import huggingface_hub
        # An explicit submodule import. huggingface_hub builds its package lazily, so `huggingface_hub.constants` is
        # not an attribute until some code has imported that submodule, and on a cold boot (this call comes before
        # upstream is imported) nothing has. Reading the attribute would raise AttributeError.
        import huggingface_hub.constants as constants
        constants.HF_HUB_OFFLINE = False
        return Path(huggingface_hub.snapshot_download(repo, revision=revision, cache_dir=DOWNLOAD_CACHE,
                                                      ignore_patterns=list(ignore) or None))
    except Exception as error:
        print(f'{repo} download failed: {type(error).__name__}', flush=True)
        return None
    finally:
        os.environ['HF_HUB_OFFLINE'] = previous if previous is not None else '1'
        if constants is not None:
            # huggingface_hub reads the variable once, when it is first imported. If that import happened inside
            # this call, the constant was computed from the temporary '0'; set it from the restored value instead
            # (yue_handler.fetch_model leaves it False after a download).
            constants.HF_HUB_OFFLINE = os.environ['HF_HUB_OFFLINE'].upper() in ('1', 'ON', 'YES', 'TRUE')


def resolve_models(download=True, only=None):
    """Find each repo's snapshot folder (pinned, newest cached, or downloaded once) and write it back to its env var."""
    report = []
    for env, repo, key in MODELS:
        if only is not None and key not in only:
            continue
        if RESOLVED.get(key, {}).get('folder') is not None:
            continue
        pinned = os.environ.get(env)
        folder, how = model_folder(pinned, repo, check=READY[key])
        if folder is None and download:
            revision = Path(pinned).name if pinned else None
            folder = fetch_model(repo, revision, DOWNLOAD_SKIP[key])
            if folder is not None and not READY[key](folder):
                folder = None
            how = 'downloaded' if folder is not None else 'missing'
        RESOLVED[key] = {'folder': folder, 'how': how}
        if folder is not None:
            os.environ[env] = str(folder)
        report.append(f'{repo.split("/")[1]} {how}')
    if report:
        print('ACE-Step models:', '; '.join(report), flush=True)


# ---------------------------------------------------------------- the checkpoints folder upstream expects

def checkpoints_dir():
    return Path(os.environ.get('ACE_CHECKPOINTS_DIR', '/tmp/ace-checkpoints'))


def _link(source, destination):
    os.symlink(str(source), str(destination), target_is_directory=Path(source).is_dir())


def _clear(path):
    path = Path(path)
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def upstream_models_dir():
    """acestep/models of the upstream clone: the folder whose .py files upstream compares and copies."""
    found = []
    try:
        import importlib.util
        spec = importlib.util.find_spec('acestep')
        if spec is not None and spec.submodule_search_locations:
            found += [Path(p) / 'models' for p in spec.submodule_search_locations]
    except (ImportError, ValueError):
        pass
    if os.environ.get('ACE_STEP_DIR'):
        found.append(Path(os.environ['ACE_STEP_DIR']) / 'acestep' / 'models')
    found.append(Path('/opt/ace-step/acestep/models'))
    for folder in found:
        if folder.is_dir():
            return folder
    raise WorkerError('The music engine is not installed on this worker.')


def build_checkpoints(target, main=None, dits=None, code_dir=None):
    """Lay out upstream's flat checkpoints folder from the RunPod snapshots without touching the shared cache.

    The VAE, text encoder and planner are symlinks (upstream only reads them). Each DiT gets a REAL folder: its
    weights and json are symlinks, but its .py files are COPIES of the pinned upstream ones. Upstream compares those
    files before loading and overwrites any that differ; a copy keeps that write inside this scratch folder instead
    of through a symlink into the shared model cache. `dits` maps a folder name to (snapshot, variant)."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    if main is not None:
        for name in MAIN_PARTS:
            _clear(target / name)
            _link(Path(main) / name, target / name)
        placeholder = target / PLACEHOLDER_DIT
        _clear(placeholder)
        placeholder.mkdir()
        (placeholder / 'model.safetensors').write_bytes(b'')   # upstream only tests that this file exists
    for name, (snapshot, variant) in (dits or {}).items():
        source = Path(code_dir) / variant
        code = sorted(p for p in source.glob('*.py') if p.name != '__init__.py') if source.is_dir() else []
        if not code:
            raise WorkerError('The music engine is not installed on this worker.')
        dest = target / name
        _clear(dest)
        dest.mkdir()
        for entry in sorted(Path(snapshot).iterdir()):
            if entry.suffix in LINK_SUFFIXES:
                _link(entry, dest / entry.name)
        for path in code:
            shutil.copy2(path, dest / path.name)


def ensure_checkpoints(models=('xl-turbo',), main=True):
    """Build (once per process) the checkpoints folder parts a load is about to need."""
    target = checkpoints_dir()
    dits, main_dir, code_dir = {}, None, None
    for model in models:
        spec = DIT_MODELS[model]
        if spec['folder'] in STATE['built']:
            continue
        folder = RESOLVED.get(spec['key'], {}).get('folder')
        if folder is None:
            raise WorkerError(f'The {spec["label"]} music model is not available on this worker yet.')
        dits[spec['folder']] = (folder, spec['variant'])
    if main and 'main' not in STATE['built']:
        main_dir = RESOLVED.get('main', {}).get('folder')
        if main_dir is None:
            raise WorkerError('The music models are not available on this worker yet.')
    if dits:
        code_dir = upstream_models_dir()
    if dits or main_dir is not None:
        build_checkpoints(target, main_dir, dits, code_dir)
        STATE['built'].update(dits)
        if main_dir is not None:
            STATE['built'].add('main')
    STATE['ckpt'] = target
    return target


# ---------------------------------------------------------------- the card and the engine

def gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name() if torch.cuda.is_available() else None
    except Exception:
        return None


def gpu_peak(reset=False):
    """PyTorch allocator peak in GiB for this process, to size cheaper 24 and 32 GB cards."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        if reset:
            torch.cuda.reset_peak_memory_stats()
            return None
        return round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    except Exception:
        return None


def gpu_total_gib():
    try:
        import torch
        return torch.cuda.get_device_properties(0).total_memory / 2 ** 30 if torch.cuda.is_available() else None
    except Exception:
        return None


def empty_cache():
    torch = sys.modules.get('torch')   # never import torch just to tidy up
    try:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def versions():
    version = None
    try:
        import torch
        version = torch.__version__
    except Exception:
        pass
    return {'ace_step_commit': os.environ.get('ACE_STEP_COMMIT', UPSTREAM_COMMIT), 'torch': version,
            'worker': os.environ.get('ACE_WORKER_VERSION', WORKER_VERSION)}


def configure_environment():
    """Process settings that must be in place before upstream is imported (it reads them while it loads)."""
    if STATE['env']:
        return
    os.environ.setdefault('ACESTEP_PROJECT_ROOT', '/tmp/ace-root')
    root = Path(os.environ['ACESTEP_PROJECT_ROOT'])
    os.environ['ACESTEP_CHECKPOINTS_DIR'] = str(checkpoints_dir())   # upstream must agree with the folder built here
    # ACESTEP_GENERATION_TIMEOUT: upstream abandons a render after 600 s by default, which a long xl-sft song
    # (50 steps with guidance) can exceed; the endpoint's own job timeout is the real limit.
    for name, value in (('HF_HUB_OFFLINE', '1'), ('TRANSFORMERS_OFFLINE', '1'), ('HF_HUB_DISABLE_TELEMETRY', '1'),
                        ('TOKENIZERS_PARALLELISM', 'false'), ('ACESTEP_DISABLE_TQDM', '1'),
                        ('ACESTEP_SAVE_MEMORY', '1'), ('ACESTEP_GENERATION_TIMEOUT', '3600'),
                        ('HF_MODULES_CACHE', '/tmp/hf-modules')):
        os.environ.setdefault(name, value)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    STATE['env'] = True


def quiet_logs():
    """loguru prints captions and lyrics to its default sink at INFO; RunPod keeps worker logs. Keep warnings only.

    backtrace and diagnose are off on purpose. With loguru's defaults (both on), every logger.exception() that
    upstream calls around a render (inference.py "Music generation failed" and others) prints the VALUES of the
    variables on each traceback line, and those frames hold the whole caption and lyrics."""
    try:
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, level='WARNING', backtrace=False, diagnose=False)
    except Exception:
        pass


def upstream():
    """The three upstream pieces used, imported on first use."""
    if STATE['api'] is None:
        configure_environment()
        quiet_logs()
        from acestep.handler import AceStepHandler
        from acestep.inference import GenerationConfig, GenerationParams, generate_music
        from acestep.llm_inference import LLMHandler
        STATE['api'] = SimpleNamespace(AceStepHandler=AceStepHandler, LLMHandler=LLMHandler,
                                       GenerationParams=GenerationParams, GenerationConfig=GenerationConfig,
                                       generate_music=generate_music)
    return STATE['api']


def require_gpu():
    try:
        import torch
        found = torch.cuda.is_available()
    except Exception:
        found = False
    if not found:
        raise WorkerError('No graphics card is available on this worker. Try again in a moment.')


def log_status(label, status):
    """Boot log: the attention and dtype lines of upstream's init status (an eager attention fallback is slow)."""
    lines = [line.strip() for line in str(status).splitlines()
             if line.startswith(('Attention:', 'Dtype:', 'Backend:', 'Device:'))]
    print(f'ACE-Step {label}:', '; '.join(lines) or 'ready', flush=True)
    if str(status).find('Attention:') >= 0 and 'Attention: sdpa' not in str(status):
        print(f'ACE-Step {label}: WARNING attention is not sdpa; renders will be slower', flush=True)


def release_dit(handler):
    """Free a DiT handler's weights (upstream has no unload; this mirrors its own failure path)."""
    for field in ('model', 'vae', 'text_encoder', 'text_tokenizer', 'config', 'silence_latent', '_base_decoder'):
        try:
            setattr(handler, field, None)
        except Exception:
            pass
    gc.collect()
    empty_cache()


def load_dit(model):
    spec = DIT_MODELS[model]
    api = upstream()
    handler = api.AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=os.environ['ACESTEP_PROJECT_ROOT'], config_path=spec['folder'], device='cuda',
        use_flash_attention=False, compile_model=False, offload_to_cpu=False, offload_dit_to_cpu=False,
        quantization=None, prefer_source=None, use_mlx_dit=False)
    if not ok:
        print(f'ACE-Step {model} load failed:', scrub(status, limit=300), flush=True)
        release_dit(handler)
        raise WorkerError(f'The {spec["label"]} music model could not be loaded.')
    log_status(model, status)
    try:
        from acestep.model_downloader import _check_code_mismatch
        differs = _check_code_mismatch(spec['folder'], STATE['ckpt'])
        if differs:
            print(f'ACE-Step {model}: upstream code differs from the checkpoint copy:', differs, flush=True)
    except Exception:
        pass
    return handler


def engine(model):
    """The loaded DiT handler for a model, loading it (and releasing the other DiT on a small card) if needed."""
    handler = STATE['dit'].get(model)
    if handler is not None and getattr(handler, 'model', None) is not None:
        return handler
    require_gpu()
    resolve_models(download=True, only=(DIT_MODELS[model]['key'], 'main'))
    ensure_checkpoints((model,))
    total = gpu_total_gib()
    if total is None or total < BOTH_MODELS_GIB:
        for other in [name for name in STATE['dit'] if name != model]:
            release_dit(STATE['dit'].pop(other))   # one DiT at a time: free the other before loading this one
    stale = STATE['dit'].pop(model, None)
    if stale is not None:
        release_dit(stale)      # a handler whose model was lost may still hold the VAE or text encoder
    began = time.monotonic()
    STATE['dit'][model] = load_dit(model)
    print(f'ACE-Step {model} loaded in {time.monotonic() - began:.1f}s', flush=True)
    return STATE['dit'][model]


def lm_backend():
    backend = os.environ.get('ACE_LM_BACKEND', 'pt').strip().lower()
    return backend if backend in ('pt', 'vllm') else 'pt'


def planner():
    """The loaded planner LM, or None when it is not available (the song is then made without it)."""
    llm = STATE['llm']
    if llm is not None and getattr(llm, 'llm_initialized', False):
        return llm
    if STATE['planner_tries'] >= 2:
        return None
    STATE['planner_tries'] += 1
    try:
        require_gpu()
        resolve_models(download=True, only=('main',))
        ensure_checkpoints(())
        began = time.monotonic()
        llm = upstream().LLMHandler()
        status, ok = llm.initialize(checkpoint_dir=str(STATE['ckpt']), lm_model_path=LM_FOLDER, backend=lm_backend(),
                                    device='cuda', offload_to_cpu=False)
        if not ok:
            print('ACE-Step planner load failed:', scrub(status, limit=300), flush=True)
            return None
        STATE['llm'] = llm
        print(f'ACE-Step planner loaded in {time.monotonic() - began:.1f}s, backend {getattr(llm, "llm_backend", None)}',
              flush=True)
        return llm
    except Exception as error:
        print('ACE-Step planner load failed:', type(error).__name__, flush=True)
        return None


def boot():
    """Load what a first job needs: xl-turbo, the planner (with its tokenizer) and the VAE."""
    began = time.monotonic()
    require_gpu()
    resolve_models(download=True, only=('turbo', 'main'))
    resolve_models(download=False, only=('sft',))   # sft is only looked for here; it loads, and downloads, on first use
    engine('xl-turbo')
    planner()
    return time.monotonic() - began


def warm():
    # FlashBoot snapshots a booted worker. Loading here, not inside the first job, puts a ready model in that
    # snapshot and keeps loading out of the billed song. A failure only falls back to loading on first use.
    try:
        print(f'ACE-Step ready at boot in {boot():.1f}s', flush=True)
    except Exception as error:
        print('ACE-Step boot load skipped:', type(error).__name__, flush=True)


# ---------------------------------------------------------------- request checks

def trim_caption(text, limit=CAPTION_MAX):
    """(caption, note): whitespace tidied, and cut at a word boundary when longer than the limit."""
    text = ' '.join(str(text).split())
    if len(text) <= limit:
        return text, None
    cut = text[:limit]
    if not text[limit].isspace():
        space = cut.rfind(' ')
        if space > 0:
            cut = cut[:space]
    cut = cut.rstrip(' ,;:-/&')
    return cut, f'The music description was shortened to fit {limit} characters.'


KEYSCALE = re.compile(r'^([A-Ga-g])\s*([#b\u266f\u266d]?)\s*([A-Za-z]*)$')


def normalize_keyscale(text):
    """'A Minor' -> 'A minor', 'f# min' -> 'F# minor', 'Bbm' -> 'Bb minor'; None when it is not a key."""
    match = KEYSCALE.match(str(text).strip())
    if not match:
        return None
    note, accidental, mode = match.groups()
    mode = {'major': 'major', 'maj': 'major', 'minor': 'minor', 'min': 'minor'}.get(mode.lower())
    if mode is None and match.group(3) in ('m', 'M'):
        mode = 'minor' if match.group(3) == 'm' else 'major'
    if mode is None:
        return None
    accidental = {'\u266f': '#', '\u266d': 'b'}.get(accidental, accidental)
    return f'{note.upper()}{accidental} {mode}'


TIME_SIGNATURES = {'2': '2', '3': '3', '4': '4', '6': '6', '2/4': '2', '3/4': '3', '4/4': '4', '6/8': '6'}


def _whole(value):
    if type(value) is int:
        return value
    if type(value) is float and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _real(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def request_input(raw):
    """Check a render request. Raises RequestError(plain sentence); returns the cleaned request and its notes."""
    mode = raw.get('mode')
    mode = 'render' if mode is None else mode
    if mode not in ('render', 'probe'):
        raise RequestError('Mode must be render or probe.')
    model = raw.get('model')
    model = 'xl-turbo' if model is None else model
    if not isinstance(model, str) or model not in DIT_MODELS:   # a list or dict is unhashable: no dict lookup first
        raise RequestError('Model must be xl-turbo or xl-sft.')
    spec = DIT_MODELS[model]
    notes = []

    caption = raw.get('caption')
    if not isinstance(caption, str):
        raise RequestError('Describe the music in a few words.' if caption is None
                           else 'The music description must be text.')
    caption, note = trim_caption(caption)
    if not caption:
        raise RequestError('Describe the music in a few words.')
    if note:
        notes.append(note)

    lyrics = raw.get('lyrics')
    lyrics = '' if lyrics is None else lyrics
    if not isinstance(lyrics, str):
        raise RequestError('Lyrics must be text.')
    lyrics = lyrics.replace('\r\n', '\n').replace('\r', '\n').strip()
    if len(lyrics) > LYRICS_MAX:
        raise RequestError(f'Lyrics must be {LYRICS_MAX} characters or fewer.')
    instrumental = not lyrics or lyrics.lower() == INSTRUMENTAL.lower()
    if instrumental:
        lyrics = INSTRUMENTAL

    duration = _real(raw.get('duration'))
    if duration is None or not DURATION_MIN <= duration <= DURATION_MAX:
        raise RequestError(f'Length must be between {DURATION_MIN} and {DURATION_MAX} seconds.')

    seed = raw.get('seed')
    if seed is not None:
        seed = _whole(seed)
        if seed is None or not 0 <= seed < SEED_LIMIT:
            raise RequestError(f'Seed must be a whole number from 0 to {SEED_LIMIT - 1}.')

    batch = raw.get('batch_size')
    batch = 1 if batch is None else _whole(batch)
    if batch is None or not 1 <= batch <= BATCH_MAX:
        raise RequestError(f'Batch size must be a whole number from 1 to {BATCH_MAX}.')

    bpm = raw.get('bpm')
    if bpm is not None:
        bpm = _whole(bpm)
        if bpm is None or not BPM_MIN <= bpm <= BPM_MAX:
            raise RequestError(f'Tempo must be a whole number from {BPM_MIN} to {BPM_MAX} BPM.')

    keyscale = raw.get('keyscale')
    if keyscale in (None, ''):
        keyscale = ''
    elif not isinstance(keyscale, str):
        raise RequestError('Key must be text, such as A minor.')
    else:
        keyscale = normalize_keyscale(keyscale) or ''
        if not keyscale:
            notes.append('The key was not understood, so it was left out.')

    signature = raw.get('timesignature')
    if signature in (None, ''):
        signature = ''
    else:
        signature = _whole(signature) if type(signature) is float else signature
        signature = TIME_SIGNATURES.get(str(signature).strip()) if type(signature) in (int, str) else None
        if signature is None:
            raise RequestError('Time signature must be 2, 3, 4 or 6.')

    language = raw.get('vocal_language')
    language = 'unknown' if language in (None, '') else language
    if not isinstance(language, str) or language.strip().lower() not in VALID_LANGUAGES:
        raise RequestError('Vocal language must be a short code such as en.')
    language = language.strip().lower()

    thinking = raw.get('thinking')
    thinking = True if thinking is None else thinking
    if type(thinking) is not bool:
        raise RequestError('Thinking must be true or false.')

    steps = raw.get('steps')
    if steps is not None:
        steps = _whole(steps)
        if steps is None or not spec['min_steps'] <= steps <= spec['max_steps']:
            raise RequestError(f'Steps must be a whole number from {spec["min_steps"]} to {spec["max_steps"]}.')
    guidance = raw.get('guidance_scale')
    if guidance is not None:
        guidance = _real(guidance)
        if guidance is None or not 1 <= guidance <= 15:
            raise RequestError('Guidance must be a number from 1 to 15.')
    shift = raw.get('shift')
    if shift is not None:
        shift = _real(shift)
        if shift is None or not 1 <= shift <= 5:
            raise RequestError('Shift must be a number from 1 to 5.')

    return dict(mode=mode, model=model, caption=caption, lyrics=lyrics, instrumental=instrumental,
                duration=float(duration), seed=seed, batch_size=batch, bpm=bpm, keyscale=keyscale,
                timesignature=signature, vocal_language=language, thinking=thinking, steps=steps,
                guidance_scale=guidance, shift=shift, notes=notes)


def refuse_adapter(raw):
    """lora_key is reserved for a later phase: refuse it before any work is done."""
    if raw.get('lora_key') not in (None, ''):
        raise RequestError(ADAPTER_SENTENCE)


def apply_adapter(pipe, key):
    """RESERVED (LoRA phase). Today a request with lora_key is refused before the pipeline is touched.

    Where it will go: fetch the adapter file for `key` from B2 into the job's temp folder, then on the DiT handler
    `pipe`: add_lora(folder, name), set_lora_scale(name, scale) (acestep core/generation/handler/lora_manager.py),
    render, and in a finally block unload_lora() so the next job starts from the plain model. peft is not in the
    image yet, and the LoRA manager refuses quantized models, which is why INT8 stays off."""
    raise NotImplementedError(ADAPTER_SENTENCE)


def take_seeds(seed, count):
    """The seed of each take: seed + i, wrapping inside 0..2147483647. No seed means a random one."""
    base = secrets.randbelow(SEED_LIMIT) if seed is None else seed
    return [(base + index) % SEED_LIMIT for index in range(count)]


def settings_for(request):
    """The values actually sent to the DiT, and plain sentences about them."""
    spec = DIT_MODELS[request['model']]
    steps = request['steps'] or spec['steps']
    shift = request['shift'] if request['shift'] is not None else spec['shift']
    notes = []
    if request['model'] == 'xl-turbo':
        guidance = spec['guidance']
        if request['guidance_scale'] not in (None, guidance):
            notes.append('The fast model ignores the guidance setting.')
        notes.append(f'Made with the fast model in {steps} steps.')
    else:
        guidance = request['guidance_scale'] if request['guidance_scale'] is not None else spec['guidance']
        notes.append(f'Made with the best-quality model in {steps} steps at guidance {guidance:g}.')
    return dict(steps=steps, guidance_scale=float(guidance), shift=float(shift), dcw_enabled=spec['dcw']), notes


# ---------------------------------------------------------------- one render

def build_generation(api, request, settings, seeds, thinking):
    """GenerationParams and GenerationConfig for one request (field names checked against the pinned source)."""
    rewrite = thinking and os.environ.get('ACE_PLANNER_REWRITES_CAPTION') == '1'
    strengths = {}
    if settings['dcw_enabled']:
        scaler, high_scaler = DCW_STRENGTHS[bool(thinking)]
        strengths = dict(dcw_scaler=scaler, dcw_high_scaler=high_scaler)
    params = api.GenerationParams(
        task_type='text2music', caption=request['caption'], lyrics=request['lyrics'],
        instrumental=request['instrumental'], vocal_language=request['vocal_language'], bpm=request['bpm'],
        keyscale=request['keyscale'], timesignature=request['timesignature'], duration=request['duration'],
        inference_steps=settings['steps'], guidance_scale=settings['guidance_scale'], shift=settings['shift'],
        dcw_enabled=settings['dcw_enabled'], seed=seeds[0], **strengths,
        # thinking False must skip the planner altogether: upstream still runs it for any use_cot_* flag left on.
        # The caller's wording and language are kept (use_cot_caption / use_cot_language off); the planner fills
        # tempo, key and time signature, and writes the 5 Hz audio codes the DiT follows.
        thinking=thinking, use_cot_metas=thinking, use_cot_caption=rewrite, use_cot_language=False)
    # use_random_seed must be False or upstream ignores the seeds list and draws its own.
    config = api.GenerationConfig(batch_size=len(seeds), use_random_seed=False, seeds=list(seeds))
    return params, config


def seed_everything(seed):
    """The planner is unseeded in upstream for a single take; seed the global generators so `seed` reproduces it."""
    random.seed(seed)
    torch = sys.modules.get('torch')
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


PHASES = (('Phase 1', 'Planning your song'), ('Phase 2', 'Writing the musical ideas'),
          ('Preparing inputs', 'Rendering your song'), ('Preparing audio', 'Finishing the audio'))


def announce(job, text):
    """Tell the booth, in a plain phrase, what the worker is doing now. Never raises."""
    try:
        import runpod
        runpod.serverless.progress_update(job, text)
    except Exception:
        pass


def make_progress(job, send=None):
    """A callback upstream can call as (ratio, 'text') or (ratio, desc='text'); forwards a few plain phrases."""
    send = send or (lambda text: announce(job, text))
    last = []

    def progress(*args, **kwargs):
        try:
            text = kwargs.get('desc') or (args[1] if len(args) > 1 else '')
            for marker, phrase in PHASES:
                if marker in str(text) and phrase not in last:
                    last[:] = [phrase]
                    send(phrase)
                    break
        except Exception:
            pass
    return progress


LINK = re.compile(r"""https?://[^\s'"<>()\[\]]*[^\s'"<>()\[\].,;:!?]""", re.IGNORECASE)
HUB_TOKEN = re.compile(r'\bhf_[A-Za-z0-9]{8,}')


def scrub(text, *private, limit=200):
    """First line of an upstream message for the worker log, with the caption and lyrics, web links and Hugging Face
    tokens removed (the contract allows the exception type; this keeps upstream's one-line reason as well)."""
    lines = str(text or '').strip().splitlines()
    line = lines[0] if lines else ''
    for secret in private:
        if secret and len(secret) >= 6:
            line = line.replace(secret, '[hidden]')
    line = HUB_TOKEN.sub('[token]', LINK.sub('[link]', line))
    return line[:limit]


def plain_failure(error):
    low = str(error or '').lower()
    return TOO_BIG if 'insufficient free vram' in low or 'out of memory' in low else FAILED


def storage():
    import boto3
    client = boto3.client('s3', endpoint_url=os.environ['AWS_ENDPOINT_URL'],
                          region_name=os.environ.get('AWS_REGION', 'us-east-005'))
    return client, os.environ['AWS_BUCKET_NAME']


def ffmpeg(*args):
    subprocess.run(['ffmpeg', '-nostdin', '-y', '-v', 'error', *map(str, args)],
                   check=True, timeout=180, capture_output=True)


def write_float_wav(path, tensor, rate):
    """Upstream's CPU tensor [channels, samples] as a float WAV for ffmpeg."""
    import soundfile as sf
    data = tensor.detach().cpu().float().numpy()
    data = data[:, None] if data.ndim == 1 else data.T
    sf.write(str(path), data, int(rate), subtype='FLOAT')


def audio_seconds(tensor, rate):
    return tensor.shape[-1] / float(rate)


def encode_take(audio, folder):
    """master.wav (24-bit PCM, 48 kHz) and master.mp3 (320k) for one take, made with ffmpeg as yue_handler does."""
    rate = audio.get('sample_rate') or SAMPLE_RATE
    wav, mp3, source = folder / 'master.wav', folder / 'master.mp3', folder / 'audio.wav'
    write_float_wav(source, audio['tensor'], rate)
    ffmpeg('-i', source, '-c:a', 'pcm_s24le', wav)
    source.unlink()
    ffmpeg('-i', wav, '-c:a', 'libmp3lame', '-b:a', '320k', mp3)
    return wav, mp3, round(audio_seconds(audio['tensor'], rate), 2)


def upload_take(client, bucket, wav, mp3):
    """Upload one take under ace/<uuid4>/ and return its keys and 7-day links."""
    prefix = f'ace/{uuid.uuid4()}'
    for path, mime in ((mp3, 'audio/mpeg'), (wav, 'audio/wav')):
        client.upload_file(str(path), bucket, prefix + '/' + path.name, ExtraArgs={'ContentType': mime})

    def url(name):
        return client.generate_presigned_url('get_object', Params={'Bucket': bucket, 'Key': prefix + '/' + name},
                                             ExpiresIn=URL_SECONDS)
    return {'key': prefix + '/master.mp3', 'wav_key': prefix + '/master.wav',
            'url': url('master.mp3'), 'wav_url': url('master.wav')}


def join_names(names):
    """['tempo', 'key', 'time signature'] -> 'tempo, key and time signature'."""
    return names[0] if len(names) < 2 else ', '.join(names[:-1]) + ' and ' + names[-1]


def planned(metadata, request):
    """What the planner decided that the caller did not set: tempo, key, time signature. None when it decided nothing."""
    if not isinstance(metadata, dict):
        return None
    plan = {}
    try:
        if request['bpm'] is None and metadata.get('bpm') not in (None, '', 'N/A'):
            bpm = _whole(float(metadata['bpm']))
            if bpm is not None and BPM_MIN <= bpm <= BPM_MAX:
                plan['bpm'] = bpm
    except (TypeError, ValueError):
        pass
    if not request['keyscale'] and isinstance(metadata.get('keyscale'), str):
        key = normalize_keyscale(metadata['keyscale'])
        if key:
            plan['keyscale'] = key
    if not request['timesignature'] and metadata.get('timesignature') not in (None, '', 'N/A'):
        signature = TIME_SIGNATURES.get(str(metadata['timesignature']).strip())
        if signature:
            plan['timesignature'] = signature
    return plan or None


def render(job, request, start):
    notes = list(request['notes'])
    settings, setting_notes = settings_for(request)
    notes += setting_notes
    seeds = take_seeds(request['seed'], request['batch_size'])
    progress = make_progress(job)
    with tempfile.TemporaryDirectory() as td:
        client, bucket = storage()   # a missing setting fails here, before any GPU time is spent
        load_began = time.monotonic()
        if request['model'] not in STATE['dit']:
            announce(job, 'Loading the music model')   # xl-sft loads on its first use and takes a while
        dit = engine(request['model'])
        llm = planner() if request['thinking'] else None
        if request['thinking'] and llm is None:
            notes.append('The planner was not available, so the song was made without it.')
        elif not request['thinking']:
            notes.append('The planner was skipped.')
        model_load_s = round(time.monotonic() - load_began, 1)

        # LoRA seam: a request's lora_key would be applied here (apply_adapter(dit, key)) and removed in a finally
        # after generate_music. request_input refuses lora_key today, so nothing is applied.
        api = upstream()
        params, config = build_generation(api, request, settings, seeds, llm is not None)
        seed_everything(seeds[0])
        gpu_peak(reset=True)
        began = time.monotonic()
        result = api.generate_music(dit, llm, params, config, save_dir=None, progress=progress)
        generate_s = time.monotonic() - began
        peak = gpu_peak()
        audios = [a for a in (result.audios or []) if a.get('tensor') is not None] if result.success else []
        if not audios:
            print('ACE-Step generation failed:', scrub(result.error, request['caption'], request['lyrics']), flush=True)
            raise WorkerError(plain_failure(result.error))
        if len(audios) < len(seeds):
            notes.append(f'Only {len(audios)} of {len(seeds)} takes fit in memory.')
        extra = result.extra_outputs or {}
        costs = extra.get('time_costs') or {}
        plan_s = round(float(costs['lm_total_time']), 1) if costs.get('lm_total_time') is not None else None
        plan = planned(extra.get('lm_metadata'), request) if plan_s is not None else None
        if plan_s is not None:
            names = {'bpm': 'tempo', 'keyscale': 'key', 'timesignature': 'time signature'}
            chosen = [names[key] for key in names if plan and key in plan]
            notes.append(f'The planner chose the {join_names(chosen)}.' if chosen else 'The planner shaped the song.')

        takes, encode_s, upload_s = [], 0.0, 0.0
        for index, audio in enumerate(audios):
            folder = Path(td) / f'take{index}'
            folder.mkdir()
            began = time.monotonic()
            wav, mp3, seconds = encode_take(audio, folder)
            encode_s += time.monotonic() - began
            began = time.monotonic()
            links = upload_take(client, bucket, wav, mp3)
            upload_s += time.monotonic() - began
            takes.append({'index': index, 'seed': seeds[index], **links, 'duration_s': seconds,
                          'bytes': mp3.stat().st_size})
    timing = {'model_load_s': model_load_s, 'render_s': round(max(generate_s - (plan_s or 0), 0), 1),
              'encode_s': round(encode_s, 1), 'upload_s': round(upload_s, 1)}
    if plan_s is not None:
        timing['plan_s'] = plan_s
    first = takes[0]
    output = {'engine': 'ace', 'mode': 'render', 'model': request['model'], 'key': first['key'],
              'wav_key': first['wav_key'], 'url': first['url'], 'wav_url': first['wav_url'],
              'duration_s': first['duration_s'], 'bytes': first['bytes'], 'seed': first['seed'],
              'processing_ms': int((time.monotonic() - start) * 1000), 'truncated': False, 'takes': takes,
              'instrumental': request['instrumental'], 'thinking': llm is not None,
              'controls': {'steps': settings['steps'], 'guidance_scale': settings['guidance_scale'],
                           'shift': settings['shift']},
              'timing': timing, 'memory': {'render_peak_gib': peak}, 'gpu': gpu_name(),
              'features': list(FEATURES), 'versions': versions(), 'worker_notes': notes}
    if plan:
        output['plan'] = plan
    return output


def probe(start):
    """No render and no weights read: which model folders this worker found, versions and the card."""
    resolve_models(download=False)
    models = {}
    for _env, repo, key in MODELS:
        found = RESOLVED.get(key, {'folder': None, 'how': 'missing'})
        models[key] = {'repo': repo, 'source': found['how'],
                       'revision': found['folder'].name if found['folder'] is not None else None}
    loaded = {'xl-turbo': 'xl-turbo' in STATE['dit'], 'xl-sft': 'xl-sft' in STATE['dit'],
              'planner': STATE['llm'] is not None}
    return {'engine': 'ace', 'mode': 'probe', 'models': models, 'loaded': loaded, 'versions': versions(),
            'gpu': gpu_name(), 'features': list(FEATURES), 'processing_ms': int((time.monotonic() - start) * 1000)}


def handler(job):
    start = time.monotonic()
    try:
        raw = job.get('input') if isinstance(job, dict) else None
        raw = {} if raw is None else raw
        if not isinstance(raw, dict):
            raise RequestError('Send the request as named fields.')
        refuse_adapter(raw)
        if raw.get('mode') == 'probe':
            return probe(start)
        return render(job, request_input(raw), start)
    except RequestError as error:
        return {'error': str(error)}
    except WorkerError as error:
        print('ACE-Step failed:', type(error).__name__, flush=True)
        return {'error': str(error)}
    except Exception as error:
        if isinstance(error, NotImplementedError) and str(error) == ADAPTER_SENTENCE:
            return {'error': ADAPTER_SENTENCE}   # apply_adapter's own refusal, not a torch "not implemented"
        # only the type is logged: a message can carry the caption, the lyrics or a URL
        print('ACE-Step failed:', type(error).__name__, flush=True)
        gc.collect()
        empty_cache()
        if 'OutOfMemory' in type(error).__name__ or 'out of memory' in str(error).lower():
            return {'error': TOO_BIG}
        return {'error': f'ACE-Step could not finish ({type(error).__name__}). Your writing is saved.'}


if __name__ == '__main__':
    import runpod
    warm()
    runpod.serverless.start({'handler': handler})
