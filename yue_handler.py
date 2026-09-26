"""One full-quality YuE2 candidate per request; private audio and score outputs.

Part 295 adds upstream's official cover and instrumental recipe (YuE #202) as
optional request fields: keep_harmony, instrumental, length_guard,
match_score_tempo and mode. A request without them renders exactly as before."""
import os
import json
import subprocess
import tempfile
import time
import uuid
import math
import re
import sys
from dataclasses import replace
from pathlib import Path

# Upstream's MIT yue2-music instrumental helpers, standard library only, kept verbatim.
# Appended, not prepended, so their plain module names never shadow an installed package.
sys.path.append(str(Path(__file__).resolve().parent / 'instrumental'))
from abc_tools import parse_abc  # noqa: E402
from common import sha256  # noqa: E402
from instrumentalize import convert_score, section_starts  # noqa: E402

PIPE = None
STYLE = {'applied': None, 'plain': None}
STYLE_KEY = re.compile(r'^yue2-loras/[A-Za-z0-9._-]{1,80}\.pt$')
# What this worker can do; the booth checks this before offering an option.
FEATURES = ('keep-harmony', 'instrumental', 'transcribe', 'score-cache', 'length-guard',
            'match-tempo', 'sections', 'lyric-fit')
TOKENS_PER_SECOND, TOKEN_CAP, TOKEN_FLOOR, GUARD_HEADROOM = 25, 9000, 200, 1.10
NO_SINGING = ('no vocals', 'no singing', 'no choir', 'no spoken words')
PLANNING_TAGS = '[Intro]\n\n[Verse]\n\n[Chorus]\n\n[Outro]\n'
UNREADABLE = 'The melody could not be read from this recording. Try a clearer recording of the song.'
UNMOVABLE = {
    'recording': "This recording's melody could not be moved onto an instrument. Try a sung cover, or another recording.",
    'score': "This score's melody could not be moved onto an instrument. Try a sung version, or another score.",
    'YuE2': "YuE2's score for this instrumental could not be moved onto an instrument. Try another seed.",
}
BPM = re.compile(r'\b\d{2,3}(?:\.\d+)?\s*bpm\b', re.I)
TAG = re.compile(r'^\s*\[([^\]\n]+)\]\s*$')
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")


def sound_controls(inp):
    def number(key, default, low, high, integer=False):
        value = inp.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high or (integer and type(value) is not int):
            raise ValueError(f'{key} must be between {low} and {high}' + (' in whole numbers.' if integer else '.'))
        return value
    return dict(weirdness=number('weirdness', 50, 0, 100, True),
                steps=number('steps', 32, 16, 64, True),
                guidance=number('guidance', 1.0, 1.0, 3.0))


def render_song(pipe, inp, controls, max_tokens=None):
    # Reset per request: a previous song's advanced settings must not leak.
    original = pipe.generation_config
    pipe.generation_config = replace(original, ode_steps=controls['steps'])
    variation = controls['weirdness']
    # Score-free songs use YuE2's own 1.01 default; at exactly 1.0 that mode loses its guidance pass.
    guidance = 1.01 if inp.get('cot') == 'off' and controls['guidance'] == 1.0 else controls['guidance']
    semantic = {'temperature': round(.7 + .006 * variation, 3)}
    if max_tokens is not None:
        semantic['max_tokens'] = max_tokens  # length guard; without it YuE2 allows 9,000 (six minutes)
    try:
        return pipe(**inp, cfg_scale=guidance,
                    abc_sampling={'temperature': round(.4 + .006 * variation, 3)},
                    semantic_sampling=semantic)
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


def cover_options(inp):
    """Part 295 fields, all optional. Leaving them out keeps today's request exactly.

    keep_harmony: a recording is transcribed with its chords and rendered with cot=full.
    instrumental: the singer's notes move to the instrument part before rendering.
    length_guard: caps the render near the score's length (on by default only
    when a new field is used). match_score_tempo: the style names the score's BPM.
    mode: render (default) or transcribe (read the melody only, no song)."""
    mode = inp.get('mode', 'render')
    if not isinstance(mode, str) or mode not in ('render', 'transcribe'):
        raise ValueError('That kind of request is not available on this music worker yet.')

    def flag(key):
        value = inp.get(key)
        if value is not None and type(value) is not bool:
            raise ValueError(f'{key} must be true or false.')
        return value
    keep, instrumental, guard, tempo = map(flag, ('keep_harmony', 'instrumental', 'length_guard', 'match_score_tempo'))
    new_shape = bool(keep or instrumental or tempo)
    return dict(mode=mode, keep_harmony=keep, instrumental=bool(instrumental),
                length_guard=new_shape if guard is None else guard, match_score_tempo=bool(tempo))


def check_combination(inp, options, reference):
    fixed = bool(reference or inp['abc'])
    if options['instrumental'] and inp['cot'] == 'off':
        raise ValueError('An instrumental needs YuE2 to write a score first.')
    if options['keep_harmony'] and not fixed and not options['instrumental']:
        raise ValueError('Keeping the original chords needs a source recording or a score.')
    if options['match_score_tempo'] and not fixed and not options['instrumental']:
        raise ValueError("Matching the score's tempo needs a source recording, a score or an instrumental.")


def instrumental_style(style):
    """Upstream prepare(): the exact prefix and suffix of an instrumental request."""
    style = style.strip().rstrip('.,') or 'Expressive instrumental music'
    if not re.match(r'^instrumental\b', style, re.I):
        style = 'Instrumental, ' + style
    for condition in NO_SINGING:
        if condition not in style.lower():
            style += ', ' + condition
    return style + '.'


def lyric_tags(text):
    """Upstream lyric_tags(): the score's section names, never sung."""
    labels = [line[2:] for line in text.splitlines() if line.startswith('% ')]
    return '\n\n'.join('[' + x.title() + ']' for x in labels) + ('\n' if labels else '')


def tempo_style(style, bpm):
    """Name the score's own tempo in the style; upstream says keep the two consistent."""
    if BPM.search(style):
        return BPM.sub(f'{bpm} BPM', style)
    return style.rstrip().rstrip('.,') + f', {bpm} BPM'


def semantic_budget(*seconds):
    """Report protocol: 25 acoustic tokens per source second, capped at 9,000, plus 10% headroom."""
    known = [s for s in seconds if s]
    if not known:
        return None
    return max(TOKEN_FLOOR, min(TOKEN_CAP, math.ceil(round(TOKENS_PER_SECOND * max(known) * GUARD_HEADROOM, 6))))


def score_sections(abc, score):
    """[{name, bars, seconds, sung_notes}] from the score's native % section comments."""
    try:
        starts = section_starts(abc, score)
    except ValueError:
        return []
    vocal, parts = score.voices['Vocal'], []
    for start, length, _ in vocal.bars:
        if start in starts or not parts:
            parts.append(dict(name=starts.get(start), start=start, bars=0, quarters=0, sung_notes=0))
        parts[-1]['bars'] += 1
        parts[-1]['quarters'] += length
    for onset, _, _ in vocal.notes:
        next(p for p in reversed(parts) if p['start'] <= onset)['sung_notes'] += 1
    return [dict(name=p['name'], bars=p['bars'], seconds=round(float(p['quarters'] * 60 / score.bpm), 1),
                 sung_notes=p['sung_notes']) for p in parts]


def score_facts(abc):
    """Tempo, key, meter, nominal length and sections; empty when the score cannot be read."""
    try:
        score = parse_abc(abc)
        vocal = score.voices['Vocal']
        n, d = vocal.bars[0][2]
        return dict(score_bpm=score.bpm, score_musical_key=vocal.keys[0][1], score_meter=f'{n}/{d}',
                    score_seconds=round(float(vocal.time * 60 / score.bpm), 2),
                    sections=score_sections(abc, score))
    except (ValueError, KeyError, TypeError, IndexError, ZeroDivisionError):
        return {}


def syllables(word):
    """Rough English syllable count: vowel groups, less a silent final e, at least one."""
    word = word.lower()
    count = len(re.findall(r'[aeiouy]+', word))
    if count > 1 and word.endswith('e') and not word.endswith(('le', 'ee', 'ye')):
        count -= 1
    return max(1, count)


def section_name(tag):
    name = re.split(r'[:(\[,]', (tag or '').lower())[0]
    name = re.sub(r'[\s_-]*\d+\s*$', '', name).strip()
    return re.sub(r'^pre[\s_-]*chorus$', 'pre-chorus', name)


def lyric_sections(lyrics):
    parts = []
    for line in lyrics.splitlines():
        tag = TAG.match(line)
        if tag:
            parts.append(dict(tag=tag.group(1).strip(), syllables=0))
            continue
        words = WORD.findall(line)
        if words:
            if not parts:
                parts.append(dict(tag=None, syllables=0))
            parts[-1]['syllables'] += sum(map(syllables, words))
    return [p for p in parts if p['syllables']]


def sung_sections(sections, pickup=8):
    """Sections that carry singing. A handful of notes (a pickup into the next
    section, or an ad-lib) joins the next sung section instead of counting alone."""
    sung, carry = [], 0
    for section in sections:
        notes = section['sung_notes'] + carry
        if notes >= pickup:
            sung.append(dict(name=section['name'], sung_notes=notes))
            carry = 0
        else:
            carry = notes
    if carry and sung:
        sung[-1]['sung_notes'] += carry
    elif carry:
        sung.append(dict(name=sections[-1]['name'], sung_notes=carry))
    return sung


def lyric_fit(lyrics, sections):
    """Report only, never blocks: each lyric section's words against the notes of its tune."""
    sung = sung_sections(sections)
    words = lyric_sections(lyrics)
    if not sung or not words:
        return None
    rows = []
    for index in range(max(len(sung), len(words))):
        tune = sung[index] if index < len(sung) else None
        text = words[index] if index < len(words) else None
        row = dict(score_section=tune and tune['name'], lyrics_section=text and text['tag'],
                   sung_notes=tune and tune['sung_notes'], syllables=text and text['syllables'])
        if tune and text:
            ratio = text['syllables'] / tune['sung_notes']
            row['fit'] = 'short' if ratio < .6 else 'long' if ratio > 1.4 else 'close'
        else:
            row['fit'] = 'no words' if tune else 'no tune'
        rows.append(row)
    return dict(scope='rough syllable count against sung notes; a guide only', sections=rows,
                same_order=[section_name(w['tag']) for w in words] == [s['name'] for s in sung])


def fixed_score_render(inp, options, abc, origin, source_seconds=None):
    """The exact request YuE2 renders once its score is fixed: a recording's
    transcription, a given score or a YuE2 plan. Pure: no GPU, files or network.

    Returns (render request, semantic token budget or None, score facts, extras, notes)."""
    render, extras, notes = dict(inp, abc=abc), {}, []
    facts = score_facts(abc)
    style = inp['style']
    if options['match_score_tempo'] and facts:
        style = tempo_style(style, facts['score_bpm'])
    if options['instrumental']:
        keep = options['keep_harmony']
        if keep is None:
            keep = origin != 'recording'  # upstream: covers drop chords; given scores and plans keep them
        try:
            converted, check = convert_score(abc, keep_chords=keep)
        except (ValueError, KeyError, TypeError, IndexError) as error:
            print('Instrumental conversion failed:', type(error).__name__, str(error)[:300], flush=True)
            raise ValueError(UNMOVABLE[origin]) from None
        chords = bool(parse_abc(converted).voices['Vocal'].chords)
        render.update(abc=converted, lyrics=lyric_tags(converted), style=instrumental_style(style),
                      cot='full' if chords else 'melody')
        extras['transfer'] = dict(vocal_notes_moved=check['vocal_notes_before'],
                                  original_ins_notes=check['original_ins_notes'],
                                  unaltered_ins_notes=check['unaltered_ins_notes'],
                                  ins_notes_trimmed=len(check['affected_ins_notes']),
                                  output_ins_notes=check['output_ins_notes'], chords_kept=chords,
                                  overlap_policy=check['overlap_policy'])
        if lyric_sections(inp['lyrics']):
            notes.append("Your words guided YuE2's plan only; an instrumental sings none of them." if origin == 'YuE2'
                         else "An instrumental sings no words, so the score's section names replaced your lyrics.")
    else:
        render['style'] = style
        if origin == 'recording':
            render['cot'] = 'full' if options['keep_harmony'] else 'melody'
        elif options['keep_harmony']:
            render['cot'] = 'full'
        extras['lyric_fit'] = lyric_fit(inp['lyrics'], facts.get('sections', []))
    budget = semantic_budget(source_seconds, facts.get('score_seconds')) if options['length_guard'] else None
    return render, budget, facts, extras, notes


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


def park_pipeline():
    """Before SheetSage2 shares the GPU, make sure YuE2 holds no GPU memory.

    The pipeline itself is kept, so a cover no longer re-hashes and reloads about
    7.8 GB of weights. decode() already parks the composer on the CPU after
    every song; only a render that failed midway can leave weights on the GPU."""
    if PIPE is None:
        return
    try:
        import torch
        for part in (getattr(PIPE, '_model', None), getattr(PIPE, '_vae', None)):
            if part is not None and next(part.parameters()).device.type != 'cpu':
                part.to('cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as error:
        print('YuE2 parking skipped:', type(error).__name__, flush=True)


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
        return round(torch.cuda.max_memory_allocated() / 2**30, 2)
    except Exception:
        return None


def token_count(pipe, abc):
    try:
        return len(pipe.tokenizer.encode(abc)) if abc else None
    except Exception:
        return None


def storage():
    import boto3
    client = boto3.client('s3', endpoint_url=os.environ['AWS_ENDPOINT_URL'],
                          region_name=os.environ.get('AWS_REGION', 'us-east-005'))
    return client, os.environ['AWS_BUCKET_NAME']


def score_cache_key(digest, task):
    revision = Path(os.environ.get('YUE_SCORE_DIR', 'unknown')).name[:12]
    return f'yue2-scores/{digest}/{task}-{revision}'


def read_recording(reference, td, task, client, bucket):
    """Download, check and transcribe a source recording, once per recording and task.

    The score is cached in the private bucket by the recording's hash, the task
    (melody or full) and the SheetSage2 revision, so takes 2 to 4 of a cover
    skip transcription. Only the few-KB score is stored, never the audio."""
    import soundfile as sf
    from auk_handler import ffmpeg, download
    timing, began = {}, time.monotonic()
    source = Path(td) / 'source'
    download(reference, source)
    decoded = Path(td) / 'source.wav'
    ffmpeg('-i', source, '-ar', 24000, '-ac', 1, decoded)
    seconds = sf.info(decoded).duration
    if seconds > 360:
        raise ValueError('Use a source recording up to six minutes long.')
    timing['download_s'] = round(time.monotonic() - began, 1)
    key = score_cache_key(sha256(source), task)
    began = time.monotonic()
    try:
        abc = client.get_object(Bucket=bucket, Key=key + '.abc')['Body'].read().decode('utf-8')
        meta = json.loads(client.get_object(Bucket=bucket, Key=key + '.json')['Body'].read())
        timing['transcribe_s'] = round(time.monotonic() - began, 1)
        return dict(abc=abc, warnings=meta.get('warnings', []), cached=True, key=key,
                    seconds=seconds, timing=timing, peak=None)
    except Exception as error:
        if type(error).__name__ not in ('NoSuchKey', 'ClientError'):
            print('Score cache read skipped:', type(error).__name__, flush=True)
    park_pipeline()
    score_dir = Path(td) / 'transcription'
    env = {**os.environ, 'LD_LIBRARY_PATH': '/opt/cover/lib', 'PATH': '/opt/cover/bin:'+os.environ['PATH']}
    try:
        subprocess.run(['/opt/cover/bin/python', '/app/cover.py', str(decoded), str(score_dir), task],
                       env=env, check=True, timeout=600, capture_output=True)
        abc = (score_dir/'score.abc').read_text()
        warnings = json.loads((score_dir/'warnings.json').read_text())
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        tail = getattr(error, 'stderr', None) or b''
        print('SheetSage2 failed:', type(error).__name__, tail[-1500:].decode('utf-8', 'replace'), flush=True)
        raise ValueError(UNREADABLE) from None
    timing['transcribe_s'] = round(time.monotonic() - began, 1)
    peak = None
    try:
        peak = json.loads((score_dir/'memory.json').read_text())['peak_allocated_gib']
    except Exception:
        pass
    try:
        client.put_object(Bucket=bucket, Key=key + '.abc', Body=abc.encode('utf-8'), ContentType='text/plain')
        client.put_object(Bucket=bucket, Key=key + '.json', ContentType='application/json', Body=json.dumps(
            dict(warnings=warnings, task=task, source_seconds=round(seconds, 2),
                 sheetsage2=Path(os.environ.get('YUE_SCORE_DIR', 'unknown')).name)).encode('utf-8'))
    except Exception as error:
        print('Score cache write skipped:', type(error).__name__, flush=True)
        key = None
    return dict(abc=abc, warnings=warnings, cached=False, key=key, seconds=seconds, timing=timing, peak=peak)


def planning_request(inp):
    """Text to instrumental (upstream plan_with_model): what YuE2 plans from.

    The planning lyrics are section tags, or her words, and are never sung."""
    return dict(style=instrumental_style(inp['style']), lyrics=inp['lyrics'] + '\n' if inp['lyrics'] else PLANNING_TAGS,
                cot=inp['cot'], seed=inp['seed'])


def plan_instrumental(pipe, inp, controls):
    """YuE2 writes the instrumental's score first; a failed plan is an error, never a fallback."""
    from yue2.protocol import SongRequest
    request = SongRequest(**planning_request(inp))
    # Same score temperature render_song uses for this weirdness.
    plan = pipe.plan(request=request, abc_sampling={'temperature': round(.4 + .006 * controls['weirdness'], 3)})
    if plan.truncated or not plan.abc:
        raise ValueError('YuE2 could not write a usable score for this instrumental. Try another seed.')
    return plan.abc


def transcribe_only(job, inp, options, start):
    """mode=transcribe: read the recording's melody and return the score, no song."""
    reference = inp.get('reference_voice_url')
    if not reference or not isinstance(reference, str) or len(reference) > 12000:
        raise ValueError('Reading a melody needs a source recording. Import it again.')
    import runpod
    task = 'full' if options['keep_harmony'] else 'melody'
    with tempfile.TemporaryDirectory() as td:
        client, bucket = storage()
        runpod.serverless.progress_update(job, 'Reading the melody of your recording')
        began = time.monotonic()
        read = read_recording(reference, td, task, client, bucket)
        key = read['key']
        if key is None:
            key = f'yue2/{uuid.uuid4()}/score'
            client.put_object(Bucket=bucket, Key=key + '.abc', Body=read['abc'].encode('utf-8'),
                              ContentType='text/plain')
        facts = score_facts(read['abc'])
        return {'engine': 'yue2', 'mode': 'transcribe', 'abc': read['abc'], 'score_key': key + '.abc',
                'score_url': client.generate_presigned_url('get_object', Params={'Bucket': bucket, 'Key': key + '.abc'},
                                                           ExpiresIn=604800),
                'cover_mode': 'harmony' if options['keep_harmony'] else 'melody',
                'score_bpm': facts.get('score_bpm'), 'score_musical_key': facts.get('score_musical_key'),
                'score_meter': facts.get('score_meter'), 'score_seconds': facts.get('score_seconds'),
                'sections': facts.get('sections', []), 'source_seconds': round(read['seconds'], 2),
                'transcription_warnings': read['warnings'], 'transcription_cached': read['cached'],
                'processing_ms': int((time.monotonic() - start) * 1000),
                'timing': {'cover_s': round(time.monotonic() - began, 1), **read['timing']},
                'gpu': gpu_name(), 'memory': {'transcribe_peak_gib': read['peak']}, 'features': list(FEATURES)}


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
        raw = job.get('input') or {}
        options = cover_options(raw)
        if options['mode'] == 'transcribe':
            return transcribe_only(job, raw, options, start)
        inp = request_input(raw)
        controls = sound_controls(raw)
        wanted = style_request(raw)
        reference = raw.get('reference_voice_url')
        check_combination(inp, options, reference)
        import runpod
        import soundfile as sf
        from auk_handler import ffmpeg
        with tempfile.TemporaryDirectory() as td:
            client, bucket = storage()
            warnings, notes, read, timing = [], [], None, {}
            cover_began = time.monotonic()
            if reference:
                runpod.serverless.progress_update(job, 'Transcribing the source melody for your cover')
                read = read_recording(reference, td, 'full' if options['keep_harmony'] else 'melody', client, bucket)
                warnings = read['warnings']
                timing.update(read['timing'])
            cover_s = round(time.monotonic() - cover_began, 1) if reference else 0
            runpod.serverless.progress_update(job, 'Loading YuE2 and recording your song')
            load_began = time.monotonic()
            pipe = engine()

            def fetch(key):
                local = Path('/tmp') / key.replace('/', '_')
                if not local.exists():
                    part = local.with_suffix('.part')
                    client.download_file(bucket, key, str(part))
                    part.rename(local)
                return str(local)
            apply_style(pipe, wanted, fetch)
            load_s = round(time.monotonic() - load_began, 1)
            fixed, origin = None, None
            if read:
                fixed, origin = read['abc'], 'recording'
            elif inp['abc']:
                fixed, origin = inp['abc'], 'score'
            elif options['instrumental']:
                runpod.serverless.progress_update(job, 'Writing a score for your instrumental')
                began = time.monotonic()
                fixed, origin = plan_instrumental(pipe, inp, controls), 'YuE2'
                timing['plan_s'] = round(time.monotonic() - began, 1)
            render, budget, facts, extras = inp, None, {}, {}
            if fixed is not None:
                began = time.monotonic()
                render, budget, facts, extras, notes = fixed_score_render(
                    inp, options, fixed, origin, read['seconds'] if read else None)
                timing['convert_s'] = round(time.monotonic() - began, 2)
            else:
                origin = 'YuE2' if inp['cot'] != 'off' else None
                if options['length_guard']:
                    notes.append('The length guard needs a score before the render, so this new song kept the six-minute limit.')
            score_tokens = token_count(pipe, render.get('abc'))
            if score_tokens and score_tokens > 4096:
                notes.append(f'The score is {score_tokens} tokens, above the usual 4096; the song may drift.')
            gpu_peak(reset=True)
            render_began = time.monotonic()
            result = render_song(pipe, render, controls, budget)
            render_s = round(time.monotonic() - render_began, 1)
            render_peak = gpu_peak()
            work = Path(td) / 'song'
            result.save_artifacts(work)
            if options['instrumental'] and fixed is not None:
                (work / 'source-score.abc').write_text(fixed)
                (work / 'transfer.json').write_text(json.dumps(extras.get('transfer')))
            if not facts and result.abc:
                facts = score_facts(result.abc)
            wav = work / 'master.wav'
            mp3 = work / 'master.mp3'
            ffmpeg('-i', work / 'audio.flac', '-c:a', 'pcm_s24le', wav)
            ffmpeg('-i', wav, '-c:a', 'libmp3lame', '-b:a', '320k', mp3)
            duration = sf.info(wav).duration
            prefix = f'yue2/{uuid.uuid4()}'
            files = [(mp3, 'audio/mpeg'), (wav, 'audio/wav')]
            files += [(p, 'text/plain' if p.suffix == '.abc' else 'application/json')
                      for p in work.iterdir() if p.suffix in ('.abc', '.json')]
            for p, mime in files:
                client.upload_file(str(p), bucket, prefix+'/'+p.name, ExtraArgs={'ContentType': mime})
            def url(name):
                return client.generate_presigned_url('get_object', Params={'Bucket':bucket,'Key':prefix+'/'+name}, ExpiresIn=604800)
            output = {'engine':'yue2', 'quality':'bf16-default', 'key':prefix+'/master.mp3',
                      'wav_key':prefix+'/master.wav', 'url':url('master.mp3'), 'wav_url':url('master.wav'),
                      'score_key':prefix+'/score.abc' if (work/'score.abc').exists() else None,
                      'duration_s':round(duration,2), 'bytes':mp3.stat().st_size,
                      'processing_ms':int((time.monotonic()-start)*1000), 'seed':inp['seed'],
                      'truncated':any(result.truncated.values()), 'truncation_flags':result.truncated,
                      'cover':bool(reference), 'transcription_warnings':warnings,
                      'controls': controls, 'lora_key': wanted[0] if wanted else None,
                      'timing': {'cover_s': cover_s, 'model_load_s': load_s, 'render_s': render_s, **timing}}
            output.update({
                'mode': 'render', 'instrumental': options['instrumental'], 'score_origin': origin,
                'cover_mode': ('harmony' if options['keep_harmony'] else 'melody') if reference else None,
                'transcription_cached': read['cached'] if read else None,
                'source_seconds': round(read['seconds'], 2) if read else None,
                'source_score_key': prefix+'/source-score.abc' if (work/'source-score.abc').exists() else None,
                'score_bpm': facts.get('score_bpm'), 'score_musical_key': facts.get('score_musical_key'),
                'score_meter': facts.get('score_meter'), 'score_seconds': facts.get('score_seconds'),
                'score_tokens': score_tokens, 'sections': facts.get('sections', []),
                'lyric_fit': extras.get('lyric_fit'), 'transfer': extras.get('transfer'),
                'semantic_budget_tokens': budget, 'worker_notes': notes, 'gpu': gpu_name(),
                'memory': {'render_peak_gib': render_peak, 'transcribe_peak_gib': read['peak'] if read else None},
                'features': list(FEATURES)})
            if render['lyrics'] != inp['lyrics']:
                output['lyrics_used'] = render['lyrics']
            if render['style'] != inp['style']:
                output['style_used'] = render['style']
            return output
    except ValueError as error:
        return {'error':str(error)}
    except Exception as error:
        print('YuE2 failed:',type(error).__name__,flush=True)
        return {'error':f'YuE2 could not finish ({type(error).__name__}). Your writing is saved.'}


if __name__ == '__main__':
    import runpod
    warm()
    runpod.serverless.start({'handler': handler})
