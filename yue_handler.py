"""One full-quality YuE2 candidate per request; private audio and score outputs.

Part 295 adds upstream's official cover and instrumental recipe (YuE #202) as
optional request fields: keep_harmony, instrumental, length_guard,
match_score_tempo and mode. A request without them renders exactly as before.
The lyric sync follow-up adds fit_lyrics, measure_fit, fit_score_touchup and mode=measure
(lyric_sync.py, align.py); a request without them also renders exactly as before, except that
the lyric_fit report now pairs sections by name and meter_check reports a style whose meter
contradicts the score."""
import difflib
import os
import hashlib
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
import lyric_sync  # noqa: E402

PIPE = None
STYLE = {'applied': None, 'plain': None}
STYLE_KEY = re.compile(r'^yue2-loras/[A-Za-z0-9._-]{1,80}\.pt$')
TAKE_KEY = re.compile(r'^yue2/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/master\.(?:wav|mp3)$')
# What this worker can do; the booth checks this before offering an option.
# lyric-sync: fit_lyrics="timing"; fit-score: measure_fit and mode="measure"; score-touchup:
# fit_score_touchup; lyric-fit-v2: lyric_fit rows paired by section name; meter-check: meter_check.
FEATURES = ('keep-harmony', 'instrumental', 'transcribe', 'score-cache', 'length-guard',
            'match-tempo', 'sections', 'lyric-fit', 'chord-check', 'lyric-sync', 'fit-score',
            'score-touchup', 'lyric-fit-v2', 'meter-check')
ALIGN_REVISION = 'mmsfa-star-hdemucs-2'  # word timing cache revision (align.py's models and recipe)
TOKENS_PER_SECOND, TOKEN_CAP, TOKEN_FLOOR, GUARD_HEADROOM = 25, 9000, 200, 1.10
NO_SINGING = ('no vocals', 'no singing', 'no choir', 'no spoken words')
PLANNING_TAGS = '[Intro]\n\n[Verse]\n\n[Chorus]\n\n[Outro]\n'
UNREADABLE = 'The melody could not be read from this recording. Try a clearer recording of the song.'
UNMOVABLE = {
    'recording': "This recording's melody could not be moved onto an instrument. Try a sung cover, or another recording.",
    'score': "This score's melody could not be moved onto an instrument. Try a sung version, or another score.",
    'YuE2': "YuE2's score for this instrumental could not be moved onto an instrument. Try another seed.",
}
# Keeping chords needs chords: a cappella or a voice into a phone transcribes without any.
NO_CHORDS = {
    'recording': 'No chords were heard in this recording, so its melody was used with a new accompaniment.',
    'score': 'This score has no chord symbols, so its melody was used with a new accompaniment.',
}
CHORD_SYMBOL = re.compile(r'"[A-G][^"\n]*"')
BPM = re.compile(r'\b\d{2,3}(?:\.\d+)?\s*bpm\b', re.I)
TAG = re.compile(r'^\s*\[([^\]\n]+)\]\s*$')
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
METER = re.compile(r'(?<![\d/])([1-9]|1[0-2])\s*/\s*(2|4|8|16)(?![\d/])')
WALTZ = re.compile(r'\bwaltz', re.I)
NOT_THE_TAKES_SCORE = ("That take's score is not one this recording was read into, so it cannot be scored "
                       'against the recording.')


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
    if not isinstance(mode, str) or mode not in ('render', 'transcribe', 'measure'):
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


def sync_options(inp):
    """Lyric sync fields (SYNC_DESIGN.md fixes 1 and 2), both optional. Leaving them out keeps
    today's request exactly.

    fit_lyrics: "timing" re-breaks the lyric lines to the recording's phrases from measured
    word timing; her words and their order never change. measure_fit: after the render, time
    the take's words too and score how well they sit on their notes (report only).
    fit_score_touchup (A/B only, with fit_lyrics): fold a quick lead-in note into the held note
    after it where a held word would otherwise stretch over more notes than it has syllables."""
    fit = inp.get('fit_lyrics')
    if fit is not None and fit is not False and fit != 'timing':
        raise ValueError('fit_lyrics must be "timing" or false.')
    measure = inp.get('measure_fit')
    if measure is not None and type(measure) is not bool:
        raise ValueError('measure_fit must be true or false.')
    touchup = inp.get('fit_score_touchup')
    if touchup is not None and type(touchup) is not bool:
        raise ValueError('fit_score_touchup must be true or false.')
    if touchup and fit != 'timing':
        raise ValueError('fit_score_touchup needs fit_lyrics "timing": the score is touched up only where '
                         'the fitted words need it.')
    return dict(fit_lyrics=fit == 'timing', measure_fit=bool(measure), fit_score_touchup=bool(touchup))


def check_sync(inp, options, sync, reference):
    if not (sync['fit_lyrics'] or sync['measure_fit']):
        return
    if options['instrumental']:
        raise ValueError('An instrumental sings no words, so there are no lyrics to fit to its tune.')
    if not reference:
        raise ValueError('Fitting words to the tune needs the source recording they were sung to.')
    if not lyric_sync.lyric_words(inp['lyrics'])[0]:
        raise ValueError('Add the words to sing before fitting them to the tune.')


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


def score_has_chords(abc):
    """Whether the score's Vocal voice carries chord symbols (upstream picks cot=full only then).

    A score upstream's parser cannot read falls back to a plain look for quoted
    chord symbols, so this never blocks a render."""
    try:
        return bool(parse_abc(abc).voices['Vocal'].chords)
    except (ValueError, KeyError, TypeError, IndexError, ZeroDivisionError):
        return bool(CHORD_SYMBOL.search(abc or ''))


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
    """Report only, never blocks: each lyric section's words against the notes of its tune.

    Sections are paired by name and order (difflib), so words under [Intro] where the
    recording's intro has no sung tune are marked "no tune" instead of shifting every later
    pair by one; sections whose names differ are paired in order within that stretch."""
    sung = sung_sections(sections)
    words = lyric_sections(lyrics)
    if not sung or not words:
        return None

    def row(text, tune):
        out = dict(score_section=tune and tune['name'], lyrics_section=text and text['tag'],
                   sung_notes=tune and tune['sung_notes'], syllables=text and text['syllables'])
        if tune and text:
            ratio = text['syllables'] / tune['sung_notes']
            out['fit'] = 'short' if ratio < .6 else 'long' if ratio > 1.4 else 'close'
        else:
            out['fit'] = 'no words' if tune else 'no tune'
        return out
    names = [section_name(w['tag']) for w in words]
    rows = []
    for _, a0, a1, b0, b1 in difflib.SequenceMatcher(None, names, [s['name'] for s in sung],
                                                     autojunk=False).get_opcodes():
        pairs = min(a1 - a0, b1 - b0)
        rows += [row(words[a0 + d], sung[b0 + d]) for d in range(pairs)]
        rows += [row(words[a], None) for a in range(a0 + pairs, a1)]
        rows += [row(None, sung[b]) for b in range(b0 + pairs, b1)]
    return dict(scope='rough syllable count against sung notes; a guide only', sections=rows,
                same_order=names == [s['name'] for s in sung], paired='by section name and order')


def meter_class(numerator, denominator):
    """Compound (6/8, 9/8, 12/8), triple (3/4) or duple (2/4, 4/4, 2/2) feel."""
    if denominator == 8 and numerator in (6, 9, 12):
        return 'compound'
    if numerator == 3:
        return 'triple'
    if numerator in (2, 4):
        return 'duple'
    return f'{numerator}/{denominator}'


def meter_check(style, score_meter):
    """When the style names a meter or feel the score's M: contradicts (upstream: describe
    tempo and meter consistently), the two meters; otherwise None. A style of "6/8 feel" over a
    4/4 score asks for triple accents on a straight grid, so phrasing can feel off the beat."""
    if not style or not score_meter or not re.fullmatch(r'\d+/\d+', score_meter):
        return None
    theirs = meter_class(*map(int, score_meter.split('/')))
    for match in METER.finditer(style):
        numerator, denominator = int(match.group(1)), int(match.group(2))
        if numerator < 2:
            continue
        if meter_class(numerator, denominator) != theirs:
            return dict(style_meter=f'{numerator}/{denominator}', score_meter=score_meter)
    if WALTZ.search(style) and theirs != 'triple':
        return dict(style_meter='waltz', score_meter=score_meter)
    return None


def touchup_note(report):
    """What fit_score_touchup changed, in words."""
    places = len(report['places'])
    return (f"The score was touched up in {places} place{'s' if places != 1 else ''}: a quick note leading "
            'into a held note became part of it, so the held word keeps that note. The rest of the melody '
            'is unchanged.')


def apply_sync(inp, abc, sync, render, extras, notes):
    """Fix 1: her lyric lines re-broken to the recording's phrases (only when fit_lyrics asked),
    and the timing plan kept for measure_fit; fix 3 when fit_score_touchup asked. Pure: the
    timing was measured beforehand. A failure here never stops the take: it renders her lines
    as written."""
    try:
        result = lyric_sync.fit(abc, inp['lyrics'], sync.get('timing'), sync.get('words'), failed=sync.get('error'),
                                touchup=bool(sync.get('touchup')))
    except Exception as error:
        print('Lyric sync fit failed:', type(error).__name__, flush=True)
        result = lyric_sync.fit(abc, inp['lyrics'], None, None, failed='error')
    report = dict(result['report'], lyrics_fitted=False)
    if sync.get('fit'):
        notes.extend(result['notes'])
        if result['applied']:
            render['lyrics'] = result['lyrics']
            report['lyrics_fitted'] = True
            if result.get('abc') and (report.get('score_touchup') or {}).get('applied'):
                render['abc'] = result['abc']
                notes.append(touchup_note(report['score_touchup']))
    elif not result['applied']:
        notes.append(f"This take could not be scored against the tune: {lyric_sync.REASONS[report['reason']]}.")
    words = lyric_sync.lyric_words(inp['lyrics'])[0]
    fitted = report['lyrics_fitted']
    order = result['order'] if fitted else list(range(len(words)))
    stars = result.get('line_ends') if fitted else lyric_sync.star_after(words)
    extras.update(lyric_sync=report, sync_plan=result['plan'], sync_order=order,
                  sync_words=[words[i]['norm'] for i in order], sync_stars=stars)


def fixed_score_render(inp, options, abc, origin, source_seconds=None, sync=None):
    """The exact request YuE2 renders once its score is fixed: a recording's
    transcription, a given score or a YuE2 plan. Pure: no GPU, files or network.
    sync: the recording's measured timing when fit_lyrics or measure_fit was asked.

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
        if options['keep_harmony'] and not chords:
            notes.append(NO_CHORDS['recording' if origin == 'recording' else 'score'])
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
        if options['keep_harmony']:
            # Upstream refuses cot=full for a score without chords; render it as the melody cover instead.
            if score_has_chords(abc):
                render['cot'] = 'full'
            else:
                render['cot'] = 'melody'
                notes.append(NO_CHORDS['recording' if origin == 'recording' else 'score'])
        elif origin == 'recording':
            render['cot'] = 'melody'
        if sync is not None:
            apply_sync(inp, abc, sync, render, extras, notes)
        extras['lyric_fit'] = lyric_fit(render['lyrics'], facts.get('sections', []))
    extras['meter_check'] = meter_check(render['style'], facts.get('score_meter'))
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


def cover_env():
    return {**os.environ, 'LD_LIBRARY_PATH': '/opt/cover/lib', 'PATH': '/opt/cover/bin:' + os.environ['PATH']}


def run_sheetsage(decoded, out_dir, task):
    subprocess.run(['/opt/cover/bin/python', '/app/cover.py', str(decoded), str(out_dir), task],
                   env=cover_env(), check=True, timeout=600, capture_output=True)


def midi_timing(folder, abc):
    """Real times of abc's sung notes from the MIDI SheetSage2 wrote beside it, or None.

    SheetSage2 writes its melody MIDI (real seconds) with every transcription; the score
    itself only has beats on one steady tempo, which drifts from the recording."""
    try:
        folder = Path(folder)
        midis = [(p.relative_to(folder).as_posix(), p.read_bytes()) for p in sorted(folder.rglob('*.mid'))]
        timing = lyric_sync.note_times(lyric_sync.score_plan(abc), midis)
        return dict(timing, revision=lyric_sync.TIMING_REVISION,
                    abc_sha=hashlib.sha256(abc.encode('utf-8')).hexdigest()[:16])
    except Exception as error:
        print('Note timing skipped:', type(error).__name__, str(error)[:200], flush=True)
        return None


def cached_timing(client, bucket, key, abc):
    """A cached score's note timing, when it was made for exactly this score."""
    try:
        timing = json.loads(client.get_object(Bucket=bucket, Key=key + '.timing.json')['Body'].read())
        if timing.get('abc_sha') == hashlib.sha256(abc.encode('utf-8')).hexdigest()[:16] \
                and timing.get('revision') == lyric_sync.TIMING_REVISION:
            return timing
    except Exception as error:
        if type(error).__name__ not in ('NoSuchKey', 'ClientError'):
            print('Note timing cache read skipped:', type(error).__name__, flush=True)
    return None


def cache_timing(client, bucket, key, timing):
    try:
        client.put_object(Bucket=bucket, Key=key + '.timing.json', ContentType='application/json',
                          Body=json.dumps(timing).encode('utf-8'))
    except Exception as error:
        print('Note timing cache write skipped:', type(error).__name__, flush=True)


def fetch_recording(reference, td):
    """Download and decode a source recording once: (source path, decoded path, seconds)."""
    import soundfile as sf
    from auk_handler import ffmpeg, download
    source = Path(td) / 'source'
    download(reference, source)
    decoded = Path(td) / 'source.wav'
    ffmpeg('-i', source, '-ar', 24000, '-ac', 1, decoded)
    seconds = sf.info(decoded).duration
    if seconds > 360:
        raise ValueError('Use a source recording up to six minutes long.')
    return source, decoded, seconds


def read_recording(reference, td, task, client, bucket, want_timing=False, fetched=None, cache_only=False):
    """Download, check and transcribe a source recording, once per recording and task.

    The score is cached in the private bucket by the recording's hash, the task
    (melody or full) and the SheetSage2 revision, so takes 2 to 4 of a cover
    skip transcription. Only the few-KB score is stored, never the audio. A fresh
    transcription also caches its note timing (a few KB, from SheetSage2's MIDI);
    a cached score's timing is read only when lyric sync asks for it.
    fetched: fetch_recording()'s result, to skip the download. cache_only: None instead of a
    fresh transcription when nothing is cached for this task."""
    timing, began = {}, time.monotonic()
    source, decoded, seconds = fetched or fetch_recording(reference, td)
    timing['download_s'] = round(time.monotonic() - began, 1)
    key = score_cache_key(sha256(source), task)
    began = time.monotonic()
    hit = None
    try:
        abc = client.get_object(Bucket=bucket, Key=key + '.abc')['Body'].read().decode('utf-8')
        meta = json.loads(client.get_object(Bucket=bucket, Key=key + '.json')['Body'].read())
        timing['transcribe_s'] = round(time.monotonic() - began, 1)
        hit = dict(abc=abc, warnings=meta.get('warnings', []), cached=True, key=key,
                   seconds=seconds, timing=timing, peak=None, task=task)
    except Exception as error:
        if type(error).__name__ not in ('NoSuchKey', 'ClientError'):
            print('Score cache read skipped:', type(error).__name__, flush=True)
    if hit:
        hit.update(source=source, decoded=decoded,
                   note_times=cached_timing(client, bucket, key, hit['abc']) if want_timing else None)
        return hit
    if cache_only:
        return None
    park_pipeline()
    score_dir = Path(td) / 'transcription'
    try:
        run_sheetsage(decoded, score_dir, task)
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
    note_times = midi_timing(score_dir, abc)
    if note_times and key:
        cache_timing(client, bucket, key, note_times)
    return dict(abc=abc, warnings=warnings, cached=False, key=key, seconds=seconds, timing=timing, peak=peak,
                source=source, decoded=decoded, note_times=note_times, task=task)


class SyncError(Exception):
    """Word timing could not be measured; the take goes ahead with the lyrics as written."""


def align_audio(path, words, td, tag, stars=None):
    """GPU: time known words on a recording's vocal stem (align.py in the cover env).
    words: aligner text per word (lyric_sync.aligner_text); stars: per word, whether vocals the
    lyrics do not name (an ad-lib) may follow it (lyric_sync.star_after). Returns align.py's
    result."""
    from auk_handler import ffmpeg
    raw, spec, out = Path(td) / f'{tag}.f32', Path(td) / f'{tag}-words.json', Path(td) / f'{tag}-timing.json'
    try:
        ffmpeg('-i', path, '-f', 'f32le', '-acodec', 'pcm_f32le', '-ac', 2, '-ar', 44100, raw)
        body = {'words': words}
        if stars is not None and len(stars) == len(words):
            body['star_after'] = [bool(x) for x in stars]
        spec.write_text(json.dumps(body))
        park_pipeline()
        subprocess.run(['/opt/cover/bin/python', '/app/align.py', str(raw), str(spec), str(out)],
                       env=cover_env(), check=True, timeout=600, capture_output=True)
        result = json.loads(out.read_text())
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        detail = ''
        try:
            detail = json.loads(out.read_text()).get('error', '')
        except Exception:
            pass
        tail = getattr(error, 'stderr', None) or b''
        print('Word timing failed:', type(error).__name__, detail, tail[-1200:].decode('utf-8', 'replace'), flush=True)
        raise SyncError(detail or type(error).__name__) from None
    finally:
        raw.unlink(missing_ok=True)
    if len(result.get('words') or []) != len(words):
        raise SyncError('word count')
    return result


def source_sync(read, lyrics, td, client, bucket, task):
    """The recording's measured timing for lyric sync. Never raises.

    Note times come from the score cache (or one SheetSage2 re-read of a score cached before
    timing was kept, about 15 s of GPU once per recording). Word times come from align.py on
    the recording, cached per recording and lyrics, so takes 2 to 4 cost nothing extra.
    Returns dict(timing, words, error, words_cached, align); error is a lyric_sync.REASONS key."""
    state = dict(timing=read.get('note_times'), words=None, error=None, words_cached=False, align=None)
    if state['timing'] is None:
        folder = Path(td) / 'retime'
        park_pipeline()
        try:
            run_sheetsage(read['decoded'], folder, task)
            state['timing'] = midi_timing(folder, read['abc'])
        except (subprocess.SubprocessError, OSError) as error:
            tail = getattr(error, 'stderr', None) or b''
            print('SheetSage2 re-read failed:', type(error).__name__, tail[-800:].decode('utf-8', 'replace'), flush=True)
        if state['timing'] and read.get('key'):
            cache_timing(client, bucket, read['key'], state['timing'])
    if state['timing'] is None:
        state['error'] = 'notes'
        return state
    words = lyric_sync.lyric_words(lyrics)[0]
    words_key = None
    if read.get('key'):
        digest = hashlib.sha256(lyrics.encode('utf-8')).hexdigest()[:16]
        words_key = f"{read['key'].rsplit('/', 1)[0]}/words-{digest}-{ALIGN_REVISION}.json"
        try:
            cached = json.loads(client.get_object(Bucket=bucket, Key=words_key)['Body'].read())
            if len(cached.get('words') or []) == len(words):
                state.update(words=cached['words'], words_cached=True)
                return state
        except Exception as error:
            if type(error).__name__ not in ('NoSuchKey', 'ClientError'):
                print('Word timing cache read skipped:', type(error).__name__, flush=True)
    try:
        result = align_audio(read['source'], [w['norm'] for w in words], td, 'source',
                             stars=lyric_sync.star_after(words))
    except SyncError:
        state['error'] = 'align'
        return state
    state.update(words=result['words'], align={k: result.get(k) for k in ('timing', 'peak_gib', 'device')})
    if words_key:
        try:
            client.put_object(Bucket=bucket, Key=words_key, ContentType='application/json', Body=json.dumps(
                dict(words=result['words'], revision=ALIGN_REVISION)).encode('utf-8'))
        except Exception as error:
            print('Word timing cache write skipped:', type(error).__name__, flush=True)
    return state


def measure_take(path, extras, td):
    """Fix 2: time the take's own words and score them against the recording's plan.

    The take is aligned with the words it was given (take_words, or sync_words), in their own
    line layout; take_positions picks the ones that are her words, in sync_order."""
    take = align_audio(path, extras.get('take_words') or extras['sync_words'], td, 'take',
                       stars=extras.get('take_stars') or extras.get('sync_stars'))
    timed = take['words']
    if extras.get('take_positions') is not None:
        timed = [timed[p] for p in extras['take_positions']]
    score = lyric_sync.measure(extras['sync_plan'], extras['sync_order'], timed)
    score['take_align'] = {k: take.get(k) for k in ('timing', 'peak_gib')}
    return score


def safe_measure(path, extras, td, notes):
    """measure_take for a finished take: any failure is a note, never a failed take."""
    try:
        return measure_take(path, extras, td)
    except SyncError:
        pass
    except Exception as error:
        print('Take scoring failed:', type(error).__name__, flush=True)
    notes.append('This take could not be scored against the tune this time.')
    return None


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
                'cover_mode': 'harmony' if options['keep_harmony'] and score_has_chords(read['abc']) else 'melody',
                'score_bpm': facts.get('score_bpm'), 'score_musical_key': facts.get('score_musical_key'),
                'score_meter': facts.get('score_meter'), 'score_seconds': facts.get('score_seconds'),
                'sections': facts.get('sections', []), 'source_seconds': round(read['seconds'], 2),
                'transcription_warnings': read['warnings'], 'transcription_cached': read['cached'],
                'processing_ms': int((time.monotonic() - start) * 1000),
                'timing': {'cover_s': round(time.monotonic() - began, 1), **read['timing']},
                'gpu': gpu_name(), 'memory': {'transcribe_peak_gib': read['peak']}, 'features': list(FEATURES)}


def measure_request(inp):
    """mode=measure: a finished take (take_key in the private bucket), the source recording it
    covers and her lyrics as written (the recording is timed against them). lyrics_used: the
    lines the take was given when they differ (a fitted take's lyrics_used), in any layout.
    cover_mode ("harmony" or "melody") narrows which of the recording's scores the take was
    rendered from; by default the take's own score.abc decides. Nothing is rendered."""
    reference, take, lyrics = inp.get('reference_voice_url'), inp.get('take_key'), inp.get('lyrics')
    if not reference or not isinstance(reference, str) or len(reference) > 12000:
        raise ValueError('Scoring a take needs the source recording it covers. Import it again.')
    if not isinstance(take, str) or not TAKE_KEY.match(take):
        raise ValueError('That take is not available to score.')
    if not isinstance(lyrics, str) or len(lyrics) > 8000 or not lyric_sync.lyric_words(lyrics)[0]:
        raise ValueError('Scoring a take needs the lyrics it sang, up to 8000 characters.')
    used = inp.get('lyrics_used')
    if used is not None and (not isinstance(used, str) or len(used) > 8000 or not lyric_sync.lyric_words(used)[0]):
        raise ValueError('lyrics_used must be the words the take was given, up to 8000 characters.')
    mode = inp.get('cover_mode')
    if mode is not None and mode not in ('harmony', 'melody'):
        raise ValueError('cover_mode must be "harmony" or "melody".')
    return dict(reference=reference, take=take, lyrics=lyrics.strip(), lyrics_used=used.strip() if used else None,
                cover_mode=mode)


def take_score(client, bucket, take_key):
    """The score a take was rendered from (score.abc beside its master), or None."""
    try:
        key = take_key.rsplit('/', 1)[0] + '/score.abc'
        return client.get_object(Bucket=bucket, Key=key)['Body'].read().decode('utf-8')
    except Exception as error:
        if type(error).__name__ not in ('NoSuchKey', 'ClientError'):
            print('Take score read skipped:', type(error).__name__, flush=True)
        return None


def measure_read(req, take_abc, td, client, bucket):
    """The recording read into the score its take was rendered from: the recording's cached
    full and melody scores (cover_mode narrows them) matched against the take's score.abc, as
    it is or touched up (fit_score_touchup). Nothing is transcribed afresh unless the take has
    no score of its own and was a harmony cover; a melody score is never transcribed here.
    ValueError when the take's score is not one of this recording's."""
    tasks = {'harmony': ['full'], 'melody': ['melody']}.get(req['cover_mode'], ['full', 'melody'])
    fetched = fetch_recording(req['reference'], td)
    for task in tasks:
        read = read_recording(req['reference'], td, task, client, bucket, want_timing=True, fetched=fetched,
                              cache_only=True)
        if read and (take_abc is None or read['abc'].strip() == take_abc.strip()):
            return dict(read, touched=False)
        if read and lyric_sync.touched_from(read['abc'], take_abc):
            return dict(read, touched=True)
    if take_abc is not None or 'full' not in tasks:
        raise ValueError(NOT_THE_TAKES_SCORE)
    return read_recording(req['reference'], td, 'full', client, bucket, want_timing=True, fetched=fetched)


def measure_only(job, inp, options, start):
    """mode=measure (fix 0 and 2 on a take that already exists): which words the take sang on
    the recording's held notes, and whether its phrases start after a pause. About the GPU cost
    of two word timings; the recording's side is cached for later takes."""
    req = measure_request(inp)
    take_key, lyrics = req['take'], req['lyrics']
    import runpod
    with tempfile.TemporaryDirectory() as td:
        client, bucket = storage()
        runpod.serverless.progress_update(job, 'Timing the words of the recording')
        began = time.monotonic()
        take_abc = take_score(client, bucket, take_key)
        read = measure_read(req, take_abc, td, client, bucket)
        state = source_sync(read, lyrics, td, client, bucket, read['task'])
        sync_s = round(time.monotonic() - began, 1)
        inp_like = dict(lyrics=lyrics)
        extras, notes = {}, []
        apply_sync(inp_like, read['abc'], dict(state, fit=False, measure=True), dict(inp_like), extras, notes)
        report = extras['lyric_sync']
        report.update(words_cached=state['words_cached'], source_align=state['align'], score_task=read['task'],
                      score_is_takes=take_abc is not None, score_touched=bool(read.get('touched')))
        if req['lyrics_used'] and extras['sync_plan']:
            order, positions = lyric_sync.sung_order(req['lyrics_used'], lyrics)
            used = lyric_sync.lyric_words(req['lyrics_used'])[0]
            extras.update(sync_order=order, take_positions=positions, take_words=[w['norm'] for w in used],
                          take_stars=lyric_sync.star_after(used))
            report['take_words'] = dict(given=len(used), hers=len(order))
        measure_s = None
        if extras['sync_plan']:
            runpod.serverless.progress_update(job, 'Timing the words of the take')
            began = time.monotonic()
            local = Path(td) / ('take' + Path(take_key).suffix)
            try:
                client.download_file(bucket, take_key, str(local))
            except Exception as error:
                print('Take download failed:', type(error).__name__, flush=True)
                raise ValueError('That take is not available to score.') from None
            report.update(safe_measure(local, extras, td, notes) or {})
            measure_s = round(time.monotonic() - began, 1)
        return {'engine': 'yue2', 'mode': 'measure', 'take_key': take_key, 'lyric_sync': report,
                'worker_notes': notes, 'transcription_cached': read['cached'],
                'source_seconds': round(read['seconds'], 2),
                'processing_ms': int((time.monotonic() - start) * 1000),
                'timing': {**read['timing'], 'sync_s': sync_s, 'measure_s': measure_s},
                'gpu': gpu_name(), 'features': list(FEATURES)}


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
        if options['mode'] == 'measure':
            return measure_only(job, raw, options, start)
        inp = request_input(raw)
        controls = sound_controls(raw)
        wanted = style_request(raw)
        reference = raw.get('reference_voice_url')
        check_combination(inp, options, reference)
        sync_opts = sync_options(raw)
        check_sync(inp, options, sync_opts, reference)
        wants_sync = sync_opts['fit_lyrics'] or sync_opts['measure_fit']
        import runpod
        import soundfile as sf
        from auk_handler import ffmpeg
        with tempfile.TemporaryDirectory() as td:
            client, bucket = storage()
            warnings, notes, read, timing = [], [], None, {}
            cover_began = time.monotonic()
            if reference:
                runpod.serverless.progress_update(job, 'Transcribing the source melody for your cover')
                task = 'full' if options['keep_harmony'] else 'melody'
                read = read_recording(reference, td, task, client, bucket, want_timing=wants_sync)
                warnings = read['warnings']
                timing.update(read['timing'])
            cover_s = round(time.monotonic() - cover_began, 1) if reference else 0
            sync = None
            if wants_sync:
                runpod.serverless.progress_update(job, 'Timing your words against the recording')
                began = time.monotonic()
                state = source_sync(read, inp['lyrics'], td, client, bucket, task)
                sync = dict(state, fit=sync_opts['fit_lyrics'], measure=sync_opts['measure_fit'],
                            touchup=sync_opts['fit_score_touchup'])
                timing['sync_s'] = round(time.monotonic() - began, 1)
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
                    inp, options, fixed, origin, read['seconds'] if read else None, sync=sync)
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
            if sync is not None:
                extras['lyric_sync'].update(words_cached=sync['words_cached'], source_align=sync['align'])
            if sync is not None and sync['measure'] and extras.get('sync_plan'):
                runpod.serverless.progress_update(job, 'Checking which words landed on the long notes')
                began = time.monotonic()
                extras['lyric_sync'].update(safe_measure(wav, extras, td, notes) or {})
                timing['measure_s'] = round(time.monotonic() - began, 1)
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
                # What was rendered, not what was asked: chords are kept only when the score has them.
                'cover_mode': ('harmony' if render['cot'] == 'full' else 'melody') if reference else None,
                'transcription_cached': read['cached'] if read else None,
                'source_seconds': round(read['seconds'], 2) if read else None,
                'source_score_key': prefix+'/source-score.abc' if (work/'source-score.abc').exists() else None,
                'score_bpm': facts.get('score_bpm'), 'score_musical_key': facts.get('score_musical_key'),
                'score_meter': facts.get('score_meter'), 'score_seconds': facts.get('score_seconds'),
                'score_tokens': score_tokens, 'sections': facts.get('sections', []),
                'lyric_fit': extras.get('lyric_fit'), 'transfer': extras.get('transfer'),
                'meter_check': extras.get('meter_check'),
                'semantic_budget_tokens': budget, 'worker_notes': notes, 'gpu': gpu_name(),
                'memory': {'render_peak_gib': render_peak, 'transcribe_peak_gib': read['peak'] if read else None},
                'features': list(FEATURES)})
            if sync is not None:
                output['lyric_sync'] = extras.get('lyric_sync')
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
