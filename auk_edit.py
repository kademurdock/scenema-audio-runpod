"""Word and lyric edits in AuK's trained wording. No GPU imports.

AuK follows only its trained edit templates (Tencent-Hunyuan/AuK
src/auk/infer/pe.config.yaml, content_edit and vocal_edit), and one
instruction carries one edit. Free-form text such as 'Replace ... Preserve the
speaker' or two changes in one request did not change the words (Nov 14 2026
diagnosis). So a word edit is built here from old words, new words and a kind,
never passed through as typed.

Job input (auk_task "edit"):
  reference_voice_url  the imported recording (required)
  old_words, new_words the one change (or a single free-form 'Replace "a" with
                       "b".' in instruction, which is mapped; anything else
                       about words is refused with a plain reason)
  edit_kind            "speech" (default) or "vocal" (sung lyrics)
  edit_start, edit_end seconds around the old words; needed over 30 s. A
                       range may be up to 30 s; padding shrinks to fit
  pad_seconds          context each side of the range, default 1.0 (0-3)
  seeds | candidates   explicit seeds (1-6), or a count (default 3) from seed
                       (fresh random base when seed is absent)
  source_text          optional transcript of the padded clip; enables
                       upstream's ASR length rule
  target_seconds       optional override of the generated clip length (0.3-30)
  crossfade_ms         splice crossfade, default 70 (0-200)
  rank                 true: transcribe each take with AuK's own Qwen Thinker
                       and put the best first; default false (bridge ranks)
Edits without words (whisper, accent, timbre, emotion, noise) keep the older
path. Only an edit verb aimed at two quoted phrases, or three or more quoted
phrases, is treated as a word change; a single quoted description (timbre,
'sound like "..."', 'the speaker who says "..."') is a style edit.

Known limits: the model works at 24 kHz mono, so the spliced stretch (the range
plus about 0.15 s each side) is mono at the model's bandwidth in a stereo or
wideband recording. Upstream trims silence off a speech clip before sizing it
(pe.py content_edit); this worker sizes the padded crop as cut.
"""
import math
import re
import secrets

# pe.config.yaml content_edit.templates.*.replace and vocal_edit.templates.
TEMPLATES = {
    ('speech', 'en'): "Replace '{orig}' with '{new}'.",
    ('speech', 'zh'): '把‘{orig}’改成‘{new}’',
    ('vocal', 'en'): 'Change "{orig}" to "{new}" in the vocal recording.',
    ('vocal', 'zh'): '把这段歌词中的“{orig}”改成“{new}”。',
}
KINDS = ('speech', 'vocal')
MAX_WORDS_CHARS = 200
MAX_CROP_SECONDS = 30.0
MAX_CANDIDATES = 6
DEFAULT_CANDIDATES = 3
DEFAULT_PAD = 1.0
DEFAULT_CROSSFADE_MS = 70
# Splice this far outside the marked range; the rest of the padding stays original.
JOIN_MARGIN_S = 0.15
# Length ratio new/old allowed for the marked range (upstream TTS check: 0.45-2.2).
MIN_RATIO, MAX_RATIO = 0.4, 2.5
LEVEL_WINDOW_S = 0.3
MAX_LEVEL_DB = 6.0

_CJK_RE = re.compile(r'[㐀-䶿一-鿿]')
_EN_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
# Single quotes must not touch letters, so "speaker's" is not a quote mark;
# inside one, an apostrophe between letters ("can't") is part of the words.
_Q = (r"""(?:(?<![A-Za-z])'((?:[^'\n]|(?<=[A-Za-z])'(?=[A-Za-z]))+)'(?![A-Za-z])"""
      r"""|"([^"\n]+)"|‘([^’\n]+)’|“([^”\n]+)”)""")
_QUOTED = re.compile(_Q)
_PATTERNS = (
    # (regex, old group block, new group block); blocks index into the two _Q groups.
    (re.compile(r'^\s*(?:please\s+)?replace\s+' + _Q + r'\s+with\s+' + _Q, re.I), 0, 1),
    (re.compile(r'^\s*(?:please\s+)?change\s+' + _Q + r'\s+(?:to|into)\s+' + _Q, re.I), 0, 1),
    (re.compile(r'^\s*(?:please\s+)?(?:say|sing)\s+' + _Q + r'\s+instead\s+of\s+' + _Q, re.I), 1, 0),
    (re.compile(r'^\s*(?:请)?把(?:这段歌词中的|这段话中的|这句话中的)?\s*' + _Q +
                r'\s*(?:改成|换成|改为|替换成|替换为)\s*' + _Q), 0, 1),
)
# A word change in free form: an edit verb aimed at two quoted phrases.
_WORD_CHANGE = (
    re.compile(r'\b(?:replace|change|swap|switch|substitute)\b[^.\n]{0,40}?' + _Q +
               r'[^.\n]{0,20}?\b(?:with|to|into|for|by)\b\s*' + _Q, re.I),
    re.compile(r'\b(?:say|says|saying|sing|sings|singing)\b[^.\n]{0,40}?' + _Q +
               r'\s*,?\s*instead\s+of\s+' + _Q, re.I),
    re.compile(_Q + r'\s*(?:->|→|=>)\s*' + _Q),
    re.compile(_Q + r'[^。\n]{0,8}?(?:改成|换成|改为|替换成|替换为)\s*' + _Q),
)
# A quoted description, not words to change: 'timbre to: "..."', 'sound like "..."'.
_STYLE_QUOTE = re.compile(r'(?:\b(?:timbre|description|describ\w*|match\w*|like|(?:who|that)\s+(?:says|said))\b'
                          r'[^"“‘\'\n.]{0,12}?|:\s*)' + _Q, re.I)
_VOCAL_HINT = re.compile(r'\b(?:sung|sing|singing|lyrics?|vocal|song)\b|歌词|唱', re.I)
_EDIT_VERB = re.compile(r'\b(?:replace|change|remove|delete|insert|add|say|sing|then)\b', re.I)
# A tail clause that only asks to keep things as they are.
_KEEP_CLAUSE = re.compile(r"\b(?:keep|keeping|preserve|preserving|maintain|retain|"
                          r"don['’]?t|do\s+not|without|unchanged|same)\b", re.I)

REFUSE_FREEFORM = ('AuK only follows its trained wording for word changes. Send the old words '
                   'and the new words (one change per render) and the worker builds the instruction.')
REFUSE_MULTI = 'One word change per render. Send each change as its own job.'


def _clean_words(value, label):
    text = re.sub(r'\s+', ' ', str(value or '')).strip()
    if not text:
        raise ValueError(f'Write the {label} words for this change.')
    if len(text) > MAX_WORDS_CHARS:
        raise ValueError(f'The {label} words are too long. Change one short phrase at a time.')
    return text


def language_of(*texts):
    return 'zh' if any(_CJK_RE.search(t or '') for t in texts) else 'en'


def tidy_words(kind, value):
    """Sung words lose double quotes, as the bridge's template does; the
    template quotes them itself."""
    text = str(value or '')
    if kind == 'vocal':
        text = re.sub(r'["“”]', '', text)
    return text


def build_instruction(kind, old, new):
    if kind not in KINDS:
        raise ValueError('Choose speech or vocal for a word change.')
    old = _clean_words(tidy_words(kind, old), 'old')
    new = _clean_words(tidy_words(kind, new), 'new')
    if old.casefold() == new.casefold():
        raise ValueError('The old and new words are the same. Nothing to change.')
    return TEMPLATES[(kind, language_of(old, new))].format(orig=old, new=new)


def parse_freeform(instruction, kind_hint=None):
    """Map one free-form word change onto the template, or say plainly why not.

    Returns None for edits that are not about words (whisper, accent, timbre,
    emotion): those keep the older path.
    """
    text = str(instruction or '').strip()
    for regex, old_at, new_at in _PATTERNS:
        match = regex.match(text)
        if not match:
            continue
        groups = match.groups()
        pick = lambda block: next(g for g in groups[block * 4:block * 4 + 4] if g is not None)
        old, new = pick(old_at), pick(new_at)
        rest = text[match.end():]
        # A tail like 'in the vocal recording.' or 'Keep the voice, don't change
        # the timing.' is fine; another quoted phrase, or an edit verb in a
        # clause that is not about keeping things, means a second change.
        if _QUOTED.search(rest):
            raise ValueError(REFUSE_MULTI)
        for clause in re.split(r'[.;,!?\n]+', rest):
            if _EDIT_VERB.search(clause) and not _KEEP_CLAUSE.search(clause):
                raise ValueError(REFUSE_MULTI)
        kind = kind_hint if kind_hint in KINDS else ('vocal' if _VOCAL_HINT.search(text) else 'speech')
        return {'kind': kind, 'old': old, 'new': new}
    # Style edits may quote a description; those quotes are not words to change.
    words_part = _STYLE_QUOTE.sub(' ', text)
    if len(_QUOTED.findall(words_part)) > 2:
        raise ValueError(REFUSE_MULTI)
    if any(regex.search(words_part) for regex in _WORD_CHANGE):
        raise ValueError(REFUSE_FREEFORM)
    return None


def spoken_duration(text):
    """pe.py _spoken_duration: 0.21 s per CJK character, 0.30 s per English word."""
    value = str(text or '')
    duration = len(_CJK_RE.findall(value)) * 0.21 + len(_EN_WORD_RE.findall(value)) * 0.30
    if duration > 0:
        return duration
    units = len(value.split()) if language_of(value) == 'en' else len(re.findall(r'\S', value))
    return units * (0.30 if language_of(value) == 'en' else 0.21)


def target_seconds(crop_seconds, range_seconds, old, new, source_text=None):
    """Upstream content_scaled rule (pe.py _content_scaled_duration).

    With a transcript of the cropped clip: crop * (said + new - old) / said.
    Without one, only the marked range is scaled by new/old; the padding keeps
    its length. With no range marked, range == crop, exactly upstream's rule.
    The new/old ratio is held to 0.4-2.5 and the clip to twice the crop and
    30 s, so a one-word-to-ten-words change cannot ask for minutes of audio.
    """
    if source_text and spoken_duration(source_text) > 0:
        said = spoken_duration(source_text)
        edited = said + spoken_duration(new) - spoken_duration(old)
        ratio = min(max(max(0.05, edited) / said, MIN_RATIO), MAX_RATIO)
        seconds = crop_seconds * ratio
    else:
        o, n = spoken_duration(old), spoken_duration(new)
        ratio = min(max(n / o, MIN_RATIO), MAX_RATIO) if o > 0 and n > 0 else 1.0
        seconds = crop_seconds - range_seconds + max(0.3, range_seconds * ratio)
    cap = min(MAX_CROP_SECONDS, max(crop_seconds, 0.25) * 2)
    return round(max(min(seconds, cap), 0.5), 3)


def _number(inp, key, default=None):
    value = inp.get(key, default)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{key} must be a number.') from None
    if not math.isfinite(value):
        raise ValueError(f'{key} must be a number.')
    return value


def crop_bounds(total, inp):
    """Range plus padding, kept inside the recording and short enough to edit."""
    start = _number(inp, 'edit_start')
    end = _number(inp, 'edit_end')
    pad = _number(inp, 'pad_seconds', DEFAULT_PAD)
    if not 0 <= pad <= 3:
        raise ValueError('Padding must be between 0 and 3 seconds.')
    if not math.isfinite(total) or total <= 0:
        raise ValueError('The recording is empty.')
    if start is None and end is None:
        if total > MAX_CROP_SECONDS:
            raise ValueError('Mark where the words are (start and end seconds). '
                             f'Recordings over {int(MAX_CROP_SECONDS)} seconds need a range.')
        return {'range_start': 0.0, 'range_end': total, 'crop_start': 0.0, 'crop_end': total}
    start = 0.0 if start is None else start
    end = total if end is None else min(end, total)
    if start < 0 or start >= total or end - start < 0.2:
        raise ValueError('Choose an edit range inside the recording, at least 0.2 seconds long.')
    if end - start > MAX_CROP_SECONDS:
        raise ValueError(f'Mark only the words to change. The range must be '
                         f'{int(MAX_CROP_SECONDS)} seconds or less.')
    # The bridge allows ranges up to 30 s: the padding shrinks to fit.
    pad = min(pad, max(0.0, (MAX_CROP_SECONDS - (end - start)) / 2))
    crop_start = max(0.0, start - pad)
    crop_end = min(total, end + pad)
    return {'range_start': start, 'range_end': end, 'crop_start': crop_start, 'crop_end': crop_end}


def edit_seeds(inp):
    seeds = inp.get('seeds')
    if seeds is not None:
        if (not isinstance(seeds, list) or not 1 <= len(seeds) <= MAX_CANDIDATES or
                any(not isinstance(s, int) or isinstance(s, bool) or not 0 <= s < 2**32 for s in seeds) or
                len(set(seeds)) != len(seeds)):
            raise ValueError(f'Seeds must be 1 to {MAX_CANDIDATES} different whole numbers.')
        return list(seeds)
    count = inp.get('candidates', DEFAULT_CANDIDATES)
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= MAX_CANDIDATES:
        raise ValueError(f'Candidates must be 1 to {MAX_CANDIDATES}.')
    base = inp.get('seed')
    if base is None:
        # A new roll every attempt: a fixed project seed retried the same dice.
        base = secrets.randbelow(2**32)
    if not isinstance(base, int) or isinstance(base, bool) or not 0 <= base < 2**32:
        raise ValueError('Seed must be an integer from 0 to 4294967295.')
    return [(base + i * 7919) % 2**32 for i in range(count)]


def words_edit_spec(inp):
    """The plan for one word or lyric edit, or None for other edit kinds."""
    if inp.get('auk_task') != 'edit':
        return None
    has_words = inp.get('old_words') is not None or inp.get('new_words') is not None
    kind = inp.get('edit_kind')
    if kind is not None and kind not in KINDS:
        raise ValueError('Choose speech or vocal for a word change.')
    if has_words:
        parsed = {'kind': kind or 'speech', 'old': inp.get('old_words'), 'new': inp.get('new_words')}
    else:
        parsed = parse_freeform(inp.get('instruction'), kind)
        if parsed is None:
            if kind is not None:
                raise ValueError('A word change needs the old words and the new words.')
            return None
    if not inp.get('reference_voice_url'):
        raise ValueError('Editing needs an imported recording.')
    instruction = build_instruction(parsed['kind'], parsed['old'], parsed['new'])
    crossfade = _number(inp, 'crossfade_ms', DEFAULT_CROSSFADE_MS)
    if not 0 <= crossfade <= 200:
        raise ValueError('Crossfade must be between 0 and 200 ms.')
    source_text = inp.get('source_text')
    if source_text is not None and (not isinstance(source_text, str) or len(source_text) > 2000):
        raise ValueError('source_text must be the transcript of the marked clip.')
    override = _number(inp, 'target_seconds')
    if override is not None and not 0.3 <= override <= MAX_CROP_SECONDS:
        raise ValueError(f'Target seconds must be between 0.3 and {int(MAX_CROP_SECONDS)}.')
    rank = inp.get('rank', False)
    if not isinstance(rank, bool):
        raise ValueError('rank must be true or false.')
    return {
        'kind': parsed['kind'], 'old': _clean_words(tidy_words(parsed['kind'], parsed['old']), 'old'),
        'new': _clean_words(tidy_words(parsed['kind'], parsed['new']), 'new'), 'instruction': instruction,
        'seeds': edit_seeds(inp), 'crossfade_ms': crossfade, 'source_text': source_text,
        'target_override': override, 'rank': rank,
    }


def splice(source, generated, crop_start, crop_end, fade):
    """Put the generated crop back between the untouched edges.

    source and generated are numpy arrays at the same rate and channel count;
    crop_start/crop_end/fade are sample counts. The first and last `fade`
    samples of the generated clip re-sing the padding, so they blend with the
    original padding under a raised-cosine crossfade (gains sum to one).
    """
    import numpy as np
    total = source.shape[0]
    out = np.array(generated, dtype=np.float64, copy=True)
    length = out.shape[0]
    fade = int(max(0, min(fade, length // 2, (crop_end - crop_start) // 2)))
    if fade:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade))
        if out.ndim == 2:
            ramp = ramp[:, None]
        if crop_start > 0:
            out[:fade] = source[crop_start:crop_start + fade] * (1 - ramp) + out[:fade] * ramp
        if crop_end < total:
            out[length - fade:] = out[length - fade:] * (1 - ramp) + source[crop_end - fade:crop_end] * ramp
    return np.concatenate([source[:crop_start], out, source[crop_end:]], axis=0)


def _tokens(text):
    value = str(text or '').casefold().replace('’', "'")
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*|[㐀-䶿一-鿿]", value)


def _lcs(a, b):
    if not a or not b:
        return 0
    row = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], prev + 1 if x == y else max(row[j], row[j - 1])
    return row[-1]


def _run(a, b):
    """Longest stretch of a heard as one unbroken run in b."""
    best = 0
    row = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], prev + 1 if x == y else 0
            best = max(best, row[j])
    return best


def score_transcript(transcript, old, new):
    """How well a take says the new words and drops the old ones (0 to 1 each).

    Like the bridge, the new words count only as one unbroken run.
    """
    heard = _tokens(transcript)
    new_tokens = _tokens(new)
    old_only = [t for t in _tokens(old) if t not in set(new_tokens)]
    new_hit = _run(new_tokens, heard) / len(new_tokens) if new_tokens else 0.0
    old_hit = _lcs(old_only, heard) / len(old_only) if old_only else 0.0
    return {'new_words_heard': round(new_hit, 3), 'old_words_heard': round(old_hit, 3),
            'score': round(new_hit - 0.5 * old_hit, 3),
            'matched': new_hit == 1.0 and old_hit < 0.5}


def rank_candidates(candidates):
    """Best first; unscored takes keep their order after scored ones."""
    order = {id(c): n for n, c in enumerate(candidates)}
    return sorted(candidates, key=lambda c: (c.get('score') is None, -(c.get('score') or 0), order[id(c)]))


def splice_points(rate, bounds, crop_start, crop_end, generated_frames, fade):
    """How much of the take's padding to drop at each end, in samples.

    The length rule keeps the padding lengths, so the original is kept until
    JOIN_MARGIN_S before the range and again from JOIN_MARGIN_S after it. Less
    of the 24 kHz mono take replaces the original's full-band stereo.
    """
    margin = JOIN_MARGIN_S * rate
    head = max(0, int(round(bounds['range_start'] * rate - margin)) - crop_start)
    tail = max(0, crop_end - int(round(bounds['range_end'] * rate + margin)))
    if generated_frames - head - tail < 2 * fade + 1:
        return 0, 0
    return head, tail


def _rms(x):
    import numpy as np
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def level_gain(source, generated, crop_start, crop_end, head_pad, tail_pad, window):
    """Gain that brings the take to the original's level at the joins.

    Compares the first and last `window` samples of the take with the same
    stretch of original padding, only where padding exists and neither side is
    near silence. Median of the ratios, held to +-MAX_LEVEL_DB; 1.0 if unknown.
    """
    import numpy as np
    pairs = []
    if head_pad > 0:
        w = int(min(window, head_pad, generated.shape[0] // 2))
        if w > 0:
            pairs.append((_rms(source[crop_start:crop_start + w]), _rms(generated[:w])))
    if tail_pad > 0:
        w = int(min(window, tail_pad, generated.shape[0] // 2))
        if w > 0:
            pairs.append((_rms(source[crop_end - w:crop_end]), _rms(generated[generated.shape[0] - w:])))
    usable = [s / g for s, g in pairs if s > 1e-4 and g > 1e-4]
    if not usable:
        return 1.0
    limit = 10 ** (MAX_LEVEL_DB / 20)
    return float(min(max(float(np.median(usable)), 1 / limit), limit))
