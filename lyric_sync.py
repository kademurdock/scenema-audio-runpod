"""Lyric-to-tune sync for YuE2 covers (Part 295 follow-up; SYNC_DESIGN.md fixes 1 to 3).

YuE2 has no field that says which syllable goes on which note. It lines the lyrics up with
the fixed score by itself, the way its training songs did: in YuE2's own songs every lyric
line end and every comma falls on a phrase end of the tune, and the word before a comma or
line end gets the phrase's last, often held, note. When a person's line breaks fall inside
the tune's phrases, held notes slip onto a neighbouring word.

fit() re-breaks the lyric lines to the recording's phrases from measured timing: the real
time of every score note (SheetSage2's own MIDI, matched to the score by note_times()) and
the real time of every word (forced alignment on the recording's vocal stem, align.py). Her
words and their order never change. Only line breaks and clause commas move, section tags
follow the score, and words in a section with no sung tune are set aside and reported.

Round 2 (after Kade's ear): a word starts only on a note onset, never deep inside a note, so a
held note belongs to the word that reaches it; sections with the same words and tune share one
layout taken from the instance heard most clearly; a section whose words were mostly not heard
keeps her own lines; her line breaks inside a phrase stay unless a phrase end is within two
words; no fitted line grows past her longest line. Round 3 (review): words heard clearly inside
a phrase stay there even where SheetSage2 wrote too few notes for them, and a word heard deep
inside a held note never takes it, even when the held note repeats the pitch before it.

measure() scores a finished take against the same plan: did each held note keep its word,
and does each phrase start after a pause. It never changes a take. touch_up() optionally (A/B
only) folds a quick slide into the held note after it and ties same-pitch lead-in notes, so a
phrase has as many notes before a held note as syllables sung before it.

Standard library only, plus the vendored upstream parser. Reports carry counts and indices,
never lyric text; only fit()'s `lyrics` field holds her words, for the render itself."""
import bisect
import difflib
import re
import struct
import sys
import unicodedata
from fractions import Fraction
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / 'instrumental'))
from abc_tools import NATURAL, TOKEN, parse_abc  # noqa: E402
from instrumentalize import section_starts  # noqa: E402

REVISION = 3                  # the fit's own rules (reports carry it)
TIMING_REVISION = 1           # note_times() output, cached per score; unchanged since round 1
PHRASE_GAP = Fraction(1, 2)   # a rest of an eighth note or longer ends a phrase
CLAUSE_HOLD = Fraction(3)     # a note of three beats or more sung straight on ends a clause: a comma
HELD = Fraction(3, 2)         # long notes whose word a take is checked for
SHORT_PHRASE = 3              # a phrase of this many notes or fewer joins the line before it
LEAD = 0.3                    # seconds of slack around a phrase's note onsets for its words
FAR = 3.0                     # a clearly heard word this many seconds away cannot sing in a phrase
TIME_WEIGHT = 4.0             # cost per second a word's onset sits outside its phrase (or off its note)
COUNT_WEIGHT = 0.2            # light pull toward as many rough syllables as notes
LOW_WEIGHT_BOOST = 4.0        # that pull grows up to five-fold where the words were not clearly heard
MAX_PER_NOTE = 1.4            # syllables per note a phrase takes before its words look misplaced
OVERFULL = 5.0                # cost per syllable over that, where the words were not clearly heard
OVERFULL_FLOOR = 0.25         # and the share of it that still applies where they were
MIX = 3.0                     # cost of one phrase singing the end of one of her sections and the start of the next
LOW_SCORE = 0.15              # aligner confidence below this is "not clearly heard"
UNHEARD_WEIGHT = 0.3          # how far an unclear word's own aligned time is trusted (a clear one: 1)
PAUSE = 0.15                  # seconds of quiet vocal before a word that make a pause
MAX_GROUP = 40                # words in one phrase, a bound for speed
MIN_NOTE_MATCH = 0.5          # score notes that must match SheetSage2's MIDI notes
MIN_HEARD = 0.5               # sung words the aligner must hear clearly, across the song
MIN_SECTION_HEARD = 0.5       # and within one section, or that section keeps her lines
MIN_COVERAGE = 0.5            # phrase notes that must end up with words
MAX_OFFSET = 0.75             # median seconds between a phrase's first note and its first word
SNAP = 0.15                   # a word heard further than this inside a note does not start on it
REFUSE = 50.0                 # cost of starting a clearly heard word on a note it was heard well inside
LAG_WINDOW = 0.3              # word onsets this close to a note onset measure the song's word-to-note lag
CRAM = 2.0                    # cost per syllable left without a note of its own
MELISMA = 0.1                 # cost per extra note a word stretches over
SHARE = 3.0                   # cost of two words starting on one note (the second heard inside it)
FORCED_SHARE = 20.0           # and of a share the aligner does not support (a phrase with too few notes)
MOVE_BREAK = 2                # her line break moves onto a phrase end at most this many words away
REPEAT_MATCH = 0.8            # tune similarity for sections with the same words to share one layout
SLIDE = Fraction(1, 4)        # a note this short touching a held note is SheetSage2's slide into it
TAG = re.compile(r'^\s*\[([^\]\n]+)\]\s*$')
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
CLAUSE_MARK = re.compile(r'[,;:.…]+$')
ENDS_CLAUSE = re.compile(r'[,;:.!?…]["”’)\]]*$')
LONE_DASH = re.compile(r'^[-‐‑‒–—―]+$')
ORDINALS = ('first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth')
REASONS = {
    'score': "the score's phrases could not be read",
    'notes': "the recording's note timing could not be read",
    'heard': 'too few of your words could be heard clearly in the recording',
    'words': 'your words did not line up with the tune',
    'empty': 'there are no words to sing',
    'align': 'your words could not be timed against the recording this time',
    'error': 'your words could not be fitted to the tune this time',
}


def syllables(word):
    """Same rough rule as yue_handler.syllables: vowel groups, less a silent final e, at least one."""
    word = word.lower()
    count = len(re.findall(r'[aeiouy]+', word))
    if count > 1 and word.endswith('e') and not word.endswith(('le', 'ee', 'ye')):
        count -= 1
    return max(1, count)


def section_name(tag):
    """Same normalisation as yue_handler.section_name: '[Verse 2]' and 'verse' are both 'verse'."""
    name = re.split(r'[:(\[,]', (tag or '').lower())[0]
    name = re.sub(r'[\s_-]*\d+\s*$', '', name).strip()
    return re.sub(r'^pre[\s_-]*chorus$', 'pre-chorus', name)


def aligner_text(token):
    """What the aligner reads for one token: lower-case a-z and apostrophes, accents removed.
    A token with no letters (a number, a dash) reads as empty and is timed from its neighbours."""
    text = unicodedata.normalize('NFKD', token.replace('’', "'").replace('‘', "'"))
    text = ''.join(c for c in text if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z']", '', text).strip("'")


def lyric_words(lyrics):
    """Every whitespace token outside [tag] lines, in order, and her sections.

    Each word: text (as written), line, section (index), syllables (rough), eol (the last token
    of her line), brk (her text breaks after it: a line end or a clause mark) and norm (what
    the aligner reads)."""
    words, sections = [], []
    for line_no, line in enumerate((lyrics or '').splitlines()):
        tag = TAG.match(line)
        if tag:
            sections.append(dict(index=len(sections), tag=tag.group(1).strip(), name=section_name(tag.group(1))))
            continue
        tokens = line.split()
        if not tokens:
            continue
        if not sections:
            sections.append(dict(index=0, tag=None, name=None))
        for k, token in enumerate(tokens):
            eol = k == len(tokens) - 1
            words.append(dict(text=token, line=line_no, section=len(sections) - 1,
                              syllables=sum(syllables(w) for w in WORD.findall(token)), eol=eol,
                              brk=eol or bool(ENDS_CLAUSE.search(token)), norm=aligner_text(token)))
    for section in sections:
        section['words'] = sum(1 for w in words if w['section'] == section['index'])
    return words, sections


def star_after(words):
    """Where the aligner may place a star token (vocals the lyrics do not name, such as an ad-lib
    between lines): after the last word of each of her lines."""
    return [bool(w['eol']) for w in words]


def score_plan(abc):
    """The score's sung notes, phrases and section instances (ValueError when unreadable).

    A phrase ends at a rest of an eighth note or longer. A phrase that straddles a section bar
    line belongs to the section of its last note: YuE2's own plans start a verse's pickup inside
    the section before while its words sit under [Verse]."""
    try:
        score = parse_abc(abc)
        vocal = score.voices['Vocal']
        starts = sorted(section_starts(abc, score).items())
    except (ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as error:
        raise ValueError(REASONS['score']) from error
    if not vocal.notes:
        raise ValueError(REASONS['score'])
    sections = [dict(name=name, start=start, phrases=0, notes=0) for start, name in starts]
    if not sections or sections[0]['start'] > 0:
        sections.insert(0, dict(name=None, start=Fraction(0), phrases=0, notes=0))
    bounds = [s['start'] for s in sections]
    notes = []
    for onset, pitch, duration in vocal.notes:
        index = max(bisect.bisect_right(bounds, onset) - 1, 0)
        notes.append(dict(on=onset, pitch=pitch, dur=duration, section=index))
        sections[index]['notes'] += 1
    phrases, end = [], None
    for k, note in enumerate(notes):
        if end is None or note['on'] - end >= PHRASE_GAP:
            phrases.append(dict(first=k, last=k))
        phrases[-1]['last'] = k
        end = note['on'] + note['dur']
    for phrase in phrases:
        phrase['section'] = notes[phrase['last']]['section']
        phrase['count'] = phrase['last'] - phrase['first'] + 1
        sections[phrase['section']]['phrases'] += 1
    return dict(notes=notes, phrases=phrases, sections=sections, bpm=score.bpm)


def _vlq(data, i):
    value = 0
    for _ in range(4):
        byte = data[i]
        i += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, i
    raise ValueError('bad MIDI length')


def midi_notes(data):
    """{(track, channel): [(onset_s, pitch, duration_s)]} from a Standard MIDI File (bytes)."""
    if data[:4] != b'MThd':
        raise ValueError('not a MIDI file')
    size = int.from_bytes(data[4:8], 'big')
    _, _, division = struct.unpack('>HHH', data[8:14])
    pos, tracks, tempos = 8 + size, [], []
    while pos + 8 <= len(data):
        kind, length = data[pos:pos + 4], int.from_bytes(data[pos + 4:pos + 8], 'big')
        body, pos = data[pos + 8:pos + 8 + length], pos + 8 + length
        if kind != b'MTrk':
            continue
        events, i, tick, status = [], 0, 0, None
        while i < len(body):
            delta, i = _vlq(body, i)
            tick += delta
            byte = body[i]
            if byte == 0xFF:
                meta, (length, j) = body[i + 1], _vlq(body, i + 2)
                if meta == 0x51 and length == 3:
                    tempos.append((tick, int.from_bytes(body[j:j + 3], 'big')))
                i = j + length
                if meta == 0x2F:
                    break
                continue
            if byte in (0xF0, 0xF7):
                length, j = _vlq(body, i + 1)
                i = j + length
                continue
            if byte & 0x80:
                status, i = byte, i + 1
            elif status is None:
                raise ValueError('MIDI running status without a status byte')
            command, channel = status & 0xF0, status & 0x0F
            width = 1 if command in (0xC0, 0xD0) else 2
            values, i = body[i:i + width], i + width
            if command == 0x90 and values[1] > 0:
                events.append((tick, True, channel, values[0]))
            elif command == 0x80 or command == 0x90:
                events.append((tick, False, channel, values[0]))
        tracks.append(events)
    if division & 0x8000:
        per_tick = 1.0 / ((256 - (division >> 8)) * (division & 0xFF))

        def seconds(tick):
            return tick * per_tick
    else:
        marks, clock, last, rate = [], 0.0, 0, 500000
        for tick, tempo in sorted(tempos):
            clock += (tick - last) * rate / 1e6 / division
            marks.append((tick, clock, tempo))
            last, rate = tick, tempo
        ticks = [m[0] for m in marks]

        def seconds(tick):
            k = bisect.bisect_right(ticks, tick) - 1
            if k < 0:
                return tick * 500000 / 1e6 / division
            base, clock, tempo = marks[k]
            return clock + (tick - base) * tempo / 1e6 / division
    result = {}
    for number, events in enumerate(tracks):
        open_notes = {}
        for tick, on, channel, pitch in events:
            key = (channel, pitch)
            if key in open_notes:
                start = open_notes.pop(key)
                result.setdefault((number, channel), []).append((seconds(start), pitch, seconds(tick) - seconds(start)))
            if on:
                open_notes[key] = tick
    return {key: sorted(notes) for key, notes in result.items()}


def _linear(points):
    n = len(points)
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    slope = sum((p[0] - mx) * (p[1] - my) for p in points) / sxx if sxx else 0.0
    return slope, my - slope * mx


def note_times(plan, midis):
    """Real [onset_s, end_s] for every sung score note, from SheetSage2's own MIDI.

    midis: [(name, MIDI bytes)]. The MIDI part whose pitch sequence best matches the score's
    Vocal notes is matched note for note (difflib); matched onsets become anchors of a
    monotone map from score time to recording time, so notes the MIDI lacks still get a time.
    Returns dict(notes=[[onset_s, end_s], ...], matched=share of notes anchored, source=name)."""
    wanted = [n['pitch'] for n in plan['notes']]
    best = None
    for name, data in midis:
        try:
            parts = midi_notes(data)
        except (ValueError, IndexError, struct.error):
            continue
        for key, notes in parts.items():
            if len(notes) < 4:
                continue
            for classes in (False, True):
                a = [p % 12 for p in wanted] if classes else wanted
                b = [n[1] % 12 if classes else n[1] for n in notes]
                matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
                ratio = matcher.ratio()
                # SheetSage2's raw vocal melody keeps real timing; the notation companions the
                # score was built from are grid-quantised, so they win only when it is missing.
                rank = (name.rsplit('/', 1)[-1] == 'melody_vocal.mid' and ratio >= MIN_NOTE_MATCH, ratio)
                if best is None or rank > best[0]:
                    best = (rank, f'{name}#{key[0]}/{key[1]}', notes, matcher)
    if best is None:
        raise ValueError(REASONS['notes'])
    _, source, midi, matcher = best
    blocks = [b for b in matcher.get_matching_blocks() if b.size]
    size = 3 if sum(b.size for b in blocks if b.size >= 3) >= len(wanted) * MIN_NOTE_MATCH else 2
    anchors = [(float(plan['notes'][b.a + k]['on']), midi[b.b + k][0])
               for b in blocks if b.size >= size for k in range(b.size)]
    for _ in range(2):  # drop anchors far off the song's overall tempo line (a mismatched repeat)
        if len(anchors) < 4:
            break
        slope, offset = _linear(anchors)
        errors = sorted(abs(s - (slope * q + offset)) for q, s in anchors)
        limit = max(1.5, 4 * errors[len(errors) // 2])
        anchors = [(q, s) for q, s in anchors if abs(s - (slope * q + offset)) <= limit]
    kept = []
    for q, s in anchors:
        if not kept or (q > kept[-1][0] and s > kept[-1][1]):
            kept.append((q, s))
    matched = len(kept) / len(wanted)
    if len(kept) < 2 or matched < MIN_NOTE_MATCH:
        raise ValueError(REASONS['notes'])
    slope, _ = _linear(kept)
    slope = slope if slope > 0 else 60.0 / plan['bpm']
    qs = [q for q, _ in kept]

    def when(q):
        k = bisect.bisect_right(qs, q) - 1
        if k < 0:
            return kept[0][1] + (q - kept[0][0]) * slope
        if k >= len(kept) - 1:
            return kept[-1][1] + (q - kept[-1][0]) * slope
        (q0, s0), (q1, s1) = kept[k], kept[k + 1]
        return s0 + (q - q0) * (s1 - s0) / (q1 - q0)
    times = [[round(when(float(n['on'])), 3), round(when(float(n['on'] + n['dur'])), 3)] for n in plan['notes']]
    return dict(notes=times, matched=round(matched, 3), source=source)


def mismatch(syllable_count, notes):
    """Relative squared mismatch; cramming syllables costs three times a melisma."""
    diff = syllable_count - notes
    return (3.0 if diff > 0 else 1.0) * diff * diff / max(notes, 1)


def sections_without_tune(sections, plan, words=None, times=None):
    """Her sections paired by name and order with the score's (difflib) whose score section
    owns no phrase, such as words under [Intro] where the recording's intro is instrumental.

    Returns (set aside, kept): a paired section is kept after all when most of its words are
    heard clearly on the tune's phrases, as when a song's last line is written under [Outro]
    but SheetSage2 transcribed its notes inside the verse. Needs plan notes with times ('s',
    'e') for that check; without words and times every such pair is set aside."""
    hers = [s for s in sections if s['words'] and s['name']]
    theirs = plan['sections']
    matcher = difflib.SequenceMatcher(None, [s['name'] for s in hers], [s['name'] for s in theirs], autojunk=False)
    spans = [(plan['notes'][p['first']].get('s'), plan['notes'][p['last']].get('e')) for p in plan['phrases']]
    aside, kept = set(), set()
    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        if op != 'equal':
            continue
        for a, b in zip(range(a0, a1), range(b0, b1)):
            if theirs[b]['phrases']:
                continue
            index, on_tune = hers[a]['index'], 0
            for i, word in enumerate(words or []):
                t = times[i] if times and i < len(times) else None
                if word['section'] != index or not t or (t.get('score') or 0) < LOW_SCORE:
                    continue
                start = t.get('start')
                if isinstance(start, (int, float)) and any(
                        s is not None and s - 1.0 <= start <= e + 1.0 for s, e in spans):
                    on_tune += 1
            (kept if on_tune * 2 > hers[a]['words'] else aside).add(index)
    return aside, kept


def _heard(times, i):
    t = times[i] if i < len(times) else None
    return bool(t) and isinstance(t.get('start'), (int, float)) and (t.get('score') or 0) >= LOW_SCORE


def _onsets(words, sung, times):
    """Onset and trust per sung word; words the aligner could not place are interpolated."""
    onset, weight = [None] * len(sung), [0.1] * len(sung)
    for k, i in enumerate(sung):
        t = times[i] if i < len(times) else None
        if t and isinstance(t.get('start'), (int, float)):
            onset[k] = float(t['start'])
            weight[k] = 1.0 if (t.get('score') or 0) >= LOW_SCORE else UNHEARD_WEIGHT
    known = [k for k, t in enumerate(onset) if t is not None]
    if not known:
        return None, None
    for k in range(len(onset)):
        if onset[k] is not None:
            continue
        before = max((j for j in known if j < k), default=None)
        after = min((j for j in known if j > k), default=None)
        if before is None:
            onset[k] = onset[after] - 0.3 * (after - k)
        elif after is None:
            onset[k] = onset[before] + 0.3 * (k - before)
        else:
            onset[k] = onset[before] + (onset[after] - onset[before]) * (k - before) / (after - before)
    return onset, weight


def _assign(plan, words, sung, onset, weight):
    """Sung words (in order) to phrases (in order): the fewest seconds of onsets outside their
    phrase, plus a pull toward as many syllables as notes. The pull is light where the words
    were heard clearly and up to five times stronger where they were not, and a phrase cannot
    take much more than MAX_PER_NOTE syllables per note unless its words were heard clearly
    there (a stretch SheetSage2 wrote too few notes for). A phrase that sings words of two of
    her sections costs MIX, so unclear words do not drift across her section tags. None when
    impossible."""
    notes, phrases = plan['notes'], plan['phrases']
    P, W = len(phrases), len(sung)
    lo = [notes[p['first']]['s'] - LEAD for p in phrases]
    hi = [notes[p['last']]['s'] + LEAD for p in phrases]
    syl, pull, mixed, unclear = [0], [0.0], [0], [0.0]
    for k, i in enumerate(sung):
        unclear.append(unclear[-1] + (1.0 - weight[k]))
        syl.append(syl[-1] + words[i]['syllables'])
        pull.append(pull[-1] + COUNT_WEIGHT * (1.0 + LOW_WEIGHT_BOOST * (1.0 - weight[k])))
        mixed.append(mixed[-1] + (k > 0 and words[i]['section'] != words[sung[k - 1]]['section']))
    INF = float('inf')
    best = [[INF] * (P + 1) for _ in range(W + 1)]
    back = [[None] * (P + 1) for _ in range(W + 1)]
    best[0][0] = 0.0
    for j in range(P):
        count = phrases[j]['count']
        for i in range(W + 1):
            here = best[i][j]
            if here == INF:
                continue
            # An empty phrase: as costly as the words that could have sung it were unclear.
            near = pull[min(i + 1, W)] - pull[min(i, W - 1)] if W else COUNT_WEIGHT
            cost = here + near * mismatch(0, count)
            if cost < best[i][j + 1]:
                best[i][j + 1], back[i][j + 1] = cost, i
            timing = 0.0
            for k in range(i, min(W, i + MAX_GROUP)):
                t = onset[k]
                gap = lo[j] - t if t < lo[j] else t - hi[j] if t > hi[j] else 0.0
                if gap > FAR and weight[k] >= 1.0:
                    break
                timing += TIME_WEIGHT * weight[k] * gap
                size = syl[k + 1] - syl[i]
                weight_here = (pull[k + 1] - pull[i]) / (k + 1 - i)
                over = max(0.0, size - max(MAX_PER_NOTE * count, count + 1))
                # Overfull is weighed by how unclear the group's words are: words heard clearly
                # inside a phrase stay there even where SheetSage2 wrote too few notes for them.
                share = max(OVERFULL_FLOOR, (unclear[k + 1] - unclear[i]) / (k + 1 - i))
                cost = (here + timing + weight_here * mismatch(size, count) + OVERFULL * share * over
                        + MIX * (mixed[k + 1] - mixed[i + 1]))
                if cost < best[k + 1][j + 1]:
                    best[k + 1][j + 1], back[k + 1][j + 1] = cost, i
    if best[W][P] == INF:
        return None
    groups, i = [None] * P, W
    for j in range(P, 0, -1):
        start = back[i][j]
        if start < i:
            groups[j - 1] = (start, i)
        i = start
    return groups


def _repeat_note(notes, n):
    """Whether note n repeats the pitch of the touching note before it: SheetSage2 splits one
    sung pitch at its own guess, so that boundary is not heard evidence for a word start."""
    return n > 0 and notes[n - 1]['pitch'] == notes[n]['pitch'] and notes[n - 1]['on'] + notes[n - 1]['dur'] == notes[n]['on']


def _snap(notes, first, last, ts, ws, syls):
    """Start note of each word of one phrase group: the first word starts the phrase, starts
    rise note by note, and every start is a note onset. A clearly heard word never starts more
    than SNAP seconds before it was heard: a word heard inside a note (or after it) does not
    take that note, which belongs to the word before it, unless the note only repeats the pitch
    before it. Two words may start on one note (the second re-singing its pitch) at a cost, and
    only where the second is heard inside that note; anywhere else a share costs FORCED_SHARE, so
    a phrase with fewer notes than words still fits. Returns absolute note indices."""
    count, size = last - first + 1, len(ts)
    INF = float('inf')

    def place(k, n, shared):
        note = notes[first + n]
        t, s, e = ts[k], note['s'], note['e']
        if shared:  # starting inside a note the word before started
            if syls[k] == 0 or s - SNAP <= t <= e + SNAP:
                return 0.0
            return FORCED_SHARE + TIME_WEIGHT * ws[k] * (s - t if t < s else t - e)
        cost = TIME_WEIGHT * ws[k] * abs(t - s)
        # The same-pitch exception covers a word heard inside a quick repeat note only: never a
        # word heard after that note ended, and never a held note (it belongs to the word that
        # reaches it).
        if t - s > SNAP and not (t <= e + SNAP and note['dur'] < HELD and _repeat_note(notes, first + n)):
            cost += REFUSE * ws[k]
        return cost

    def span(k, owned):
        if owned <= 0:
            return 0.0 if syls[k] == 0 else SHARE + CRAM * syls[k]
        return CRAM * max(0, syls[k] - owned) + MELISMA * max(0, owned - max(syls[k], 1))
    best = [[INF] * count for _ in range(size)]
    back = [[None] * count for _ in range(size)]
    best[0][0] = place(0, 0, False)
    for k in range(1, size):
        for n in range(count):
            for m in range(n + 1):
                if best[k - 1][m] == INF:
                    continue
                cost = best[k - 1][m] + span(k - 1, n - m) + place(k, n, n == m)
                if cost < best[k][n]:
                    best[k][n], back[k][n] = cost, m
    n = min(range(count), key=lambda x: best[size - 1][x] + span(size - 1, count - x))
    out = [0] * size
    for k in range(size - 1, -1, -1):
        out[k] = first + n
        if k:
            n = back[k][n]
    return out


def word_lag(plan, onset, weight):
    """The median seconds clearly heard words sit after their nearest note onset (0 with fewer
    than eight such words). The snap refuses a word heard more than SNAP inside a note, so a
    steady lag between the aligner and SheetSage2 is taken out first."""
    ons = sorted(n['s'] for n in plan['notes'])
    diffs = []
    for t, w in zip(onset, weight):
        if w < 1.0:
            continue
        j = bisect.bisect_left(ons, t)
        near = min(((abs(t - ons[x]), t - ons[x]) for x in (j - 1, j) if 0 <= x < len(ons)), default=None)
        if near and near[0] <= LAG_WINDOW:
            diffs.append(near[1])
    if len(diffs) < 8:
        return 0.0
    diffs.sort()
    return diffs[len(diffs) // 2]


def _starts(plan, groups, words, sung, onset, weight):
    """Start note (absolute index) of every sung word, phrase group by phrase group."""
    start = [None] * len(sung)
    for j, group in enumerate(groups):
        if group is None:
            continue
        a, b = group
        phrase = plan['phrases'][j]
        start[a:b] = _snap(plan['notes'], phrase['first'], phrase['last'], onset[a:b], weight[a:b],
                           [words[sung[k]]['syllables'] for k in range(a, b)])
    return start


def _phrase_of(plan):
    """Phrase index of every note."""
    out = [0] * len(plan['notes'])
    for j, phrase in enumerate(plan['phrases']):
        for n in range(phrase['first'], phrase['last'] + 1):
            out[n] = j
    return out


def _regroup(plan, start):
    """Phrase groups again from word start notes: each word sings in the phrase of its start."""
    phrase_of = _phrase_of(plan)
    groups = [None] * len(plan['phrases'])
    for k, n in enumerate(start):
        j = phrase_of[n]
        groups[j] = (groups[j][0], k + 1) if groups[j] else (k, k + 1)
    return groups


def _owners(plan, groups, start, syllable_counts=None):
    """The word (sung position) singing each note of a phrase with words: the last word to
    start on or before it. On a note two words start on, the first one with a syllable owns its
    onset (a dash or a number never owns a note another word sings)."""
    own = {}
    for j, group in enumerate(groups):
        if group is None:
            continue
        a, b = group
        phrase = plan['phrases'][j]
        for n in range(phrase['first'], phrase['last'] + 1):
            on = [k for k in range(a, b) if start[k] == n]
            if on:
                voiced = [k for k in on if not syllable_counts or syllable_counts[k]]
                own[n] = (voiced or on)[0]
            else:
                before = [k for k in range(a, b) if start[k] <= n]
                own[n] = before[-1] if before else a
    return own


def _note_map(src, tgt):
    """Source note index -> target note index for two runs of notes with the same tune shape
    (lists of absolute note indices and their pitches), by difflib on pitch."""
    matcher = difflib.SequenceMatcher(None, [p for _, p in src], [p for _, p in tgt], autojunk=False)
    pairs = [(block.a + d, block.b + d) for block in matcher.get_matching_blocks() for d in range(block.size)]
    mapping = {}
    for a in range(len(src)):
        exact = next((b for x, b in pairs if x == a), None)
        if exact is None:
            before = [(x, b) for x, b in pairs if x < a]
            after = [b for x, b in pairs if x > a]
            b = before[-1][1] + (a - before[-1][0]) if before else 0
            if after:
                b = min(b, after[0] - 1 if after[0] > 0 else 0)
            exact = max(0, min(b, len(tgt) - 1))
        mapping[src[a][0]] = tgt[exact][0]
    return mapping, matcher.ratio()


def _repeats(words, sections, sung, start, plan, times, trusted):
    """Sections with the same words and the same tune share one layout: the word starts of the
    instance heard most clearly, carried onto the other instances note for note. Changes start
    and trusted in place; returns (report rows, {target position: source position})."""
    notes, phrase_of = plan['notes'], _phrase_of(plan)
    position = {i: k for k, i in enumerate(sung)}
    families = {}
    for s in sections:
        mine = [i for i, w in enumerate(words) if w['section'] == s['index']]
        if len(mine) < 4 or any(i not in position for i in mine) or sum(1 for i in mine if words[i]['norm']) < 4:
            continue
        families.setdefault(tuple(words[i]['norm'] for i in mine), []).append(
            dict(section=s['index'], positions=[position[i] for i in mine],
                 heard=sum(1 for i in mine if _heard(times, i)),
                 score=sum((times[i].get('score') or 0) for i in mine if i < len(times) and times[i]) / len(mine)))
    rows, copied = [], {}
    for members in families.values():
        if len(members) < 2:
            continue
        for m in members:
            ks = m['positions']
            m['notes'] = list(range(start[ks[0]], plan['phrases'][phrase_of[start[ks[-1]]]]['last'] + 1))
        source = max(members, key=lambda m: (m['heard'], m['score']))
        row = dict(section=next(s['name'] for s in sections if s['index'] == source['section']),
                   instances=[m['section'] for m in members], source=source['section'],
                   heard=[m['heard'] for m in members], words=len(source['positions']),
                   fitted=trusted[source['section']], disagreements=[])
        src_notes = [(n, notes[n]['pitch']) for n in source['notes']]
        for m in members:
            if m is source:
                continue
            ks = m['positions']
            mapping, ratio = _note_map(src_notes, [(n, notes[n]['pitch']) for n in m['notes']])
            moved = [mapping[start[s]] for s in source['positions']]
            for r in range(1, len(moved)):  # starts rise wherever the source's rise, and never fall
                src_rises = start[source['positions'][r]] > start[source['positions'][r - 1]]
                moved[r] = max(moved[r], moved[r - 1] + 1 if src_rises else moved[r - 1])
            fits = (ratio >= REPEAT_MATCH and moved[-1] <= m['notes'][-1]
                    and (ks[0] == 0 or start[ks[0] - 1] < moved[0])
                    and (ks[-1] + 1 >= len(start) or moved[-1] < start[ks[-1] + 1]))
            own_held = {n: max((r for r in range(len(ks)) if start[ks[r]] <= n), default=0)
                        for n in m['notes'] if notes[n]['dur'] >= HELD}
            new_held = {n: max((r for r in range(len(ks)) if moved[r] <= n), default=0) for n in own_held}
            row['disagreements'].append(dict(
                instance=m['section'], tune_match=round(ratio, 3), shared=fits,
                starts=sum(1 for r, k in enumerate(ks) if start[k] != moved[r]),
                held_owners=sum(1 for n in own_held if own_held[n] != new_held[n])))
            if not fits:
                continue
            for r, k in enumerate(ks):
                start[k] = moved[r]
                copied[k] = source['positions'][r]
            trusted[m['section']] = trusted[source['section']]
        rows.append(row)
    return rows, copied


def _voiced_after(order, times):
    """Seconds each word is sung before the next word starts, less any quiet before that next
    word: long for a held note, short for a quick one, never counting a rest."""
    spans = {}
    timed = [i for i in order if times.get(i)]
    for k, i in enumerate(timed):
        t = times[i]
        nxt = times[timed[k + 1]] if k + 1 < len(timed) else None
        if nxt:
            spans[i] = max(0.0, nxt['start'] - t['start'] - (nxt.get('quiet_before') or 0.0))
        else:
            spans[i] = max(0.0, (t.get('end') or t['start']) - t['start'])
    return spans


def _tag(name):
    return f'[{name.title()}]' if name else None


def _named(sections, indices):
    """'the bridge', 'the second verse' ...: her sections by name, with an ordinal when the
    name comes back."""
    names = []
    for index in indices:
        s = next(x for x in sections if x['index'] == index)
        same = [x['index'] for x in sections if x['name'] == s['name'] and x['words']]
        name = s['name'] or 'untitled'
        if len(same) > 1:
            nth = same.index(index)
            name = f'{ORDINALS[nth] if nth < len(ORDINALS) else str(nth + 1) + "th"} {name}'
        names.append('the ' + name)
    return names[0] if len(names) == 1 else ', '.join(names[:-1]) + ' and ' + names[-1]


def _split_long(line, words, sung, her_eol, cap, owned):
    """Split one line (sung positions) until no part has more syllables than both her longest
    line (cap) and MAX_PER_NOTE per note it sings (owned: notes per position): at her own line
    end nearest the middle, else after a clause mark, else nearest the syllable middle."""
    count = sum(words[sung[k]]['syllables'] for k in line)
    if count <= max(cap, MAX_PER_NOTE * sum(owned.get(k, 0) for k in line)) or len(line) < 2:
        return [line]
    half, running, best = count / 2.0, 0, None
    for x, k in enumerate(line[:-1]):
        running += words[sung[k]]['syllables']
        rank = (0 if k in her_eol else 1 if ENDS_CLAUSE.search(words[sung[k]]['text']) else 2, abs(running - half))
        if best is None or rank < best[0]:
            best = (rank, x)
    cut = best[1] + 1
    return (_split_long(line[:cut], words, sung, her_eol, cap, owned)
            + _split_long(line[cut:], words, sung, her_eol, cap, owned))


def fit(abc, lyrics, timing, word_times, failed=None, touchup=False):
    """Her lyrics re-broken to the recording's phrases from measured timing.

    timing: note_times() output for this score. word_times: per word of lyric_words(lyrics),
    {'start', 'end', 'score', 'quiet_before'} or None, from align.py on the recording.
    failed: a REASONS key when the timing could not be measured at all. touchup: also build a
    touched-up score (touch_up()) for the fitted layout.
    Returns dict(applied, lyrics, order, report, notes, plan, abc, line_ends). When applied is
    False the lyrics are hers unchanged and report['reason'] says why; plan is then None. abc is
    the touched-up score, or None; line_ends marks, per word of order, a line end in lyrics."""
    words, sections = lyric_words(lyrics)
    report = dict(revision=REVISION, timing_source='recording', applied=False, reason=None)
    out = dict(applied=False, lyrics=lyrics, order=list(range(len(words))), report=report, notes=[], plan=None,
               abc=None)

    def refuse(reason):
        report['reason'] = reason
        out['notes'].append(f'Your line breaks were kept as written: {REASONS[reason]}.')
        return out
    if not words:
        return refuse('empty')
    if failed in REASONS:
        return refuse(failed)
    try:
        plan = score_plan(abc)
    except ValueError:
        return refuse('score')
    if not timing or len(timing.get('notes') or []) != len(plan['notes']):
        return refuse('notes')
    for note, (start_s, end_s) in zip(plan['notes'], timing['notes']):
        note['s'], note['e'] = start_s, max(end_s, start_s)
    report['note_timing'] = dict(matched=timing.get('matched'), source=timing.get('source'))
    times = word_times or []
    aside, kept_by_timing = sections_without_tune(sections, plan, words, times)
    sung = [i for i, w in enumerate(words) if w['section'] not in aside]
    report['words_without_tune'] = [dict(section=s['name'], words=s['words'],
                                         lines=len({w['line'] for w in words if w['section'] == s['index']}))
                                    for s in sections if s['index'] in aside]
    report['sections_sung_elsewhere'] = len(kept_by_timing)
    if not sung:
        return refuse('empty')
    heard = sum(1 for i in sung if _heard(times, i))
    report['words'] = dict(sung=len(sung), heard=heard, set_aside=len(words) - len(sung))
    if heard < MIN_HEARD * len(sung):
        return refuse('heard')
    onset, weight = _onsets(words, sung, times)
    if onset is None:
        return refuse('heard')
    notes, phrases = plan['notes'], plan['phrases']
    far = 0
    for k, t in enumerate(onset):  # a word far from every phrase is placed by its neighbours
        gap = min(max(notes[p['first']]['s'] - LEAD - t, t - notes[p['last']]['s'] - LEAD, 0.0) for p in phrases)
        if gap > FAR:
            far += 1
            weight[k] = min(weight[k], 0.3)
    report['words']['far_from_notes'] = far
    groups = _assign(plan, words, sung, onset, weight)
    if groups is None:
        return refuse('words')
    covered = sum(p['count'] for p, g in zip(phrases, groups) if g)
    report['coverage'] = round(covered / len(notes), 3)
    if covered < MIN_COVERAGE * len(notes):
        return refuse('words')
    # Note times and word times must agree: a clearly heard first word sits near its phrase's
    # first note. A large gap means the note timing followed the score's steady grid instead.
    offsets = sorted(abs(onset[g[0]] - notes[phrases[j]['first']]['s'])
                     for j, g in enumerate(groups) if g and weight[g[0]] >= 1.0)
    if offsets:
        middle = offsets[len(offsets) // 2]
        report['onset_offset_s'] = dict(median=round(middle, 3), max=round(offsets[-1], 3))
        if middle > MAX_OFFSET:
            return refuse('notes')
    her_breaks = [k for k, i in enumerate(sung) if words[i]['brk'] and k < len(sung) - 1]
    before_groups = groups
    lag = word_lag(plan, onset, weight)
    report['word_lag_s'] = round(lag, 3)
    start = _starts(plan, groups, words, sung, [t - lag for t in onset], weight)

    # A section whose words were mostly not heard keeps her own lines (and repeats inherit the
    # verdict of the instance they copy).
    section_of = [words[i]['section'] for i in sung]
    heard_in = {s['index']: [0, 0] for s in sections}
    for k, i in enumerate(sung):
        heard_in[section_of[k]][1] += 1
        heard_in[section_of[k]][0] += _heard(times, i)
    trusted = {s: h >= MIN_SECTION_HEARD * n for s, (h, n) in heard_in.items()}
    repeats, copied = _repeats(words, sections, sung, start, plan, times, trusted)
    groups = _regroup(plan, start)
    own = _owners(plan, groups, start, [words[i]['syllables'] for i in sung])
    phrase_of = _phrase_of(plan)
    fitted = [trusted[s] for s in section_of]
    if not any(fitted):
        return refuse('heard')

    # Line breaks and commas, word by word; copied instances take their source's layout below.
    last_of_group = {g[1] - 1 for g in groups if g}
    her_eol = {k for k, i in enumerate(sung) if words[i]['eol']}
    section_end = {k for k in range(len(sung)) if k == len(sung) - 1 or section_of[k + 1] != section_of[k]}
    breaks, commas, moved, kept_breaks = set(), set(), 0, 0
    for k in range(len(sung)):
        if not fitted[k]:
            if k in her_eol:
                breaks.add(k)
            continue
        if k in last_of_group or k in section_end:
            breaks.add(k)
        elif k in her_eol:
            near = [p for p in last_of_group
                    if abs(p - k) <= MOVE_BREAK and section_of[p] == section_of[k] and fitted[p]]
            if near:
                moved += 1
            else:
                breaks.add(k)
                kept_breaks += 1
    # A very short phrase joins the line before it with a comma, or the line after it when it
    # opens one of her sections.
    merged = 0
    present = [g for g in groups if g]

    def join(p, q):
        """Join the line ending at position p onto the next one (q is a word on that next line)."""
        nonlocal merged
        if fitted[p] and fitted[q] and section_of[p] == section_of[q] and p not in section_end and p in breaks:
            breaks.discard(p)
            commas.add(p)
            merged += 1
            return True
        return False
    for x, group in enumerate(present):
        j = phrase_of[start[group[0]]]
        if phrases[j]['count'] > SHORT_PHRASE:
            continue
        if not (x and join(present[x - 1][1] - 1, group[0])) and x + 1 < len(present):
            if x == 0 or section_of[present[x - 1][1] - 1] != section_of[group[0]]:
                join(group[1] - 1, present[x + 1][0])
    for j, group in enumerate(groups):
        if group is None:
            continue
        for n in range(phrases[j]['first'], phrases[j]['last']):
            k = own[n]
            if notes[n]['dur'] >= CLAUSE_HOLD and k != group[1] - 1 and fitted[k] and k not in breaks:
                commas.add(k)
    for k, src in copied.items():  # one layout for every instance of a repeated section
        if not fitted[k]:
            continue
        for marks in (breaks, commas):
            marks.discard(k)
            if src in marks:
                marks.add(k)

    # Lines, split where one grows past her longest line.
    her_lines = {}
    for w in words:
        her_lines.setdefault(w['line'], 0)
        her_lines[w['line']] += w['syllables']
    cap = max(her_lines.values()) if her_lines else 0
    lines, line = [], []
    for k in range(len(sung)):
        line.append(k)
        if k in breaks or k == len(sung) - 1:
            lines.append(line)
            line = []
    owned = {}
    for n, k in own.items():
        owned[k] = owned.get(k, 0) + 1
    split = []
    for line in lines:
        parts = _split_long(line, words, sung, her_eol, cap, owned) if all(fitted[k] for k in line) else [line]
        split.extend(parts)
        for part in parts[:-1]:
            breaks.add(part[-1])
            commas.discard(part[-1])
    report['lines_split_at_cap'] = len(split) - len(lines)
    lines = split

    # The text: each of her sections opens with the tag of the score section most of its words
    # sing in; inside it, a line whose first word sings in a later score section opens that one.
    home = {}
    for k in range(len(sung)):
        home.setdefault(section_of[k], []).append(phrases[phrase_of[start[k]]]['section'])
    home = {s: max(set(v), key=lambda x: (v.count(x), -x)) for s, v in home.items()}
    tagged = {s['index']: bool(s['name']) for s in sections}
    blocks, removed, added = [], 0, 0
    for line in lines:
        first = line[0]
        if (first == 0 or section_of[first - 1] != section_of[first]) and tagged[section_of[first]]:
            section = home[section_of[first]]
        elif not blocks:
            section = phrases[phrase_of[start[first]]]['section']
        else:
            section = max(phrases[phrase_of[start[first]]]['section'], blocks[-1]['section'])
        if not blocks or blocks[-1]['section'] != section:
            blocks.append(dict(section=section, lines=[]))
        tokens = []
        for x, k in enumerate(line):
            token = words[sung[k]]['text']
            if fitted[k] and k not in breaks:
                bare = CLAUSE_MARK.sub('', token)
                # No comma after a lone dash or before one: the dash already marks the clause.
                dash_next = x + 1 < len(line) and LONE_DASH.match(words[sung[line[x + 1]]]['text'])
                comma = (k in commas and bool(bare) and not ENDS_CLAUSE.search(bare) and not dash_next
                         and not LONE_DASH.match(bare))
                if bare != token and not comma:
                    removed += 1
                token = bare
                if comma:
                    if not ENDS_CLAUSE.search(words[sung[k]]['text']):
                        added += 1
                    token += ','
            if token:
                tokens.append(token)
        blocks[-1]['lines'].append(' '.join(tokens))
    text = '\n\n'.join('\n'.join(([_tag(plan['sections'][b['section']]['name'])]
                                  if plan['sections'][b['section']]['name'] else []) + b['lines']) for b in blocks)

    # Held notes: whose word each one plans, and whether the recording shows it clearly.
    src_times = {i: times[i] for i in sung if i < len(times) and times[i]}
    src_spans = _voiced_after(sung, src_times)
    held = []
    for j, group in enumerate(groups):
        if group is None:
            continue
        phrase = phrases[j]
        for n in range(phrase['first'], phrase['last'] + 1):
            if notes[n]['dur'] < HELD:
                continue
            k = own[n]
            near = [m for m in (k - 1, k + 1) if 0 <= m < len(sung)]
            mine = src_spans.get(sung[k])
            clear = mine is not None and all(src_spans.get(sung[m]) is None or mine > src_spans[sung[m]] for m in near)
            verified = clear and fitted[k] and _heard(times, sung[k]) and all(_heard(times, sung[m]) for m in near)
            held.append(dict(section=plan['sections'][phrase['section']]['name'], phrase=j,
                             note=n - phrase['first'], beats=float(notes[n]['dur']),
                             seconds=round(notes[n]['e'] - notes[n]['s'], 2), word=sung[k],
                             word_in_line=k - next(x for x in range(k, -1, -1) if x == 0 or (x - 1) in breaks),
                             last=n == phrase['last'], clear=clear, verified=verified))
    with_words = sum(1 for g in groups if g)
    order = list(sung)
    report.update(
        applied=True, phrases=len(phrases), phrases_with_words=with_words,
        lines=len(lines), short_phrases_joined=merged, commas_added=added, commas_removed=removed,
        her_breaks=len(her_breaks),
        her_breaks_inside_phrases=sum(1 for k in her_breaks if k not in {g[1] - 1 for g in before_groups if g}),
        her_breaks_kept=kept_breaks, her_breaks_moved=moved,
        breaks_on_phrase_ends=dict(before=sum(1 for g in before_groups if g and words[sung[g[1] - 1]]['brk']),
                                   after=sum(1 for g in groups if g and (g[1] - 1 in breaks or g[1] - 1 in commas)),
                                   of=with_words),
        line_cap=cap, longest_line=max((sum(words[sung[k]]['syllables'] for k in line) for line in lines), default=0),
        sections=[dict(index=s['index'], section=s['name'], words=heard_in[s['index']][1],
                       heard=heard_in[s['index']][0], fitted=trusted[s['index']])
                  for s in sections if s['index'] not in aside and heard_in[s['index']][1]],
        repeats=repeats, held_notes=held)
    kept = dict(groups=[[sung[k] for k in range(g[0], g[1])] if g else None for g in groups],
                sections=[plan['sections'][p['section']]['name'] for p in phrases], held=held)
    kept['source_pauses'] = source_pauses(kept, times)
    out.update(applied=True, lyrics=text, order=order, plan=kept,
               line_ends=[k in breaks or k == len(sung) - 1 for k in range(len(sung))])
    lines_left = sum(g['lines'] for g in report['words_without_tune'])
    left = ''
    if report['words_without_tune']:
        parts = [f"{g['lines']} {g['section'] or 'untitled'} line{'s' if g['lines'] != 1 else ''}"
                 for g in report['words_without_tune']]
        said = parts[0] if len(parts) == 1 else ', '.join(parts[:-1]) + ' and ' + parts[-1]
        left = (f"; {said} {'were' if lines_left != 1 else 'was'} left out, because the recording has no sung tune "
                'there')
    out['notes'].append(f"Your line breaks now follow the tune's phrases: your words keep their order{left}.")
    unfitted = [s['index'] for s in sections if s['index'] not in aside and heard_in[s['index']][1]
                and not trusted[s['index']]]
    if unfitted:
        their = 'its' if len(unfitted) == 1 else 'their'
        named = _named(sections, unfitted)
        out['notes'].append(f"{named[0].upper()}{named[1:]} {'keeps' if len(unfitted) == 1 else 'keep'} your own "
                            f'line breaks: too few of {their} words could be heard clearly in the recording.')
    if touchup:
        touched, touch_report = touch_up(abc, plan, groups, start, own, fitted, words, sung)
        report['score_touchup'] = touch_report
        out['abc'] = touched
    return out


def _vocal_tokens(abc):
    """Each sounding Vocal note's tokens in the ABC text: [(line index, start, end, match)],
    walking the native two-voice layout the way abc_tools.parse does."""
    lines = abc.splitlines()
    notes, pending, cursor = [], False, 8
    while cursor < len(lines):
        while cursor < len(lines) and lines[cursor].startswith('% '):
            cursor += 1
        for name in ('Vocal', 'Ins'):
            cursor += 1
            while cursor < len(lines) and lines[cursor].startswith(('M:', 'K:')):
                cursor += 1
            if cursor >= len(lines):
                return notes
            if name == 'Vocal':
                line, offset = lines[cursor], 0
                for bar in line[:-1].split('|'):
                    if not re.fullmatch(r'Z([2-4])?', bar.strip()):
                        position = offset
                        while position < offset + len(bar):
                            if bar[position - offset].isspace():
                                position += 1
                                continue
                            match = TOKEN.match(line, position)
                            position = match.end()
                            if match.group('note') is None:
                                continue
                            if match.group('note') == 'z':
                                pending = False
                                continue
                            if not pending:
                                notes.append([])
                            notes[-1].append((cursor, match.start(), match.end(), match))
                            pending = bool(match.group('tie'))
                    offset += len(bar) + 1
            cursor += 1
    return notes


def _spell(pitch, match):
    """Ways to write MIDI pitch with the duration of token match, tied onward: the plain letter
    first (the key and bar decide its accidental), then with an explicit accidental."""
    out = []
    for explicit in (False, True):
        for letter, natural in NATURAL.items():
            for alter, mark in ((0, '='), (1, '^'), (-1, '_')):
                if (natural + alter - pitch) % 12:
                    continue
                octave = (pitch - alter - natural - 60) // 12
                text = letter.lower() + "'" * (octave - 1) if octave >= 1 else letter + ',' * (-octave)
                out.append((mark if explicit else '') + text + match.group('duration') + '-')
    return out


def _merge_note(abc, index):
    """The score with Vocal note index tied onto the note after it (at that note's pitch), or
    None when no spelling keeps every other note, bar, chord and the Ins voice exactly."""
    before = parse_abc(abc)
    vocal = before.voices['Vocal'].notes
    if index + 1 >= len(vocal):
        return None
    on, _, dur = vocal[index]
    _, pitch, dur2 = vocal[index + 1]
    expected = vocal[:index] + [[on, pitch, dur + dur2]] + vocal[index + 2:]
    tokens = _vocal_tokens(abc)[index]
    lines = abc.splitlines(keepends=True)
    for spelling in ([None] if vocal[index][1] == pitch else []) + list(range(len(_spell(pitch, tokens[0][3])))):
        edited = list(lines)
        for line_no, a, b, match in reversed(tokens):
            body = edited[line_no]
            if spelling is None:
                new = match.group(0) if match.group('tie') else match.group(0) + '-'
            else:
                new = _spell(pitch, match)[spelling]
            edited[line_no] = body[:a] + new + body[b:]
        text = ''.join(edited)
        try:
            after = parse_abc(text)
        except ValueError:
            continue
        if (after.voices['Vocal'].notes == expected and after.voices['Ins'].notes == before.voices['Ins'].notes
                and after.voices['Vocal'].bars == before.voices['Vocal'].bars
                and after.voices['Vocal'].chords == before.voices['Vocal'].chords
                and after.voices['Vocal'].keys == before.voices['Vocal'].keys and after.bpm == before.bpm):
            return text
    return None


def touch_up(abc, plan, groups, start, own, fitted, words, sung):
    """Fix 3 (A/B only): where a phrase has more notes before a held note than syllables sung
    before it, so that one syllable per note would put the next word on the held note (her
    chorus: two words over four quick notes before the held word), remove the extra onsets:
    1. the held word's own quick notes before the held note fold into it when they only repeat
       the held pitch (a tie: the melody is unchanged) or are a sixteenth-note slide into it
       (the slide's pitch goes);
    2. then touching same-pitch notes before the held word's first note, among the two words
       before it, are tied into one (every pitch stays; one re-attack goes).
    Never a word's own first note after its start, never another held note. Only in fitted
    sections; every change is checked with the native parser, which must find every other note,
    bar, chord and the Ins voice unchanged. Returns (new score or None, report)."""
    notes = plan['notes']
    merges, taken = [], set()
    for j, group in enumerate(groups):
        if group is None:
            continue
        phrase = plan['phrases'][j]
        for n in range(phrase['first'], phrase['last'] + 1):
            k = own.get(n)
            if notes[n]['dur'] < HELD or k is None or not fitted[k] or start[k] > n:
                continue
            place = dict(section=plan['sections'][phrase['section']]['name'], phrase=j, held_note=n - phrase['first'])
            lead = [x for x in range(group[0], k) if start[x] >= phrase['first']]
            extra = ((n - phrase['first']) - sum(words[sung[x]]['syllables'] for x in lead)
                     - max(words[sung[k]]['syllables'] - 1, 0))
            m, target = n - 1, n
            while extra > 0 and m > start[k] and own.get(m) == k and m not in taken and notes[m]['dur'] < HELD:
                touching = notes[m]['on'] + notes[m]['dur'] == notes[target]['on']
                if not touching or not (notes[m]['pitch'] == notes[n]['pitch'] or notes[m]['dur'] <= SLIDE):
                    break
                merges.append(dict(place, note=m, kind='tie' if notes[m]['pitch'] == notes[n]['pitch'] else 'slide'))
                taken.add(m)
                extra -= 1
                target, m = m, m - 1
            window = [x for x in lead[-2:] if fitted[x]]
            if extra <= 0 or not window:
                continue
            for m in range(start[k] - 2, start[window[0]] - 1, -1):  # tie note m onto note m + 1
                if extra <= 0:
                    break
                a, b = notes[m], notes[m + 1]
                if (m in taken or m + 1 in taken or a['dur'] >= HELD or b['dur'] >= HELD
                        or a['pitch'] != b['pitch'] or a['on'] + a['dur'] != b['on']):
                    continue
                merges.append(dict(place, note=m, kind='lead-in tie'))
                taken.add(m)
                extra -= 1
    report = dict(applied=False, ties=0, folds=0, lead_in_ties=0, notes_before=len(notes), notes_after=len(notes),
                  places=[])
    if not merges:
        report['reason'] = 'nothing to touch up'
        return None, report
    text = abc
    for merge in sorted(merges, key=lambda x: -x['note']):  # from the end, so earlier indices hold
        try:
            new = _merge_note(text, merge['note'])
        except (ValueError, KeyError, IndexError, TypeError):
            new = None
        if new is None:
            continue
        text = new
        report[{'tie': 'ties', 'slide': 'folds'}.get(merge['kind'], 'lead_in_ties')] += 1
        report['places'].append(dict(section=merge['section'], phrase=merge['phrase'], held_note=merge['held_note'],
                                     kind=merge['kind']))
    if text == abc:
        report['reason'] = 'no change passed the score check'
        return None, report
    report['places'].sort(key=lambda x: (x['phrase'], x['held_note']))
    report.update(applied=True, notes_after=len(parse_abc(text).voices['Vocal'].notes))
    return text, report


def touched_from(original, touched):
    """Whether score touched is score original with some Vocal notes merged into the touching
    note after them, touch_up()'s only change: the same bars, chords, keys, tempo and Ins voice,
    and every Vocal note either unchanged or a run of touching notes ending on its pitch."""
    try:
        a, b = parse_abc(original), parse_abc(touched)
    except ValueError:
        return False
    va, vb = a.voices['Vocal'], b.voices['Vocal']
    if (va.bars != vb.bars or va.chords != vb.chords or va.keys != vb.keys or a.bpm != b.bpm
            or a.voices['Ins'].notes != b.voices['Ins'].notes or len(vb.notes) >= len(va.notes)):
        return False
    i = 0
    for on, pitch, dur in vb.notes:
        total, end, last = 0, on, None
        while i < len(va.notes) and total < dur:
            if va.notes[i][0] != end:
                return False
            total, end, last = total + va.notes[i][2], va.notes[i][0] + va.notes[i][2], va.notes[i][1]
            i += 1
        if total != dur or last != pitch:
            return False
    return i == len(va.notes)


def measure(plan, order, take, speed=1.0):
    """How well a finished take kept the plan (fit()'s plan). order: her word indices in the
    order the take sang them. take: per position of order, align.py's timing or None.
    speed: how much faster than its score's written tempo the take was sung (fit_tempo); its
    pauses shrink by the same factor, so the pause test is PAUSE / speed. 1.0 otherwise.

    Held notes: the planned word must be sung longer than the word either side of it (sung
    time up to the next word, less any quiet before it). A note counts toward the score only
    when the recording itself was heard clearly there (the planned word and both neighbours,
    and the held shape: plan 'verified'); the rest are reported apart as unverified. Phrase
    starts: a pause of PAUSE seconds before each phrase's first word, checked only where the
    recording pauses. Words heard: aligner confidence."""
    timed = {i: t for i, t in zip(order, take) if t and isinstance(t.get('start'), (int, float))}
    heard = {i: t for i, t in timed.items() if (t.get('score') or 0) >= LOW_SCORE}
    spans = _voiced_after(order, timed)
    position = {i: k for k, i in enumerate(order)}
    counts = {True: [0, 0], False: [0, 0]}
    misses = []
    for note in plan['held']:
        word = note['word']
        verified = bool(note.get('verified', note.get('clear')))
        if word not in heard or word not in position:
            continue
        k = position[word]
        near = [order[m] for m in (k - 1, k + 1) if 0 <= m < len(order) and order[m] in heard]
        if not near:
            continue
        winner = max([word] + near, key=lambda i: (spans.get(i, 0.0), i == word))
        counts[verified][1] += 1
        if winner == word:
            counts[verified][0] += 1
        elif verified:
            misses.append(dict(kind='held', section=note['section'], phrase=note['phrase'], note=note['note'],
                               sung_word_offset=position[winner] - k))
    held_hits, held_of = counts[True]
    pause_hits = pause_of = 0
    pause = PAUSE / speed
    groups = plan['groups']
    for j, group in enumerate(groups):
        if not group or j == 0 or not any(groups[:j]):
            continue
        first = group[0]
        if first not in heard or not plan.get('source_pauses', {}).get(j, True):
            continue
        pause_of += 1
        if (heard[first].get('quiet_before') or 0.0) >= pause:
            pause_hits += 1
        else:
            misses.append(dict(kind='pause', section=plan['sections'][j], phrase=j))
    sung = [i for i, t in zip(order, take) if t is not None]
    parts = [(0.5, held_hits, held_of), (0.3, pause_hits, pause_of), (0.2, len(heard), len(sung))]
    weight = sum(w for w, _, of in parts if of)
    score = round(100 * sum(w * hits / of for w, hits, of in parts if of) / weight) if weight else None
    return dict(fit_score=score, held_words_on_note=dict(hits=held_hits, of=held_of),
                held_unverified=dict(hits=counts[False][0], of=counts[False][1]),
                phrase_starts_after_pause=dict(hits=pause_hits, of=pause_of),
                words_heard=dict(hits=len(heard), of=len(sung)), misses=misses[:24])


def source_pauses(plan, times):
    """Phrases whose first word follows a pause in the recording itself (where a take is
    expected to pause too). times: per her word index, align.py timing or None."""
    out = {}
    for j, group in enumerate(plan['groups']):
        if group:
            t = times[group[0]] if group[0] < len(times) else None
            out[j] = bool(t) and (t.get('quiet_before') or 0.0) >= PAUSE
    return out


def sung_order(lyrics_used, lyrics):
    """Her word indices in the order a take sang them, from the lyrics it was given (such as
    fit()'s re-broken lines): (order, positions) where positions are the matching indices into
    lyric_words(lyrics_used). Words she never wrote are left out."""
    used = [w['norm'] for w in lyric_words(lyrics_used)[0]]
    hers = [w['norm'] for w in lyric_words(lyrics)[0]]
    order, positions = [], []
    for block in difflib.SequenceMatcher(None, used, hers, autojunk=False).get_matching_blocks():
        for d in range(block.size):
            positions.append(block.a + d)
            order.append(block.b + d)
    return order, positions
