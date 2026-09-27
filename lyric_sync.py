"""Lyric-to-tune sync for YuE2 covers (Part 295 follow-up; SYNC_DESIGN.md fixes 1 and 2).

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

measure() scores a finished take against the same plan: did each held note keep its word,
and does each phrase start after a pause. It never changes a take.

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
from abc_tools import parse_abc  # noqa: E402
from instrumentalize import section_starts  # noqa: E402

REVISION = 1
PHRASE_GAP = Fraction(1, 2)   # a rest of an eighth note or longer ends a phrase
CLAUSE_HOLD = Fraction(3)     # a note of three beats or more sung straight on ends a clause: a comma
HELD = Fraction(3, 2)         # long notes whose word a take is checked for
SHORT_PHRASE = 3              # a phrase of this many notes or fewer joins the line before it
LEAD = 0.3                    # seconds of slack around a phrase's note onsets for its words
FAR = 3.0                     # a clearly heard word this many seconds away cannot sing in a phrase
TIME_WEIGHT = 4.0             # cost per second a word's onset sits outside its phrase
COUNT_WEIGHT = 0.2            # light pull toward as many rough syllables as notes
LOW_SCORE = 0.15              # aligner confidence below this is "not clearly heard"
PAUSE = 0.15                  # seconds of quiet vocal before a word that make a pause
MAX_GROUP = 40                # words in one phrase, a bound for speed
MIN_NOTE_MATCH = 0.5          # score notes that must match SheetSage2's MIDI notes
MIN_HEARD = 0.5               # sung words the aligner must hear clearly
MIN_COVERAGE = 0.5            # phrase notes that must end up with words
MAX_OFFSET = 0.75             # median seconds between a phrase's first note and its first word
TAG = re.compile(r'^\s*\[([^\]\n]+)\]\s*$')
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
CLAUSE_MARK = re.compile(r'[,;:.…]+$')
ENDS_CLAUSE = re.compile(r'[,;:.!?…]["”’)\]]*$')
REASONS = {
    'score': "the score's phrases could not be read",
    'notes': "the recording's note timing could not be read",
    'heard': 'too few of your words could be heard clearly in the recording',
    'words': 'your words did not line up with the tune',
    'empty': 'there are no words to sing',
    'align': 'your words could not be timed against the recording this time',
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

    Each word: text (as written), line, section (index), syllables (rough), brk (her text
    breaks after it: a line end or a clause mark) and norm (what the aligner reads)."""
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
            words.append(dict(text=token, line=line_no, section=len(sections) - 1,
                              syllables=sum(syllables(w) for w in WORD.findall(token)),
                              brk=k == len(tokens) - 1 or bool(ENDS_CLAUSE.search(token)),
                              norm=aligner_text(token)))
    for section in sections:
        section['words'] = sum(1 for w in words if w['section'] == section['index'])
    return words, sections


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


def _onsets(words, sung, times):
    """Onset and trust per sung word; words the aligner could not place are interpolated."""
    onset, weight = [None] * len(sung), [0.1] * len(sung)
    for k, i in enumerate(sung):
        t = times[i] if i < len(times) else None
        if t and isinstance(t.get('start'), (int, float)):
            onset[k] = float(t['start'])
            weight[k] = 1.0 if (t.get('score') or 0) >= LOW_SCORE else 0.3
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
    phrase, plus a light pull toward as many syllables as notes. None when impossible."""
    notes, phrases = plan['notes'], plan['phrases']
    P, W = len(phrases), len(sung)
    lo = [notes[p['first']]['s'] - LEAD for p in phrases]
    hi = [notes[p['last']]['s'] + LEAD for p in phrases]
    syl = [0]
    for i in sung:
        syl.append(syl[-1] + words[i]['syllables'])
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
            cost = here + COUNT_WEIGHT * mismatch(0, count)
            if cost < best[i][j + 1]:
                best[i][j + 1], back[i][j + 1] = cost, i
            timing = 0.0
            for k in range(i, min(W, i + MAX_GROUP)):
                t = onset[k]
                gap = lo[j] - t if t < lo[j] else t - hi[j] if t > hi[j] else 0.0
                if gap > FAR and weight[k] >= 1.0:
                    break
                timing += TIME_WEIGHT * weight[k] * gap
                cost = here + timing + COUNT_WEIGHT * mismatch(syl[k + 1] - syl[i], count)
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


def _owners(plan, groups, onset):
    """For each note of a phrase with words, the word (sung position) whose sung span
    [onset, next onset) overlaps the note most; ties go to the earlier word."""
    notes = plan['notes']
    ends = [onset[k + 1] if k + 1 < len(onset) else onset[k] + 2.0 for k in range(len(onset))]
    own = {}
    for j, group in enumerate(groups):
        if group is None:
            continue
        a, b = group
        phrase = plan['phrases'][j]
        for n in range(phrase['first'], phrase['last'] + 1):
            on, end = notes[n]['s'], notes[n]['e']
            overlap, pick = 0.0, None
            for k in range(a, b):
                shared = min(end, ends[k]) - max(on, onset[k])
                if shared > overlap + 1e-9:
                    overlap, pick = shared, k
            if pick is None:
                pick = a
                for k in range(a, b):
                    if onset[k] <= on + 0.04:
                        pick = k
            own[n] = pick
    return own


def _voiced_after(order, times):
    """Seconds each word is sung before the next word starts, less any quiet before that next
    word: long for a held note, short for a quick one, never counting a rest."""
    spans = {}
    for k, i in enumerate(order):
        t = times.get(i)
        if not t:
            continue
        nxt = times.get(order[k + 1]) if k + 1 < len(order) else None
        if nxt:
            spans[i] = max(0.0, nxt['start'] - t['start'] - (nxt.get('quiet_before') or 0.0))
        else:
            spans[i] = max(0.0, (t.get('end') or t['start']) - t['start'])
    return spans


def _tag(name):
    return f'[{name.title()}]' if name else None


def fit(abc, lyrics, timing, word_times, failed=None):
    """Her lyrics re-broken to the recording's phrases from measured timing.

    timing: note_times() output for this score. word_times: per word of lyric_words(lyrics),
    {'start', 'end', 'score', 'quiet_before'} or None, from align.py on the recording.
    failed: a REASONS key when the timing could not be measured at all.
    Returns dict(applied, lyrics, order, report, notes, plan). When applied is False the lyrics
    are hers unchanged and report['reason'] says why; plan is then None."""
    words, sections = lyric_words(lyrics)
    report = dict(revision=REVISION, timing_source='recording', applied=False, reason=None)
    out = dict(applied=False, lyrics=lyrics, order=list(range(len(words))), report=report, notes=[], plan=None)

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
    for note, (start, end) in zip(plan['notes'], timing['notes']):
        note['s'], note['e'] = start, max(end, start)
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
    heard = sum(1 for i in sung if i < len(times) and times[i] and (times[i].get('score') or 0) >= LOW_SCORE)
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
    own = _owners(plan, groups, onset)
    commas = set()
    for j, group in enumerate(groups):
        if group is None:
            continue
        for n in range(phrases[j]['first'], phrases[j]['last']):
            if notes[n]['dur'] >= CLAUSE_HOLD and own[n] != group[1] - 1:
                commas.add(own[n])
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
            near = [src_spans.get(sung[m]) for m in (k - 1, k + 1) if 0 <= m < len(sung)]
            mine = src_spans.get(sung[k])
            clear = mine is not None and all(s is None or mine > s for s in near)
            held.append(dict(section=plan['sections'][phrase['section']]['name'], phrase=j,
                             note=n - phrase['first'], beats=float(notes[n]['dur']),
                             seconds=round(notes[n]['e'] - notes[n]['s'], 2), word=sung[k],
                             last=n == phrase['last'], clear=clear))
    blocks, current, removed, added, merged = [], None, 0, 0, 0
    order = []
    for j, group in enumerate(groups):
        if group is None:
            continue
        a, b = group
        tokens = []
        for k in range(a, b):
            token = words[sung[k]]['text']
            order.append(sung[k])
            if k < b - 1:
                bare = CLAUSE_MARK.sub('', token)
                if bare != token and k not in commas:
                    removed += 1
                token = bare
                if k in commas:
                    if not ENDS_CLAUSE.search(words[sung[k]]['text']):
                        added += 1
                    token += ','
            if token:
                tokens.append(token)
        line = ' '.join(tokens)
        section = phrases[j]['section']
        if current is None or current['section'] != section:
            current = dict(section=section, lines=[line])
            blocks.append(current)
        elif phrases[j]['count'] <= SHORT_PHRASE:
            previous = current['lines'][-1]
            if not ENDS_CLAUSE.search(previous):
                previous += ','
                added += 1
            current['lines'][-1] = previous + ' ' + line
            merged += 1
        else:
            current['lines'].append(line)
    text = '\n\n'.join('\n'.join(([_tag(plan['sections'][b['section']]['name'])]
                                  if plan['sections'][b['section']]['name'] else []) + b['lines']) for b in blocks)
    with_words = sum(1 for g in groups if g)
    last_of_group = {g[1] - 1 for g in groups if g}
    her_breaks = [k for k, i in enumerate(sung) if words[i]['brk'] and k < len(sung) - 1]
    report.update(
        applied=True, phrases=len(phrases), phrases_with_words=with_words,
        lines=sum(len(b['lines']) for b in blocks), short_phrases_joined=merged,
        commas_added=added, commas_removed=removed,
        her_breaks=len(her_breaks), her_breaks_inside_phrases=sum(1 for k in her_breaks if k not in last_of_group),
        breaks_on_phrase_ends=dict(before=sum(1 for g in groups if g and words[sung[g[1] - 1]]['brk']),
                                   after=with_words, of=with_words),
        held_notes=held)
    kept = dict(groups=[[sung[k] for k in range(g[0], g[1])] if g else None for g in groups],
                sections=[plan['sections'][p['section']]['name'] for p in phrases], held=held)
    kept['source_pauses'] = source_pauses(kept, times)
    out.update(applied=True, lyrics=text, order=order, plan=kept)
    lines = report['lines']
    out['notes'].append(f"Your line breaks now follow the tune's phrases: {lines} line{'s' if lines != 1 else ''}"
                        f", your words and their order unchanged.")
    for gone in report['words_without_tune']:
        what = f"The {gone['section']} words" if gone['section'] else 'Some words'
        count = gone['lines']
        out['notes'].append(f"{what} ({count} line{'s' if count != 1 else ''}) have no sung tune in the recording, "
                            'so this take leaves them out.')
    return out


def measure(plan, order, take):
    """How well a finished take kept the plan (fit()'s plan). order: her word indices in the
    order the take sang them. take: per position of order, align.py's timing or None.

    Held notes: the planned word must be sung longer than the word either side of it (sung
    time up to the next word, less any quiet before it), checked only where the recording
    itself shows that shape. Phrase starts: a pause of PAUSE seconds before each phrase's first
    word, checked only where the recording pauses. Words heard: aligner confidence."""
    timed = {i: t for i, t in zip(order, take) if t and isinstance(t.get('start'), (int, float))}
    heard = {i: t for i, t in timed.items() if (t.get('score') or 0) >= LOW_SCORE}
    spans = _voiced_after(order, timed)
    position = {i: k for k, i in enumerate(order)}
    held_hits = held_of = pause_hits = pause_of = 0
    misses = []
    for note in plan['held']:
        word = note['word']
        if not note['clear'] or word not in heard or word not in position:
            continue
        k = position[word]
        near = [order[m] for m in (k - 1, k + 1) if 0 <= m < len(order) and order[m] in heard]
        if not near:
            continue
        winner = max([word] + near, key=lambda i: (spans.get(i, 0.0), i == word))
        held_of += 1
        if winner == word:
            held_hits += 1
        else:
            misses.append(dict(kind='held', section=note['section'], phrase=note['phrase'], note=note['note'],
                               sung_word_offset=position[winner] - k))
    groups = plan['groups']
    for j, group in enumerate(groups):
        if not group or j == 0 or not any(groups[:j]):
            continue
        first = group[0]
        if first not in heard or not plan.get('source_pauses', {}).get(j, True):
            continue
        pause_of += 1
        if (heard[first].get('quiet_before') or 0.0) >= PAUSE:
            pause_hits += 1
        else:
            misses.append(dict(kind='pause', section=plan['sections'][j], phrase=j))
    sung = [i for i, t in zip(order, take) if t is not None]
    parts = [(0.5, held_hits, held_of), (0.3, pause_hits, pause_of), (0.2, len(heard), len(sung))]
    weight = sum(w for w, _, of in parts if of)
    score = round(100 * sum(w * hits / of for w, hits, of in parts if of) / weight) if weight else None
    return dict(fit_score=score, held_words_on_note=dict(hits=held_hits, of=held_of),
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
