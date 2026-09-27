"""CPU tests for lyric_sync (Part 295 follow-up). Every lyric here is invented for these tests."""
import json
import unittest

import lyric_sync as S
import yue_handler as Y

# An invented native two-voice score at 60 BPM (one quarter note = one second of score time).
# The intro is instrumental apart from one pickup note that leads into the verse. Phrases:
# verse 6 notes (with the pickup) and 5; chorus 6 notes with a four-beat note sung straight on
# as its third note, then a three-note phrase, then 5 notes. The outro has no tune.
SCORE = '''X:1
T:
M:4/4
L:1/8
Q:1/4=60
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
% intro
V: Vocal
z8|z6z1G1|
V: Ins
C8|C8|
% verse
V: Vocal
c2c2d2e2|f4z4|e2d2c2d2|e4z4|
V: Ins
Z4|
% chorus
V: Vocal
z4g2a2|b8|a2g2e2z2|c2d2e2z2|
V: Ins
Z4|
V: Vocal
g2g2a2b2|c'4z4|
V: Ins
Z2|
% outro
V: Vocal
Z|
V: Ins
C8|
'''
COUNTS = [6, 5, 6, 3, 5]
# Her words, broken in the wrong places and without the clause comma, plus an intro line
# where the recording has no sung tune.
MISBROKEN = '''[Intro]
Oh oh here we go

[Verse 1]
Grey clouds roll past
the town cold wind at
the door

[Chorus]
Sing it loud let
it ring all night long we
dance till the dawn
'''
FITTED = '''[Verse]
Grey clouds roll past the town
cold wind at the door

[Chorus]
Sing it loud, let it ring, all night long
we dance till the dawn'''
# Which score note each sung word starts on (verse and chorus words, in order).
STARTS = list(range(6)) + list(range(6, 11)) + list(range(11, 25))


def real(q):
    """The recording runs 1% slower than the score's tempo and starts 0.2 s later."""
    return 0.2 + 1.01 * float(q)


def word_times(plan, lyrics, starts, intro=5, late=0.05, score=0.8):
    """What align.py would report: onsets just after their notes, quiet before phrase starts."""
    words, _ = S.lyric_words(lyrics)
    notes = plan['notes']
    times = [dict(start=1.0 + 0.5 * k, end=1.3 + 0.5 * k, score=0.4, quiet_before=0.5) for k in range(intro)]
    firsts = {p['first'] for p in plan['phrases']}
    for k, n in enumerate(starts):
        n, extra = n if isinstance(n, tuple) else (n, 0.0)
        on = real(notes[n]['on']) + late + extra
        quiet = 0.0
        if n in firsts and n > 0:
            quiet = real(notes[n]['on']) - real(notes[n - 1]['on'] + notes[n - 1]['dur']) - 0.1
        times.append(dict(start=round(on, 3), end=round(on + 0.3, 3), score=score, quiet_before=round(max(quiet, 0.0), 3)))
    assert len(times) == len(words)
    return times


def vlq(n):
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def smf(parts, division=480, tempo=500000):
    """A format 1 Standard MIDI File: a tempo track, then one track per [(onset_s, pitch, dur_s)]."""
    per_second = division * 1e6 / tempo

    def chunk(events):
        body, last = b'', 0
        for tick, data in sorted(events, key=lambda e: (e[0], e[1][0] == 0x90)):
            body += vlq(tick - last) + data
            last = tick
        body += b'\x00\xff\x2f\x00'
        return b'MTrk' + len(body).to_bytes(4, 'big') + body
    tracks = [chunk([(0, b'\xff\x51\x03' + tempo.to_bytes(3, 'big'))])]
    for channel, notes in enumerate(parts):
        events = []
        for onset, pitch, duration in notes:
            events.append((round(onset * per_second), bytes([0x90 | channel, pitch, 90])))
            events.append((round((onset + duration) * per_second), bytes([0x80 | channel, pitch, 0])))
        tracks.append(chunk(events))
    head = b'MThd' + (6).to_bytes(4, 'big') + (1).to_bytes(2, 'big') + len(tracks).to_bytes(2, 'big') + division.to_bytes(2, 'big')
    return head + b''.join(tracks)


def recording_midi(plan):
    """SheetSage2-like output: the vocal melody in real time, missing its pickup note and with
    one extra note, beside an instrumental part."""
    vocal = [(real(n['on']), n['pitch'], 1.01 * float(n['dur']) * 0.95) for n in plan['notes'][1:]]
    vocal.insert(9, (real(20.5), 50, 0.2))
    vocal.sort()
    band = [(real(q), 36 + (q % 5), 0.9) for q in range(0, 48, 2)]
    return smf([vocal, band])


def sung_words(text, skip=None):
    out, tag = [], None
    for line in text.splitlines():
        m = Y.TAG.match(line)
        if m:
            tag = S.section_name(m.group(1))
        elif tag != skip:
            out += Y.WORD.findall(line)
    return out


class ScorePlanTest(unittest.TestCase):
    def test_phrases_and_sections(self):
        plan = S.score_plan(SCORE)
        self.assertEqual([p['count'] for p in plan['phrases']], COUNTS)
        names = [plan['sections'][p['section']]['name'] for p in plan['phrases']]
        self.assertEqual(names, ['verse', 'verse', 'chorus', 'chorus', 'chorus'])
        self.assertEqual([(s['name'], s['phrases'], s['notes']) for s in plan['sections']],
                         [('intro', 0, 1), ('verse', 2, 10), ('chorus', 3, 14), ('outro', 0, 0)])
        with self.assertRaises(ValueError):
            S.score_plan('not a score')

    def test_helpers_match_the_worker(self):
        for word in ('river', 'dance', 'little', 'free', 'eye', "don't", 'rhythm'):
            self.assertEqual(S.syllables(word), Y.syllables(word))
        for tag in ('Verse 2', 'Pre Chorus', 'chorus: x2', None):
            self.assertEqual(S.section_name(tag), Y.section_name(tag))
        self.assertEqual(S.TAG.pattern, Y.TAG.pattern)
        self.assertEqual(S.WORD.pattern, Y.WORD.pattern)

    def test_aligner_text(self):
        self.assertEqual(S.aligner_text('Don’t,'), "don't")
        self.assertEqual(S.aligner_text('Café!'), 'cafe')
        self.assertEqual(S.aligner_text('(ooh)'), 'ooh')
        self.assertEqual(S.aligner_text('10,000'), '')
        self.assertEqual(S.aligner_text("'til"), 'til')

    def test_lyric_words_keep_every_token(self):
        words, sections = S.lyric_words('[Verse]\nOne, two 3 four\n\nfive six')
        self.assertEqual([w['text'] for w in words], ['One,', 'two', '3', 'four', 'five', 'six'])
        self.assertEqual([w['brk'] for w in words], [True, False, False, True, False, True])
        self.assertEqual(words[2]['syllables'], 0)
        self.assertEqual(sections, [dict(index=0, tag='Verse', name='verse', words=6)])


class MidiTest(unittest.TestCase):
    def test_notes_round_trip(self):
        notes = [(0.5, 60, 1.0), (1.5, 62, 0.25), (2.0, 64, 2.0)]
        parts = S.midi_notes(smf([notes], tempo=400000))
        self.assertEqual(list(parts), [(1, 0)])
        for got, want in zip(parts[(1, 0)], notes):
            self.assertAlmostEqual(got[0], want[0], places=2)
            self.assertEqual(got[1], want[1])
            self.assertAlmostEqual(got[2], want[2], places=2)

    def test_running_status_and_zero_velocity_offs(self):
        track = (b'\x00\x90\x3c\x40' b'\x83\x60\x3c\x00' b'\x00\x3e\x40' b'\x83\x60\x3e\x00' b'\x00\xff\x2f\x00')
        data = (b'MThd' + (6).to_bytes(4, 'big') + b'\x00\x00\x00\x01\x01\xe0'
                + b'MTrk' + len(track).to_bytes(4, 'big') + track)
        notes = S.midi_notes(data)[(0, 0)]
        self.assertEqual([(round(a, 3), p, round(d, 3)) for a, p, d in notes], [(0.0, 60, 0.5), (0.5, 62, 0.5)])
        with self.assertRaises(ValueError):
            S.midi_notes(b'RIFF0000')

    def test_the_raw_vocal_melody_wins_over_grid_notation(self):
        plan = S.score_plan(SCORE)
        grid = smf([[(float(n['on']), n['pitch'], float(n['dur'])) for n in plan['notes']]])
        timing = S.note_times(plan, [('notation/song_melody.mid', grid), ('melody_vocal.mid', recording_midi(plan))])
        self.assertEqual(timing['source'], 'melody_vocal.mid#1/0')
        self.assertEqual(S.note_times(plan, [('notation/song_melody.mid', grid)])['source'], 'notation/song_melody.mid#1/0')

    def test_note_times_follow_the_recording(self):
        plan = S.score_plan(SCORE)
        timing = S.note_times(plan, [('melody_vocal.mid', recording_midi(plan)), ('broken.mid', b'nonsense')])
        self.assertEqual(timing['source'], 'melody_vocal.mid#1/0')
        self.assertGreaterEqual(timing['matched'], 0.9)
        for note, (start, end) in zip(plan['notes'], timing['notes']):
            self.assertAlmostEqual(start, real(note['on']), delta=0.02)
            self.assertAlmostEqual(end, real(note['on'] + note['dur']), delta=0.02)

    def test_unmatched_midi_is_refused(self):
        plan = S.score_plan(SCORE)
        noise = smf([[(k * 0.5, 30 + (k * 7) % 11, 0.2) for k in range(40)]])
        with self.assertRaisesRegex(ValueError, 'note timing'):
            S.note_times(plan, [('x.mid', noise)])
        with self.assertRaisesRegex(ValueError, 'note timing'):
            S.note_times(plan, [])


class FitTest(unittest.TestCase):
    def setUp(self):
        self.plan = S.score_plan(SCORE)
        self.timing = S.note_times(self.plan, [('melody_vocal.mid', recording_midi(self.plan))])

    def test_misbroken_lines_follow_the_tune(self):
        result = S.fit(SCORE, MISBROKEN, self.timing, word_times(self.plan, MISBROKEN, STARTS))
        self.assertTrue(result['applied'])
        self.assertEqual(result['lyrics'], FITTED)
        self.assertEqual(sung_words(result['lyrics']), sung_words(MISBROKEN, skip='intro'))
        self.assertEqual(result['order'], list(range(5, 30)))
        report = result['report']
        self.assertEqual(report['words_without_tune'], [dict(section='intro', words=5, lines=1)])
        self.assertEqual((report['phrases'], report['phrases_with_words'], report['lines']), (5, 5, 4))
        self.assertEqual(report['short_phrases_joined'], 1)
        self.assertEqual(report['commas_added'], 2)
        self.assertEqual(report['breaks_on_phrase_ends'], dict(before=2, after=5, of=5))
        self.assertEqual(report['her_breaks_inside_phrases'], 4)
        self.assertLess(report['onset_offset_s']['median'], 0.1)
        held = [(h['phrase'], h['note'], h['word'], h['last'], h['clear']) for h in report['held_notes']]
        self.assertIn((2, 2, 18, False, True), held)          # the four-beat note keeps its word
        self.assertEqual(result['notes'], ["Your line breaks now follow the tune's phrases: your words keep their "
                                           'order; 1 intro line was left out, because the recording has no sung '
                                           'tune there.'])
        self.assertEqual([(s['section'], s['fitted']) for s in report['sections']], [('verse', True), ('chorus', True)])
        self.assertEqual(report['repeats'], [])
        self.assertEqual(result['line_ends'], [k in (5, 10, 19, 24) for k in range(25)])

    def test_reports_carry_no_lyric_text(self):
        result = S.fit(SCORE, MISBROKEN, self.timing, word_times(self.plan, MISBROKEN, STARTS))
        dumped = json.dumps([result['report'], result['notes'], result['plan']]).lower()
        for word in set(w.lower() for w in Y.WORD.findall(MISBROKEN)):
            if len(word) >= 4 and word not in ('intro', 'verse', 'chorus'):
                self.assertNotRegex(dumped, r'\b' + word + r'\b')

    def test_timing_decides_which_word_holds_the_note(self):
        # The singer stretched "Sing" over two notes, so "it" lands on the four-beat note.
        starts = STARTS[:11] + [11, 13, 14, 15, 16, (16, 0.2)] + list(range(17, 25))
        result = S.fit(SCORE, MISBROKEN, self.timing, word_times(self.plan, MISBROKEN, starts))
        self.assertTrue(result['applied'])
        self.assertIn('Sing it, loud let it ring', result['lyrics'])

    def test_timing_overrides_misleading_syllable_counts(self):
        # "town" is a two-note melisma: by counts alone "cold" would join the first phrase.
        lyrics = ('[Verse]\nGrey clouds roll town cold\nwind at the door\n'
                  '[Chorus]\nSing it loud let it ring all night long we dance till the dawn\n')
        starts = [0, 1, 2, 4, 6, 7, 8, 9, 10] + list(range(11, 25))
        words, _ = S.lyric_words(lyrics)
        counted = [w['syllables'] for w in words[:9]]
        self.assertLess(S.mismatch(sum(counted[:5]), 6) + S.mismatch(sum(counted[5:]), 5),
                        S.mismatch(sum(counted[:4]), 6) + S.mismatch(sum(counted[4:]), 5))
        result = S.fit(SCORE, lyrics, self.timing, word_times(self.plan, lyrics, starts, intro=0))
        self.assertTrue(result['lyrics'].startswith('[Verse]\nGrey clouds roll town\ncold wind at the door\n'))

    def test_failures_keep_her_lines(self):
        times = word_times(self.plan, MISBROKEN, STARTS)
        cases = [(S.fit(SCORE, MISBROKEN, None, times), 'notes'),
                 (S.fit(SCORE, MISBROKEN, self.timing, None), 'heard'),
                 (S.fit(SCORE, MISBROKEN, self.timing, [dict(t, score=0.01) for t in times]), 'heard'),
                 (S.fit('not a score', MISBROKEN, self.timing, times), 'score'),
                 (S.fit(SCORE, '[Verse]\n', self.timing, []), 'empty'),
                 (S.fit(SCORE, MISBROKEN, dict(self.timing, notes=self.timing['notes'][:-1]), times), 'notes'),
                 (S.fit(SCORE, MISBROKEN, dict(self.timing, notes=[[a + 1.2, b + 1.2] for a, b in self.timing['notes']]),
                        times), 'notes')]
        for result, reason in cases:
            with self.subTest(reason=reason):
                self.assertFalse(result['applied'])
                self.assertEqual(result['lyrics'], result['lyrics'] if reason == 'empty' else MISBROKEN)
                self.assertEqual(result['report']['reason'], reason)
                self.assertIsNone(result['plan'])
                self.assertIn('kept as written', result['notes'][0])

    def test_words_far_from_the_tune_are_placed_by_neighbours(self):
        times = word_times(self.plan, MISBROKEN, STARTS)
        times[8] = dict(times[8], start=60.0, end=60.3)   # one word mis-heard a minute away
        result = S.fit(SCORE, MISBROKEN, self.timing, times)
        self.assertTrue(result['applied'])
        self.assertEqual(result['report']['words']['far_from_notes'], 1)

    def test_a_last_line_under_outro_sung_in_the_chorus_is_kept(self):
        # The score's outro has no tune, but these words are heard on the chorus's last phrase.
        lyrics = MISBROKEN.replace('it ring all night long we\ndance till the dawn\n',
                                   'it ring all night long\n\n[Outro]\nwe dance till the dawn\n')
        result = S.fit(SCORE, lyrics, self.timing, word_times(self.plan, lyrics, STARTS))
        self.assertEqual(result['lyrics'], FITTED)
        self.assertEqual(result['report']['sections_sung_elsewhere'], 1)
        self.assertEqual(result['report']['words_without_tune'], [dict(section='intro', words=5, lines=1)])
        # Intro words heard clearly on the verse's tune are kept too; unheard ones are not.
        times = word_times(self.plan, lyrics, STARTS)
        on_tune = [dict(t, start=times[5]['start'] - 0.2 + 0.01 * k, score=0.6) for k, t in enumerate(times[:5])]
        plan = S.score_plan(SCORE)
        for note, (start, end) in zip(plan['notes'], self.timing['notes']):
            note['s'], note['e'] = start, end
        words, sections = S.lyric_words(lyrics)
        self.assertEqual(S.sections_without_tune(sections, plan, words, on_tune + times[5:]), (set(), {0, 3}))
        quiet = [dict(t, score=0.05) for t in on_tune]
        self.assertEqual(S.sections_without_tune(sections, plan, words, quiet + times[5:]), ({0}, {3}))
        self.assertEqual(S.sections_without_tune(sections, plan), ({0, 3}, set()))

    def test_untagged_lyrics_still_fit(self):
        body = '\n'.join(line for line in MISBROKEN.splitlines()[3:] if not line.startswith('['))
        times = word_times(self.plan, body, STARTS, intro=0)
        result = S.fit(SCORE, body, self.timing, times)
        self.assertEqual(result['lyrics'], FITTED)


class MeasureTest(unittest.TestCase):
    def setUp(self):
        plan = S.score_plan(SCORE)
        timing = S.note_times(plan, [('melody_vocal.mid', recording_midi(plan))])
        self.result = S.fit(SCORE, MISBROKEN, timing, word_times(plan, MISBROKEN, STARTS))
        self.plan = self.result['plan']
        self.source = word_times(plan, MISBROKEN, STARTS)

    def take(self, shift_hold=False, no_pauses=False):
        """A take 3% faster than the recording, as renders are."""
        times = []
        for i in self.result['order']:
            t = dict(self.source[i])
            t['start'], t['end'] = round(t['start'] * 0.97, 3), round(t['end'] * 0.97, 3)
            if no_pauses:
                t['quiet_before'] = 0.0
            times.append(t)
        if shift_hold:   # the held note slides from "loud" onto "let"
            k = self.result['order'].index(18)
            times[k + 1]['start'] = round(times[k]['start'] + 0.4, 3)
            times[k + 2]['start'] = round(times[k + 1]['start'] + 3.8, 3)
            times[k + 3]['start'] = round(times[k + 2]['start'] + 0.4, 3)
        return times

    def test_a_faithful_take_scores_full_marks(self):
        m = S.measure(self.plan, self.result['order'], self.take())
        self.assertEqual(m['fit_score'], 100)
        self.assertEqual(m['held_words_on_note']['hits'], m['held_words_on_note']['of'])
        self.assertGreater(m['held_words_on_note']['of'], 0)
        self.assertEqual(m['phrase_starts_after_pause'], dict(hits=4, of=4))
        self.assertEqual(m['misses'], [])

    def test_a_slipped_hold_is_named_by_position(self):
        m = S.measure(self.plan, self.result['order'], self.take(shift_hold=True))
        self.assertLess(m['fit_score'], 100)
        slip = [x for x in m['misses'] if x['kind'] == 'held']
        self.assertEqual(slip[0], dict(kind='held', section='chorus', phrase=2, note=2, sung_word_offset=1))

    def test_missing_pauses_and_unheard_words_lower_the_score(self):
        take = self.take(no_pauses=True)
        m = S.measure(self.plan, self.result['order'], take)
        self.assertEqual(m['phrase_starts_after_pause']['hits'], 0)
        take[0] = None
        m2 = S.measure(self.plan, self.result['order'], take)
        self.assertEqual(m2['words_heard']['of'], len(take) - 1)
        self.assertIsNone(S.measure(dict(groups=[], held=[], sections=[]), [], [])['fit_score'])


# The booth's request shape today (yue.ts yueInput for a sung cover of a recording).
COVER = {'style': 'Warm folk ballad', 'title': 'Test', 'count': 2, 'weirdness': 50, 'steps': 32, 'guidance': 1,
         'lyrics': MISBROKEN.strip(), 'band': None, 'reference_voice_url': 'https://example.invalid/original.mp3',
         'cot': 'full', 'seed': 7, 'keep_harmony': True, 'match_score_tempo': True, 'length_guard': True}
TAKE = 'yue2/3f2b8c1e-0a4d-4e8b-9c7a-1b2c3d4e5f60/master.wav'


class NoSuchKey(Exception):
    pass


class Bucket:
    """Just enough of boto3's S3 client for the caches."""

    def __init__(self, objects=None):
        self.objects, self.puts = dict(objects or {}), []

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise NoSuchKey(Key)

        class Body:
            def __init__(self, data):
                self.data = data

            def read(self):
                return self.data
        return {'Body': Body(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, ContentType):
        self.objects[Key] = Body
        self.puts.append(Key)


class HandlerSyncTest(unittest.TestCase):
    def setUp(self):
        self.plan = S.score_plan(SCORE)
        self.timing = S.note_times(self.plan, [('melody_vocal.mid', recording_midi(self.plan))])
        self.words = word_times(self.plan, MISBROKEN, STARTS)

    def test_todays_requests_ask_for_no_sync(self):
        off = dict(fit_lyrics=False, measure_fit=False, fit_score_touchup=False)
        for shape in ({}, COVER, {'style': 'pop', 'lyrics': 'la', 'cot': 'off'}, {'fit_lyrics': False, 'measure_fit': False},
                      {'fit_score_touchup': False}):
            self.assertEqual(Y.sync_options(shape), off)
        self.assertEqual(Y.sync_options({'fit_lyrics': 'timing', 'measure_fit': True}),
                         dict(fit_lyrics=True, measure_fit=True, fit_score_touchup=False))
        self.assertEqual(Y.sync_options({'fit_lyrics': 'timing', 'fit_score_touchup': True})['fit_score_touchup'], True)
        self.assertTrue({'lyric-sync', 'fit-score', 'score-touchup', 'lyric-fit-v2', 'meter-check'} <= set(Y.FEATURES))
        self.assertEqual(Y.cover_options({'mode': 'measure'})['mode'], 'measure')

    def test_bad_values_are_refused_before_any_gpu_work(self):
        for key, value in (('fit_lyrics', 'yes'), ('fit_lyrics', True), ('fit_lyrics', 0), ('fit_lyrics', 'counts'),
                           ('measure_fit', 'true'), ('measure_fit', 1), ('fit_score_touchup', 'yes'),
                           ('fit_score_touchup', True)):
            with self.subTest(key=key, value=value):
                with self.assertRaisesRegex(ValueError, key):
                    Y.sync_options({key: value})
                answer = Y.handler({'input': dict(COVER, **{key: value})})
                self.assertIn(key, answer['error'])

    def test_sync_needs_a_sung_cover_of_a_recording(self):
        sync = Y.sync_options({'fit_lyrics': 'timing'})
        cover = Y.request_input(COVER)
        with self.assertRaisesRegex(ValueError, 'instrumental'):
            Y.check_sync(cover, Y.cover_options(dict(COVER, instrumental=True)), sync, COVER['reference_voice_url'])
        score = Y.request_input({'style': 'pop', 'lyrics': 'la la', 'abc': SCORE})
        with self.assertRaisesRegex(ValueError, 'source recording'):
            Y.check_sync(score, Y.cover_options({}), sync, None)
        with self.assertRaisesRegex(ValueError, 'words'):
            Y.check_sync(Y.request_input(dict(COVER, lyrics='[Verse]')), Y.cover_options(COVER), sync, 'https://x')
        Y.check_sync(cover, Y.cover_options(COVER), sync, COVER['reference_voice_url'])
        Y.check_sync(score, Y.cover_options({}), Y.sync_options({}), None)

    def render(self, **sync):
        inp = Y.request_input(COVER)
        state = dict(timing=self.timing, words=self.words, error=None, fit=False, measure=False)
        state.update(sync)
        return inp, Y.fixed_score_render(inp, Y.cover_options(COVER), SCORE, 'recording', 50.0, sync=state)

    def test_fitted_lyrics_reach_the_render(self):
        inp, (render, budget, facts, extras, notes) = self.render(fit=True)
        self.assertEqual(render['lyrics'], FITTED)
        self.assertEqual({k: v for k, v in render.items() if k != 'lyrics'},
                         {k: v for k, v in Y.fixed_score_render(inp, Y.cover_options(COVER), SCORE, 'recording', 50.0)[0].items()
                          if k != 'lyrics'})
        self.assertTrue(extras['lyric_sync']['lyrics_fitted'])
        self.assertTrue(any('follow the tune' in note for note in notes))
        self.assertTrue(extras['lyric_fit']['same_order'])       # the fitted tags follow the score
        self.assertEqual(extras['sync_order'], list(range(5, 30)))
        json.dumps(extras['lyric_sync'])

    def test_measure_only_keeps_her_lyrics(self):
        inp, (render, _, _, extras, notes) = self.render(measure=True)
        self.assertEqual(render['lyrics'], inp['lyrics'])
        self.assertFalse(extras['lyric_sync']['lyrics_fitted'])
        self.assertEqual(extras['sync_order'], list(range(30)))
        self.assertEqual(len(extras['sync_words']), 30)
        self.assertEqual(notes, [Y.NO_CHORDS['recording']])

    def test_unmeasured_timing_renders_her_lines(self):
        inp, (render, _, _, extras, notes) = self.render(fit=True, error='align', words=None)
        self.assertEqual(render['lyrics'], inp['lyrics'])
        self.assertEqual(extras['lyric_sync']['reason'], 'align')
        self.assertIsNone(extras['sync_plan'])
        self.assertIn('kept as written', notes[-1])
        _, (_, _, _, _, notes) = self.render(measure=True, error='notes', timing=None)
        self.assertIn('could not be scored', notes[-1])

    def test_measure_request_checks(self):
        good = {'mode': 'measure', 'reference_voice_url': 'https://x/a.mp3', 'take_key': TAKE, 'lyrics': MISBROKEN}
        self.assertEqual(Y.measure_request(good), dict(reference='https://x/a.mp3', take=TAKE, lyrics=MISBROKEN.strip(),
                                                       lyrics_used=None, cover_mode=None))
        self.assertEqual(Y.measure_request(dict(good, lyrics_used=FITTED + '\n', cover_mode='harmony'))['lyrics_used'],
                         FITTED)
        for bad in ({'take_key': '../yue2/x/master.wav'}, {'take_key': 'yue2-scores/a/full.abc'},
                    {'take_key': TAKE.replace('.wav', '.flac')}, {'lyrics': '[Verse]'}, {'lyrics': None},
                    {'reference_voice_url': ''}, {'lyrics_used': '[Chorus]'}, {'lyrics_used': 7},
                    {'cover_mode': 'full'}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    Y.measure_request(dict(good, **bad))
                self.assertIn('error', Y.handler({'input': dict(good, **bad)}))

    def test_note_timing_is_read_from_sheetsage_midi_and_cached_per_score(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / 'notation').mkdir()
            (Path(td) / 'notation' / 'song_melody.mid').write_bytes(recording_midi(self.plan))
            (Path(td) / 'score.abc').write_text(SCORE)
            timing = Y.midi_timing(td, SCORE)
            self.assertIsNone(Y.midi_timing(Path(td) / 'notation', 'not a score'))
        self.assertEqual(timing['source'], 'notation/song_melody.mid#1/0')
        bucket = Bucket()
        Y.cache_timing(bucket, 'b', 'yue2-scores/abc/full-80af707174fc', timing)
        self.assertEqual(bucket.puts, ['yue2-scores/abc/full-80af707174fc.timing.json'])
        self.assertEqual(Y.cached_timing(bucket, 'b', 'yue2-scores/abc/full-80af707174fc', SCORE), timing)
        self.assertIsNone(Y.cached_timing(bucket, 'b', 'yue2-scores/abc/full-80af707174fc', SCORE + '\n'))
        self.assertIsNone(Y.cached_timing(bucket, 'b', 'yue2-scores/other/full-80af707174fc', SCORE))

    def test_word_timing_is_measured_once_per_recording_and_lyrics(self):
        calls = []

        def aligned(path, words, td, tag, stars=None):
            calls.append((path, len(words), tag))
            self.assertEqual(stars, S.star_after(S.lyric_words(COVER['lyrics'])[0]))
            return dict(words=self.words, timing={'align_s': 1.0}, peak_gib=2.0, device='cuda')
        original = Y.align_audio
        Y.align_audio = aligned
        try:
            bucket = Bucket()
            read = dict(note_times=self.timing, key='yue2-scores/abc/full-80af707174fc', abc=SCORE,
                        source='/tmp/source', decoded='/tmp/source.wav')
            first = Y.source_sync(read, COVER['lyrics'], '/tmp', bucket, 'b', 'full')
            second = Y.source_sync(read, COVER['lyrics'], '/tmp', bucket, 'b', 'full')
        finally:
            Y.align_audio = original
        self.assertEqual(calls, [('/tmp/source', 30, 'source')])
        self.assertEqual((first['error'], first['words_cached'], second['words_cached']), (None, False, True))
        self.assertEqual(second['words'], self.words)
        self.assertRegex(bucket.puts[0], r'^yue2-scores/abc/words-[0-9a-f]{16}-mmsfa-star-hdemucs-2\.json$')
        stored = bucket.objects[bucket.puts[0]].decode('utf-8').lower()
        for word in set(w.lower() for w in Y.WORD.findall(MISBROKEN)):
            if len(word) >= 4:
                self.assertNotRegex(stored, r'\b' + word + r'\b')

    def test_failed_measurements_name_their_reason(self):
        def broken(*args, **kwargs):
            raise Y.SyncError('fixture')

        def no_reread(*args):
            raise Y.subprocess.SubprocessError('fixture')
        saved = Y.align_audio, Y.run_sheetsage
        Y.align_audio, Y.run_sheetsage = broken, no_reread
        try:
            read = dict(note_times=self.timing, key=None, abc=SCORE, source='s', decoded='d')
            self.assertEqual(Y.source_sync(read, COVER['lyrics'], '/tmp', Bucket(), 'b', 'full')['error'], 'align')
            read['note_times'] = None
            self.assertEqual(Y.source_sync(read, COVER['lyrics'], '/tmp', Bucket(), 'b', 'full')['error'], 'notes')
        finally:
            Y.align_audio, Y.run_sheetsage = saved


# ---- Round 2 (after Kade's ear): the review's fixes, each on invented lyrics. ----

def with_times(plan, timing):
    for note, (start, end) in zip(plan['notes'], timing['notes']):
        note['s'], note['e'] = start, end
    return plan


def fake_notes(spec):
    """Notes for _snap: [(onset beats, duration beats, pitch)] at 60 BPM (one beat = one second)."""
    from fractions import Fraction
    return [dict(on=Fraction(on), dur=Fraction(dur), pitch=pitch, s=float(on), e=float(on + dur)) for on, dur, pitch in spec]


class SnapTest(unittest.TestCase):
    NOTES = [(0, 1, 60), (1, 4, 62), (5, 0.5, 64), (5.5, 0.5, 65)]

    def test_a_word_heard_inside_a_held_note_starts_on_the_next_note(self):
        notes = fake_notes(self.NOTES)
        # The third word is heard well inside the held note: the held note stays with the second word.
        self.assertEqual(S._snap(notes, 0, 3, [0.0, 1.0, 4.6, 5.5], [1, 1, 1, 1], [1, 1, 1, 1]), [0, 1, 2, 3])
        # Heard far from any later note, it may re-sing the held pitch inside it (a same-pitch repeat
        # SheetSage2 wrote as one note), but the second word still owns the held note's onset.
        starts = S._snap(notes, 0, 3, [0.0, 1.0, 2.0, 5.5], [1, 1, 1, 1], [1, 1, 1, 1])
        self.assertEqual(starts[:2], [0, 1])
        self.assertEqual(starts.index(1), 1)
        # Heard 0.1 s into a note is heard on it.
        self.assertEqual(S._snap(notes, 0, 3, [0.0, 1.1, 5.0, 5.6], [1, 1, 1, 1], [1, 1, 1, 1]), [0, 1, 2, 3])
        # A word heard after a note ended never takes that note either (the round-1 slip), nor shares the
        # start of the word before it: a word, a slide, a held note, then a word heard a second inside it.
        slide = fake_notes([(0, 1, 60), (1, 0.25, 64), (1.25, 4, 62), (5.25, 0.5, 65)])
        self.assertEqual(S._snap(slide, 0, 3, [0.0, 2.25, 5.3], [1, 1, 1], [1, 1, 1]), [0, 3, 3])

    def test_a_same_pitch_repeat_may_take_a_late_heard_word(self):
        repeat = fake_notes([(0, 1, 60), (1, 1, 60), (2, 1, 64)])
        other = fake_notes([(0, 1, 60), (1, 1, 62), (2, 1, 64)])
        self.assertEqual(S._snap(repeat, 0, 2, [0.0, 1.3], [1, 1], [1, 1]), [0, 1])
        self.assertEqual(S._snap(other, 0, 2, [0.0, 1.3], [1, 1], [1, 1]), [0, 2])

    def test_more_words_than_notes_share_a_note_instead_of_failing(self):
        notes = fake_notes([(0, 1, 60), (1, 1, 62)])
        starts = S._snap(notes, 0, 1, [0.0, 0.5, 1.0], [1, 1, 1], [1, 1, 1])
        self.assertEqual(len(starts), 3)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts, sorted(starts))


class FitRoundTwoTest(unittest.TestCase):
    def setUp(self):
        self.plan = S.score_plan(SCORE)
        self.timing = S.note_times(self.plan, [('melody_vocal.mid', recording_midi(self.plan))])

    def test_the_next_word_heard_inside_the_held_note_does_not_take_it(self):
        # The aligner hears "let" (the word after the held word) a full second inside the held note.
        times = word_times(self.plan, MISBROKEN, STARTS)
        held = next(n for n in self.plan['notes'] if n['dur'] == 4)
        times[19] = dict(times[19], start=round(real(held['on']) + 1.0, 3))
        result = S.fit(SCORE, MISBROKEN, self.timing, times)
        self.assertEqual(result['lyrics'], FITTED)
        own = [h for h in result['report']['held_notes'] if h['beats'] == 4.0]
        self.assertEqual([(h['word'], h['word_in_line']) for h in own], [(18, 2)])

    def test_a_section_mostly_not_heard_keeps_her_lines(self):
        times = word_times(self.plan, MISBROKEN, STARTS)
        for i in range(5, 16):                      # the verse's words
            times[i] = dict(times[i], score=0.05)
        result = S.fit(SCORE, MISBROKEN, self.timing, times)
        self.assertTrue(result['applied'])
        self.assertTrue(result['lyrics'].startswith('[Verse]\nGrey clouds roll past\nthe town cold wind at\nthe door\n'))
        self.assertTrue(result['lyrics'].endswith('[Chorus]\nSing it loud, let it ring, all night long\nwe dance till the dawn'))
        self.assertEqual([(s['section'], s['heard'], s['words'], s['fitted']) for s in result['report']['sections']],
                         [('verse', 0, 11, False), ('chorus', 14, 14, True)])
        self.assertEqual(result['notes'][1], 'The verse keeps your own line breaks: too few of its words could be heard '
                                             'clearly in the recording.')
        self.assertFalse(any(h['verified'] for h in result['report']['held_notes'] if h['section'] == 'verse'))

    def test_her_break_far_from_a_phrase_end_is_kept(self):
        lyrics = MISBROKEN.replace('Grey clouds roll past\nthe town cold wind at\nthe door',
                                   'Grey\nclouds roll past the town cold wind at the door')
        result = S.fit(SCORE, lyrics, self.timing, word_times(self.plan, lyrics, STARTS))
        self.assertTrue(result['lyrics'].startswith('[Verse]\nGrey\nclouds roll past the town\ncold wind at the door\n'))
        self.assertEqual((result['report']['her_breaks_kept'], result['report']['her_breaks_moved']), (1, 2))

    def test_a_lone_dash_takes_no_comma(self):
        lyrics = MISBROKEN.replace('Sing it loud let', 'Sing it loud - let')
        times = word_times(self.plan, MISBROKEN, STARTS)
        times.insert(19, None)                      # the dash is timed from its neighbours
        result = S.fit(SCORE, lyrics, self.timing, times)
        self.assertIn('Sing it loud - let it ring, all night long', result['lyrics'])

    def test_repeated_sections_share_the_clearest_layout(self):
        chorus = """V: Vocal
z4g2a2|b8|a2g2e2z2|c2d2e2z2|
V: Ins
Z4|
V: Vocal
g2g2a2b2|c'4z4|
V: Ins
Z2|
"""
        score = SCORE.replace('% outro\n', '% chorus\n' + chorus + '% outro\n')
        lyrics = MISBROKEN + '\n[Chorus]\nSing it loud let\nit ring all night long we\ndance till the dawn\n'
        plan = S.score_plan(score)
        timing = S.note_times(plan, [('melody_vocal.mid', smf([[(real(n['on']), n['pitch'], float(n['dur']) * 0.95)
                                                                  for n in plan['notes']]]))])
        with_times(plan, timing)
        starts = STARTS + list(range(25, 39))
        times = word_times(plan, lyrics, starts)
        for k in range(30, 44):                     # the second chorus: barely heard, and heard early
            times[k] = dict(times[k], score=0.05, start=round(times[k]['start'] - 0.45, 3))
        result = S.fit(score, lyrics, timing, times)
        self.assertTrue(result['applied'])
        block = '[Chorus]\nSing it loud, let it ring, all night long\nwe dance till the dawn'
        self.assertEqual(result['lyrics'].count(block), 2)
        (row,) = result['report']['repeats']
        self.assertEqual((row['section'], row['instances'], row['source'], row['fitted']), ('chorus', [2, 3], 2, True))
        self.assertEqual(row['heard'], [14, 0])
        self.assertTrue(row['disagreements'][0]['shared'])
        self.assertEqual(row['disagreements'][0]['tune_match'], 1.0)
        held = [(h['phrase'], h['word'], h['verified']) for h in result['report']['held_notes'] if h['beats'] == 4.0]
        self.assertEqual(held, [(2, 18, True), (5, 32, False)])    # the same word of each chorus; only the heard one counts
        self.assertEqual(len(result['notes']), 1)

    def test_lines_are_capped_at_her_longest_or_the_notes_they_sing(self):
        words, _ = S.lyric_words('[Verse]\n' + ' '.join(['la'] * 10) + '\nla la')
        sung, line = list(range(12)), list(range(10))
        # Ten syllables on ten notes: within 1.4 per note, so longer than her longest line is fine.
        self.assertEqual(S._split_long(line, words, sung, {9, 11}, 2, {k: 1 for k in line}), [line])
        # Ten syllables crammed on three notes: split at her own line end first, then to her longest.
        parts = S._split_long(line, words, sung, {4, 9, 11}, 5, {0: 1, 5: 1, 9: 1})
        self.assertEqual(parts, [list(range(5)), list(range(5, 10))])
        parts = S._split_long(line, words, sung, {4, 9, 11}, 3, {})
        self.assertEqual(parts[0][-1], 1)
        self.assertEqual(sum(parts, []), line)
        self.assertTrue(all(len(p) <= 3 for p in parts))

    def test_unclear_words_follow_the_syllable_count_more(self):
        plan = S.score_plan(SCORE)
        with_times(plan, self.timing)
        words, _ = S.lyric_words('[Verse]\n' + ' '.join(['la'] * 11))
        sung = list(range(11))
        first = plan['phrases'][0]
        onset = [plan['notes'][first['first']]['s'] + 0.1 * k for k in range(11)]   # all heard inside phrase one
        clear = S._assign(dict(plan, phrases=plan['phrases'][:2]), words, sung, onset, [1.0] * 11)
        unclear = S._assign(dict(plan, phrases=plan['phrases'][:2]), words, sung, onset, [0.1] * 11)
        self.assertEqual(clear, [(0, 11), None])                # heard clearly there: timing decides
        self.assertEqual(unclear, [(0, 8), (8, 11)])            # not heard: the six-note phrase stops at 1.4 per note


# L:1/16 so a sixteenth-note slide can lead into the held note; one beat is one second.
TOUCH = """X:1
T:
M:4/4
L:1/16
Q:1/4=60
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
% verse
V: Vocal
c4c4d4e4|f8z8|
V: Ins
Z2|
% chorus
V: Vocal
z4g2a2b3a1g4-|g8e4f4|
V: Ins
Z2|
V: Vocal
z4c4d4e4|f4z12|
V: Ins
Z2|
"""
TOUCH_LYRICS = '[Verse]\nRiver runs\nslow\n[Chorus]\nSing it loud and\nbright we dance till dawn\n'
# Word k starts on note TOUCH_STARTS[k]: "loud" is sung over b, the slide and the held g.
TOUCH_STARTS = [0, 2, 3, 5, 6, 7, 10, 11, 12, 13, 14, 15]


def touch_times(score):
    plan = S.score_plan(score)
    timing = dict(notes=[[round(0.1 + float(n['on']), 3), round(0.1 + float(n['on'] + n['dur']), 3)]
                         for n in plan['notes']], matched=1.0, source='fixture')
    times = [dict(start=round(timing['notes'][n][0] + 0.03, 3), end=round(timing['notes'][n][0] + 0.3, 3), score=0.8,
                  quiet_before=0.5 if n in (0, 5, 12) else 0.0) for n in TOUCH_STARTS]
    return plan, timing, times


class TouchUpTest(unittest.TestCase):
    def fitted(self, score, touchup=True):
        plan, timing, times = touch_times(score)
        self.assertEqual(len(S.lyric_words(TOUCH_LYRICS)[0]), len(TOUCH_STARTS))
        return plan, S.fit(score, TOUCH_LYRICS, timing, times, touchup=touchup)

    def test_a_slide_into_the_held_note_folds_into_it(self):
        plan, result = self.fitted(TOUCH)
        self.assertEqual(result['lyrics'], '[Verse]\nRiver runs slow\n\n[Chorus]\nSing it loud, and bright\nwe dance till dawn')
        touch = result['report']['score_touchup']
        self.assertEqual((touch['applied'], touch['ties'], touch['folds']), (True, 0, 1))
        self.assertEqual(touch['places'], [dict(section='chorus', phrase=1, held_note=4, kind='slide')])
        self.assertEqual((touch['notes_before'], touch['notes_after']), (16, 15))
        before = S.parse_abc(TOUCH)
        after = S.parse_abc(result['abc'])
        vocal = before.voices['Vocal'].notes
        slide = next(k for k, n in enumerate(vocal) if n[2] == S.Fraction(1, 4))
        self.assertEqual(after.voices['Vocal'].notes,
                         vocal[:slide] + [[vocal[slide][0], vocal[slide + 1][1], vocal[slide][2] + vocal[slide + 1][2]]]
                         + vocal[slide + 2:])
        self.assertEqual(after.voices['Ins'].notes, before.voices['Ins'].notes)
        self.assertEqual(after.voices['Vocal'].bars, before.voices['Vocal'].bars)

    def test_a_same_pitch_lead_in_is_tied_and_the_melody_keeps_every_pitch(self):
        score = TOUCH.replace('b3a1g4-', 'b3g1g4-')
        _, result = self.fitted(score)
        touch = result['report']['score_touchup']
        self.assertEqual((touch['ties'], touch['folds']), (1, 0))
        pitches = [p for _, p, _ in S.parse_abc(result['abc']).voices['Vocal'].notes]
        original = [p for _, p, _ in S.parse_abc(score).voices['Vocal'].notes]
        self.assertEqual(original[8], original[9])
        self.assertEqual(pitches, original[:9] + original[10:])

    def test_a_touched_up_score_is_recognised_as_this_scores(self):
        _, result = self.fitted(TOUCH)
        self.assertTrue(S.touched_from(TOUCH, result['abc']))
        self.assertFalse(S.touched_from(TOUCH, TOUCH))                      # nothing merged
        self.assertFalse(S.touched_from(result['abc'], TOUCH))              # the other way round
        self.assertFalse(S.touched_from(TOUCH, TOUCH.replace('b3a1g4-', 'b3a1g4-').replace('f4z12', 'e4z12')))
        self.assertFalse(S.touched_from(TOUCH, 'not a score'))

    def test_nothing_to_touch_leaves_the_score_alone(self):
        score = TOUCH.replace('b3a1g4-', 'b2a2g4-')
        _, result = self.fitted(score)
        self.assertIsNone(result['abc'])
        self.assertEqual(result['report']['score_touchup']['reason'], 'nothing to touch up')
        _, plain = self.fitted(TOUCH, touchup=False)
        self.assertNotIn('score_touchup', plain['report'])
        self.assertIsNone(plain['abc'])


class MeasureRoundTwoTest(unittest.TestCase):
    def test_unverified_held_notes_are_reported_apart(self):
        plan = dict(groups=[[0, 1, 2]], sections=['chorus'], source_pauses={0: False},
                    held=[dict(word=1, verified=True, section='chorus', phrase=0, note=1),
                          dict(word=0, verified=False, section='chorus', phrase=0, note=0)])
        take = [dict(start=0.0, end=0.2, score=0.9, quiet_before=0.0), dict(start=0.3, end=2.0, score=0.9, quiet_before=0.0),
                dict(start=2.4, end=2.6, score=0.9, quiet_before=0.0)]
        m = S.measure(plan, [0, 1, 2], take)
        self.assertEqual(m['held_words_on_note'], dict(hits=1, of=1))
        self.assertEqual(m['held_unverified'], dict(hits=0, of=1))
        self.assertEqual(m['fit_score'], 100)
        self.assertEqual(m['misses'], [])

    def test_the_words_a_take_was_given_map_back_to_hers(self):
        order, positions = S.sung_order(FITTED, MISBROKEN)
        self.assertEqual(order, list(range(5, 30)))
        self.assertEqual(positions, list(range(25)))
        order, positions = S.sung_order('[Verse]\nhey Grey clouds roll', MISBROKEN)
        self.assertEqual((order, positions), ([5, 6, 7], [1, 2, 3]))


class HandlerRoundTwoTest(unittest.TestCase):
    def setUp(self):
        self.plan = S.score_plan(SCORE)
        self.timing = S.note_times(self.plan, [('melody_vocal.mid', recording_midi(self.plan))])
        self.words = word_times(self.plan, MISBROKEN, STARTS)

    def render(self, **sync):
        inp = Y.request_input(COVER)
        state = dict(timing=self.timing, words=self.words, error=None, fit=True, measure=False)
        state.update(sync)
        return inp, Y.fixed_score_render(inp, Y.cover_options(COVER), SCORE, 'recording', 50.0, sync=state)

    def test_a_fit_that_breaks_renders_her_lines(self):
        saved = Y.lyric_sync.fit

        def broken(*args, **kwargs):
            if kwargs.get('failed') == 'error':
                return saved(*args, **kwargs)
            raise ZeroDivisionError('fixture')
        Y.lyric_sync.fit = broken
        try:
            inp, (render, _, _, extras, notes) = self.render()
        finally:
            Y.lyric_sync.fit = saved
        self.assertEqual(render['lyrics'], inp['lyrics'])
        self.assertEqual(extras['lyric_sync']['reason'], 'error')
        self.assertIn('kept as written: your words could not be fitted to the tune this time.', notes[-1])

    def test_a_take_that_cannot_be_scored_is_a_note(self):
        saved = Y.align_audio

        def broken(*args, **kwargs):
            raise KeyError('fixture')
        Y.align_audio = broken
        notes = []
        try:
            self.assertIsNone(Y.safe_measure('/tmp/take.wav', dict(sync_words=['la'], sync_plan={}, sync_order=[0]),
                                             '/tmp', notes))
        finally:
            Y.align_audio = saved
        self.assertEqual(notes, ['This take could not be scored against the tune this time.'])

    def test_the_touched_up_score_reaches_the_render_only_when_asked(self):
        inp, (render, _, _, extras, notes) = self.render(touchup=True)
        self.assertEqual(render['abc'], SCORE)                  # nothing in this score to touch up
        self.assertEqual(extras['lyric_sync']['score_touchup']['reason'], 'nothing to touch up')
        _, timing, times = touch_times(TOUCH)
        cover = dict(COVER, lyrics=TOUCH_LYRICS)
        inp = Y.request_input(cover)
        state = dict(timing=timing, words=times, error=None, fit=True, measure=False, touchup=True)
        render, _, _, extras, notes = Y.fixed_score_render(inp, Y.cover_options(cover), TOUCH, 'recording', 20.0, sync=state)
        self.assertNotEqual(render['abc'], TOUCH)
        self.assertEqual(len(S.parse_abc(render['abc']).voices['Vocal'].notes), 15)
        self.assertIn('touched up in 1 place:', notes[-1])
        state['touchup'] = False
        render, _, _, extras, notes = Y.fixed_score_render(inp, Y.cover_options(cover), TOUCH, 'recording', 20.0, sync=state)
        self.assertEqual(render['abc'], TOUCH)

    def test_the_fitted_layout_marks_where_stars_may_go(self):
        inp, (render, _, _, extras, _) = self.render()
        self.assertEqual(extras['sync_stars'], [k in (5, 10, 19, 24) for k in range(25)])
        inp, (render, _, _, extras, _) = self.render(fit=False, measure=True)
        self.assertEqual(extras['sync_stars'], S.star_after(S.lyric_words(MISBROKEN)[0]))


class MeasureModeTest(unittest.TestCase):
    """mode=measure picks the score the take was rendered from and never transcribes a melody score."""

    def run_read(self, cached, take_abc, cover_mode=None):
        calls = []
        saved = Y.fetch_recording, Y.read_recording

        def read(reference, td, task, client, bucket, want_timing=False, fetched=None, cache_only=False):
            calls.append((task, cache_only))
            if task in cached:
                return dict(abc=cached[task], task=task, cached=True)
            return None if cache_only else dict(abc='fresh-' + task, task=task, cached=False)
        Y.fetch_recording, Y.read_recording = (lambda reference, td: ('s', 'd', 10.0)), read
        try:
            req = dict(reference='https://x/a.mp3', take=TAKE, lyrics=MISBROKEN, lyrics_used=None, cover_mode=cover_mode)
            return Y.measure_read(req, take_abc, '/tmp', None, 'b'), calls
        finally:
            Y.fetch_recording, Y.read_recording = saved

    def test_the_takes_own_score_decides(self):
        read, calls = self.run_read({'full': 'FULL', 'melody': 'MELODY'}, 'MELODY\n')
        self.assertEqual(read['task'], 'melody')
        self.assertEqual(calls, [('full', True), ('melody', True)])
        read, _ = self.run_read({'full': 'FULL', 'melody': 'MELODY'}, 'FULL')
        self.assertEqual(read['task'], 'full')

    def test_a_touched_up_take_scores_against_the_original(self):
        _, result = TouchUpTest().fitted(TOUCH)
        read, calls = self.run_read({'full': TOUCH, 'melody': 'MELODY'}, result['abc'])
        self.assertEqual((read['task'], read['touched'], read['abc']), ('full', True, TOUCH))

    def test_a_score_this_recording_never_had_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not one this recording was read into"):
            self.run_read({'full': 'FULL'}, 'SOMETHING ELSE')

    def test_never_a_fresh_melody_transcription(self):
        read, calls = self.run_read({}, None)
        self.assertEqual((read['task'], read['cached']), ('full', False))
        self.assertEqual(calls, [('full', True), ('melody', True), ('full', False)])
        with self.assertRaisesRegex(ValueError, 'not one this recording'):
            self.run_read({}, None, cover_mode='melody')
        read, calls = self.run_read({'melody': 'MELODY'}, None, cover_mode='melody')
        self.assertEqual(calls, [('melody', True)])


class ReportRoundTwoTest(unittest.TestCase):
    SECTIONS = [dict(name='intro', bars=5, seconds=13.0, sung_notes=1), dict(name='verse', bars=10, seconds=28.0, sung_notes=46),
                dict(name='chorus', bars=8, seconds=22.3, sung_notes=40), dict(name='outro', bars=4, seconds=11.0, sung_notes=0)]

    def test_lyric_fit_pairs_sections_by_name_when_the_intro_has_no_tune(self):
        lyrics = ('[Intro]\nOh oh here we go now\n[Verse 1]\n' + 'la ' * 44 + '\n[Chorus]\n' + 'sing ' * 38)
        fit = Y.lyric_fit(lyrics, self.SECTIONS)
        self.assertEqual([(r['lyrics_section'], r['score_section'], r['fit']) for r in fit['sections']],
                         [('Intro', None, 'no tune'), ('Verse 1', 'verse', 'close'), ('Chorus', 'chorus', 'close')])
        self.assertFalse(fit['same_order'])
        self.assertEqual(fit['paired'], 'by section name and order')

    def test_meter_check(self):
        self.assertEqual(Y.meter_check('Smooth and floating with a 6/8 feel, 86 BPM', '4/4'),
                         dict(style_meter='6/8', score_meter='4/4'))
        self.assertEqual(Y.meter_check('A slow waltz', '4/4'), dict(style_meter='waltz', score_meter='4/4'))
        self.assertEqual(Y.meter_check('Swaying 12/8 blues', '3/4'), dict(style_meter='12/8', score_meter='3/4'))
        for style, meter in (('Straight 4/4 rock', '4/4'), ('Marching 2/4', '4/4'), ('A waltz in 3/4', '3/4'),
                             ('24/7 radio pop at 120 BPM', '4/4'), ('Half-time 1/2 feel', '4/4'), ('No meter named', '4/4'),
                             ('6/8 ballad', None), ('6/8 ballad', 'odd')):
            with self.subTest(style=style):
                self.assertIsNone(Y.meter_check(style, meter))

    def test_the_meter_check_rides_with_a_cover(self):
        styled = dict(COVER, style='Warm folk ballad with a 6/8 feel, 90 BPM')
        inp = Y.request_input(styled)
        _, _, facts, extras, _ = Y.fixed_score_render(inp, Y.cover_options(styled), SCORE, 'recording', 50.0)
        self.assertEqual(facts['score_meter'], '4/4')
        self.assertEqual(extras['meter_check'], dict(style_meter='6/8', score_meter='4/4'))
        inp = Y.request_input(COVER)
        self.assertIsNone(Y.fixed_score_render(inp, Y.cover_options(COVER), SCORE, 'recording', 50.0)[3]['meter_check'])


class StarTokenTest(unittest.TestCase):
    def test_stars_sit_between_her_lines_only(self):
        import align
        words, _ = S.lyric_words('[Verse]\nla la\nlo\n[Chorus]\nli -')
        self.assertEqual(S.star_after(words), [False, True, True, False, True])
        self.assertEqual(align.targets([[1], [1], [2], [3], []], S.star_after(words), 9),
                         ([9, 1, 1, 9, 2, 9, 3, 9], [None, 0, 1, None, 2, None, 3, None]))
        self.assertEqual(align.targets([[1], [2]]), ([1, 2], [0, 1]))           # no stars: exactly the round-1 targets


if __name__ == '__main__':
    unittest.main()
