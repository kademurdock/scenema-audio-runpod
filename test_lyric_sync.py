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
        held = [(h['phrase'], h['note'], h['word'], h['last'], h['clear']) for h in report['held_notes']]
        self.assertIn((2, 2, 18, False, True), held)          # the four-beat note keeps its word
        self.assertEqual(len(result['notes']), 2)
        self.assertIn('intro words (1 line)', result['notes'][1])

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
                 (S.fit(SCORE, MISBROKEN, dict(self.timing, notes=self.timing['notes'][:-1]), times), 'notes')]
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
        for shape in ({}, COVER, {'style': 'pop', 'lyrics': 'la', 'cot': 'off'}, {'fit_lyrics': False, 'measure_fit': False}):
            self.assertEqual(Y.sync_options(shape), dict(fit_lyrics=False, measure_fit=False))
        self.assertEqual(Y.sync_options({'fit_lyrics': 'timing', 'measure_fit': True}),
                         dict(fit_lyrics=True, measure_fit=True))
        self.assertTrue({'lyric-sync', 'fit-score'} <= set(Y.FEATURES))
        self.assertEqual(Y.cover_options({'mode': 'measure'})['mode'], 'measure')

    def test_bad_values_are_refused_before_any_gpu_work(self):
        for key, value in (('fit_lyrics', 'yes'), ('fit_lyrics', True), ('fit_lyrics', 0), ('fit_lyrics', 'counts'),
                           ('measure_fit', 'true'), ('measure_fit', 1)):
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
        self.assertEqual(Y.measure_request(good), ('https://x/a.mp3', TAKE, MISBROKEN.strip()))
        for bad in ({'take_key': '../yue2/x/master.wav'}, {'take_key': 'yue2-scores/a/full.abc'},
                    {'take_key': TAKE.replace('.wav', '.flac')}, {'lyrics': '[Verse]'}, {'lyrics': None},
                    {'reference_voice_url': ''}):
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

        def aligned(path, words, td, tag):
            calls.append((path, len(words), tag))
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
        self.assertRegex(bucket.puts[0], r'^yue2-scores/abc/words-[0-9a-f]{16}-mmsfa-hdemucs-1\.json$')
        stored = bucket.objects[bucket.puts[0]].decode('utf-8').lower()
        for word in set(w.lower() for w in Y.WORD.findall(MISBROKEN)):
            if len(word) >= 4:
                self.assertNotRegex(stored, r'\b' + word + r'\b')

    def test_failed_measurements_name_their_reason(self):
        def broken(*args):
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


if __name__ == '__main__':
    unittest.main()
