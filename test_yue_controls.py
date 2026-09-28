import hashlib
import sys
import types
import unittest
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from unittest import mock
import yue_handler
from yue_handler import sound_controls, render_song, request_input, style_request
from yue_handler import (cover_options, check_combination, fixed_score_render, instrumental_style, lyric_tags,
                         tempo_style, semantic_budget, score_facts, lyric_fit, planning_request, PLANNING_TAGS,
                         FEATURES, UNMOVABLE, NO_CHORDS, score_has_chords)
from yue_handler import (fit_option, check_fit, source_limit, tempo_plan, with_tempo, score_tempo, scale_style_bpm,
                         fetch_recording, spoken_length, handler, SOURCE_MAX, FIT_SOURCE_MAX)
from abc_tools import compare, parse_abc


@dataclass(frozen=True)
class Config:
    ode_steps: int = 32


class Pipeline:
    generation_config = Config()

    def __call__(self, **kwargs):
        return self.generation_config.ode_steps, kwargs


# A YuE2 plan from Sep 17 (platform render; melody plus chords) and SheetSage2's
# melody-only transcription of the same song: the real shapes the worker sees.
PLANNED = '''X:1
T:
M:3/4
L:1/16
Q:1/4=84
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:Bb
% intro
V: Vocal
"Bb"z12|"Bb"z4"Bb/D"z8|"Eb"z12|"Eb"z4"Eb/F"z4"F"z4|
V: Ins
B,2FB2FF3B,2F|B,3B,2FDF2F2B|E3B,2G4B,2G-|G3B,2GF3A,2F|
V: Vocal
"Bb"z12|"Bb"z4"Bb/D"z8|"Eb"z12|"Eb"z4"Eb/F"zBd2"F"d2dd-|
V: Ins
B,3B,2F4B,2F|B,z2B,2FDF2F2B|E3B,2G4B,2G|Z|
% verse
V: Vocal
"Bb"d2BBz8|"Bb"z4"Bb/D"zBd2f2BB-|"Eb"B3z8z|"Eb"z4"Eb/F"zBd2"F"d2d2|
V: Ins
Z4|
V: Vocal
"Gm7"d6c4Bd-|"F"d4-dzc4BB-|"Eb"B3z8z|"Eb"z4zBd2d2dd-|
V: Ins
Z4|
V: Vocal
"Bb"d2cBz8|"Bb"z4"Bb/D"zBd2f2BB-|"Eb"B3z8z|"Eb"z4"Eb/F"zBd2"F"d2dd-|
V: Ins
Z4|
V: Vocal
"Gm7"d3zc4B2d2-|"F"d6c2BBBz|"Eb"z12|"Eb"z6d2d2dd-|
V: Ins
Z4|
V: Vocal
"Cm7"d2cBz8|"F"z6dc3BB-|
V: Ins
Z2|
% outro
V: Vocal
"Bb"B3z8z|"Bb"z4"Bb/D"z8|"Eb"z12|"Eb"z4"Eb/F"z4"F"z4|
V: Ins
Z|z6f6|g12-|g4z2f6|
V: Vocal
"Gm7"z12|"F"z12|"Bb"z12|"Bb"z12|
V: Ins
d12|c12|B12-|B4z8|
'''

TRANSCRIBED = '''X:1
T:
M:3/4
L:1/32
Q:1/4=83
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:Bb
% intro
V: Vocal
Z4|
V: Ins
Z|B,4F2B4f6F2B4f2|B4F2B4f2d2f4f4B2|E4G2B4g6G2B4g2|
V: Vocal
Z4|
V: Ins
E4G2B4g2F4z2A4f2|B4F2B4f6F2B4f2|B4F2B4f2d2f4f4B2|E4G2B4g2z12|
V: Vocal
z8z2B2d4d4d2d2-|
V: Ins
z8B2z12z2|
% verse
V: Vocal
d4B4z16|z8z2B2d4f4B2B2-|B4z16z4|z8z2B2d4d4d4|
V: Ins
Z4|
V: Vocal
d12c8B2d2-|d8-d2z2c8B2B2-|B4z16z4|z8z2B2d4d4d2d2-|
V: Ins
Z4|
V: Vocal
d4c2B2z16|z8z2B2d4f4B2B2-|B4z16z4|z8z2B2d4d4d2d2-|
V: Ins
Z4|
V: Vocal
d6z2c8B4d4-|d12c2B2B2B6|Z|z12d4d4d2d2-|
V: Ins
Z4|
V: Vocal
d4c2B2z16|z12d2c6B2B2-|
V: Ins
Z2|
% outro
V: Vocal
B4z16z4|Z3|
V: Ins
Z|d12f12|g24-|g8z4f12|
V: Vocal
Z4|
V: Ins
d24|c24|B24-|B8z16|
'''

TAGS = '[Intro]\n\n[Verse]\n\n[Outro]\n'
NO_SINGING = ', no vocals, no singing, no choir, no spoken words.'
# Every request shape the booth sends today (yue.ts yueInput), including the
# fields the worker ignores (title, count, band, lora_scale without a key).
BOOTH_COVER = {'style': 'Warm soul ballad', 'title': 'Test', 'count': 2, 'weirdness': 50, 'steps': 32,
               'guidance': 1, 'lyrics': '[Verse]\nHold on to me\n[Chorus]\nAll night long', 'band': None,
               'reference_voice_url': 'https://example.invalid/original.mp3', 'cot': 'melody', 'seed': 7}


def harmony(chords):
    result = []
    for when, symbol in chords:
        if not result or result[-1][1] != symbol:
            result.append((when, symbol))
    return result


class ControlsTest(unittest.TestCase):
    def test_defaults_preserve_release_sound(self):
        pipe = Pipeline()
        steps, args = render_song(pipe, {'seed': 17}, sound_controls({}))
        self.assertEqual(steps, 32)
        self.assertEqual(args, {'seed': 17, 'cfg_scale': 1.0,
                               'abc_sampling': {'temperature': .7},
                               'semantic_sampling': {'temperature': 1.0}})

    def test_overrides_reach_pipeline_and_reset(self):
        pipe = Pipeline()
        controls = sound_controls({'weirdness': 100, 'steps': 64, 'guidance': 2})
        steps, args = render_song(pipe, {'seed': 42}, controls)
        self.assertEqual(steps, 64)
        self.assertEqual(args['cfg_scale'], 2)
        self.assertEqual(args['semantic_sampling']['temperature'], 1.3)
        self.assertEqual(pipe.generation_config.ode_steps, 32)
        class Broken(Pipeline):
            def __call__(self, **kwargs):
                raise RuntimeError('fixture')
        broken = Broken()
        with self.assertRaises(RuntimeError):
            render_song(broken, {}, controls)
        self.assertEqual(broken.generation_config.ode_steps, 32)

    def test_bad_controls_are_rejected_before_gpu_load(self):
        for key, value in [('steps', 0), ('steps', 65), ('steps', 16.5),
                           ('weirdness', 101), ('weirdness', True),
                           ('guidance', float('nan')), ('guidance', 0)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                sound_controls({key: value})


class StyleTest(unittest.TestCase):
    def test_score_free_song_keeps_upstream_guidance(self):
        _, args = render_song(Pipeline(), {'seed': 1, 'cot': 'off'}, sound_controls({}))
        self.assertEqual(args['cfg_scale'], 1.01)

    def test_style_key_and_strength_are_checked(self):
        self.assertIsNone(style_request({}))
        self.assertEqual(style_request({'lora_key': 'yue2-loras/kids-step1200.pt'}), ('yue2-loras/kids-step1200.pt', 1.0))
        for bad in ({'lora_key': '../secrets.pt'}, {'lora_key': 'yue2-loras/a.pt', 'lora_scale': 9}):
            with self.assertRaises(ValueError):
                style_request(bad)

    def test_score_free_refuses_a_cover(self):
        ok = request_input({'style': 'soulful', 'lyrics': 'la', 'cot': 'off'})
        self.assertEqual(ok['cot'], 'off')
        with self.assertRaises(ValueError):
            request_input({'style': 'soulful', 'lyrics': 'la', 'cot': 'off', 'abc': 'X:1'})


class CoverOptionsTest(unittest.TestCase):
    TODAY = dict(mode='render', keep_harmony=None, instrumental=False, length_guard=False, match_score_tempo=False)

    def test_todays_requests_get_todays_behaviour(self):
        shapes = [{}, BOOTH_COVER, {'style': 'pop', 'lyrics': 'la', 'cot': 'off', 'lora_key': 'yue2-loras/kids.pt',
                                    'lora_scale': 1.2, 'band': 'kids'},
                  {'style': 'pop', 'abc': PLANNED, 'cot': 'full', 'weirdness': 80, 'steps': 64, 'guidance': 2.5}]
        for shape in shapes:
            with self.subTest(shape=sorted(shape)):
                self.assertEqual(cover_options(shape), self.TODAY)
        self.assertEqual(cover_options({'keep_harmony': False, 'instrumental': False}),
                         dict(self.TODAY, keep_harmony=False))

    def test_new_fields_turn_the_length_guard_on_unless_told(self):
        self.assertTrue(cover_options({'instrumental': True})['length_guard'])
        self.assertTrue(cover_options({'keep_harmony': True})['length_guard'])
        self.assertTrue(cover_options({'match_score_tempo': True})['length_guard'])
        self.assertFalse(cover_options({'keep_harmony': True, 'length_guard': False})['length_guard'])
        self.assertTrue(cover_options({'length_guard': True})['length_guard'])

    def test_bad_values_are_refused_before_any_gpu_work(self):
        for key, value in [('keep_harmony', 'yes'), ('keep_harmony', 1), ('instrumental', 0),
                           ('length_guard', 'true'), ('match_score_tempo', [True])]:
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'true or false'):
                cover_options({key: value})
        for mode in ('cover', 'Render', 5, ['render'], None, ''):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'not available'):
                cover_options({'mode': mode})
        self.assertEqual(cover_options({'mode': 'transcribe'})['mode'], 'transcribe')

    def test_impossible_combinations_are_refused(self):
        song = request_input({'style': 'pop', 'lyrics': 'la'})
        off = request_input({'style': 'pop', 'lyrics': 'la', 'cot': 'off'})
        score = request_input({'style': 'pop', 'abc': PLANNED})
        with self.assertRaisesRegex(ValueError, 'chords'):
            check_combination(song, cover_options({'keep_harmony': True}), None)
        with self.assertRaisesRegex(ValueError, 'write a score first'):
            check_combination(off, cover_options({'instrumental': True}), None)
        with self.assertRaisesRegex(ValueError, 'tempo'):
            check_combination(song, cover_options({'match_score_tempo': True}), None)
        check_combination(score, cover_options({'keep_harmony': True}), None)
        check_combination(song, cover_options({'keep_harmony': True}), 'https://example.invalid/a.mp3')
        check_combination(song, cover_options({'instrumental': True, 'keep_harmony': True}), None)
        check_combination(off, cover_options({}), None)

    def test_features_are_advertised(self):
        self.assertTrue({'keep-harmony', 'instrumental', 'transcribe', 'score-cache', 'length-guard'} <= set(FEATURES))


class BackwardsCompatibilityTest(unittest.TestCase):
    def test_recording_cover_renders_exactly_as_before(self):
        # Before Part 295: inp['abc'] = transcription; inp['cot'] = 'melody'; render_song(pipe, inp, controls).
        for extra in ({}, {'lora_key': 'yue2-loras/kids-step1200.pt', 'lora_scale': 0.8},
                      {'weirdness': 90, 'steps': 48, 'guidance': 2.0}):
            raw = dict(BOOTH_COVER, **extra)
            inp, controls = request_input(raw), sound_controls(raw)
            before = dict(inp, abc=TRANSCRIBED, cot='melody')
            render, budget, facts, extras, notes = fixed_score_render(inp, cover_options(raw), TRANSCRIBED,
                                                                      'recording', 72.6)
            self.assertEqual(render, before)
            self.assertIsNone(budget)
            self.assertEqual(notes, [])
            self.assertNotIn('transfer', extras)
            self.assertEqual(render_song(Pipeline(), render, controls), render_song(Pipeline(), before, controls))
            self.assertNotIn('max_tokens', render_song(Pipeline(), render, controls)[1]['semantic_sampling'])

    def test_given_score_renders_exactly_as_before(self):
        for cot in ('melody', 'full'):
            raw = {'style': 'pop', 'lyrics': 'la la', 'abc': PLANNED, 'cot': cot, 'seed': 5}
            inp = request_input(raw)
            render, budget, _, _, notes = fixed_score_render(inp, cover_options(raw), inp['abc'], 'score')
            self.assertEqual(render, inp)
            self.assertIsNone(budget)
            self.assertEqual(notes, [])

    def test_score_facts_never_block_a_sung_cover(self):
        broken = TRANSCRIBED.replace('K:Bb', 'K:A#')
        inp = request_input(BOOTH_COVER)
        render, budget, facts, extras, _ = fixed_score_render(inp, cover_options(BOOTH_COVER), broken, 'recording', 72.6)
        self.assertEqual(render, dict(inp, abc=broken, cot='melody'))
        self.assertEqual(facts, {})
        self.assertIsNone(extras['lyric_fit'])


class InstrumentalTest(unittest.TestCase):
    def test_recording_becomes_an_upstream_instrumental(self):
        raw = dict(BOOTH_COVER, instrumental=True)
        render, budget, facts, extras, notes = fixed_score_render(request_input(raw), cover_options(raw),
                                                                  TRANSCRIBED, 'recording', 72.6)
        before, after = parse_abc(TRANSCRIBED), parse_abc(render['abc'])
        self.assertEqual(after.voices['Vocal'].notes, [])
        for note in before.voices['Vocal'].notes:
            self.assertIn(note, after.voices['Ins'].notes)
        self.assertEqual(after.bpm, before.bpm)
        self.assertEqual(after.voices['Ins'].bars, before.voices['Ins'].bars)
        self.assertEqual(render['lyrics'], TAGS)
        self.assertEqual(render['lyrics'], lyric_tags(render['abc']))
        self.assertEqual(render['style'], 'Instrumental, Warm soul ballad' + NO_SINGING)
        self.assertEqual(render['cot'], 'melody')
        self.assertEqual(extras['transfer']['vocal_notes_moved'], len(before.voices['Vocal'].notes))
        self.assertFalse(extras['transfer']['chords_kept'])
        self.assertEqual(budget, semantic_budget(72.6, facts['score_seconds']))
        self.assertEqual(len(notes), 1)

    def test_kept_harmony_renders_with_the_full_score(self):
        raw = {'style': 'Piano trio', 'abc': PLANNED, 'instrumental': True, 'keep_harmony': True}
        render, _, _, extras, notes = fixed_score_render(request_input(raw), cover_options(raw), PLANNED, 'score')
        self.assertEqual(render['cot'], 'full')
        self.assertEqual(harmony(parse_abc(render['abc']).voices['Vocal'].chords),
                         harmony(parse_abc(PLANNED).voices['Vocal'].chords))
        self.assertTrue(extras['transfer']['chords_kept'])
        self.assertEqual(notes, [])

    def test_a_score_keeps_its_chords_unless_told(self):
        for keep, cot in ((None, 'full'), (False, 'melody')):
            raw = {'style': 'Piano trio', 'abc': PLANNED, 'instrumental': True}
            if keep is not None:
                raw['keep_harmony'] = keep
            render, _, _, _, _ = fixed_score_render(request_input(raw), cover_options(raw), PLANNED, 'score')
            with self.subTest(keep=keep):
                self.assertEqual(render['cot'], cot)

    def test_yue2_plan_keeps_chords_and_words_only_guide_it(self):
        raw = {'style': 'Banjo breakdown', 'lyrics': 'Down by the river', 'instrumental': True}
        inp = request_input(raw)
        self.assertEqual(planning_request(inp), dict(style='Instrumental, Banjo breakdown' + NO_SINGING,
                                                     lyrics='Down by the river\n', cot='full', seed=42))
        self.assertEqual(planning_request(request_input({'style': 'Banjo breakdown'}))['lyrics'], PLANNING_TAGS)
        render, budget, _, _, notes = fixed_score_render(inp, cover_options(raw), PLANNED, 'YuE2')
        self.assertEqual(render['cot'], 'full')
        self.assertEqual(render['lyrics'], TAGS)
        self.assertEqual(budget, semantic_budget(score_facts(PLANNED)['score_seconds']))
        self.assertIn('guided', notes[0])

    def test_style_prefix_and_suffix_match_upstream(self):
        upstream = ('Instrumental, expressive chamber string ensemble, lyrical violin melody, warm viola and cello '
                    'accompaniment, G major, 96 BPM, a repeated opening phrase followed by a short gentle ending, '
                    'natural concert hall acoustics, no vocals, no singing, no choir, no spoken words.')
        self.assertEqual(instrumental_style(upstream), upstream)
        self.assertEqual(instrumental_style('Warm soul ballad.'), 'Instrumental, Warm soul ballad' + NO_SINGING)
        self.assertEqual(instrumental_style('instrumental jazz trio, no vocals'),
                         'instrumental jazz trio, no vocals, no singing, no choir, no spoken words.')

    def test_unmovable_melody_is_a_plain_error_not_a_fallback(self):
        raw = dict(BOOTH_COVER, instrumental=True)
        broken = TRANSCRIBED.replace('K:Bb', 'K:A#')
        with self.assertRaises(ValueError) as caught:
            fixed_score_render(request_input(raw), cover_options(raw), broken, 'recording', 72.6)
        self.assertEqual(str(caught.exception), UNMOVABLE['recording'])


class LengthAndTempoTest(unittest.TestCase):
    def test_budget_follows_the_report_with_headroom_and_caps(self):
        self.assertEqual(semantic_budget(60), 1650)
        self.assertEqual(semantic_budget(72.6, 75.9), 2088)
        self.assertEqual(semantic_budget(None, 100), 2750)
        self.assertEqual(semantic_budget(400), 9000)
        self.assertEqual(semantic_budget(3), 200)
        self.assertIsNone(semantic_budget())
        self.assertIsNone(semantic_budget(None, None))

    def test_budget_reaches_the_pipeline(self):
        _, args = render_song(Pipeline(), {'seed': 1}, sound_controls({}), 2088)
        self.assertEqual(args['semantic_sampling'], {'temperature': 1.0, 'max_tokens': 2088})

    def test_guarded_cover_gets_a_budget(self):
        # PLANNED stands in for SheetSage2's full transcription: melody plus chord symbols.
        raw = dict(BOOTH_COVER, keep_harmony=True)
        render, budget, _, _, notes = fixed_score_render(request_input(raw), cover_options(raw), PLANNED, 'recording', 72.6)
        self.assertEqual(render['cot'], 'full')
        self.assertEqual(budget, semantic_budget(72.6, 72.86))
        self.assertEqual(notes, [])
        raw = dict(BOOTH_COVER, match_score_tempo=True)
        _, budget, _, _, _ = fixed_score_render(request_input(raw), cover_options(raw), TRANSCRIBED, 'recording', 72.6)
        self.assertEqual(budget, 2088)

    def test_tempo_names_the_score(self):
        self.assertEqual(tempo_style('Soul ballad, 120 BPM, warm', 83), 'Soul ballad, 83 BPM, warm')
        self.assertEqual(tempo_style('90bpm groove', 83), '83 BPM groove')
        self.assertEqual(tempo_style('Soul ballad.', 83), 'Soul ballad, 83 BPM')
        raw = dict(BOOTH_COVER, style='Soul ballad at 120 BPM', match_score_tempo=True)
        render, _, _, _, _ = fixed_score_render(request_input(raw), cover_options(raw), TRANSCRIBED, 'recording', 72.6)
        self.assertEqual(render['style'], 'Soul ballad at 83 BPM')
        self.assertEqual(render['cot'], 'melody')


class ChordCheckTest(unittest.TestCase):
    """Keeping chords needs chords. A cappella, or a voice sung into a phone, transcribes
    without any; upstream refuses cot=full for such a score, so it renders as the melody cover."""

    def test_score_has_chords(self):
        self.assertTrue(score_has_chords(PLANNED))
        self.assertFalse(score_has_chords(TRANSCRIBED))
        # A key label upstream's parser cannot read falls back to a plain look for chord symbols.
        self.assertTrue(score_has_chords(PLANNED.replace('K:Bb', 'K:A#')))
        self.assertFalse(score_has_chords(TRANSCRIBED.replace('K:Bb', 'K:A#')))
        self.assertFalse(score_has_chords(''))

    def test_recording_without_chords_renders_as_the_melody_cover(self):
        raw = dict(BOOTH_COVER, keep_harmony=True)
        inp = request_input(raw)
        render, budget, facts, extras, notes = fixed_score_render(inp, cover_options(raw), TRANSCRIBED, 'recording', 72.6)
        self.assertEqual(render, dict(inp, abc=TRANSCRIBED, cot='melody'))
        self.assertEqual(notes, [NO_CHORDS['recording']])
        self.assertEqual(budget, 2088)  # the length guard still applies
        self.assertIsNotNone(extras['lyric_fit'])
        self.assertEqual(facts['score_bpm'], 83)

    def test_score_without_chords_renders_from_its_melody(self):
        raw = {'style': 'pop', 'lyrics': 'la la', 'abc': TRANSCRIBED, 'keep_harmony': True}
        inp = request_input(raw)
        render, _, _, _, notes = fixed_score_render(inp, cover_options(raw), inp['abc'], 'score')
        self.assertEqual(render['cot'], 'melody')
        self.assertEqual(notes, [NO_CHORDS['score']])
        raw['abc'] = PLANNED
        inp = request_input(raw)
        render, _, _, _, notes = fixed_score_render(inp, cover_options(raw), inp['abc'], 'score')
        self.assertEqual(render['cot'], 'full')
        self.assertEqual(notes, [])

    def test_unreadable_scores_never_fail_a_sung_cover(self):
        raw = dict(BOOTH_COVER, keep_harmony=True)
        for abc, cot in ((PLANNED.replace('K:Bb', 'K:A#'), 'full'), (TRANSCRIBED.replace('K:Bb', 'K:A#'), 'melody')):
            with self.subTest(cot=cot):
                render, _, facts, _, _ = fixed_score_render(request_input(raw), cover_options(raw), abc, 'recording', 72.6)
                self.assertEqual(render['cot'], cot)
                self.assertEqual(facts, {})

    def test_chordless_instrumental_still_uses_melody(self):
        raw = dict(BOOTH_COVER, instrumental=True, keep_harmony=True)
        render, _, _, extras, notes = fixed_score_render(request_input(raw), cover_options(raw), TRANSCRIBED, 'recording', 72.6)
        self.assertEqual(render['cot'], 'melody')
        self.assertFalse(extras['transfer']['chords_kept'])
        self.assertIn(NO_CHORDS['recording'], notes)

    def test_feature_is_advertised(self):
        self.assertIn('chord-check', FEATURES)


class ScoreReportTest(unittest.TestCase):
    def test_sections_tempo_key_and_length(self):
        facts = score_facts(PLANNED)
        self.assertEqual((facts['score_bpm'], facts['score_musical_key'], facts['score_meter']), (84, 'Bb', '3/4'))
        self.assertEqual(facts['score_seconds'], 72.86)
        self.assertEqual([(s['name'], s['bars'], s['sung_notes']) for s in facts['sections']],
                         [('intro', 8, 5), ('verse', 18, 52), ('outro', 8, 0)])
        self.assertEqual(score_facts('not a score'), {})

    def test_lyric_fit_reports_without_blocking(self):
        sections = score_facts(TRANSCRIBED)['sections']
        fit = lyric_fit('[Verse 1]\nHold on to me, hold on tight\nAll the way home tonight\n[Chorus]\nLa', sections)
        self.assertEqual(fit['sections'][0]['score_section'], 'verse')
        self.assertEqual(fit['sections'][0]['sung_notes'], 56)
        self.assertEqual(fit['sections'][0]['fit'], 'short')
        self.assertEqual(fit['sections'][1]['fit'], 'no tune')
        self.assertFalse(fit['same_order'])
        self.assertTrue(lyric_fit('[Verse]\n' + 'la ' * 50, sections)['same_order'])
        self.assertIsNone(lyric_fit('', sections))
        self.assertIsNone(lyric_fit('words', []))


def long_score(groups, bpm=86):
    """An invented native score of four-bar 4/4 groups, 16 quarter notes each, with chords;
    sections change every four groups. 36 groups at 86 BPM run 401.86 s, a 6:40 recording's."""
    head = ['X:1', 'T:', 'M:4/4', 'L:1/8', f'Q:1/4={bpm}', 'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
            'V: Ins clef=treble name="Ins Melody" snm="Inst."', 'K:C']
    body = []
    for group in range(groups):
        if group % 4 == 0:
            body.append('% ' + ('verse', 'chorus')[(group // 4) % 2])
        body += ['V: Vocal', '"C"c2d2e2f2|"G"g4z4|"F"e2d2c2d2|"C"e4z4|', 'V: Ins', 'Z4|']
    return '\n'.join(head + body) + '\n'


LONG = long_score(36)
LONG_COVER = dict(BOOTH_COVER, keep_harmony=True, length_guard=True, fit_tempo=True)
SPED_NOTE = "Sped up 15% to fit YuE2's six-minute limit (86 to 99 BPM), in the same key; the notes are unchanged."


def fetch(seconds, limit):
    """fetch_recording with the download and decoder faked: the file measures `seconds`."""
    fakes = {'soundfile': types.SimpleNamespace(info=lambda path: types.SimpleNamespace(duration=seconds)),
             'auk_handler': types.SimpleNamespace(ffmpeg=lambda *args: None, download=lambda url, dest: None)}
    with mock.patch.dict(sys.modules, fakes):
        return fetch_recording('https://example.invalid/original.mp3', '/tmp', limit)


class FitTempoTest(unittest.TestCase):
    """fit_tempo: a score too long for YuE2's six minutes is sung a little faster instead of cut."""

    def test_the_option_is_checked_and_todays_requests_ignore_it(self):
        self.assertIn('fit-tempo', FEATURES)
        self.assertFalse(fit_option({}))
        self.assertFalse(fit_option(BOOTH_COVER))
        self.assertFalse(fit_option({'fit_tempo': False}))
        self.assertTrue(fit_option({'fit_tempo': True}))
        for bad in ('yes', 1, 'true', [True]):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'fit_tempo must be true or false'):
                fit_option({'fit_tempo': bad})
            self.assertIn('fit_tempo', handler({'input': dict(BOOTH_COVER, fit_tempo=bad)})['error'])
        self.assertEqual((source_limit(False), source_limit(True)), (360, 402))
        # A new field, so it turns the length guard on unless told; without it nothing changes.
        self.assertTrue(cover_options({'fit_tempo': True})['length_guard'])
        self.assertFalse(cover_options({'fit_tempo': True, 'length_guard': False})['length_guard'])
        self.assertEqual(cover_options({'fit_tempo': False}), CoverOptionsTest.TODAY)
        song = request_input({'style': 'pop', 'lyrics': 'la'})
        with self.assertRaisesRegex(ValueError, 'needs a source recording, a score or an instrumental'):
            check_fit(song, cover_options({}), None, True)
        check_fit(song, cover_options({'instrumental': True}), None, True)
        check_fit(song, cover_options({}), 'https://example.invalid/a.mp3', True)
        check_fit(song, cover_options({}), None, False)

    def test_without_the_field_nothing_changes(self):
        for abc, seconds in ((TRANSCRIBED, 72.6), (LONG, 399.0)):
            raw = dict(BOOTH_COVER, keep_harmony=True, length_guard=True)
            inp = request_input(raw)
            render, budget, facts, extras, notes = fixed_score_render(inp, cover_options(raw), abc, 'recording', seconds)
            self.assertEqual(render['abc'], abc)
            self.assertNotIn('tempo_fit', extras)
            self.assertNotIn('take_speed', extras)
            self.assertEqual(budget, semantic_budget(seconds, score_facts(abc)['score_seconds']))
            self.assertEqual(fixed_score_render(inp, cover_options(raw), abc, 'recording', seconds, fit_tempo=False)[:3],
                             (render, budget, facts))

    def test_a_score_that_fits_is_untouched(self):
        raw = dict(BOOTH_COVER, keep_harmony=True, length_guard=True)
        inp = request_input(raw)
        plain = fixed_score_render(inp, cover_options(raw), PLANNED, 'recording', 72.6)
        fitted = fixed_score_render(inp, cover_options(dict(raw, fit_tempo=True)), PLANNED, 'recording', 72.6,
                                    fit_tempo=True)
        self.assertEqual(fitted[:3], plain[:3])
        self.assertEqual(fitted[4], plain[4])
        self.assertEqual(fitted[3]['tempo_fit'], dict(
            applied=False, limit_seconds=360, fit_seconds=352, source_seconds=72.6, factor=1.0, from_bpm=84, to_bpm=84,
            percent=0, score_seconds_before=72.86, score_seconds_after=72.86, style_bpm=[]))
        self.assertNotIn('take_speed', fitted[3])

    def test_a_long_score_is_sped_up_just_enough_in_its_tempo_line_only(self):
        inp = request_input(LONG_COVER)
        render, budget, facts, extras, notes = fixed_score_render(inp, cover_options(LONG_COVER), LONG, 'recording',
                                                                  399.0, fit_tempo=True)
        before, after = LONG.splitlines(), render['abc'].splitlines()
        self.assertEqual([k for k in range(len(before)) if before[k] != after[k]], [4])
        self.assertEqual(after[4], 'Q:1/4=99')
        self.assertEqual(len(after), len(before))
        self.assertTrue(compare(parse_abc(LONG), parse_abc(render['abc']), allow_tempo_change=True)['match'])
        self.assertFalse(compare(parse_abc(LONG), parse_abc(render['abc']))['match'])
        self.assertEqual((facts['score_bpm'], facts['score_seconds']), (99, 349.09))
        self.assertLessEqual(facts['score_seconds'], 352)
        self.assertEqual(sum(s['seconds'] for s in facts['sections']) // 1, 349)   # sections describe the take
        self.assertEqual(budget, 9000)
        self.assertEqual(render['cot'], 'full')
        self.assertEqual(render['style'], 'Warm soul ballad')
        self.assertEqual(extras['take_speed'], 99 / 86)
        self.assertEqual(extras['tempo_fit'], dict(
            applied=True, limit_seconds=360, fit_seconds=352, source_seconds=399.0, factor=1.1512, from_bpm=86,
            to_bpm=99, percent=15, score_seconds_before=401.86, score_seconds_after=349.09, style_bpm=[]))
        self.assertEqual(notes, [SPED_NOTE])
        # The sped score and the six-minute budget reach the pipeline.
        _, args = render_song(Pipeline(), render, sound_controls(LONG_COVER), budget)
        self.assertEqual(args['abc'].splitlines()[4], 'Q:1/4=99')
        self.assertEqual(args['semantic_sampling']['max_tokens'], 9000)
        # Asked to leave the length guard off, the score is still sped and YuE2 keeps its own 9,000.
        raw = dict(LONG_COVER, length_guard=False)
        render, budget, _, _, _ = fixed_score_render(request_input(raw), cover_options(raw), LONG, 'recording', 399.0,
                                                     fit_tempo=True)
        self.assertIsNone(budget)
        self.assertEqual(render['abc'].splitlines()[4], 'Q:1/4=99')

    def test_worked_examples_and_the_limit(self):
        # Four-bar groups at 86 BPM: 30 fit; 32, 34 and 36 need 2%, 8% and 15%; 38 would need 21%.
        for groups, bpm, percent in ((30, 86, 0), (32, 88, 2), (34, 93, 8), (36, 99, 15)):
            with self.subTest(groups=groups):
                plan = tempo_plan(long_score(groups))
                self.assertEqual(plan['to_bpm'], bpm)
                self.assertLessEqual(plan['after'], 352)
                if percent:  # the smallest whole tempo that fits: one BPM slower would not
                    self.assertGreater(Fraction(groups * 16 * 60, bpm - 1), 352)
                raw = dict(LONG_COVER)
                _, _, _, extras, _ = fixed_score_render(request_input(raw), cover_options(raw), long_score(groups),
                                                        'recording', None, fit_tempo=True)
                self.assertEqual(extras['tempo_fit']['percent'], percent)
        with self.assertRaises(ValueError) as caught:
            fixed_score_render(request_input(LONG_COVER), cover_options(LONG_COVER), long_score(38), 'recording', 420.0,
                               fit_tempo=True)
        self.assertEqual(str(caught.exception), "This song's score runs 7 minutes 4 seconds. Fitting it into YuE2's six "
                         'minutes would mean speeding it up 21%, more than the 20% a cover allows. Import a shorter '
                         'recording or an excerpt; nothing was trimmed.')
        with self.assertRaisesRegex(ValueError, 'Use a shorter score; nothing was trimmed.'):
            tempo_plan(long_score(38), None, 'score')
        with self.assertRaisesRegex(ValueError, 'Try another seed; nothing was trimmed.'):
            tempo_plan(long_score(38), None, 'YuE2')
        # Just under 20% and exactly 20% faster are allowed.
        self.assertEqual(tempo_plan(long_score(36, bpm=83))['factor'], Fraction(99, 83))
        self.assertEqual(tempo_plan(long_score(22, bpm=50))['factor'], Fraction(60, 50))

    def test_a_given_score_past_the_limit_is_refused_before_any_gpu_work(self):
        # The handler answers before importing runpod or loading YuE2: this test has neither.
        raw = {'style': 'pop', 'lyrics': 'la la', 'abc': long_score(38), 'cot': 'melody', 'fit_tempo': True}
        answer = handler({'input': raw})
        self.assertIn('speeding it up 21%', answer['error'])
        self.assertIn('Use a shorter score', answer['error'])

    def test_the_style_follows_the_new_tempo(self):
        raw = dict(LONG_COVER, match_score_tempo=True)
        render, *_ = fixed_score_render(request_input(raw), cover_options(raw), LONG, 'recording', 399.0, fit_tempo=True)
        self.assertEqual(render['style'], 'Warm soul ballad, 99 BPM')
        raw = dict(LONG_COVER, style='Soul ballad at 86 BPM, a slow 86bpm groove, 70.5 bpm hi-hats')
        render, _, _, extras, notes = fixed_score_render(request_input(raw), cover_options(raw), LONG, 'recording', 399.0,
                                                         fit_tempo=True)
        self.assertEqual(render['style'], 'Soul ballad at 99 BPM, a slow 99bpm groove, 81 bpm hi-hats')
        self.assertEqual(extras['tempo_fit']['style_bpm'], [[86, 99], [86, 99], [70.5, 81]])
        self.assertEqual(notes, [SPED_NOTE + ' The style named 86 BPM, so it now names 99 BPM to match. The style '
                                 'named 70.5 BPM, so it now names 81 BPM to match.'])
        self.assertEqual(scale_style_bpm('No tempo named', Fraction(99, 86)), ('No tempo named', []))

    def test_instrumentals_and_given_scores_are_sped_the_same_way(self):
        for raw, origin in (({'style': 'Piano trio', 'abc': LONG, 'instrumental': True, 'fit_tempo': True}, 'score'),
                            ({'style': 'Piano trio', 'instrumental': True, 'fit_tempo': True}, 'YuE2'),
                            ({'style': 'pop', 'lyrics': 'la la', 'abc': LONG, 'cot': 'full', 'fit_tempo': True}, 'score')):
            with self.subTest(origin=origin, instrumental=raw.get('instrumental')):
                render, budget, facts, extras, notes = fixed_score_render(request_input(raw), cover_options(raw), LONG,
                                                                          origin, fit_tempo=True)
                sped = parse_abc(render['abc'])
                self.assertEqual(sped.bpm, 99)
                self.assertEqual(budget, 9000)
                self.assertEqual(extras['tempo_fit']['percent'], 15)
                self.assertIn(SPED_NOTE, notes)
                if raw.get('instrumental'):
                    self.assertEqual(sped.voices['Vocal'].notes, [])
                    self.assertEqual(sped.voices['Ins'].notes, parse_abc(LONG).voices['Vocal'].notes)
                else:
                    self.assertEqual(render['abc'], with_tempo(LONG, 99))

    def test_source_limits_with_and_without_the_field(self):
        self.assertEqual(fetch(360.0, SOURCE_MAX)[2], 360.0)
        with self.assertRaisesRegex(ValueError, '^Use a source recording up to six minutes long.$'):
            fetch(390.0, SOURCE_MAX)
        self.assertEqual(fetch(390.0, FIT_SOURCE_MAX)[2], 390.0)
        self.assertEqual(fetch(401.9, FIT_SOURCE_MAX)[2], 401.9)
        with self.assertRaisesRegex(ValueError, '^Use a source recording up to 6 minutes 40 seconds long.$'):
            fetch(402.5, FIT_SOURCE_MAX)

    def test_an_unreadable_score_is_never_guessed_at(self):
        broken = LONG.replace('K:C', 'K:H')
        self.assertIsNone(tempo_plan(broken, 300.0))
        raw = dict(LONG_COVER)
        render, _, _, extras, notes = fixed_score_render(request_input(raw), cover_options(raw), broken, 'recording', 300.0,
                                                         fit_tempo=True)
        self.assertEqual(render['abc'], broken)
        self.assertEqual(extras['tempo_fit'], dict(applied=False, limit_seconds=360, fit_seconds=352, source_seconds=300.0,
                                                   reason='score unreadable'))
        self.assertEqual(notes, [])
        with self.assertRaisesRegex(ValueError, 'runs over six minutes, and its melody score could not be read'):
            tempo_plan(broken, 380.0)

    def test_tempo_line_helpers(self):
        self.assertEqual(score_tempo(LONG), 86)
        self.assertIsNone(score_tempo('not a score'))
        self.assertEqual(with_tempo(LONG.replace('\n', '\r\n'), 99), with_tempo(LONG, 99).replace('\n', '\r\n'))
        with self.assertRaises(ValueError):
            with_tempo('X:1\nT:\n', 99)
        self.assertEqual((spoken_length(424.19), spoken_length(400), spoken_length(360), spoken_length(61)),
                         ('7 minutes 4 seconds', '6 minutes 40 seconds', '6 minutes', '1 minute 1 second'))
        self.assertEqual(yue_handler.FIT_HARD, Fraction(6, 5))


class VendoredHelpersTest(unittest.TestCase):
    # Upstream YuE ab2e5a3 skills/yue2-music/instrumental, as listed in its bundle-manifest.json.
    UPSTREAM = {
        'abc_tools.py': 'ea04b922dacebec7ad257a2f8d83bdb5dfecb7a23110c1a3121c5c41c313930e',
        'common.py': '2929619bd37a176c87c37fdcde9864613d4d5172ca2d89d724aa450a4d7053be',
        'compile_score.py': 'c4eec7f68e3f09b93b45d568a8af640b3f10bb47d31ef6d3b6ce4ae23ac1dea3',
        'instrumentalize.py': 'a769111656f418cf4298841dc4a7f6be1a50bfae1e66b1bccfaca3eacf948681',
        'LICENSE': '689d887cb61b76599b8bded2066dbd4cd728d5345533925585ee1270ae7315d1',
    }

    def test_helpers_are_upstream_verbatim(self):
        folder = Path(__file__).resolve().parent / 'instrumental'
        for name, digest in self.UPSTREAM.items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((folder / name).read_bytes()).hexdigest(), digest)
        import abc_tools
        self.assertEqual(Path(abc_tools.__file__).resolve().parent, folder)


if __name__ == '__main__':
    unittest.main()


class ModelFolderTest(unittest.TestCase):
    """Sep 28 2026: a machine whose model cache holds another snapshot than the pinned one still finds the model."""

    def test_pinned_then_cached_then_missing(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            hub = Path(td) / 'hub'
            pinned = hub / 'models--m-a-p--SheetSage2' / 'snapshots' / ('a' * 40)
            other = hub / 'models--m-a-p--SheetSage2' / 'snapshots' / ('b' * 40)
            other.mkdir(parents=True)
            (other / 'config.json').write_text('{}')
            folder, how = yue_handler.model_folder(pinned, 'm-a-p/SheetSage2', roots=[hub])
            self.assertEqual((folder, how), (other, 'cached ' + 'b' * 12))
            pinned.mkdir(parents=True)
            self.assertEqual(yue_handler.model_folder(pinned, 'm-a-p/SheetSage2', roots=[hub])[1], 'cached ' + 'b' * 12, 'an empty pinned folder is not the model')
            (pinned / 'config.json').write_text('{}')
            self.assertEqual(yue_handler.model_folder(pinned, 'm-a-p/SheetSage2', roots=[hub]), (pinned, 'pinned'))
            self.assertEqual(yue_handler.model_folder(hub / 'models--m-a-p--MERT-v2-FullSong' / 'snapshots' / ('c' * 40), 'm-a-p/MERT-v2-FullSong', roots=[hub]), (None, 'missing'))

    def test_resolve_points_the_environment_at_the_folder_found(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            hub = Path(td) / 'hub'
            have = hub / 'models--m-a-p--MERT-v2-FullSong' / 'snapshots' / ('d' * 40)
            have.mkdir(parents=True)
            (have / 'config.json').write_text('{}')
            pinned = str(hub / 'models--m-a-p--MERT-v2-FullSong' / 'snapshots' / ('e' * 40))
            env = {'YUE_MERT_DIR': pinned, 'HF_HUB_CACHE': str(hub)}
            with mock.patch.dict(os.environ, env, clear=False), mock.patch.dict(yue_handler.RESOLVED, {'done': False}), \
                    mock.patch.object(yue_handler, 'MODELS', (('YUE_MERT_DIR', 'm-a-p/MERT-v2-FullSong'),)):
                yue_handler.resolve_models(download=False)
                self.assertEqual(os.environ['YUE_MERT_DIR'], str(have))
                self.assertTrue(yue_handler.RESOLVED['done'])
