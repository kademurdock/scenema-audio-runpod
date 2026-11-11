"""Nov 10 2026, round two, CPU only: a length ceiling for a song with no score (max_seconds), a
cover that names its score's key (match_score_key) and a cover whose lyric tags take the score's
own section names (score_section_tags). No model, GPU or network.

Every field is optional; the first class proves that a request without them is built exactly as
it was before."""
import unittest

import yue_handler as yue
from test_yue_controls import BOOTH_COVER, PLANNED, TRANSCRIBED


def sections(*names, notes=24):
    return [dict(name=name, bars=4, seconds=10.0, sung_notes=notes) for name in names]


class OldRequestsTest(unittest.TestCase):
    def test_a_request_without_the_new_fields_asks_for_nothing_new(self):
        for raw in ({}, BOOTH_COVER, {'style': 'pop', 'lyrics': 'la', 'cot': 'off'}):
            self.assertIsNone(yue.length_cap(raw))
            self.assertEqual(yue.score_options(raw), dict(match_score_key=False, score_section_tags=False))

    def test_no_ceiling_leaves_every_budget_as_it_was(self):
        for budget in (None, 200, 2088, 9000):
            self.assertEqual(yue.capped_budget(budget, None), budget)

    def test_a_fixed_score_renders_the_same_request_with_and_without_the_empty_options(self):
        for raw, abc, origin, seconds in ((BOOTH_COVER, TRANSCRIBED, 'recording', 72.6),
                                          (dict(BOOTH_COVER, keep_harmony=True, match_score_tempo=True, length_guard=True),
                                           TRANSCRIBED, 'recording', 72.6),
                                          ({'style': 'pop, 84 BPM', 'lyrics': '[Verse]\nla la', 'abc': PLANNED, 'cot': 'full'},
                                           PLANNED, 'score', None)):
            inp, options = yue.request_input(raw), yue.cover_options(raw)
            before = yue.fixed_score_render(inp, options, abc, origin, seconds)
            # nothing names the score's key or renames a tag unless it was asked for
            key = yue.key_words(before[2]['score_musical_key'])
            self.assertIsNotNone(key)
            self.assertNotIn(key, before[0]['style'])
            self.assertEqual(before[0]['lyrics'], inp['lyrics'])
            self.assertNotIn('score_tags', before[3])
            for more in (None, {}, yue.score_options(raw)):
                self.assertEqual(yue.fixed_score_render(inp, options, abc, origin, seconds, more=more), before)

    def test_the_worker_says_what_it_can_do(self):
        self.assertLessEqual({'length-cap', 'match-key', 'score-tags'}, set(yue.FEATURES))


class LengthCapTest(unittest.TestCase):
    def test_only_a_number_of_seconds_from_one_to_six_minutes(self):
        self.assertEqual(yue.length_cap({'max_seconds': 222}), 222.0)
        self.assertEqual(yue.length_cap({'max_seconds': 60}), 60.0)
        self.assertEqual(yue.length_cap({'max_seconds': 287.5}), 287.5)
        for bad in (59, 361, '222', True, float('nan'), float('inf'), [222], -1):
            with self.assertRaises(ValueError, msg=repr(bad)):
                yue.length_cap({'max_seconds': bad})

    def test_the_ceiling_is_25_tokens_a_second(self):
        self.assertEqual(yue.capped_budget(None, 222.0), 5550)
        self.assertEqual(yue.capped_budget(None, 60.0), 1500)

    def test_the_smaller_limit_wins(self):
        self.assertEqual(yue.capped_budget(5000, 222.0), 5000)
        self.assertEqual(yue.capped_budget(6000, 222.0), 5550)

    def test_a_six_minute_ceiling_is_yue2s_own_limit_and_sets_nothing(self):
        self.assertIsNone(yue.capped_budget(None, 360.0))
        self.assertEqual(yue.capped_budget(4000, 360.0), 4000)

    def test_the_budget_reaches_the_sampler_as_max_tokens(self):
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Config:
            ode_steps: int = 32

        class Pipe:
            generation_config = Config()

            def __call__(self, **kwargs):
                return kwargs
        song = {'style': 'pop', 'lyrics': '[verse]\nla la', 'abc': None, 'cot': 'off', 'seed': 7}
        controls = dict(weirdness=50, steps=32, guidance=1.0)
        sent = yue.render_song(Pipe(), song, controls, yue.capped_budget(None, 222.0))
        self.assertEqual(sent['semantic_sampling']['max_tokens'], 5550)
        self.assertNotIn('max_tokens', yue.render_song(Pipe(), song, controls, yue.capped_budget(None, None))['semantic_sampling'])


class KeyTest(unittest.TestCase):
    def test_key_names(self):
        self.assertEqual(yue.key_words('Fm'), 'F minor')
        self.assertEqual(yue.key_words('C'), 'C major')
        self.assertEqual(yue.key_words('Bb'), 'B-flat major')
        self.assertEqual(yue.key_words('C#m'), 'C-sharp minor')
        for odd in (None, '', 'Dmix', 'H', 'Cmaj7', 'c'):
            self.assertIsNone(yue.key_words(odd))

    def test_the_key_goes_before_a_tempo_that_ends_the_style(self):
        self.assertEqual(yue.key_style('soul, female lead vocal, 97 BPM', 'Fm'), 'soul, female lead vocal, F minor, 97 BPM')
        self.assertEqual(yue.key_style('soul, female lead vocal, 97 BPM.', 'Fm'), 'soul, female lead vocal, F minor, 97 BPM.')
        self.assertEqual(yue.key_style('Warm soul ballad.', 'Bb'), 'Warm soul ballad, B-flat major')

    def test_a_style_that_names_a_key_is_left_alone(self):
        for style in ('slow jam in A minor, 80 BPM', 'G major folk', 'in the key of D, strings', 'Bb major swing',
                      'C-sharp minor ballad'):
            self.assertEqual(yue.key_style(style, 'Fm'), style)

    def test_an_unreadable_key_changes_nothing(self):
        self.assertEqual(yue.key_style('soul, 97 BPM', 'Dmix'), 'soul, 97 BPM')

    def test_a_cover_names_its_scores_key_and_tempo_together(self):
        raw = dict(BOOTH_COVER, keep_harmony=True, match_score_tempo=True, length_guard=True, match_score_key=True)
        inp, options = yue.request_input(raw), yue.cover_options(raw)
        render, _, facts, _, _ = yue.fixed_score_render(inp, options, TRANSCRIBED, 'recording', 72.6,
                                                        more=yue.score_options(raw))
        words = yue.key_words(facts['score_musical_key'])
        self.assertIsNotNone(words)
        self.assertEqual(render['style'], f"Warm soul ballad, {words}, {facts['score_bpm']} BPM")
        self.assertEqual(render['lyrics'], inp['lyrics'])

    def test_her_own_key_wins(self):
        raw = dict(BOOTH_COVER, style='Warm soul ballad in E minor', match_score_key=True)
        inp, options = yue.request_input(raw), yue.cover_options(raw)
        render, _, _, _, _ = yue.fixed_score_render(inp, options, TRANSCRIBED, 'recording', 72.6, more=yue.score_options(raw))
        self.assertEqual(render['style'], 'Warm soul ballad in E minor')

    def test_flags_must_be_true_or_false(self):
        for key in ('match_score_key', 'score_section_tags'):
            with self.assertRaises(ValueError):
                yue.score_options({key: 'yes'})


class ScoreTagsTest(unittest.TestCase):
    SHEET = '[Verse 1]\nPaper boats along the drain\n\n[Verse 2]\nCarry all the rain\n\n[bridge]\nHold the line'

    def test_tags_take_the_scores_names_when_the_counts_match(self):
        sheet, changed = yue.tags_from_score(self.SHEET, sections('verse', 'chorus', 'verse'))
        self.assertEqual(changed, 2)
        self.assertEqual(sheet, '[Verse 1]\nPaper boats along the drain\n\n[Chorus]\nCarry all the rain\n\n[verse]\nHold the line')

    def test_every_sung_word_and_line_is_untouched(self):
        sheet, _ = yue.tags_from_score(self.SHEET, sections('intro', 'chorus', 'outro'))
        strip = lambda text: [line for line in text.split('\n') if not line.startswith('[')]
        self.assertEqual(strip(sheet), strip(self.SHEET))

    def test_a_different_number_of_sections_leaves_the_sheet_alone(self):
        for names in (('verse', 'chorus'), ('verse', 'chorus', 'verse', 'chorus'), ()):
            self.assertEqual(yue.tags_from_score(self.SHEET, sections(*names)), (self.SHEET, 0))

    def test_words_before_any_tag_or_an_unnamed_score_section_leave_the_sheet_alone(self):
        loose = 'An opening line\n' + self.SHEET
        self.assertEqual(yue.tags_from_score(loose, sections('verse', 'chorus', 'verse')), (loose, 0))
        self.assertEqual(yue.tags_from_score(self.SHEET, sections('verse', None, 'verse')), (self.SHEET, 0))

    def test_a_tag_with_no_words_under_it_is_skipped(self):
        sheet = '[Intro]\n\n[Verse]\nla la\n\n[Chorus]\nla la'
        got, changed = yue.tags_from_score(sheet, sections('chorus', 'chorus'))
        self.assertEqual((got, changed), ('[Intro]\n\n[Chorus]\nla la\n\n[Chorus]\nla la', 1))

    def test_matching_names_change_nothing(self):
        sheet = '[Verse 1]\nla la\n\n[Pre-Chorus]\nla\n\n[Chorus]\nla la'
        self.assertEqual(yue.tags_from_score(sheet, sections('verse', 'pre-chorus', 'chorus')), (sheet, 0))

    def test_only_a_cover_of_a_recording_asked_for_it_is_renamed(self):
        raw = dict(BOOTH_COVER, score_section_tags=True)
        inp, options = yue.request_input(raw), yue.cover_options(raw)
        facts = yue.score_facts(TRANSCRIBED)
        want, changed = yue.tags_from_score(inp['lyrics'], facts.get('sections', []))
        render, _, _, extras, notes = yue.fixed_score_render(inp, options, TRANSCRIBED, 'recording', 72.6,
                                                             more=yue.score_options(raw))
        self.assertEqual(render['lyrics'], want)
        self.assertEqual(extras['score_tags'], changed)
        self.assertEqual(any('section tag' in note for note in notes), bool(changed))
        as_score, _, _, extras, _ = yue.fixed_score_render(inp, options, TRANSCRIBED, 'score', None,
                                                           more=yue.score_options(raw))
        self.assertEqual(as_score['lyrics'], inp['lyrics'])
        self.assertNotIn('score_tags', extras)


class HandlerWiringTest(unittest.TestCase):
    """Review, Nov 10 2026: the helpers above were tested, but nothing proved that handler() hands a
    request's max_seconds, match_score_key and score_section_tags to them. test_yue_fixes.py runs
    whole takes through the handler inside the image; this reads the handler's own source, so the
    CPU step fails too if one of the three wires is cut or moved."""

    @classmethod
    def setUpClass(cls):
        import inspect
        cls.source = inspect.getsource(yue.handler)

    def order(self, *pieces):
        at = [self.source.find(piece) for piece in pieces]
        for piece, where in zip(pieces, at):
            self.assertGreater(where, -1, piece)
        self.assertEqual(at, sorted(at), pieces)

    def test_the_three_fields_are_read_from_the_raw_request_before_any_gpu_work(self):
        self.order('cap, more = length_cap(raw), score_options(raw)', 'pipe = engine()')

    def test_the_ceiling_is_applied_to_the_budget_the_render_is_given(self):
        self.order('render, budget, facts, extras = inp, None, {}, {}',
                   'budget = capped_budget(budget, cap)',
                   'result = render_song(pipe, render, controls, budget,')
        self.assertEqual(self.source.count('budget = capped_budget(budget, cap)'), 1)

    def test_the_score_options_reach_the_fixed_score_render(self):
        call = self.source[self.source.index('fixed_score_render('):]
        call = call[:call.index(')\n') + 1]
        self.assertIn('more=more', call)

    def test_the_take_reports_what_it_was_asked(self):
        self.order("output['max_seconds'] = cap", "output['score_options'] = dict(more, tags_renamed=extras.get('score_tags'))")


if __name__ == '__main__':
    unittest.main()
