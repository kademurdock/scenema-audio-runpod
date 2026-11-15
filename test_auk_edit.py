"""Word and lyric edit rules. CPU only; numpy for the splice checks."""
import unittest
from unittest.mock import patch

import auk_edit as e
import auk_handler as worker


class TemplateTests(unittest.TestCase):
    def test_templates_match_upstream_pe_config(self):
        # Tencent-Hunyuan/AuK pe.config.yaml: content_edit replace, vocal_edit.
        self.assertEqual(e.build_instruction('speech', 'today', 'tomorrow'), "Replace 'today' with 'tomorrow'.")
        self.assertEqual(e.build_instruction('vocal', 'rear view', 'like you'),
                         'Change "rear view" to "like you" in the vocal recording.')
        self.assertEqual(e.build_instruction('speech', '今天', '明天'), '把‘今天’改成‘明天’')
        self.assertEqual(e.build_instruction('vocal', '明天你好', '未来你好'), '把这段歌词中的“明天你好”改成“未来你好”。')

    def test_words_are_tidied_and_checked(self):
        self.assertEqual(e.build_instruction('speech', '  see\n you ', 'bye'), "Replace 'see you' with 'bye'.")
        for kind, old, new in (('speech', '', 'x'), ('speech', 'x', ' '), ('speech', 'Same', 'same'),
                               ('song', 'a', 'b'), ('speech', 'a' * 201, 'b')):
            with self.subTest(kind=kind, old=old[:10], new=new), self.assertRaises(ValueError):
                e.build_instruction(kind, old, new)

    def test_free_form_single_change_is_mapped(self):
        cases = (
            ('Replace "Tuesday" with "Thursday".', 'speech', 'Tuesday', 'Thursday'),
            ("replace 'a cat' with 'a dog'", 'speech', 'a cat', 'a dog'),
            ('Change “rear view” to “like you” in the vocal recording.', 'vocal', 'rear view', 'like you'),
            ('Sing "hello" instead of "goodbye".', 'vocal', 'goodbye', 'hello'),
            # The Booth's old starter: an extra sentence that only asks to keep the voice.
            ('Replace "Tuesday" with "Thursday". Preserve the speaker\'s voice, pace and room sound.',
             'speech', 'Tuesday', 'Thursday'),
        )
        for text, kind, old, new in cases:
            with self.subTest(text=text):
                self.assertEqual(e.parse_freeform(text), {'kind': kind, 'old': old, 'new': new})
        self.assertEqual(e.parse_freeform('Replace "a" with "b".', 'vocal')['kind'], 'vocal')

    def test_free_form_that_cannot_be_mapped_is_refused_plainly(self):
        multi = ('Replace "red" with "blue" and change "cat" to "dog".',
                 'Replace "red" with "blue". Then remove "very".',
                 'Change "a" to "b", "c" to "d", "e" to "f".')
        for text in multi:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, 'One word change per render'):
                e.parse_freeform(text)
        for text in ('Fix the lyrics in the chorus.', 'Make her say something nicer.',
                     'Change the words to be happier.', 'Swap "x" for "y".'):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, 'trained wording'):
                e.parse_freeform(text)

    def test_style_edits_are_left_to_the_older_path(self):
        for text in ('Make this a whisper.', 'Give this voice an Irish accent.', 'Change the emotion to happy.',
                     'Turn this voice into a man.', 'Remove the background noise.', ''):
            with self.subTest(text=text):
                self.assertIsNone(e.parse_freeform(text))


class LengthTests(unittest.TestCase):
    def test_spoken_duration_matches_pe_py(self):
        self.assertAlmostEqual(e.spoken_duration('like you'), 0.6)
        self.assertAlmostEqual(e.spoken_duration("don't stop"), 0.6)
        self.assertAlmostEqual(e.spoken_duration('明天你好'), 0.84)
        self.assertAlmostEqual(e.spoken_duration('hi 你好'), 0.72)
        self.assertEqual(e.spoken_duration(''), 0)

    def test_only_the_marked_range_is_scaled(self):
        # 1 s pads keep their length; the 2 s range scales by 1 word / 2 words.
        self.assertEqual(e.target_seconds(4.0, 2.0, 'rear view', 'you'), 3.0)
        # No range: upstream's whole-clip rule, base * new / old.
        self.assertEqual(e.target_seconds(6.0, 6.0, 'one two', 'one two three four'), 12.0)
        # With the clip's transcript: upstream's ASR rule.
        self.assertEqual(e.target_seconds(6.0, 2.0, 'two words', 'four new words here',
                                          source_text='one two three four five six seven eight nine ten'), 7.2)
        # Never collapses to nothing.
        self.assertEqual(e.target_seconds(0.4, 0.4, 'a b c d e f g h i j', 'k'), 0.5)

    def test_crop_bounds_pad_and_limits(self):
        self.assertEqual(e.crop_bounds(20, {'edit_start': 8, 'edit_end': 10}),
                         {'range_start': 8, 'range_end': 10, 'crop_start': 7, 'crop_end': 11})
        self.assertEqual(e.crop_bounds(20, {'edit_start': 0.5, 'edit_end': 25, 'pad_seconds': 0.25}),
                         {'range_start': 0.5, 'range_end': 20, 'crop_start': 0.25, 'crop_end': 20})
        self.assertEqual(e.crop_bounds(12, {})['crop_end'], 12)
        for total, inp in ((40, {}), (40, {'edit_start': 1, 'edit_end': 35}), (20, {'edit_start': 5, 'edit_end': 5.1}),
                           (20, {'edit_start': 21}), (20, {'edit_start': -1, 'edit_end': 3}),
                           (20, {'edit_start': 1, 'edit_end': 2, 'pad_seconds': 5}),
                           (20, {'edit_start': 'soon'}), (20, {'edit_end': float('nan')})):
            with self.subTest(total=total, inp=inp), self.assertRaises(ValueError):
                e.crop_bounds(total, inp)


class SeedTests(unittest.TestCase):
    def test_default_is_three_fresh_seeds_each_attempt(self):
        with patch.object(e.secrets, 'randbelow', side_effect=[100, 200]):
            first, second = e.edit_seeds({}), e.edit_seeds({})
        self.assertEqual(first, [100, 100 + 7919, 100 + 2 * 7919])
        self.assertEqual(second[0], 200)

    def test_explicit_seeds_and_counts(self):
        self.assertEqual(e.edit_seeds({'seeds': [4, 2**32 - 1]}), [4, 2**32 - 1])
        self.assertEqual(e.edit_seeds({'seed': 2**32 - 1, 'candidates': 2}), [2**32 - 1, 7918])
        for inp in ({'seeds': []}, {'seeds': [1, 1]}, {'seeds': [1.5]}, {'seeds': list(range(7))},
                    {'seeds': [True]}, {'candidates': 0}, {'candidates': 7}, {'seed': -1}, {'seed': '5'}):
            with self.subTest(inp=inp), self.assertRaises(ValueError):
                e.edit_seeds(inp)


class SpliceTests(unittest.TestCase):
    def test_edges_untouched_and_crossfade_sums_to_one(self):
        import numpy as np
        source = np.ones((1000, 2)) * 0.5
        generated = np.ones((300, 2)) * -0.25
        out = e.splice(source, generated, 400, 600, 50)
        self.assertEqual(out.shape, (1100, 2))
        np.testing.assert_array_equal(out[:400], source[:400])
        np.testing.assert_array_equal(out[700:], source[600:])
        np.testing.assert_array_equal(out[450:650], generated[50:250])
        fade_in = out[400:450, 0]
        self.assertTrue(np.all(np.diff(fade_in) <= 1e-12))  # 0.5 down to -0.25
        self.assertAlmostEqual(fade_in[0], 0.5)
        self.assertAlmostEqual(fade_in[-1], -0.25)
        fade_out = out[650:700, 0]
        self.assertAlmostEqual(fade_out[0], -0.25)
        self.assertAlmostEqual(fade_out[-1], 0.5)

    def test_no_fade_at_the_recording_ends(self):
        import numpy as np
        source = np.linspace(0, 1, 100)
        generated = np.full(80, 7.0)
        out = e.splice(source, generated, 0, 100, 10)
        np.testing.assert_array_equal(out, generated)
        out = e.splice(source, np.full(4, 7.0), 50, 52, 10)  # fade capped by the short take
        np.testing.assert_array_equal(out[:50], source[:50])
        self.assertEqual(out.shape, (102,))


class RankTests(unittest.TestCase):
    def test_score_prefers_new_words_and_penalises_old(self):
        good = e.score_transcript('I see you in my like you mirror', 'rear view', 'like you')
        old = e.score_transcript('I see you in my rear view mirror', 'rear view', 'like you')
        half = e.score_transcript('I see you in my like mirror', 'rear view', 'like you')
        self.assertEqual((good['new_words_heard'], good['old_words_heard'], good['matched']), (1.0, 0.0, True))
        self.assertFalse(old['matched'])
        self.assertLess(old['score'], half['score'])
        self.assertLess(half['score'], good['score'])
        self.assertTrue(e.score_transcript('明天上午开会', '今天下午', '明天上午')['matched'])
        self.assertTrue(e.score_transcript("Don’t stop", 'keep going', "don't stop")['matched'])

    def test_rank_is_stable_and_puts_unscored_last(self):
        takes = [{'seed': 1, 'score': 0.2}, {'seed': 2}, {'seed': 3, 'score': 0.9}, {'seed': 4, 'score': 0.2}]
        self.assertEqual([t['seed'] for t in e.rank_candidates(takes)], [3, 1, 4, 2])


class MemoryTests(unittest.TestCase):
    def test_24_gb_cards_run_without_offload(self):
        gib = 1024**3
        self.assertFalse(worker.use_cpu_offload(int(22.0 * gib), 'auto'))   # L4, A10, A5000
        self.assertFalse(worker.use_cpu_offload(int(23.6 * gib), 'auto'))   # RTX 4090, 3090
        self.assertFalse(worker.use_cpu_offload(int(44.4 * gib), 'auto'))   # A40, A6000
        self.assertTrue(worker.use_cpu_offload(int(15.6 * gib), 'auto'))    # 16 GB cards
        self.assertTrue(worker.use_cpu_offload(int(44 * gib), '1'))
        self.assertFalse(worker.use_cpu_offload(int(15 * gib), 'false'))

    def test_boot_load_and_first_job_share_one_load(self):
        import threading
        loads = []
        gate = threading.Event()

        def slow_load():
            loads.append(1)
            gate.wait(2)
            return object()

        with patch.object(worker, 'ENGINE', None), patch.object(worker, 'load_engine', slow_load), \
                patch.dict('os.environ', {'AUK_BOOT_DELAY': '0'}):
            boot = threading.Thread(target=worker.warm)
            boot.start()
            while not loads:
                pass
            got = []
            job = threading.Thread(target=lambda: got.append(worker.engine()))
            job.start()
            gate.set()
            boot.join(5)
            job.join(5)
            self.assertEqual(len(loads), 1)
            self.assertIs(got[0], worker.ENGINE)

    def test_failed_boot_load_is_retried_by_the_first_job(self):
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError('host not ready')
            return 'engine'

        with patch.object(worker, 'ENGINE', None), patch.object(worker, 'load_engine', flaky), \
                patch.dict('os.environ', {'AUK_BOOT_DELAY': '0'}):
            worker.warm()
            self.assertEqual(worker.engine(), 'engine')
        self.assertEqual(len(attempts), 2)


if __name__ == '__main__':
    unittest.main()
