"""Word and lyric edit rules. CPU only; numpy for the splice checks."""
import re
import unittest
from pathlib import Path
from unittest.mock import patch

import auk_contract as c
import auk_edit as e
import auk_handler as worker

# Tencent-Hunyuan/AuK 3253462 pe.config.yaml template_pools: emotion_edit, voice_edit and
# separate_speech by_content, English and Chinese, placeholders filled.
UPSTREAM_STYLE_TEMPLATES = (
    'Switch the emotion to happy.',
    'Change the emotion of this speech to happy.',
    'Convert the emotional tone to happy.',
    'Re-express this sentence with a happy emotion.',
    'Adjust the speaking emotion to happy.',
    'Turn the emotion into happy.',
    'Change the mood to happy.',
    'Render this speech in a happy emotion.',
    'Transform the emotion to happy.',
    'Speak this line with a happy feeling.',
    'Make this speech sound happy.',
    'Shift the emotional tone to happy.',
    'Replace the emotion in this audio with happy.',
    'Read this sentence again in a happy mood.',
    'Set the emotional tone of this clip to happy.',
    'Re-voice this line so it feels happy.',
    'Give this speech a happy emotion.',
    'Recast the emotion of this audio as happy.',
    'Modify the speaking emotion to happy.',
    'Make the overall mood happy.',
    'Express this content with a happy tone.',
    'Change the affect of this voice to happy.',
    'Switch the emotional color of this speech to happy.',
    'Deliver this line with a happy emotion.',
    'Alter the mood of the audio to happy.',
    'Reinterpret this speech with a happy feeling.',
    'Let this clip carry a happy emotion.',
    'Turn the speaking style happy.',
    'Re-perform this line in a happy emotional tone.',
    'Make the whole audio convey a happy mood.',
    '将情感切换成happy。',
    '把这段语音的情感转换为happy。',
    '请将这段话的情感改成happy。',
    '将说话情绪调整为happy。',
    '用happy的情感重新表达这段话。',
    '将情绪转变为happy。',
    '请把语气改为happy。',
    '让这段语音表现出happy的情绪。',
    '把情感换成happy的感觉。',
    '将这段音频重新演绎为happy的情感。',
    '请用happy的语气说这段话。',
    '把整段语音的情感调成happy。',
    '将这句话的情绪替换为happy。',
    '请将朗读的情感转为happy。',
    '把说话的感情改成happy。',
    '将语音情绪重新设定为happy。',
    '请让这段话带上happy的情感。',
    '把这段音频的情绪改造成happy。',
    '将整体情感基调换成happy。',
    '请以happy的情绪重新念这段话。',
    '把这段语音改成happy的腔调。',
    '将情感色彩调整为happy。',
    '请把这段话演绎成happy的情感。',
    '让说话语气切换到happy。',
    '把这段录音的情感变为happy。',
    '将这段话的感情基调改成happy。',
    '请用happy的方式表达这段内容。',
    '把语音的情绪风格改为happy。',
    '将这段话重新演绎,情感设为happy。',
    '请让整段音频呈现happy的情绪。',
    'Convert the voice in this audio to match the following description: "a deep, warm male voice".',
    'Keep the spoken content unchanged and change the timbre to: "a deep, warm male voice".',
    'Edit the voice of this speech to match: "a deep, warm male voice".',
    'Transform the speaker\'s timbre into: "a deep, warm male voice".',
    'Please change the vocal timbre to fit this description: "a deep, warm male voice".',
    'Modify the audio so the voice matches: "a deep, warm male voice".',
    'Replace the speaker timbre with the one described as: "a deep, warm male voice".',
    'Without altering the content, adjust the timbre to: "a deep, warm male voice".',
    'Re-render this speech using a voice that matches: "a deep, warm male voice".',
    'Change the timbre of this audio to: "a deep, warm male voice".',
    'Recreate this speech with a voice described as: "a deep, warm male voice".',
    'Keep the words the same, but switch the voice to: "a deep, warm male voice".',
    'Apply a timbre transfer so the voice becomes: "a deep, warm male voice".',
    'Make the speaker sound like: "a deep, warm male voice".',
    'Adjust the vocal characteristics to match: "a deep, warm male voice".',
    'Render the same content with a voice matching: "a deep, warm male voice".',
    'Swap the speaker\'s voice for one that fits: "a deep, warm male voice".',
    'Please reshape the timbre of this audio into: "a deep, warm male voice".',
    'Turn the voice in this clip into: "a deep, warm male voice".',
    'While preserving the script, change the timbre to: "a deep, warm male voice".',
    'Generate the same speech but with the voice: "a deep, warm male voice".',
    'Alter the speaking voice to match this description: "a deep, warm male voice".',
    'Re-voice this audio with a timbre described as: "a deep, warm male voice".',
    'Please edit the speaker timbre to: "a deep, warm male voice".',
    'Convert the speaking style of this audio to: "a deep, warm male voice".',
    'Give this speech a new voice that matches: "a deep, warm male voice".',
    'Change the vocal identity to fit: "a deep, warm male voice".',
    'Keep the content intact and morph the timbre into: "a deep, warm male voice".',
    'Synthesize this speech again with a timbre of: "a deep, warm male voice".',
    'Please convert the audio so the speaker sounds like: "a deep, warm male voice".',
    '请将这段音频的音色修改为符合以下描述的声音:"a deep, warm male voice"。',
    '保持说话内容不变,把音色转换成:"a deep, warm male voice"。',
    '将输入语音的音色改成这样的特征:"a deep, warm male voice"。',
    '把这段语音的嗓音替换为符合该描述的音色:"a deep, warm male voice"。',
    '请在不改变说话内容的前提下,将音色调整为:"a deep, warm male voice"。',
    '将该音频的音色编辑为:"a deep, warm male voice"。',
    '把说话人的音色修改成:"a deep, warm male voice",内容保持不变。',
    '请将音色转换为符合下述描述的声音:"a deep, warm male voice"。',
    '改变这段语音的音色,使其符合:"a deep, warm male voice"。',
    '请重新生成这段语音,音色需匹配:"a deep, warm male voice"。',
    '请把这段语音的音色编辑成:"a deep, warm male voice",台词不变。',
    '将这段录音的声线调整为:"a deep, warm male voice"。',
    '请转换说话人的声音特质,使其符合:"a deep, warm male voice"。',
    '在保留原内容的同时,把嗓音改成:"a deep, warm male voice"。',
    '请将该语音重新演绎为这样的音色:"a deep, warm male voice"。',
    '把这段话用如下音色重新说一遍:"a deep, warm male voice"。',
    '请调整音色,让它听起来像:"a deep, warm male voice"。',
    '将音频中的声音替换为符合此描述的音色:"a deep, warm male voice"。',
    '请把说话人的嗓音特征修改为:"a deep, warm male voice"。',
    '保持文字内容不变,将声音风格转为:"a deep, warm male voice"。',
    '请对这段语音做音色迁移,目标音色为:"a deep, warm male voice"。',
    '把这段语音的发声特质改造成:"a deep, warm male voice"。',
    '请将原音频的音色更换为:"a deep, warm male voice",其余保持原样。',
    '将说话声调整成符合以下特征的音色:"a deep, warm male voice"。',
    '请生成内容相同、但音色为如下描述的语音:"a deep, warm male voice"。',
    '把这条音频的声音换成:"a deep, warm male voice"。',
    '请将语音的音色风格改写为:"a deep, warm male voice"。',
    '在不修改台词的情况下,把音色变更为:"a deep, warm male voice"。',
    '请重新合成这段语音,使其音色符合:"a deep, warm male voice"。',
    '将这段语音的嗓音调成这样:"a deep, warm male voice"。',
    'Please extract the speaker who says "see you tomorrow" and keep only this person\'s voice; remove all other speakers, and meanwhile denoise the audio, producing a single-speaker clean speech of the same length as the input.',
    'Please keep only the speaker who says "see you tomorrow" in this input speech, remove the other speakers, and meanwhile denoise the audio, and output a single clean speech track.',
    'This audio contains multiple speakers. Please locate the speaker who says "see you tomorrow" and output only this speaker\'s clean voice; the other speakers should be removed as well, and meanwhile denoise the audio.',
    'This audio contains multiple speakers. Please locate the speaker who says "see you tomorrow" and output only this speaker\'s clean voice; the other speakers should be removed as well.',
    'Please keep only the speaker who says "see you tomorrow" in this input speech, remove the other speakers, and output a single clean speech track.',
    '请在这段输入语音中保留说“see you tomorrow”的那位说话人，去掉其他说话人并对音频去噪，输出单条纯净人声。',
    '请提取说“see you tomorrow”的那个人，只保留这个人的声音，同时消除其他说话人的声音并去噪，输出与输入等长的单人纯净语音。',
    '这段音频里有多位说话人，请找出说“see you tomorrow”的那位，仅输出该说话人的干净语音，其余说话人也要去掉。',
)


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
        # An edit verb aimed at two quoted phrases, worded some other way.
        for text in ('Swap "x" for "y".', 'Make her say "yes" instead of "no".',
                     'Please could you change "red" into "blue"?', '"today" -> "tomorrow"'):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, 'trained wording'):
                e.parse_freeform(text)

    def test_style_edits_are_left_to_the_older_path(self):
        for text in ('Make this a whisper.', 'Give this voice an Irish accent.', 'Change the emotion to happy.',
                     'Turn this voice into a man.', 'Remove the background noise.', '',
                     # Plain style requests that only mention words or singing.
                     'Sing this softer.', 'Say it with more emotion.', 'Whisper the words.',
                     'Fix the lyrics in the chorus.', 'Change the words to be happier.',
                     'Make this voice sound like "a deep old man".'):
            with self.subTest(text=text):
                self.assertIsNone(e.parse_freeform(text))

    def test_the_booths_own_style_texts_still_pass(self):
        # Fork kadeSoundBooth.js starters and timbre hint; its nodetest expects them as non-word edits.
        for text in ('Remove background noise and reverberation. Preserve the voice and every spoken word.',
                     'Isolate the main speaking voice and remove background sounds and music. Preserve every spoken word.',
                     'Turn the speech into a natural whisper. Preserve the words and speaker identity.',
                     'Raise the pitch by two semitones while preserving the words and speaking speed.',
                     'Keep the spoken content unchanged and change the timbre to: "a youthful, light voice with clear diction"',
                     'Keep the spoken content unchanged and change the timbre to: "a youthful, light voice with clear diction".'):
            with self.subTest(text=text):
                self.assertIsNone(e.parse_freeform(text))
                plan = c.plan({'auk_task': 'edit', 'instruction': text, 'reference_voice_url': 'https://x.invalid/a.wav'})
                self.assertNotIn('words_edit', plan[0])

    def test_upstream_style_templates_pass(self):
        # pe.config.yaml template_pools, English and Chinese (emotion, timbre, separation).
        for text in UPSTREAM_STYLE_TEMPLATES:
            with self.subTest(text=text):
                self.assertIsNone(e.parse_freeform(text))

    def test_every_upstream_template_when_the_checkout_is_here(self):
        path = Path(__file__).resolve().parent.parent / 'upstream' / 'src' / 'auk' / 'infer' / 'pe.config.yaml'
        try:
            import yaml
            config = yaml.safe_load(path.read_text(encoding='utf-8'))
        except (ImportError, OSError):
            self.skipTest('upstream pe.config.yaml is not next to this checkout')
        fill = {'emo': 'happy', 'emotion': 'happy', 'value': 'slow down to 0.5x', 'n_ord': 'first',
                'timbre_desc': 'a deep, warm male voice', 'text': 'see you tomorrow'}
        texts = []

        def walk(node, path=''):
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(value, f'{path}/{key}')
            elif path.endswith(('/en', '/zh')) and '/vocal_edit/' not in path and '/content_edit/' not in path:
                for item in node if isinstance(node, list) else [node]:
                    if isinstance(item, str):
                        texts.append(re.sub(r'\{(\w+)\}', lambda m: fill.get(m.group(1), 'x'), item))
        walk(config.get('template_pools', {}))
        self.assertGreater(len(texts), 300)
        for text in texts:
            with self.subTest(text=text[:80]):
                self.assertIsNone(e.parse_freeform(text))

    def test_keep_tails_contractions_and_chinese(self):
        self.assertEqual(e.parse_freeform("Replace 'a' with 'b'. Keep the speaker's voice, don't change the timing."),
                         {'kind': 'speech', 'old': 'a', 'new': 'b'})
        self.assertEqual(e.parse_freeform("Replace 'I can't' with 'I won't'."),
                         {'kind': 'speech', 'old': "I can't", 'new': "I won't"})
        self.assertEqual(e.parse_freeform('把‘今天’改成‘明天’'), {'kind': 'speech', 'old': '今天', 'new': '明天'})
        self.assertEqual(e.parse_freeform('把这段歌词中的“明天你好”改成“未来你好”。'),
                         {'kind': 'vocal', 'old': '明天你好', 'new': '未来你好'})
        for text in ('Replace "a" with "b". Then remove the breath.', '把“a”改成“b”，再把“c”改成“d”。'):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, 'One word change per render'):
                e.parse_freeform(text)

    def test_sung_words_lose_double_quotes_like_the_bridge(self):
        # Bridge auk-words.js wordTemplate strips " from sung words.
        self.assertEqual(e.build_instruction('vocal', 'rear view', 'like "you"'),
                         'Change "rear view" to "like you" in the vocal recording.')
        self.assertEqual(e.build_instruction('vocal', 'say “hi”', 'say hello'),
                         'Change "say hi" to "say hello" in the vocal recording.')
        self.assertEqual(e.build_instruction('speech', "don't", 'do not'), "Replace 'don't' with 'do not'.")
        spec = e.words_edit_spec({'auk_task': 'edit', 'reference_voice_url': 'https://x.invalid/a.wav',
                                  'edit_kind': 'vocal', 'old_words': 'a "b"', 'new_words': 'c'})
        self.assertEqual((spec['old'], spec['instruction']), ('a b', 'Change "a b" to "c" in the vocal recording.'))


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

    def test_ranges_up_to_30_seconds_shrink_the_padding(self):
        # The bridge allows 30 s ranges (auk-words.js MAX_RANGE_S).
        self.assertEqual(e.crop_bounds(60, {'edit_start': 10, 'edit_end': 39}),
                         {'range_start': 10, 'range_end': 39, 'crop_start': 9.5, 'crop_end': 39.5})
        self.assertEqual(e.crop_bounds(60, {'edit_start': 10, 'edit_end': 40}),
                         {'range_start': 10, 'range_end': 40, 'crop_start': 10, 'crop_end': 40})
        self.assertEqual(e.crop_bounds(60, {'edit_start': 10, 'edit_end': 38, 'pad_seconds': 3})['crop_start'], 9)
        with self.assertRaisesRegex(ValueError, '30 seconds or less'):
            e.crop_bounds(60, {'edit_start': 10, 'edit_end': 40.5})

    def test_length_ratio_and_total_are_limited(self):
        # One word to ten: the range grows by at most 2.5x, the clip stays under 2x crop and 30 s.
        self.assertEqual(e.target_seconds(4.0, 2.0, 'a', 'one two three four five six seven eight nine ten'), 7.0)
        self.assertEqual(e.target_seconds(30.0, 28.0, 'a', 'one two three four five six seven eight nine ten'), 30.0)
        # Ten words to one: at most 0.4x.
        self.assertEqual(e.target_seconds(4.0, 2.0, 'one two three four five six seven eight nine ten', 'a'), 2.8)
        self.assertEqual(e.target_seconds(6.0, 2.0, 'x', 'a b c d e f g h',
                                          source_text='one two three'), 12.0)
        with self.assertRaises(ValueError):
            e.words_edit_spec({'auk_task': 'edit', 'reference_voice_url': 'https://x.invalid/a.wav',
                               'old_words': 'a', 'new_words': 'b', 'target_seconds': 45})


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


class JoinTests(unittest.TestCase):
    def test_splice_keeps_the_original_until_near_the_range(self):
        rate = 1000
        bounds = {'range_start': 8.0, 'range_end': 10.0}
        # Crop 7-11 s, take 5 s: keep 0.85 s of original padding at each end.
        self.assertEqual(e.splice_points(rate, bounds, 7000, 11000, 5000, 70), (850, 850))
        # No padding at the recording's start: nothing to keep there.
        self.assertEqual(e.splice_points(rate, {'range_start': 0.0, 'range_end': 2.0}, 0, 3000, 3000, 70), (0, 850))
        # A take too short to trim falls back to the whole crop.
        self.assertEqual(e.splice_points(rate, bounds, 7000, 11000, 1800, 70), (0, 0))

    def test_level_matches_the_padding_within_6_db(self):
        import numpy as np
        source = np.full((1000, 2), 0.4)
        self.assertAlmostEqual(e.level_gain(source, np.full(400, 0.25), 100, 500, 100, 100, 50), 1.6)
        self.assertAlmostEqual(e.level_gain(source, np.full(400, 0.01), 100, 500, 100, 100, 50), 10 ** 0.3)
        self.assertEqual(e.level_gain(source, np.zeros(400), 100, 500, 100, 100, 50), 1.0)  # silent take
        self.assertEqual(e.level_gain(source, np.full(400, 0.2), 0, 400, 0, 0, 50), 1.0)    # no padding


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

    def test_new_words_must_be_one_unbroken_run_like_the_bridge(self):
        apart = e.score_transcript('like the view of you', 'rear view', 'like you')
        self.assertEqual(apart['new_words_heard'], 0.5)
        self.assertFalse(apart['matched'])
        self.assertTrue(e.score_transcript('so like you then', 'rear view', 'like you')['matched'])

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
        self.assertFalse(worker.use_cpu_offload(int(15 * gib), 'False'))
        self.assertFalse(worker.use_cpu_offload(int(15 * gib), ' FALSE '))
        for value, expected in (('1', True), ('False', False), ('FALSE', False), ('0', False), ('off', False)):
            with self.subTest(value=value), patch.dict('os.environ', {'AUK_BOOT_LOAD': value}):
                self.assertIs(worker._flag('AUK_BOOT_LOAD', '1'), expected)

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
