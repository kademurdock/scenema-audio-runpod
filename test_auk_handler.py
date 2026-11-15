"""Real PCM/FFmpeg checks with only GPU inference and cloud storage substituted."""
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import auk_handler as worker


@unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is required')
class AudioTests(unittest.TestCase):
    def test_full_edit_sends_instruction_and_audio_and_returns_generated_pcm(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / 'source.wav'
            rate = 24000
            t = np.arange(rate * 13) / rate
            sf.write(source, 0.3 * np.sin(t * 2 * np.pi * 330), rate, subtype='PCM_16')
            original, _ = sf.read(source)
            uploaded = {}
            calls = []

            class Engine:
                model = types.SimpleNamespace(transformer=types.SimpleNamespace(clear_cache=lambda: None))

                def generate(self, messages, **kwargs):
                    self_outer.assertEqual(len(messages), 1)
                    self_outer.assertEqual(messages[0]['role'], 'user')
                    content = messages[0]['content']
                    self_outer.assertEqual([c['type'] for c in content], ['text', 'audio'])
                    reference, ref_rate = sf.read(content[1]['audio'])
                    self_outer.assertEqual(ref_rate, rate)
                    np.testing.assert_array_equal(reference, original)
                    calls.append((content[0]['text'], kwargs))
                    # Distinct, exactly representable PCM marks model output;
                    # returning or stitching any source audio fails below.
                    return np.full(round(kwargs['gen_seconds'] * rate), 0.125), rate

            class Storage:
                def upload_file(self, path, bucket, key, **kwargs):
                    dest = root / Path(key).name
                    shutil.copy(path, dest)
                    uploaded[key] = dest

                def generate_presigned_url(self, _action, Params, **kwargs):
                    return 'https://example.invalid/' + Params['Key']

            self_outer = self
            modules = {
                'auk.infer.infer_auk': types.SimpleNamespace(save_audio=lambda data, sr, path: sf.write(path, data, sr)),
                'boto3': types.SimpleNamespace(client=lambda *a, **k: Storage()),
                'runpod': types.SimpleNamespace(serverless=types.SimpleNamespace(progress_update=lambda *a: None)),
            }
            for instruction, sent in (('Give this voice an Irish accent.', 'Give this voice an Irish accent.'),
                                      ('Turn this voice into a twelve year old child.',
                                       'Keep the spoken content unchanged and change the timbre to: "a twelve year old child".')):
                with self.subTest(instruction=instruction), patch.dict(sys.modules, modules), \
                        patch.dict('os.environ', {'AWS_ENDPOINT_URL': 'https://example.invalid', 'AWS_BUCKET_NAME': 'test'}), \
                        patch.object(worker, 'download', lambda url, dest: shutil.copy(source, dest)), \
                        patch.object(worker, 'engine', return_value=Engine()):
                    result = worker.handler({'input': {'auk_task': 'edit', 'instruction': instruction,
                        'reference_voice_url': 'https://example.invalid/source.wav', 'seed': 777,
                        'voice_description': 'This must not replace an edit instruction.', 'voice_sample': True}})
                    self.assertNotIn('error', result)
                    rendered, sr = sf.read(uploaded[result['wav_key']])
                    self.assertEqual(sr, rate)
                    self.assertEqual(rendered.shape, original.shape)
                    np.testing.assert_array_equal(rendered, np.full(original.shape, 0.125))
                    self.assertEqual(calls[-1], (sent, {
                        'gen_seconds': 13, 'nfe': 32, 'cfg_strength': 2.0, 'seed': 777}))
                    self.assertEqual(result['parts'], 1)
                    self.assertFalse(result['voice_sample'])
            self.assertEqual(len(calls), 2)

    def test_range_keeps_stereo_edges_and_requested_length(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / 'source.wav'
            rate = 48000
            t = np.arange(rate * 45) / rate
            audio = np.stack([0.3 * np.sin(t * 2 * np.pi * 330),
                              0.2 * np.sin(t * 2 * np.pi * 550)], axis=1)
            sf.write(source, audio, rate, subtype='PCM_24')
            uploaded = {}
            calls = []

            class Engine:
                model = types.SimpleNamespace(transformer=types.SimpleNamespace(clear_cache=lambda: None))

                def generate(self, messages, **kwargs):
                    calls.append(kwargs)
                    return np.full(round(kwargs['gen_seconds'] * 24000), 0.125), 24000

            class Storage:
                def upload_file(self, path, bucket, key, **kwargs):
                    dest = root / Path(key).name
                    shutil.copy(path, dest)
                    uploaded[key] = dest

                def generate_presigned_url(self, _action, Params, **kwargs):
                    return 'https://example.invalid/' + Params['Key']

            modules = {
                'auk.infer.infer_auk': types.SimpleNamespace(save_audio=lambda data, sr, path: sf.write(path, data, sr)),
                'boto3': types.SimpleNamespace(client=lambda *a, **k: Storage()),
                'runpod': types.SimpleNamespace(serverless=types.SimpleNamespace(progress_update=lambda *a: None)),
            }
            with patch.dict(sys.modules, modules), patch.dict('os.environ', {'AWS_ENDPOINT_URL': 'https://example.invalid', 'AWS_BUCKET_NAME': 'test'}), \
                    patch.object(worker, 'download', lambda url, dest: shutil.copy(source, dest)), \
                    patch.object(worker, 'engine', lambda: Engine()):
                result = worker.handler({'input': {'auk_task': 'edit', 'instruction': 'Make this a whisper.',
                    'reference_voice_url': 'https://example.invalid/source.wav', 'edit_start': 10,
                    'edit_end': 30, 'gen_seconds': 15}})
            self.assertNotIn('error', result)
            rendered, sr = sf.read(uploaded[result['wav_key']])
            original, _ = sf.read(source)
            self.assertEqual(sr, rate)
            self.assertEqual(rendered.shape, (rate * 40, 2))
            np.testing.assert_array_equal(rendered[:10 * rate], original[:10 * rate])
            np.testing.assert_array_equal(rendered[25 * rate:], original[30 * rate:])
            # Mono take copied to both channels at full level (no -3 dB upmix).
            np.testing.assert_allclose(np.median(rendered[11 * rate:24 * rate], axis=0), 0.125, atol=1e-3)
            self.assertEqual(len(calls), 2)
            self.assertEqual(sum(c['gen_seconds'] for c in calls), 15)

    def _words_edit_run(self, inp, transcripts=None, rate=48000, seconds=20, fail_seeds=()):
        """Run the word-edit path with real PCM and FFmpeg; GPU and B2 are fakes."""
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        root = Path(td.name)
        source = root / 'source.wav'
        t = np.arange(rate * seconds) / rate
        audio = np.stack([0.3 * np.sin(t * 2 * np.pi * 330), 0.2 * np.sin(t * 2 * np.pi * 550)], axis=1)
        sf.write(source, audio, rate, subtype='PCM_24')
        original, _ = sf.read(source, always_2d=True)
        uploaded, calls, heard = {}, [], list(transcripts or [])

        class Engine:
            model = types.SimpleNamespace(transformer=types.SimpleNamespace(clear_cache=lambda: None))

            def generate(self, messages, **kwargs):
                content = messages[0]['content']
                reference, ref_rate = sf.read(content[1]['audio'])
                calls.append({'text': content[0]['text'], 'ref_seconds': len(reference) / ref_rate,
                              'ref_rate': ref_rate, 'ref_ndim': reference.ndim, **kwargs})
                if kwargs['seed'] in fail_seeds:
                    # Upstream's own mid-take check (infer_auk.py) raises ValueError.
                    raise ValueError('Audio tensor contains NaN or Inf.')
                return np.full(round(kwargs['gen_seconds'] * 24000), 0.125), 24000

        class Storage:
            def upload_file(self, path, bucket, key, **kwargs):
                dest = root / Path(key).name
                shutil.copy(path, dest)
                uploaded[key] = dest

            def generate_presigned_url(self, _action, Params, **kwargs):
                return 'https://example.invalid/' + Params['Key']

        modules = {
            'auk.infer.infer_auk': types.SimpleNamespace(save_audio=lambda data, sr, path: sf.write(path, data, sr)),
            'boto3': types.SimpleNamespace(client=lambda *a, **k: Storage()),
            'runpod': types.SimpleNamespace(serverless=types.SimpleNamespace(progress_update=lambda *a: None)),
        }
        with patch.dict(sys.modules, modules), \
                patch.dict('os.environ', {'AWS_ENDPOINT_URL': 'https://example.invalid', 'AWS_BUCKET_NAME': 'test'}), \
                patch.object(worker, 'download', lambda url, dest: shutil.copy(source, dest)), \
                patch.object(worker, 'engine', return_value=Engine()), \
                patch.object(worker, 'transcribe', lambda path, kind: heard.pop(0)):
            result = worker.handler({'input': {'auk_task': 'edit',
                                               'reference_voice_url': 'https://example.invalid/s.wav', **inp}})
        return result, calls, uploaded, original

    def test_lyric_edit_crops_pads_scales_splices_and_ranks_every_seed(self):
        rate = 48000
        result, calls, uploaded, original = self._words_edit_run({
            'edit_kind': 'vocal', 'old_words': 'rear view', 'new_words': 'like you so',
            'edit_start': 8, 'edit_end': 10, 'seeds': [11, 22, 33], 'rank': True},
            transcripts=['rear view', 'like you so', 'like you'])
        self.assertNotIn('error', result)
        # One canonical instruction, the padded crop as 24 kHz mono, one call per seed.
        self.assertEqual([c['seed'] for c in calls], [11, 22, 33])
        for c in calls:
            self.assertEqual(c['text'], 'Change "rear view" to "like you so" in the vocal recording.')
            self.assertEqual((c['ref_rate'], c['ref_ndim']), (24000, 1))
            self.assertAlmostEqual(c['ref_seconds'], 4.0, places=2)
            # 1 s pad + 2 s range * (3 words / 2 words) + 1 s pad.
            self.assertEqual(c['gen_seconds'], 5.0)
            self.assertEqual((c['nfe'], c['cfg_strength']), (32, 2.0))
        self.assertEqual((result['crop_start'], result['crop_end']), (7.0, 11.0))
        self.assertEqual(result['target_seconds'], 5.0)
        # Every take is returned with its seed, best match first.
        self.assertTrue(result['ranked'])
        self.assertTrue(result['any_matched'])
        self.assertEqual([c['seed'] for c in result['candidates']], [22, 33, 11])
        self.assertTrue(result['candidates'][0]['matched'])
        self.assertEqual(result['seed'], 22)
        self.assertEqual(len(uploaded), 9)  # WAV, MP3 and the edited clip per take
        # The take is brought to the original's level in the padding.
        level = np.sqrt(np.mean(np.square(original[7 * rate:int(7.3 * rate)])))
        gain = level / 0.125
        for cand in result['candidates']:
            rendered, sr = sf.read(uploaded[cand['wav_key']], always_2d=True)
            self.assertEqual(sr, rate)
            self.assertEqual(rendered.shape, (rate * 21, 2))  # 7 s + 5 s + 9 s
            # The original stays until 0.15 s before the range and from 0.15 s after it.
            np.testing.assert_array_equal(rendered[:int(7.85 * rate)], original[:int(7.85 * rate)])
            np.testing.assert_array_equal(rendered[int(11.15 * rate):], original[int(10.15 * rate):])
            middle = rendered[8 * rate:11 * rate]
            self.assertAlmostEqual(cand['level_db'], 20 * np.log10(gain), delta=0.25)
            np.testing.assert_allclose(middle, 0.125 * 10 ** (cand['level_db'] / 20), atol=2e-3)
            # The edited stretch with its padding, for the bridge's old-words check.
            self.assertEqual(cand['clip_key'], cand['key'][:-4] + '-clip.mp3')
            self.assertIn(cand['clip_key'], uploaded)
            self.assertTrue(cand['clip_url'].endswith('-clip.mp3'))
            self.assertEqual((cand['clip_start'], cand['clip_seconds']), (7.0, 5.0))
        self.assertEqual(result['clip_url'], result['candidates'][0]['clip_url'])

    def test_a_take_that_raises_value_error_is_reported_and_the_others_kept(self):
        result, calls, uploaded, _ = self._words_edit_run({
            'old_words': 'red', 'new_words': 'blue', 'edit_start': 8, 'edit_end': 9, 'seeds': [1, 2, 3]},
            fail_seeds=(2,))
        self.assertNotIn('error', result)
        self.assertEqual([c['seed'] for c in result['candidates']], [1, 3])
        self.assertEqual(result['failed_seeds'], [{'seed': 2, 'error': 'ValueError'}])
        result, _, _, _ = self._words_edit_run({
            'old_words': 'red', 'new_words': 'blue', 'edit_start': 8, 'edit_end': 9, 'seeds': [1]},
            fail_seeds=(1,))
        self.assertNotIn('NaN', result['error'])
        self.assertIn('unchanged', result['error'])

    def test_a_30_second_range_is_accepted_with_smaller_padding(self):
        result, calls, _, _ = self._words_edit_run({
            'old_words': 'one two three', 'new_words': 'four five six', 'edit_start': 5, 'edit_end': 34,
            'candidates': 1}, seconds=45)
        self.assertNotIn('error', result)
        self.assertEqual((result['crop_start'], result['crop_end']), (4.5, 34.5))
        self.assertAlmostEqual(calls[0]['ref_seconds'], 30.0, places=2)
        self.assertEqual(calls[0]['gen_seconds'], 30.0)

    def test_unranked_takes_keep_seed_order_and_short_clip_needs_no_range(self):
        result, calls, uploaded, original = self._words_edit_run({
            'instruction': "Replace 'see you Tuesday' with 'see you Thursday'.", 'candidates': 2, 'seed': 9},
            seconds=6)
        self.assertNotIn('error', result)
        self.assertEqual([c['seed'] for c in result['candidates']], [9, 9 + 7919])
        self.assertFalse(result['ranked'])
        self.assertIsNone(result['any_matched'])
        self.assertEqual(calls[0]['text'], "Replace 'see you Tuesday' with 'see you Thursday'.")
        self.assertEqual(calls[0]['gen_seconds'], 6.0)  # Same word count, same length.
        self.assertEqual((result['crop_start'], result['crop_end']), (0.0, 6.0))

    def test_word_edits_that_cannot_work_are_refused_before_the_gpu(self):
        cases = (
            ({'instruction': 'Replace "red" with "blue" and change "cat" to "dog".'}, 'One word change per render'),
            ({'instruction': 'Swap "red" for "blue".'}, 'trained wording'),
            ({'old_words': 'red', 'new_words': 'blue'}, 'need a range'),
            ({'old_words': 'red', 'new_words': 'red', 'edit_start': 1, 'edit_end': 2}, 'the same'),
            ({'old_words': 'red', 'new_words': 'blue', 'edit_start': 1, 'edit_end': 40}, 'Mark only the words'),
        )
        for inp, message in cases:
            with self.subTest(inp=inp):
                result, calls, uploaded, _ = self._words_edit_run(inp, seconds=45)
                self.assertIn(message, result.get('error', ''))
                self.assertEqual(calls, [])
                self.assertEqual(uploaded, {})


if __name__ == '__main__':
    unittest.main()
