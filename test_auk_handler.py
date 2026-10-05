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
            edits = (
                ('Give this voice an Irish accent.', 'Give this voice an Irish accent.'),
                ('Turn this voice into a twelve year old child.',
                 'Keep the spoken content unchanged and change the timbre to: "a twelve year old child".'),
            )
            for instruction, prepared in edits:
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
                    self.assertEqual(calls[-1], (prepared, {
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
                    return np.zeros(round(kwargs['gen_seconds'] * 24000)), 24000

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
            self.assertEqual(len(calls), 2)
            self.assertEqual(sum(c['gen_seconds'] for c in calls), 15)


if __name__ == '__main__':
    unittest.main()
