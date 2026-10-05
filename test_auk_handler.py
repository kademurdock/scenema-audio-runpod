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
