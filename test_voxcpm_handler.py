"""Contract and handler checks with real FFmpeg/PCM; only the model and B2 are fakes."""
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

import voxcpm_contract as contract
import voxcpm_handler as worker


class ContractTests(unittest.TestCase):
    def test_split_keeps_order_and_limit(self):
        text = " ".join(f"Sentence number {i} is here, and it goes on a little." for i in range(40))
        chunks = contract.split_text(text, 120)
        self.assertTrue(all(len(c) <= 120 for c in chunks))
        self.assertEqual(" ".join(chunks), text)
        self.assertTrue(all(c.endswith(".") for c in chunks))

    def test_split_long_sentence_at_commas_then_spaces(self):
        long = ", ".join(["word " * 20] * 5).strip() + "."
        chunks = contract.split_text(long, 80)
        self.assertTrue(all(len(c) <= 80 for c in chunks))
        self.assertEqual(" ".join(chunks).split(), long.split())
        self.assertEqual(contract.split_text("x" * 170, 80), ["x" * 80, "x" * 80, "x" * 10])

    def test_short_text_is_one_chunk(self):
        self.assertEqual(contract.split_text("  Hello there.  How are you?  "), ["Hello there. How are you?"])

    def test_design_needs_description_and_no_clip(self):
        with self.assertRaises(ValueError):
            contract.parse({"mode": "design", "text": "Hi."})
        with self.assertRaises(ValueError):
            contract.parse({"mode": "design", "text": "Hi.", "description": "warm",
                            "reference_voice_url": "https://x/y.wav"})
        spec = contract.parse({"mode": "design", "text": "Hi.", "description": " A (warm) voice ", "seed": 7})
        self.assertEqual(spec["description"], "A warm voice")
        self.assertEqual(spec["seed"], 7)
        self.assertEqual((spec["cfg_value"], spec["inference_timesteps"], spec["normalize"]), (2.0, 10, False))

    def test_clone_fields(self):
        with self.assertRaises(ValueError):
            contract.parse({"mode": "clone", "text": "Hi."})
        spec = contract.parse({"mode": "clone", "text": "Hi.", "reference_voice_url": "https://h/a.mp3",
                               "style": "(slightly faster, cheerful tone)", "prompt_text": " Exact  words. "})
        self.assertEqual(spec["style"], "slightly faster, cheerful tone")
        self.assertEqual(spec["prompt_text"], "Exact words.")
        self.assertEqual(contract.with_prefix(spec["style"], "Hi."), "(slightly faster, cheerful tone)Hi.")
        self.assertEqual(contract.with_prefix("", "Hi."), "Hi.")

    def test_bad_knobs(self):
        base = {"mode": "design", "text": "Hi.", "description": "warm"}
        for bad in ({"mode": "talk"}, {"text": ""}, {"seed": -1}, {"seed": 1.5}, {"seed": True},
                    {"cfg_value": 9}, {"inference_timesteps": 0}, {"normalize": "yes"},
                    {"text": "a" * (contract.MAX_TEXT_CHARS + 1)}):
            with self.assertRaises(ValueError, msg=bad):
                contract.parse({**base, **bad})
        self.assertGreater(contract.parse(base)["seed"], 0)


class FakeModel:
    def __init__(self, rate=48000):
        self.tts_model = types.SimpleNamespace(sample_rate=rate)
        self.calls = []

    def generate(self, text, **kwargs):
        self.calls.append({"text": text, **kwargs})
        for key in ("reference_wav_path", "prompt_wav_path"):
            if key in kwargs:
                info = sf.info(kwargs[key])
                assert info.frames > 0
        return np.full(int(0.5 * self.tts_model.sample_rate), 0.25 * len(self.calls) / 10, dtype=np.float32)


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required")
class HandlerTests(unittest.TestCase):
    def run_job(self, inp, clip_seconds=3.0):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        model = FakeModel()
        uploads = {}
        seeds = []

        class Storage:
            def upload_file(self, path, bucket, key, **kwargs):
                dest = root / key.replace("/", "_")
                shutil.copy(path, dest)
                uploads[key] = (dest, kwargs["ExtraArgs"]["ContentType"])

            def generate_presigned_url(self, _action, Params, **kwargs):
                return "https://signed.invalid/" + Params["Key"]

        clip = root / "clip.mp3"
        t = np.arange(int(44100 * clip_seconds)) / 44100
        sf.write(root / "clip.wav", 0.3 * np.sin(2 * np.pi * 220 * t), 44100)
        worker.ffmpeg("-i", root / "clip.wav", "-c:a", "libmp3lame", clip)

        def fake_download(url, dest):
            shutil.copy(clip, dest)

        modules = {"boto3": types.SimpleNamespace(client=lambda *a, **k: Storage()),
                   "runpod": types.SimpleNamespace(serverless=types.SimpleNamespace(progress_update=lambda *a: None))}
        env = {"AWS_ENDPOINT_URL": "https://s3.example.invalid", "AWS_BUCKET_NAME": "b"}
        with patch.dict(sys.modules, modules), patch.dict("os.environ", env), \
                patch.object(worker, "ENGINE", model), patch.object(worker, "download", fake_download), \
                patch.object(worker, "seed_everything", seeds.append):
            out = worker.handler({"id": "j", "input": inp})
        return out, model, uploads, seeds

    def test_design_long_text_keeps_one_voice(self):
        text = " ".join(["This sentence is about fifty characters in length."] * 12)
        out, model, uploads, seeds = self.run_job({"mode": "design", "text": text, "seed": 42,
                                                   "description": "A warm, calm woman", "cfg_value": 2.5,
                                                   "inference_timesteps": 12})
        self.assertNotIn("error", out, out)
        self.assertGreater(len(model.calls), 1)
        first, rest = model.calls[0], model.calls[1:]
        self.assertTrue(first["text"].startswith("(A warm, calm woman)This sentence"))
        self.assertNotIn("reference_wav_path", first)
        for call in rest:
            self.assertFalse(call["text"].startswith("("))
            self.assertTrue(call["reference_wav_path"].endswith("part0.wav"))
        for call in model.calls:
            self.assertEqual((call["cfg_value"], call["inference_timesteps"], call["normalize"]), (2.5, 12, False))
        self.assertEqual(seeds, [42] * len(model.calls))
        self.assertEqual((out["mode"], out["seed"], out["sample_rate"]), ("design", 42, 48000))
        expected = round((0.5 * len(model.calls) + 0.15 * (len(model.calls) - 1)), 2)
        self.assertAlmostEqual(out["duration_seconds"], expected, places=2)
        mp3 = uploads[out["key"]][0]
        self.assertEqual(uploads[out["key"]][1], "audio/mpeg")
        self.assertEqual(sf.info(uploads[out["wav_key"]][0]).samplerate, 48000)
        probe = worker.subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate",
                                       "-of", "csv=p=0", str(mp3)], capture_output=True, text=True)
        self.assertEqual(probe.stdout.strip(), "48000")
        self.assertTrue(out["url"].startswith("https://signed.invalid/voxcpm/"))

    def test_clone_with_style(self):
        out, model, _, _ = self.run_job({"mode": "clone", "text": "Hello there. Nice day.",
                                         "reference_voice_url": "https://h/a.mp3",
                                         "style": "slightly faster, cheerful tone"}, clip_seconds=25)
        self.assertNotIn("error", out, out)
        (call,) = model.calls
        self.assertEqual(call["text"], "(slightly faster, cheerful tone)Hello there. Nice day.")
        self.assertNotIn("prompt_wav_path", call)
        info = sf.info(call["reference_wav_path"])
        self.assertEqual((info.samplerate, info.channels), (16000, 1))
        self.assertLessEqual(info.duration, worker.REF_MAX_S + 0.05)

    def test_clone_ultimate_uses_clip_twice(self):
        out, model, _, _ = self.run_job({"mode": "clone", "text": "Hello.", "reference_voice_url": "https://h/a.mp3",
                                         "prompt_text": "Exact transcript."})
        self.assertNotIn("error", out, out)
        (call,) = model.calls
        self.assertEqual(call["prompt_wav_path"], call["reference_wav_path"])
        self.assertEqual(call["prompt_text"], "Exact transcript.")
        self.assertEqual(call["text"], "Hello.")

    def test_ultimate_clip_too_long_is_refused(self):
        out, model, _, _ = self.run_job({"mode": "clone", "text": "Hello.", "reference_voice_url": "https://h/a.mp3",
                                         "prompt_text": "Words."}, clip_seconds=31)
        self.assertIn("error", out)
        self.assertEqual(model.calls, [])

    def test_bad_input_is_an_error_not_a_crash(self):
        out, model, _, _ = self.run_job({"mode": "design", "text": "Hi."})
        self.assertIn("description", out["error"])

    def test_download_refuses_other_hosts(self):
        with patch.dict("os.environ", {"AWS_ENDPOINT_URL": "https://s3.us-east-005.backblazeb2.com",
                                       "AWS_BUCKET_NAME": "Bucket", "VOXCPM_AUDIO_HOSTS": ""}):
            self.assertIn("bucket.s3.us-east-005.backblazeb2.com", worker.allowed_hosts())
            for url in ("http://s3.us-east-005.backblazeb2.com/a.mp3", "https://169.254.169.254/a",
                        "https://evil.example/a.mp3", "https://s3.us-east-005.backblazeb2.com:8443/a"):
                with self.assertRaises(ValueError):
                    worker.download(url, "/nonexistent")


if __name__ == "__main__":
    unittest.main()
