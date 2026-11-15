"""Contract and handler checks with real FFmpeg/PCM; only the model and B2 are fakes.

With MING_UPSTREAM set to a checkout of inclusionAI/Ming-3.0-tts at the pinned
commit, every task the contract picks is also run through Ming's own request
resolver, so a job the contract accepts is one Ming accepts.
"""
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

import ming_contract as contract
import ming_handler as worker

UPSTREAM = os.environ.get("MING_UPSTREAM")
if UPSTREAM:
    sys.path.insert(0, UPSTREAM)

CLIP = {"reference_voice_url": "https://h/a.mp3"}


class ContractTests(unittest.TestCase):
    def test_design_tasks(self):
        spec = contract.parse({"mode": "design", "text": "Hi.", "description": " (A warm voice) ", "seed": 7})
        self.assertEqual((spec["task"], spec["args"], spec["seed"]), ("voice_design", {"style": "A warm voice"}, 7))
        spec = contract.parse({"mode": "design", "text": "Hi.", "description": "A warm voice.",
                               "style": "slow and kind"})
        self.assertEqual((spec["task"], spec["args"]), ("freeform", {"instruction": "A warm voice. slow and kind"}))
        spec = contract.parse({"mode": "design", "text": "Hi.", "voice": "US_Emma"})
        self.assertEqual((spec["task"], spec["args"]), ("ip", {"ip": "US_Emma"}))
        self.assertEqual((spec["cfg_scale"], spec["flow_steps"], spec["normalize"]), (2.0, 10, True))
        self.assertGreater(contract.parse({"mode": "design", "text": "Hi.", "description": "x"})["seed"], 0)

    def test_design_refusals(self):
        base = {"mode": "design", "text": "Hi."}
        for bad in ({}, {"description": "warm", "reference_voice_url": "https://x/y.wav"},
                    {"description": "warm", "emotion": "happy"}, {"description": "warm", "text": "[Sigh] Hi."},
                    {"voice": "US_Emma", "description": "warm"}, {"voice": "a/b"},
                    {"description": "warm", "language": "fr"}):
            with self.assertRaises(ValueError, msg=bad):
                contract.parse({**base, **bad})

    def test_clone_tasks(self):
        def task(**extra):
            spec = contract.parse({"mode": "clone", "text": "Hello there.", **CLIP, **extra})
            return spec["task"], spec["args"], spec["uses_prompt_text"]

        self.assertEqual(task(prompt_text=" Exact  words. "),
                         ("seed", {"prompt_text": "Exact words.", "target_language": "English"}, True))
        self.assertEqual(task(style="(slightly faster, cheerful tone)"),
                         ("freeform", {"instruction": "slightly faster, cheerful tone"}, False))
        # A transcript is not used by freeform; the style wins.
        self.assertEqual(task(style="cheerful", prompt_text="Words.")[0], "freeform")
        self.assertEqual(task(emotion="Happy"), ("emotion", {"emotion": "高兴"}, False))
        self.assertEqual(task(speed="fast"), ("basic", {"speed": "快速", "pitch": "中", "volume": "中"}, False))
        self.assertEqual(task(), ("basic", {"speed": "中速", "pitch": "中", "volume": "中"}, False))
        self.assertEqual(task(prompt_text="Mots.", language="fr")[:2],
                         ("multilingual", {"prompt_text": "Mots.", "target_language": "French"}))
        spec = contract.parse({"mode": "clone", "text": "你好。", **CLIP, "prompt_text": "参考。"})
        self.assertEqual(spec["args"]["target_language"], "Chinese")

    def test_sound_tags(self):
        spec = contract.parse({"mode": "clone", **CLIP, "prompt_text": "Words.",
                               "text": "[sigh] Well. [laughs] Okay [UHM] then [Question-huh]"})
        self.assertEqual(spec["task"], "paralinguistic")
        self.assertEqual(spec["text"], "[Sigh] Well. [Laughter] Okay [Uhm] then [Question-huh]")
        self.assertEqual(spec["sound_tags"], ["[Sigh]", "[Laughter]", "[Uhm]", "[Question-huh]"])
        for bad in ({"text": "[Whistle] hi", "prompt_text": "W."}, {"text": "[Sigh] hi"},
                    {"text": "[Sigh] hi", "prompt_text": "W.", "style": "calm"}, {"text": "[Sigh]", "prompt_text": "W."}):
            with self.assertRaises(ValueError, msg=bad):
                contract.parse({"mode": "clone", **CLIP, **bad})
        self.assertEqual(len(contract.SOUND_TAGS), 14)

    def test_bad_knobs(self):
        base = {"mode": "clone", "text": "Hi.", **CLIP}
        for bad in ({"mode": "talk"}, {"text": ""}, {"seed": -1}, {"seed": 1.5}, {"seed": True},
                    {"cfg_scale": 9}, {"flow_steps": 0}, {"normalize": "yes"}, {"speed": "zippy"},
                    {"emotion": "happy", "style": "calm"}, {"emotion": "happy", "speed": "fast"},
                    {"voice": "US_Emma"}, {"reference_voice_url": ""}, {"language": "xx"},
                    {"language": "ja"}, {"text": "a" * (contract.MAX_TEXT_CHARS + 1)}):
            with self.assertRaises(ValueError, msg=bad):
                contract.parse({**base, **bad})


@unittest.skipUnless(UPSTREAM, "MING_UPSTREAM checkout not given")
class UpstreamAgreementTests(unittest.TestCase):
    """Every task the contract chooses passes Ming's own resolver unchanged."""

    def test_tags_match(self):
        from ming_tiny.task_protocol import PARALINGUISTIC_TAGS
        from frontend.toolkit import PARALINGUISTIC_TAGS as FRONTEND_TAGS
        self.assertEqual(tuple(contract.SOUND_TAGS), tuple(PARALINGUISTIC_TAGS))
        self.assertEqual(set(contract.SOUND_TAGS), set(FRONTEND_TAGS))

    def test_resolver_accepts_every_contract_task(self):
        import inspect
        from ming_tiny.api import MingTinyTTS  # noqa: F401  (imports cleanly without torch work)
        from ming_tiny.task_protocol import resolve_task_request
        params = set(inspect.signature(MingTinyTTS.generate).parameters)
        jobs = [
            {"mode": "design", "text": "Hi there.", "description": "A warm woman"},
            {"mode": "design", "text": "Hi there.", "description": "A warm woman", "style": "slow"},
            {"mode": "design", "text": "Hi there.", "voice": "US_Emma"},
            {"mode": "clone", "text": "Hi there.", **CLIP, "prompt_text": "Words."},
            {"mode": "clone", "text": "[sigh] Hi there.", **CLIP, "prompt_text": "Words."},
            {"mode": "clone", "text": "Bonjour.", **CLIP, "prompt_text": "Mots.", "language": "fr"},
            {"mode": "clone", "text": "Hi there.", **CLIP, "style": "cheerful"},
            {"mode": "clone", "text": "Hi there.", **CLIP, "emotion": "sad"},
            {"mode": "clone", "text": "Hi there.", **CLIP, "pitch": "high"},
            {"mode": "clone", "text": "Hi there.", **CLIP},
        ]
        for job in jobs:
            spec = contract.parse(job)
            args = dict(spec["args"])
            if spec["mode"] == "clone":
                args["prompt_wav"] = "/tmp/ref.wav"
            self.assertLessEqual(set(args) | {"task", "text", "seed", "cfg_scale", "flow_steps"}, params)
            request = resolve_task_request(spec["task"], text=spec["text"], **args)
            self.assertEqual(request.task, spec["task"], job)
            if spec["task"] == "paralinguistic":
                self.assertIn("[Sigh]", request.text)

    def test_tags_survive_ming_text_prep(self):
        from frontend.text_processing import prepare_text, split_text
        text = prepare_text("[Laughter] Well, okay then.", task="paralinguistic", language="English")
        self.assertIn("[Laughter]", text)
        self.assertGreater(len(split_text(" ".join(["One two three four five six."] * 10))), 1)


class FakeMing:
    sample_rate = 44100

    def __init__(self):
        self.calls = []
        self.text_normalization = True

    def generate(self, **kwargs):
        call = dict(kwargs, normalization=self.text_normalization)
        if "prompt_wav" in kwargs:
            info = sf.info(kwargs["prompt_wav"])  # read now: the job's folder is gone afterwards
            call["prompt_wav:info"] = (info.samplerate, info.channels, info.duration)
        self.calls.append(call)
        wave = np.full(int(0.5 * self.sample_rate), 0.2, dtype=np.float32)
        return types.SimpleNamespace(waveform=wave.reshape(1, -1), segments=[1, 2])


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required")
class HandlerTests(unittest.TestCase):
    def run_job(self, inp, clip_seconds=3.0):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        model = FakeMing()
        uploads = {}

        class Storage:
            def upload_file(self, path, bucket, key, **kwargs):
                dest = root / key.replace("/", "_")
                shutil.copy(path, dest)
                uploads[key] = (dest, kwargs["ExtraArgs"]["ContentType"])

            def generate_presigned_url(self, _action, Params, **kwargs):
                return "https://signed.invalid/" + Params["Key"]

        clip = root / "clip.mp3"
        t = np.arange(int(48000 * clip_seconds)) / 48000
        sf.write(root / "clip.wav", np.stack([0.3 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1), 48000)
        worker.ffmpeg("-i", root / "clip.wav", "-c:a", "libmp3lame", clip)

        def fake_download(url, dest):
            shutil.copy(clip, dest)

        modules = {"boto3": types.SimpleNamespace(client=lambda *a, **k: Storage()),
                   "runpod": types.SimpleNamespace(serverless=types.SimpleNamespace(progress_update=lambda *a: None))}
        env = {"AWS_ENDPOINT_URL": "https://s3.example.invalid", "AWS_BUCKET_NAME": "b"}
        with patch.dict(sys.modules, modules), patch.dict("os.environ", env), \
                patch.object(worker, "ENGINE", model), patch.object(worker, "download", fake_download):
            out = worker.handler({"id": "j", "input": inp})
        return out, model, uploads

    def test_design_writes_wav_and_mp3(self):
        out, model, uploads = self.run_job({"mode": "design", "text": "Well, hey there.", "seed": 42,
                                            "description": "A warm, calm woman", "cfg_scale": 2.5,
                                            "flow_steps": 12, "normalize": False})
        self.assertNotIn("error", out, out)
        (call,) = model.calls
        self.assertEqual(call, {"task": "voice_design", "text": "Well, hey there.", "seed": 42, "cfg_scale": 2.5,
                                "flow_steps": 12, "style": "A warm, calm woman", "normalization": False})
        self.assertEqual((out["mode"], out["seed"], out["sample_rate"], out["task"], out["parts"]),
                         ("design", 42, 44100, "voice_design", 2))
        self.assertAlmostEqual(out["duration_seconds"], 0.5, places=2)
        mp3 = uploads[out["key"]][0]
        self.assertEqual(uploads[out["key"]][1], "audio/mpeg")
        self.assertEqual(sf.info(uploads[out["wav_key"]][0]).samplerate, 44100)
        probe = worker.subprocess.run(["ffmpeg", "-hide_banner", "-i", str(mp3)], capture_output=True, text=True)
        self.assertIn("mp3", probe.stderr)
        self.assertIn("44100 Hz", probe.stderr)
        self.assertTrue(out["url"].startswith("https://signed.invalid/ming/"))
        self.assertEqual(out["engine"], "ming-3.0-tts-tiny")

    def test_clone_with_style_trims_clip(self):
        out, model, _ = self.run_job({"mode": "clone", "text": "Hello there. Nice day.", **CLIP,
                                      "style": "slightly faster, cheerful tone"}, clip_seconds=25)
        self.assertNotIn("error", out, out)
        (call,) = model.calls
        self.assertEqual((call["task"], call["instruction"]), ("freeform", "slightly faster, cheerful tone"))
        self.assertNotIn("prompt_text", call)
        rate, channels, seconds = call["prompt_wav:info"]
        self.assertEqual((rate, channels), (44100, 1))
        self.assertLessEqual(seconds, worker.REF_MAX_S + 0.05)

    def test_clone_with_transcript_keeps_whole_clip(self):
        out, model, _ = self.run_job({"mode": "clone", "text": "Hello.", **CLIP, "prompt_text": "Exact words."},
                                     clip_seconds=24)
        self.assertNotIn("error", out, out)
        (call,) = model.calls
        self.assertEqual((call["task"], call["prompt_text"], call["target_language"]), ("seed", "Exact words.", "English"))
        self.assertGreater(call["prompt_wav:info"][2], 23.5)

    def test_transcript_clip_too_long_is_refused(self):
        out, model, _ = self.run_job({"mode": "clone", "text": "Hello.", **CLIP, "prompt_text": "Words."},
                                     clip_seconds=31)
        self.assertIn("error", out)
        self.assertEqual(model.calls, [])

    def test_bad_input_is_an_error_not_a_crash(self):
        out, model, _ = self.run_job({"mode": "design", "text": "Hi."})
        self.assertIn("description", out["error"])
        self.assertNotIn("url", out)

    def test_download_refuses_other_hosts(self):
        with patch.dict("os.environ", {"AWS_ENDPOINT_URL": "https://s3.us-east-005.backblazeb2.com",
                                       "AWS_BUCKET_NAME": "Bucket", "MING_AUDIO_HOSTS": ""}):
            self.assertIn("bucket.s3.us-east-005.backblazeb2.com", worker.allowed_hosts())
            for url in ("http://s3.us-east-005.backblazeb2.com/a.mp3", "https://169.254.169.254/a",
                        "https://evil.example/a.mp3", "https://s3.us-east-005.backblazeb2.com:8443/a"):
                with self.assertRaises(ValueError):
                    worker.download(url, "/nonexistent")


if __name__ == "__main__":
    unittest.main()
