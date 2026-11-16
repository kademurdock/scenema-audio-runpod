"""Contract and handler checks with real FFmpeg/PCM; only the models and B2 are fakes.

With CHATTERBOX_UPSTREAM set (the image build), the contract's language and
tag lists are also checked against the installed chatterbox package.
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

import chatterbox_contract as contract
import chatterbox_handler as worker


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

    def test_split_chinese_without_spaces(self):
        text = "你好，今天天气真不错。希望你有一个愉快的周末！" * 10
        chunks = contract.split_text(text, 60)
        self.assertTrue(all(len(c) <= 60 for c in chunks))
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(c.endswith(("。", "！")) for c in chunks))

    def test_split_never_breaks_a_tag(self):
        text = ("This line is long enough to need splitting " * 3 + "[clear throat] and then more words. ") * 6
        chunks = contract.split_text(text.strip(), 70)
        joined = " ".join(chunks)
        self.assertEqual(joined.count("[clear throat]"), 6)
        for chunk in chunks:
            self.assertEqual(chunk.count("["), chunk.count("]"))

    def test_defaults_and_mode_from_reference(self):
        spec = contract.parse({"text": "Hi there."})
        self.assertEqual((spec["model"], spec["mode"], spec["language_id"]), ("turbo", "design", "en"))
        self.assertEqual((spec["temperature"], spec["exaggeration"], spec["cfg_weight"]), (0.8, None, None))
        self.assertGreater(spec["seed"], 0)
        spec = contract.parse({"model": "original", "text": "Hi.", "reference_url": " https://h/a.mp3 "})
        self.assertEqual((spec["mode"], spec["reference_url"]), ("clone", "https://h/a.mp3"))
        self.assertEqual((spec["exaggeration"], spec["cfg_weight"]), (0.5, 0.5))
        spec = contract.parse({"model": "original", "mode": "clone", "text": "Hi.",
                               "reference_voice_url": "https://h/a.mp3", "exaggeration": 0.9, "cfg_weight": 0,
                               "temperature": 1.1, "seed": 7})
        self.assertEqual((spec["exaggeration"], spec["cfg_weight"], spec["temperature"], spec["seed"]),
                         (0.9, 0.0, 1.1, 7))

    def test_mode_and_reference_must_agree(self):
        with self.assertRaises(ValueError):
            contract.parse({"mode": "clone", "text": "Hi."})
        with self.assertRaises(ValueError):
            contract.parse({"mode": "design", "text": "Hi.", "reference_url": "https://h/a.mp3"})
        with self.assertRaisesRegex(ValueError, "cannot make a voice from words"):
            contract.parse({"mode": "design", "text": "Hi.", "description": "a warm woman"})

    def test_turbo_tags(self):
        spec = contract.parse({"text": "Oh [Laughs] that is funny [clear  throat] anyway [SIGH]."})
        self.assertEqual(spec["chunks"], ["Oh [laugh] that is funny [clear throat] anyway [sigh]."])
        self.assertEqual(spec["sound_tags"], ["[laugh]", "[clear throat]", "[sigh]"])
        with self.assertRaisesRegex(ValueError, "Unknown sound tag"):
            contract.parse({"text": "Hello [burp] there."})
        with self.assertRaises(ValueError):
            contract.parse({"text": "[laugh]"})
        self.assertEqual(len(contract.SOUND_TAGS), 19)
        self.assertEqual(len(set(contract.SOUND_TAGS)), 19)

    def test_tags_only_on_turbo(self):
        for model, extra in (("original", {}), ("multilingual", {"language_id": "en"})):
            with self.assertRaisesRegex(ValueError, "only with model turbo"):
                contract.parse({"model": model, "text": "Ha [laugh] ha.", **extra})
        spec = contract.parse({"model": "original", "text": "Read [section 2] now."})
        self.assertEqual(spec["chunks"], ["Read [section 2] now."])

    def test_turbo_reports_ignored_knobs(self):
        spec = contract.parse({"text": "Hi.", "exaggeration": 0.7, "cfg_weight": 0.3, "prompt_text": "x"})
        self.assertEqual(spec["ignored"], ["exaggeration", "cfg_weight", "prompt_text"])
        self.assertIsNone(spec["exaggeration"])

    def test_languages(self):
        with self.assertRaisesRegex(ValueError, "needs language_id"):
            contract.parse({"model": "multilingual", "text": "Bonjour."})
        spec = contract.parse({"model": "multilingual", "text": "Bonjour.", "language_id": " FR "})
        self.assertEqual(spec["language_id"], "fr")
        with self.assertRaisesRegex(ValueError, "English only"):
            contract.parse({"model": "turbo", "text": "Hola.", "language_id": "es"})
        with self.assertRaises(ValueError):
            contract.parse({"model": "multilingual", "text": "Hi.", "language_id": "xx"})
        self.assertEqual(contract.parse({"model": "original", "text": "Hi.", "language_id": "en"})["language_id"], "en")
        self.assertEqual(len(contract.LANGUAGES), 23)
        dense = contract.parse({"model": "multilingual", "language_id": "zh", "text": "你好。" * 100})
        self.assertTrue(all(len(c) <= contract.DENSE_CHUNK_CHARS for c in dense["chunks"]))

    def test_bad_knobs(self):
        base = {"model": "original", "text": "Hi."}
        for bad in ({"model": "nano"}, {"mode": "talk"}, {"text": ""}, {"text": 5}, {"seed": -1}, {"seed": 1.5},
                    {"seed": True}, {"exaggeration": 3}, {"exaggeration": 0.1}, {"cfg_weight": 1.5},
                    {"temperature": 0}, {"temperature": "hot"}, {"reference_url": 5},
                    {"text": "a" * (contract.MAX_TEXT_CHARS + 1)}):
            with self.assertRaises(ValueError, msg=bad):
                contract.parse({**base, **bad})
        with self.assertRaises(ValueError):
            contract.parse({"model": "turbo", "text": "Hi.", "temperature": 2.5})
        with self.assertRaises(ValueError):
            contract.parse({"model": "turbo", "text": "Hi.", "exaggeration": "lots"})
        with self.assertRaises(ValueError):
            contract.parse(["not", "an", "object"])

    @unittest.skipUnless(os.environ.get("CHATTERBOX_UPSTREAM"), "installed chatterbox package needed")
    def test_lists_match_upstream(self):
        from chatterbox.mtl_tts import SUPPORTED_LANGUAGES, MULTILINGUAL_T3_MODELS
        self.assertEqual(contract.LANGUAGES, SUPPORTED_LANGUAGES)
        self.assertEqual(MULTILINGUAL_T3_MODELS["v3"], worker.MULTILINGUAL_T3)


class FakeConds:
    def __init__(self, t3, gen):
        self.t3, self.gen = t3, gen


class FakeModel:
    """Upstream's shape: generate() reads self.conds; prepare_conditionals() replaces it."""

    def __init__(self, variant, rate=24000):
        self.variant, self.sr = variant, rate
        self.conds = FakeConds("builtin-t3", {"voice": "builtin"})
        self.calls = []
        self.prepared = []

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5, **kwargs):
        info = sf.info(wav_fpath)  # read now: the job's folder is gone afterwards
        self.prepared.append({"path": wav_fpath, "exaggeration": exaggeration,
                              "info": (info.samplerate, info.channels, info.duration)})
        self.conds = FakeConds("clip-t3", {"voice": "clip"})

    def generate(self, text, **kwargs):
        assert "audio_prompt_path" not in kwargs
        self.calls.append({"text": text, "voice": self.conds.gen["voice"], **kwargs})
        self.conds.t3 = "touched"  # upstream reassigns .t3 when exaggeration changes
        return np.full((1, int(0.5 * self.sr)), 0.1, dtype=np.float32)


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required")
class HandlerTests(unittest.TestCase):
    def setUp(self):
        self.models = {v: FakeModel(v) for v in worker.VARIANTS}
        builtin = {v: m.conds for v, m in self.models.items()}
        self.enterContext(patch.dict(worker.ENGINES, self.models, clear=True))
        self.enterContext(patch.dict(worker.BUILTIN, builtin, clear=True))

    def run_job(self, inp, clip_seconds=8.0):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
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
                patch.object(worker, "download", fake_download), \
                patch.object(worker, "seed_everything", seeds.append):
            out = worker.handler({"id": "j", "input": inp})
        return out, uploads, seeds

    def test_turbo_design_long_text(self):
        text = " ".join(["This sentence is about fifty characters in length."] * 12) + " [chuckle] Done."
        out, uploads, seeds = self.run_job({"text": text, "seed": 42, "temperature": 0.6, "cfg_weight": 0.2})
        self.assertNotIn("error", out, out)
        model = self.models["turbo"]
        self.assertGreater(len(model.calls), 1)
        self.assertEqual(model.prepared, [])
        for call in model.calls:
            self.assertEqual(set(call) - {"text", "voice"}, {"temperature"})
            self.assertEqual((call["temperature"], call["voice"]), (0.6, "builtin"))
        self.assertIn("[chuckle]", model.calls[-1]["text"])
        self.assertEqual(seeds, [42] * len(model.calls))
        self.assertEqual((out["model"], out["mode"], out["seed"], out["sample_rate"]), ("turbo", "design", 42, 24000))
        self.assertEqual((out["sound_tags"], out["ignored_fields"]), (["[chuckle]"], ["cfg_weight"]))
        self.assertEqual(out["model_revision"], worker.TURBO_SHA)
        expected = round(0.5 * len(model.calls) + 0.15 * (len(model.calls) - 1), 2)
        self.assertAlmostEqual(out["duration_seconds"], expected, places=2)
        mp3 = uploads[out["key"]][0]
        self.assertEqual(uploads[out["key"]][1], "audio/mpeg")
        self.assertEqual(sf.info(uploads[out["wav_key"]][0]).samplerate, 24000)
        probe = worker.subprocess.run(["ffmpeg", "-hide_banner", "-i", str(mp3)], capture_output=True, text=True)
        self.assertIn("mp3", probe.stderr)
        self.assertIn("24000 Hz", probe.stderr)
        self.assertTrue(out["url"].startswith("https://signed.invalid/chatterbox/"))

    def test_original_clone_then_design_returns_to_builtin_voice(self):
        out, _, _ = self.run_job({"model": "original", "text": "Hello there. Nice day.",
                                  "reference_url": "https://h/a.mp3", "exaggeration": 0.8, "cfg_weight": 0.3},
                                 clip_seconds=25)
        self.assertNotIn("error", out, out)
        model = self.models["original"]
        (prepared,) = model.prepared
        rate, channels, seconds = prepared["info"]
        self.assertEqual((rate, channels, prepared["exaggeration"]), (24000, 1, 0.8))
        self.assertLessEqual(seconds, worker.REF_MAX_S + 0.05)
        (call,) = model.calls
        self.assertEqual(call, {"text": "Hello there. Nice day.", "voice": "clip", "exaggeration": 0.8,
                                "cfg_weight": 0.3, "temperature": 0.8})
        out, _, _ = self.run_job({"model": "original", "text": "Back to you."})
        self.assertNotIn("error", out, out)
        self.assertEqual(model.calls[-1]["voice"], "builtin")
        self.assertEqual(worker.BUILTIN["original"].t3, "builtin-t3")  # the kept voice was not touched

    def test_multilingual_passes_language(self):
        out, _, _ = self.run_job({"model": "multilingual", "language_id": "fr", "text": "Bonjour à tous.",
                                  "reference_url": "https://h/a.mp3", "cfg_weight": 0}, clip_seconds=3)
        self.assertNotIn("error", out, out)
        (call,) = self.models["multilingual"].calls
        self.assertEqual((call["language_id"], call["cfg_weight"], call["exaggeration"]), ("fr", 0.0, 0.5))
        self.assertEqual((out["language_id"], out["model_revision"]), ("fr", worker.BASE_SHA))

    def test_turbo_clip_must_be_longer_than_five_seconds(self):
        out, _, _ = self.run_job({"text": "Hello.", "reference_url": "https://h/a.mp3"}, clip_seconds=4)
        self.assertIn("longer than 5 seconds", out["error"])
        self.assertEqual(self.models["turbo"].calls, [])
        out, _, _ = self.run_job({"text": "Hello.", "reference_url": "https://h/a.mp3"}, clip_seconds=6)
        self.assertNotIn("error", out, out)
        self.assertEqual(self.models["turbo"].prepared[0]["exaggeration"], 0.5)  # upstream default, unused

    def test_bad_input_is_an_error_not_a_crash(self):
        out, _, _ = self.run_job({"model": "multilingual", "text": "Hi."})
        self.assertIn("language_id", out["error"])

    def test_download_refuses_other_hosts(self):
        with patch.dict("os.environ", {"AWS_ENDPOINT_URL": "https://s3.us-east-005.backblazeb2.com",
                                       "AWS_BUCKET_NAME": "Bucket", "CHATTERBOX_AUDIO_HOSTS": ""}):
            self.assertIn("bucket.s3.us-east-005.backblazeb2.com", worker.allowed_hosts())
            for url in ("http://s3.us-east-005.backblazeb2.com/a.mp3", "https://169.254.169.254/a",
                        "https://evil.example/a.mp3", "https://s3.us-east-005.backblazeb2.com:8443/a"):
                with self.assertRaises(ValueError):
                    worker.download(url, "/nonexistent")


class ModelDirTests(unittest.TestCase):
    def test_cache_paths(self):
        self.assertEqual(worker.default_dir("turbo"), "/runpod-volume/huggingface-cache/hub/models--ResembleAI--"
                         "chatterbox-turbo/snapshots/" + worker.TURBO_SHA)
        self.assertEqual(worker.default_dir("multilingual"), worker.default_dir("original"))
        for info in worker.VARIANTS.values():
            self.assertRegex(info["sha"], r"^[0-9a-f]{40}$")

    def test_cached_snapshot_is_used_and_missing_one_is_fetched_at_the_pinned_revision(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        cached, fetched = root / "cached", root / "fetched"
        for folder in (cached, fetched):
            folder.mkdir()
            for name in worker.VARIANTS["original"]["files"]:
                (folder / name).write_bytes(b"x")
        calls = []

        def snapshot_download(**kwargs):
            calls.append(kwargs)
            return str(fetched)

        hub = types.SimpleNamespace(snapshot_download=snapshot_download)
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            with patch.dict("os.environ", {"CHATTERBOX_DIR": str(cached)}):
                self.assertEqual(worker.model_dir("original"), cached)
            self.assertEqual(calls, [])
            with patch.dict("os.environ", {"CHATTERBOX_DIR": str(root / "missing")}):
                self.assertEqual(worker.model_dir("original"), fetched)
                (call,) = calls
                self.assertEqual((call["repo_id"], call["revision"]), (worker.BASE_REPO, worker.BASE_SHA))
                self.assertEqual(sorted(call["allow_patterns"]), sorted(worker.VARIANTS["original"]["files"]))
                with self.assertRaises(ValueError):
                    worker.model_dir("multilingual")  # the fetched folder lacks the multilingual files
            with patch.dict("os.environ", {"CHATTERBOX_DIR": str(root / "missing"), "CHATTERBOX_HF_FALLBACK": "0"}):
                with self.assertRaisesRegex(ValueError, "cache is not ready"):
                    worker.model_dir("original")


if __name__ == "__main__":
    unittest.main()
