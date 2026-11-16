"""Contract and handler checks with real FFmpeg/PCM; only the model, Perth and B2 are fakes."""
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

import dramabox_contract as contract
import dramabox_handler as worker

SCENE = 'A tired old sailor speaks slowly, "The sea gave me everything." He sighs deeply. "And took it back."'


class ContractTests(unittest.TestCase):
    def test_design_by_default(self):
        spec = contract.parse({"prompt": SCENE, "seed": 7})
        self.assertEqual(spec["mode"], "design")
        self.assertEqual(spec["seed"], 7)
        self.assertEqual((spec["cfg_scale"], spec["stg_scale"], spec["rescale_scale"], spec["duration_multiplier"],
                          spec["duration"], spec["reference_seconds"], spec["watermark"]),
                         (2.5, 1.5, "auto", 1.1, 0.0, 10.0, True))
        self.assertNotIn("reference_url", spec)
        self.assertGreater(contract.parse({"prompt": SCENE})["seed"], 0)

    def test_clone_from_reference(self):
        spec = contract.parse({"prompt": SCENE, "reference_url": " https://h/a.mp3 ", "reference_seconds": 20})
        self.assertEqual((spec["mode"], spec["reference_url"], spec["reference_seconds"]),
                         ("clone", "https://h/a.mp3", 20.0))
        self.assertEqual(contract.parse({"prompt": SCENE, "reference_voice_url": "https://h/a.mp3"})["mode"], "clone")

    def test_mode_must_match_the_clip(self):
        with self.assertRaises(ValueError):
            contract.parse({"prompt": SCENE, "mode": "design", "reference_url": "https://h/a.mp3"})
        with self.assertRaises(ValueError):
            contract.parse({"prompt": SCENE, "mode": "clone"})

    def test_prompt_rules(self):
        curly = contract.parse({"prompt": "A woman says,\n “Hello there.”"})["prompt"]
        self.assertEqual(curly, 'A woman says, "Hello there."')
        for bad in ("", "   ", "A woman speaks with no quotes at all.", 'She says, "Hello.', 'She says, " "',
                    "x" * (contract.MAX_PROMPT_CHARS + 1), 5):
            with self.assertRaises(ValueError, msg=bad):
                contract.parse({"prompt": bad})
        self.assertEqual(contract.parse({"text": SCENE})["prompt"], SCENE)

    def test_bad_knobs(self):
        for bad in ({"seed": -1}, {"seed": 1.5}, {"seed": True}, {"cfg_scale": 0.5}, {"cfg_scale": 11},
                    {"stg_scale": -1}, {"rescale_scale": 2}, {"rescale_scale": "high"},
                    {"duration_multiplier": 3}, {"duration": 46}, {"duration": 2}, {"reference_seconds": 2},
                    {"reference_seconds": 31}, {"watermark": "no"}, {"mode": "talk"}):
            with self.assertRaises(ValueError, msg=bad):
                contract.parse({"prompt": SCENE, **bad})
        spec = contract.parse({"prompt": SCENE, "rescale_scale": 0, "duration": 12, "watermark": False})
        self.assertEqual((spec["rescale_scale"], spec["duration"], spec["watermark"]), (0.0, 12.0, False))


class FakeModel:
    rate = 48000

    def __init__(self, estimate=8.0):
        self.calls = []
        self.fixed_estimate = estimate

    def estimate(self, prompt, multiplier):
        return self.fixed_estimate

    def _record(self, kind, prompt, kwargs):
        call = {"kind": kind, "prompt": prompt, **kwargs}
        if kwargs.get("voice_ref"):
            info = sf.info(kwargs["voice_ref"])  # read now: the job's folder is gone afterwards
            call["voice_ref:info"] = (info.samplerate, info.channels, info.duration)
        self.calls.append(call)

    def generate(self, prompt, **kwargs):
        self._record("generate", prompt, kwargs)
        return np.full((1, 2, int(1.5 * self.rate)), 0.2, dtype=np.float32), self.rate

    def generate_long(self, prompt, **kwargs):
        self._record("generate_long", prompt, kwargs)
        kwargs["progress_callback"](0, 3, 37.0)
        kwargs["progress_callback"](2, 3, 30.0)
        return np.full((2, int(2.0 * self.rate)), 0.1, dtype=np.float32), self.rate


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required")
class HandlerTests(unittest.TestCase):
    def run_job(self, inp, clip_seconds=12.0, estimate=8.0):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        model = FakeModel(estimate)
        uploads, marks, progress = {}, [], []

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

        def fake_watermark(audio, rate):
            marks.append((audio.shape, rate))
            return audio * 0.5

        modules = {"boto3": types.SimpleNamespace(client=lambda *a, **k: Storage()),
                   "runpod": types.SimpleNamespace(serverless=types.SimpleNamespace(
                       progress_update=lambda job, message: progress.append(message)))}
        env = {"AWS_ENDPOINT_URL": "https://s3.example.invalid", "AWS_BUCKET_NAME": "b"}
        with patch.dict(sys.modules, modules), patch.dict("os.environ", env), \
                patch.object(worker, "ENGINE", model), patch.object(worker, "download", fake_download), \
                patch.object(worker, "watermark", fake_watermark):
            out = worker.handler({"id": "j", "input": inp})
        return out, model, uploads, marks, progress

    def test_design_single_take(self):
        out, model, uploads, marks, _ = self.run_job({"prompt": SCENE, "seed": 42, "cfg_scale": 3,
                                                     "stg_scale": 1.0, "rescale_scale": 0.5})
        self.assertNotIn("error", out, out)
        (call,) = model.calls
        self.assertEqual(call["kind"], "generate")
        self.assertEqual(call["prompt"], SCENE)
        self.assertIsNone(call["voice_ref"])
        self.assertEqual((call["seed"], call["cfg_scale"], call["stg_scale"], call["rescale_scale"],
                          call["gen_duration"], call["denoise_ref"]), (42, 3.0, 1.0, 0.5, 0.0, False))
        self.assertEqual(marks, [((2, 72000), 48000)])
        self.assertEqual((out["mode"], out["seed"], out["sample_rate"], out["channels"], out["parts"]),
                         ("design", 42, 48000, 2, 1))
        self.assertAlmostEqual(out["duration_seconds"], 1.5, places=2)
        self.assertEqual(out["target_seconds"], 8.0)
        self.assertTrue(out["watermark"])
        mp3 = uploads[out["key"]][0]
        self.assertEqual(uploads[out["key"]][1], "audio/mpeg")
        wav = uploads[out["wav_key"]][0]
        data, rate = sf.read(wav)
        self.assertEqual((rate, data.shape[1]), (48000, 2))
        self.assertAlmostEqual(float(np.abs(data).max()), 0.1, places=2)  # the watermark output was written
        probe = worker.subprocess.run(["ffmpeg", "-hide_banner", "-i", str(mp3)], capture_output=True, text=True)
        self.assertIn("mp3", probe.stderr)
        self.assertIn("48000 Hz", probe.stderr)
        self.assertTrue(out["url"].startswith("https://signed.invalid/dramabox/"))
        self.assertIn("render_ms", out)

    def test_clone_uses_the_asked_seconds(self):
        out, model, _, _, _ = self.run_job({"prompt": SCENE, "reference_url": "https://h/a.mp3",
                                            "reference_seconds": 6, "watermark": False, "duration": 20})
        self.assertNotIn("error", out, out)
        (call,) = model.calls
        rate, channels, seconds = call["voice_ref:info"]
        self.assertEqual((rate, channels), (48000, 2))
        self.assertAlmostEqual(seconds, 6.0, places=1)
        self.assertEqual((call["ref_duration"], call["gen_duration"]), (6.0, 20.0))
        self.assertEqual((out["mode"], out["reference_seconds"], out["target_seconds"], out["watermark"]),
                         ("clone", 6.0, 20.0, False))

    def test_short_clip_is_used_whole(self):
        out, model, _, _, _ = self.run_job({"prompt": SCENE, "reference_url": "https://h/a.mp3"}, clip_seconds=4)
        self.assertNotIn("error", out, out)
        self.assertAlmostEqual(model.calls[0]["ref_duration"], 4.0, places=1)

    def test_too_short_clip_is_refused(self):
        out, model, _, _, _ = self.run_job({"prompt": SCENE, "reference_url": "https://h/a.mp3"}, clip_seconds=2)
        self.assertIn("at least 3 seconds", out["error"])
        self.assertEqual(model.calls, [])

    def test_long_scene_is_split(self):
        out, model, _, marks, progress = self.run_job({"prompt": SCENE, "seed": 5}, estimate=100.0)
        self.assertNotIn("error", out, out)
        (call,) = model.calls
        self.assertEqual(call["kind"], "generate_long")
        self.assertEqual((call["max_chunk_duration"], call["target_chunk_duration"], call["crossfade_ms"]),
                         (45.0, 37.0, 50.0))
        self.assertNotIn("gen_duration", call)
        self.assertEqual(out["parts"], 3)
        self.assertEqual(progress[-1], "DramaBox: part 3 of 3")
        self.assertEqual(marks, [((2, 96000), 48000)])

    def test_too_long_scene_is_refused(self):
        out, model, _, _, _ = self.run_job({"prompt": SCENE}, estimate=400.0)
        self.assertIn("too long", out["error"])
        self.assertEqual(model.calls, [])

    def test_bad_input_is_an_error_not_a_crash(self):
        out, model, _, _, _ = self.run_job({"prompt": "No quotes here."})
        self.assertIn("double quotes", out["error"])

    def test_download_refuses_other_hosts(self):
        with patch.dict("os.environ", {"AWS_ENDPOINT_URL": "https://s3.us-east-005.backblazeb2.com",
                                       "AWS_BUCKET_NAME": "Bucket", "DRAMABOX_AUDIO_HOSTS": ""}):
            self.assertIn("bucket.s3.us-east-005.backblazeb2.com", worker.allowed_hosts())
            for url in ("http://s3.us-east-005.backblazeb2.com/a.mp3", "https://169.254.169.254/a",
                        "https://evil.example/a.mp3", "https://s3.us-east-005.backblazeb2.com:8443/a"):
                with self.assertRaises(ValueError):
                    worker.download(url, "/nonexistent")

    def test_cache_paths_follow_runpod_layout(self):
        path = worker.cached_snapshot(worker.MODEL_REPO, worker.MODEL_SHA)
        self.assertEqual(path.as_posix(), "/runpod-volume/huggingface-cache/hub/models--ResembleAI--Dramabox/"
                                          "snapshots/404f967f653fa1170dc15a9d1ddd3fdb9a0a842d")
        with patch.dict("os.environ", {"DRAMABOX_ALLOW_DOWNLOAD": "0", "DRAMABOX_MODEL_DIR": "/nonexistent/a",
                                       "DRAMABOX_GEMMA_DIR": "/nonexistent/b"}):
            with self.assertRaises(ValueError):
                worker.model_dirs()


if __name__ == "__main__":
    unittest.main()
