"""CPU tests of the voice worker: numpy + ffmpeg, no torch, no GPU, no bucket, no network.

The separators and RVC are stand-ins written into a temporary folder laid out like the RVC repo (tools/pymss/cli.py, infer/cli.py),
so voice_pipeline.Runner starts real subprocesses with the real command lines, and each stand-in records the arguments and
environment it was given. The bucket is a fake with the three boto3 calls the worker makes.
Run: python -m unittest -v test_voice.py   (ffmpeg on PATH or VP_FFMPEG)"""
import json
import os
import shutil
import sys
import tempfile
import textwrap
import types
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import voice_models  # noqa: E402
import voice_pipeline  # noqa: E402
import voice_request  # noqa: E402
import vpaudio as va  # noqa: E402

SR = 44100
MODEL = "voice-models/u123/abc/model.pth"
INDEX = "voice-models/u123/abc/model.index"

PYMSS_STANDIN = textwrap.dedent('''
    """Stand-in for RVC's pymss CLI. FAKE_PLAN (JSON) says what each model does: vocals / karaoke / karaoke_weak / named /
    named_weak / dereverb / fail / silent. Records every call to FAKE_LOG."""
    import glob, json, os, sys
    sys.path.insert(0, os.environ["FAKE_VOICE_DIR"])
    import numpy as np
    import vpaudio as va
    a = sys.argv[1:]
    assert a[0] == "infer", a
    model = a[1]
    opt = {a[i]: a[i + 1] for i in range(2, len(a) - 1) if a[i].startswith("-")}
    with open(os.environ["FAKE_LOG"], "a") as f:
        f.write(json.dumps({"tool": "pymss", "argv": a}) + "\\n")
    what = json.loads(os.environ["FAKE_PLAN"]).get(model, "fail")
    if what == "fail":
        sys.exit(3)
    os.makedirs(opt["-o"], exist_ok=True)
    for path in glob.glob(os.path.join(opt["-i"], "*.wav")):
        stem = os.path.splitext(os.path.basename(path))[0]
        x, sr = va.decode(path)
        out = lambda instr, y: va.encode(os.path.join(opt["-o"], f"{stem}_{instr}.wav"), y, sr, bits=32)
        if what == "vocals":
            out("Vocals", x * 0.5); out("Instrumental", x * 0.5)
        elif what == "silent":
            out("vocals", x * 0.0); out("other", x)
        elif what == "karaoke":
            out("karaoke", x * 0.8); out("instrumental", x * 0.2)
        elif what == "karaoke_weak":
            out("karaoke", x * 0.3); out("instrumental", x * 0.3)
        elif what == "named":  # the frazer & becruily layout: Vocals is the lead, Instrumental the rest
            out("Vocals", x * 0.8); out("Instrumental", x * 0.2)
        elif what == "named_weak":
            out("Vocals", x * 0.3); out("Instrumental", x * 0.7)
        elif what == "dereverb":
            out("noreverb", x * 0.9); out("reverb", x * 0.1)
''')

RVC_STANDIN = textwrap.dedent('''
    """Stand-in for RVC's infer/cli.py: the input 25 ms late and 6 dB quieter, like a real conversion that is a little off."""
    import json, os, sys
    sys.path.insert(0, os.environ["FAKE_VOICE_DIR"])
    import numpy as np
    import vpaudio as va
    a = sys.argv[1:]
    opt = {a[i]: (a[i + 1] if i + 1 < len(a) and not a[i + 1].startswith("--") else True) for i in range(len(a)) if a[i].startswith("--")}
    with open(os.environ["FAKE_LOG"], "a") as f:
        f.write(json.dumps({"tool": "rvc", "argv": a, "PYTHONSAFEPATH": os.environ.get("PYTHONSAFEPATH"),
                            "PYTHONPATH": os.environ.get("PYTHONPATH"), "cwd": os.getcwd()}) + "\\n")
    x, sr = va.decode(opt["--input"], channels=1)
    lag = int(0.025 * sr)
    y = np.concatenate([np.zeros((lag, 1), np.float32), x[:-lag]]) * np.float32(10 ** (-6 / 20))
    va.encode(opt["--output"], va.resample(y, sr, 48000), 48000, bits=32)  # RVC answers at the model's 48 kHz
''')


def tone_song(seconds=6.0, f0=220.0, sr=SR, band_level=0.3):
    """A sung-like harmonic tone with a slow vibrato over a quiet 110 Hz band; band_level=0 gives a dry vocal."""
    t = np.arange(int(seconds * sr)) / sr
    phase = 2 * np.pi * np.cumsum(f0 * 2 ** (0.3 * np.sin(2 * np.pi * 0.5 * t) / 12)) / sr
    voice = sum(np.sin(k * phase) / k for k in range(1, 8)) * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t) ** 2)
    band = band_level * (np.sin(2 * np.pi * 110 * t) + 0.1 * np.random.default_rng(3).standard_normal(len(t)))
    mix = np.stack([0.3 * voice + band, 0.3 * voice + band], 1)
    return (mix / np.abs(mix).max() * 0.7).astype(np.float32)


class FakeS3:
    def __init__(self, objects):
        self.objects = dict(objects)
        self.downloads, self.uploads, self.heads = [], [], []

    def head_object(self, Bucket, Key):
        self.heads.append(Key)
        if Key not in self.objects:
            raise type("NoSuchKey", (Exception,), {})()
        data = self.objects[Key]
        return {"ETag": f'"{hash(data) & 0xffff:x}"', "ContentLength": len(data)}

    def download_file(self, bucket, key, dest):
        self.downloads.append(key)
        if key not in self.objects:
            raise type("NoSuchKey", (Exception,), {})()
        with open(dest, "wb") as f:
            f.write(self.objects[key])

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        self.uploads.append((key, ExtraArgs["ContentType"], os.path.getsize(path)))

    def generate_presigned_url(self, op, Params, ExpiresIn):
        return f"https://bucket.example/{Params['Key']}?X-Amz-Expires={ExpiresIn}"


def EX(name):
    return voice_request.EXTRACTORS[name]["model"]


def LEAD(name):
    return voice_request.LEAD_MODELS[name]["model"]


ROUND1 = {"extractor": "bs_roformer", "fallback": "demucs", "lead_model": "aufr33", "dereverb": True, "soft_s": False}


def request(**over):
    raw = {"mode": "song", "audio_key": "yue2/abc/master.wav", "model_key": MODEL, "index_key": INDEX, "pitch": "auto"}
    raw.update(over)
    return voice_request.parse(raw, resolve=False)


class Base(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="voicetest-")
        self.rvc = os.path.join(self.td, "rvc")
        os.makedirs(os.path.join(self.rvc, "tools", "pymss"))
        os.makedirs(os.path.join(self.rvc, "infer"))
        for p in ("tools/__init__.py", "tools/pymss/__init__.py"):
            open(os.path.join(self.rvc, p), "w").close()
        with open(os.path.join(self.rvc, "tools", "pymss", "cli.py"), "w") as f:
            f.write(PYMSS_STANDIN)
        with open(os.path.join(self.rvc, "infer", "cli.py"), "w") as f:
            f.write(RVC_STANDIN)
        self.log = os.path.join(self.td, "calls.jsonl")
        self.env_before = dict(os.environ)
        os.environ.update({"FAKE_VOICE_DIR": HERE, "FAKE_LOG": self.log, "VOICE_SELFTEST": os.path.join(self.td, "none.json"),
                           "VOICE_CACHE_DIR": os.path.join(self.td, "cache")})
        self.plan({})

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.env_before)
        shutil.rmtree(self.td, ignore_errors=True)

    def plan(self, extra):
        plan = {EX("hyperace"): "vocals", EX("bs_roformer"): "vocals", EX("demucs"): "vocals", LEAD("frazer"): "named",
                LEAD("aufr33"): "karaoke", voice_request.DEREVERB_MODEL: "dereverb"}
        plan.update(extra)
        os.environ["FAKE_PLAN"] = json.dumps(plan)

    def runner(self):
        return voice_pipeline.Runner(os.path.join(self.td, "tools.log"), env={
            "VOICE_RVC_DIR": self.rvc, "VOICE_RVC_PYTHON": sys.executable, "PYMSS_MODEL_DIR": os.path.join(self.td, "models"),
            "PYMSS_ENDPOINT": "https://models.example/pinned", "VOICE_DEVICE": "cpu"})

    def calls(self, tool=None):
        if not os.path.exists(self.log):
            return []
        with open(self.log) as f:
            rows = [json.loads(line) for line in f]
        return [r for r in rows if tool is None or r["tool"] == tool]

    def song_file(self, x=None, name="song.wav"):
        path = os.path.join(self.td, name)
        va.encode(path, tone_song() if x is None else x, SR, bits=16)
        return path

    def run_job(self, req, audio=None, index=True):
        work = os.path.join(self.td, "work")
        os.makedirs(work, exist_ok=True)
        said = []
        out = voice_pipeline.run(req, audio or self.song_file(), os.path.join(self.td, "m.pth"),
                                 os.path.join(self.td, "m.index") if index else None, work, self.runner(), progress=said.append)
        return out, said


class RequestTests(unittest.TestCase):
    def test_defaults_are_the_round2_chain(self):
        r = request()
        self.assertEqual(r["options"], {"extractor": "hyperace", "fallback": "bs_roformer", "lead_split": True,
                                        "lead_model": "frazer", "dereverb": False, "room": True, "soft_s": True, "index_rate": 0.5,
                                        "protect": 0.33, "rms_mix_rate": 0.25, "f0_method": "rmvpe"})
        self.assertEqual({k: request(options=ROUND1)["options"][k] for k in ROUND1}, ROUND1)  # round 1 is one request away
        self.assertEqual(r["pitch"], "auto")
        self.assertTrue(r["output_prefix"].startswith("voice/"))

    def test_options_and_pitch(self):
        r = request(pitch=-12, options={"extractor": "melband_kim", "fallback": "none", "lead_split": "off", "index_rate": 0.3,
                                        "protect": 0.2})
        self.assertEqual((r["pitch"], r["options"]["extractor"], r["options"]["fallback"], r["options"]["lead_split"]),
                         (-12, "melband_kim", None, False))
        self.assertEqual(request(options={"extractor": "bs_roformer"})["options"]["fallback"], None)  # never itself

    def test_refusals_are_sentences(self):
        bad = [dict(audio_key="../etc/passwd"), dict(audio_key="private/other.wav"), dict(model_key="voice-models/u1/x.bin"),
               dict(model_key="models/u1/x.pth"), dict(index_key="voice-models/u999/abc/model.index"), dict(pitch=30),
               dict(pitch=1.5), dict(pitch=True), dict(options={"extractor": "magic"}), dict(options={"protect": 0.9}),
               dict(options={"index_rate": "high"}), dict(options={"lead_split": "maybe"}), dict(mode="karaoke"),
               dict(options={"lead_model": "magic"}), dict(options={"soft_s": "sometimes"}),
               dict(output_prefix="yue2/x"), dict(voice_range={"p05": 70, "p50": 60, "p95": 80}), dict(model_sha256="xyz"),
               dict(audio_url="https://bucket.example/a.wav")]
        for over in bad:
            with self.assertRaises(ValueError, msg=over) as caught:
                request(**over)
            self.assertTrue(caught.exception.args[0].endswith("."), over)

    def test_urls_need_an_allowed_host(self):
        os.environ["VOICE_AUDIO_HOSTS"] = "s3.example.com"
        try:
            raw = {"audio_url": "https://s3.example.com/b/audios/x.mp3?X-Amz-Signature=1", "model_key": MODEL}
            self.assertEqual(voice_request.parse(raw, resolve=False)["audio_url"], raw["audio_url"])
            for url in ("http://s3.example.com/x", "https://evil.example/x", "https://user:pw@s3.example.com/x",
                        "https://s3.example.com:8443/x"):
                with self.assertRaises(ValueError):
                    voice_request.parse({"audio_url": url, "model_key": MODEL}, resolve=False)
        finally:
            del os.environ["VOICE_AUDIO_HOSTS"]


class ModelCacheTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="voicecache-")
        os.environ["VOICE_CACHE_DIR"] = self.td

    def tearDown(self):
        del os.environ["VOICE_CACHE_DIR"]
        shutil.rmtree(self.td, ignore_errors=True)

    def test_by_hash(self):
        s3 = FakeS3({MODEL: b"model-bytes"})
        import hashlib
        sha = hashlib.sha256(b"model-bytes").hexdigest()
        path, info = voice_models.fetch(s3, "b", MODEL, ".pth", sha)
        self.assertEqual((os.path.basename(path), info["cached"]), (sha + ".pth", False))
        path2, info2 = voice_models.fetch(s3, "b", MODEL, ".pth", sha)
        self.assertEqual((path2, info2["cached"], len(s3.downloads)), (path, True, 1))  # no second download

    def test_mismatch_is_refused_and_nothing_kept(self):
        s3 = FakeS3({MODEL: b"tampered"})
        with self.assertRaises(ValueError):
            voice_models.fetch(s3, "b", MODEL, ".pth", "0" * 64)
        self.assertEqual([p for p in os.listdir(self.td) if not p.startswith("alias")], [])

    def test_without_hash_uses_an_alias(self):
        s3 = FakeS3({INDEX: b"index-bytes"})
        path, info = voice_models.fetch(s3, "b", INDEX, ".index")
        path2, info2 = voice_models.fetch(s3, "b", INDEX, ".index")
        self.assertEqual((path2, info2["cached"], len(s3.downloads), len(s3.heads)), (path, True, 1, 2))

    def test_missing_model_is_a_sentence(self):
        with self.assertRaises(ValueError):
            voice_models.fetch(FakeS3({}), "b", MODEL, ".pth", "a" * 64)

    def test_prune_keeps_the_newest(self):
        os.environ["VOICE_CACHE_FILES"] = "2"
        try:
            s3 = FakeS3({f"voice-models/u/m{i}.pth": f"m{i}".encode() for i in range(4)})
            for i in range(4):
                voice_models.fetch(s3, "b", f"voice-models/u/m{i}.pth", ".pth")
            self.assertEqual(len([p for p in os.listdir(self.td) if p.endswith(".pth")]), 2)
        finally:
            del os.environ["VOICE_CACHE_FILES"]


class PipelineTests(Base):
    def test_default_song_chain(self):
        out, said = self.run_job(request())
        r = out["report"]
        self.assertEqual([c["argv"][1] for c in self.calls("pymss")], [EX("hyperace"), LEAD("frazer")])  # no dereverb
        sep = r["separation"]
        self.assertEqual((sep["vocals_by"], sep["lead_split"], sep["lead_model"], sep["dereverb"]),
                         ("hyperace", "used", "frazer", "off"))
        self.assertAlmostEqual(sep["lead_energy_share"], 0.64, delta=0.02)  # the named Vocals stem (0.8 of the vocals), not a guess
        self.assertNotIn("room_db", r["placed"])  # no dereverb, so no room added back
        self.assertIn("soft_s", r["placed"])
        self.assertTrue(r["settings"]["soft_s"])
        self.assertEqual(r["notes"], [])
        self.assertEqual(said, ["Splitting the voice from the music", "Finding the lead singer", "Singing it in your voice",
                                "Putting the song back together"])
        mix, _ = va.decode(out["files"]["mix.wav"])
        self.assertEqual(mix.shape, (len(tone_song()), 2))

    def test_named_lead_stem_missing_is_not_guessed(self):
        self.plan({LEAD("frazer"): "karaoke"})  # outputs named karaoke/instrumental: no Vocals stem to trust
        out, _ = self.run_job(request())
        self.assertTrue(out["report"]["separation"]["lead_split"].startswith("failed"))
        self.assertTrue(any("every voice was re-sung" in note for note in out["report"]["notes"]))

    def test_round1_chain_and_command_lines(self):
        out, said = self.run_job(request(pitch=3, options=ROUND1))
        r = out["report"]
        pymss = self.calls("pymss")
        self.assertEqual([c["argv"][1] for c in pymss], [EX("bs_roformer"), LEAD("aufr33"), voice_request.DEREVERB_MODEL])
        argv = pymss[0]["argv"]
        self.assertEqual(argv[2:4], ["--endpoint", "https://models.example/pinned"])  # the pinned endpoint, right after the model
        self.assertIn("--device", argv)
        rvc = self.calls("rvc")[0]
        self.assertEqual((rvc["PYTHONSAFEPATH"], rvc["PYTHONPATH"]), ("1", self.rvc))
        flags = rvc["argv"]
        self.assertEqual(flags[flags.index("--pitch") + 1], "3")
        self.assertEqual(flags[flags.index("--index-rate") + 1], "0.5")
        self.assertEqual(flags[flags.index("--protect") + 1], "0.33")
        self.assertEqual(flags[flags.index("--rms-mix-rate") + 1], "0.25")
        self.assertIn("--index", flags)
        sep = r["separation"]
        self.assertEqual((sep["vocals_by"], sep["lead_split"], sep["dereverb"]), ("bs_roformer", "used", "used"))
        self.assertAlmostEqual(sep["reverb_ratio_db"], 20 * np.log10(0.1 / 0.9), delta=0.2)
        self.assertAlmostEqual(r["placed"]["lag_ms"], 25.0, delta=2.0)  # the stand-in's 25 ms is taken back out
        self.assertTrue(-30 <= r["placed"]["room_db"] <= -6)
        self.assertNotIn("soft_s", r["placed"])
        self.assertEqual(r["notes"], [])
        self.assertEqual(said, ["Splitting the voice from the music", "Finding the lead singer", "Taking the room off the voice",
                                "Singing it in your voice", "Putting the song back together"])
        n = len(tone_song())
        for name in ("mix.mp3", "mix.wav", "vocal.mp3", "vocal.wav"):
            self.assertTrue(os.path.exists(out["files"][name]), name)
        mix, _ = va.decode(out["files"]["mix.wav"])
        vocal, _ = va.decode(out["files"]["vocal.wav"])
        self.assertEqual((len(mix), mix.shape[1], len(vocal), vocal.shape[1]), (n, 2, n, 1))
        self.assertLessEqual(float(np.abs(mix).max()), 10 ** (-1 / 20) + 1e-3)  # the limiter's ceiling
        orig = va.measure(va.decode(self.song_file())[0], SR)["I"]
        self.assertAlmostEqual(va.measure(mix, SR)["I"], orig, delta=1.0)  # back to the original's loudness

    def test_fallback_extractor(self):
        self.plan({EX("hyperace"): "fail"})
        out, _ = self.run_job(request())
        sep = out["report"]["separation"]
        self.assertEqual((sep["vocals_by"], sep["fallback_used"]), ("bs_roformer", "bs_roformer"))
        self.assertTrue(any("could not split" in note for note in out["report"]["notes"]))

    def test_no_voice_anywhere_is_a_sentence(self):
        self.plan({EX("hyperace"): "silent", EX("bs_roformer"): "fail"})
        with self.assertRaises(ValueError) as caught:
            self.run_job(request())
        self.assertIn("another vocal extractor", caught.exception.args[0])
        self.assertEqual(self.calls("rvc"), [])  # nothing converted, nothing spent on RVC

    def test_weak_lead_split_resings_every_voice(self):
        for model, kind, opts in ((LEAD("frazer"), "named_weak", {}), (LEAD("aufr33"), "karaoke_weak", {"lead_model": "aufr33"})):
            self.plan({model: kind})
            out, _ = self.run_job(request(options=opts))
            self.assertTrue(out["report"]["separation"]["lead_split"].startswith("misfired"), kind)
            self.assertTrue(any("every voice was re-sung" in note for note in out["report"]["notes"]), kind)

    def test_switches_off(self):
        out, _ = self.run_job(request(options={"lead_split": False, "dereverb": False, "soft_s": False, "index_rate": 0}), index=True)
        self.assertEqual([c["argv"][1] for c in self.calls("pymss")], [EX("hyperace")])
        self.assertNotIn("soft_s", out["report"]["placed"])
        flags = self.calls("rvc")[0]["argv"]
        self.assertNotIn("--index", flags)  # index rate 0: no index
        self.assertNotIn("room_db", out["report"]["placed"])  # no dereverb, so no room added back

    def test_vocal_mode_skips_separation(self):
        out, said = self.run_job(request(mode="vocal"), index=False)
        self.assertEqual(self.calls("pymss"), [])
        self.assertNotIn("--index", self.calls("rvc")[0]["argv"])
        self.assertEqual(out["files"]["mix.mp3"], out["files"]["vocal.mp3"])
        self.assertEqual(said, ["Singing it in your voice", "Finishing your vocal"])

    def test_extractor_that_failed_the_build_check_is_skipped(self):
        with open(os.environ["VOICE_SELFTEST"], "w") as f:
            json.dump({"extractors": {"hyperace": False, "bs_roformer": True}}, f)
        out, _ = self.run_job(request())
        self.assertEqual(out["report"]["separation"]["vocals_by"], "bs_roformer")
        self.assertTrue(any("failed the worker's own check" in note for note in out["report"]["notes"]))

    def test_octave_rule(self):
        high = tone_song(f0=880.0, band_level=0)  # a dry vocal on A5, MIDI 81
        rng = {"p05": 55.0, "p50": 62.0, "p95": 70.0}
        out, _ = self.run_job(request(mode="vocal", voice_range=rng), audio=self.song_file(high, "high.wav"), index=False)
        self.assertEqual((out["report"]["pitch"]["shift"], out["report"]["pitch"]["source"]), (-12, "auto"))
        self.assertEqual(self.calls("rvc")[0]["argv"][self.calls("rvc")[0]["argv"].index("--pitch") + 1], "-12")
        out, _ = self.run_job(request(mode="vocal"), audio=self.song_file(high, "high2.wav"), index=False)
        self.assertEqual(out["report"]["pitch"]["shift"], 0)
        self.assertIn("no voice range", out["report"]["pitch"]["why"])

    def test_silence_and_length(self):
        with self.assertRaises(ValueError):
            self.run_job(request(), audio=self.song_file(np.zeros((SR * 4, 2), np.float32), "quiet.wav"))
        long = os.path.join(self.td, "long.wav")
        va.encode(long, np.zeros((int(SR * (voice_request.MAX_SECONDS + 3)), 1), np.float32) + 0.1, SR)
        with self.assertRaises(ValueError) as caught:
            self.run_job(request(), audio=long)
        self.assertIn("up to six minutes", caught.exception.args[0])


class SoftSTests(unittest.TestCase):
    def test_s_hiss_comes_from_the_input_and_the_vowels_stay(self):
        """A sung tone with bursts of hiss (the S sounds) in; the 'converted' voice has the same tone but a louder, buzzy 6 kHz
        whistle where each S was. After the blend the S frames carry the input's own hiss at the input's S-to-vowel balance, and
        the vowels are untouched."""
        rng = np.random.default_rng(5)
        n = SR * 4
        t = np.arange(n) / SR
        tone = sum(np.sin(2 * np.pi * 220 * k * t) / k for k in range(1, 8)).astype(np.float32) * 0.3
        s_mask = ((t % 1.0) > 0.7).astype(np.float32)  # 0.3 s of S every second
        spec = np.fft.rfft(rng.standard_normal(n))
        spec[np.fft.rfftfreq(n, 1 / SR) < 4500] = 0
        hiss = (np.fft.irfft(spec, n) * 0.1).astype(np.float32)
        src = tone * (1 - s_mask) + hiss * s_mask
        conv = tone * (1 - s_mask) + (0.2 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32) * s_mask
        y, info = va.unvoiced_blend(conv, src, SR)
        self.assertEqual(len(y), n)
        self.assertGreater(info["blend_frames"], 50)
        band = va._band(SR, 2048, 4500, 11000)
        m, _ = va.sib_frames(src, SR)
        core = m & np.roll(m, 3) & np.roll(m, -3)  # away from the fades

        def hf_db(x):
            return 10 * np.log10((np.abs(va.stft(x)[: len(m)][core][:, band]) ** 2).sum(1).mean())
        self.assertLess(abs(hf_db(y) - (hf_db(src) + info["src_gain_db"])), 1.5)  # the input's hiss, level-matched
        self.assertGreater(hf_db(conv) - hf_db(y), 3)  # the whistle is gone
        vowels = s_mask[: len(y)] == 0
        vowels &= np.roll(vowels, SR // 10) & np.roll(vowels, -SR // 10)
        self.assertLess(float(np.abs(y[vowels] - conv[vowels]).max()), 1e-3)  # vowels exactly as converted

    def test_no_s_sounds_leaves_the_voice_as_it_was(self):
        t = np.arange(SR * 3) / SR
        tone = sum(np.sin(2 * np.pi * 220 * k * t) / k for k in range(1, 8)).astype(np.float32) * 0.3
        y, info = va.unvoiced_blend(tone, tone, SR)
        self.assertEqual(info["blend_frames"], 0)
        self.assertLess(float(np.abs(y - tone).max()), 1e-3)


class HandlerTests(Base):
    def test_output_shape_and_one_copy_in_vocal_mode(self):
        import voice_handler
        with open(self.song_file(), "rb") as f:
            s3 = FakeS3({"voice/abcdefgh12/in.wav": f.read(), MODEL: b"m", INDEX: b"i"})
        progress = []
        fake_runpod = types.SimpleNamespace(serverless=types.SimpleNamespace(progress_update=lambda job, text: progress.append(text)))
        sys.modules["runpod"] = fake_runpod
        runner = self.runner()
        saved = (voice_handler.storage, voice_pipeline.Runner, dict(voice_handler.GPU))
        voice_handler.storage = lambda: (s3, "bucket")
        voice_pipeline.Runner = lambda log: runner
        voice_handler.GPU.clear()
        voice_handler.GPU.update({"cuda": True, "name": "NVIDIA GeForce RTX 4090", "capability": "sm_89", "supported": True})
        try:
            out = voice_handler.handler({"input": {"mode": "vocal", "audio_key": "voice/abcdefgh12/in.wav", "model_key": MODEL,
                                                   "output_prefix": "voice/testjob0001"}})
            self.assertNotIn("error", out, out)
            self.assertEqual((out["gpu"], out["mode"], out["key"], out["vocal_key"]),
                             ("NVIDIA GeForce RTX 4090", "vocal", "voice/testjob0001/vocal.mp3", "voice/testjob0001/vocal.mp3"))
            self.assertEqual(sorted(k for k, _, _ in s3.uploads), ["voice/testjob0001/report.json", "voice/testjob0001/vocal.mp3",
                                                                   "voice/testjob0001/vocal.wav"])
            self.assertTrue(out["url"].startswith("https://bucket.example/voice/testjob0001/vocal.mp3"))
            self.assertIn("timing", out)
            self.assertEqual(progress[0], "Fetching the recording and your voice")
            bad = voice_handler.handler({"input": {"audio_key": "voice/x/in.wav", "model_key": "nope"}})
            self.assertEqual(set(bad), {"error"})
            voice_handler.GPU.update({"supported": False, "name": "NVIDIA GeForce RTX 5090"})
            refused = voice_handler.handler({"input": {"mode": "vocal", "audio_key": "voice/abcdefgh12/in.wav", "model_key": MODEL}})
            self.assertIn("RTX 50", refused["error"])
        finally:
            voice_handler.storage, voice_pipeline.Runner = saved[0], saved[1]
            voice_handler.GPU.clear()
            voice_handler.GPU.update(saved[2])
            sys.modules.pop("runpod", None)


if __name__ == "__main__":
    unittest.main()
