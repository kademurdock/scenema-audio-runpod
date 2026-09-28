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
import vocalfx  # noqa: E402
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
        work = tempfile.mkdtemp(prefix="work-", dir=self.td)  # a fresh folder per job, as each worker job gets
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
               dict(audio_url="https://bucket.example/a.wav"), dict(vocal_fx="cathedral"), dict(vocal_fx=3),
               dict(vocal_fx=True), dict(vocal_fx="")]
        for over in bad:
            with self.assertRaises(ValueError, msg=over) as caught:
                request(**over)
            self.assertTrue(caught.exception.args[0].endswith("."), over)

    def test_vocal_fx_is_optional_and_named(self):
        self.assertEqual(request()["vocal_fx"], "none")  # nothing asked: exactly today's output
        self.assertEqual(request(vocal_fx=None)["vocal_fx"], "none")
        self.assertEqual(voice_request.VOCAL_FX, vocalfx.NAMES)  # the contract and the effects list the same presets
        for name in vocalfx.NAMES:
            self.assertEqual(request(vocal_fx=name)["vocal_fx"], name)
        self.assertEqual(request(vocal_fx=" Plate ")["vocal_fx"], "plate")
        self.assertEqual(request(vocal_fx="echo")["options"], request()["options"])  # the chain before the effect is untouched

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

    def test_no_vocal_fx_is_byte_identical(self):
        """vocal_fx left out and vocal_fx "none" make the same files, byte for byte, and no effect files."""
        song = self.song_file()
        plain, said = self.run_job(request(), audio=song)
        none, _ = self.run_job(request(vocal_fx="none"), audio=song)
        for name in ("mix.wav", "mix.mp3", "vocal.wav", "vocal.mp3"):
            with open(plain["files"][name], "rb") as a, open(none["files"][name], "rb") as b:
                self.assertEqual(a.read(), b.read(), name)
        for out in (plain, none):
            self.assertNotIn("vocal_fx.wav", out["files"])
            self.assertIsNone(out["report"]["vocal_fx"])
        self.assertNotIn("Adding the vocal effect", said)

    def test_vocal_fx_goes_on_the_lead_and_the_dry_vocal_stays_dry(self):
        song = self.song_file()
        dry, _ = self.run_job(request(), audio=song)
        wet, said = self.run_job(request(vocal_fx="echo"), audio=song)
        with open(dry["files"]["vocal.wav"], "rb") as a, open(wet["files"]["vocal.wav"], "rb") as b:
            self.assertEqual(a.read(), b.read())  # the plain dry vocal is never touched
        self.assertIn("Adding the vocal effect", said)
        self.assertEqual(said[-1], "Putting the song back together")
        fx = wet["report"]["vocal_fx"]
        self.assertEqual((fx["preset"], fx["label"], fx["tempo"]["source"]), ("echo", "Echo", "assumed"))  # 6 s: too short
        self.assertAlmostEqual(fx["delay"]["delay_ms"], 375.0, delta=0.5)  # a dotted eighth at the assumed 120 BPM
        n = len(tone_song())
        voice_dry, _ = va.decode(dry["files"]["vocal.wav"])
        voice_fx, _ = va.decode(wet["files"]["vocal_fx.wav"])
        self.assertEqual((voice_dry.shape[1], voice_fx.shape[1]), (1, 2))
        self.assertGreaterEqual(len(voice_fx), n)  # same start, plus the echo's tail
        self.assertGreater(len(voice_fx), n + SR // 2)
        lead = np.repeat(voice_dry, 2, axis=1)
        self.assertAlmostEqual(va.measure(voice_fx, SR)["I"], va.measure(lead, SR)["I"], delta=0.7)  # as loud as the dry lead
        self.assertLessEqual(float(np.abs(voice_fx).max()), 10 ** (-1 / 20) + 1e-3)
        mix_dry, _ = va.decode(dry["files"]["mix.wav"])
        mix_wet, _ = va.decode(wet["files"]["mix.wav"])
        self.assertEqual(mix_wet.shape, mix_dry.shape)  # the song keeps its length
        self.assertGreater(float(np.abs(mix_wet - mix_dry).max()), 1e-3)
        self.assertAlmostEqual(va.measure(mix_wet, SR)["I"], va.measure(mix_dry, SR)["I"], delta=1.0)
        self.assertIn("vocal_fx_s", wet["report"]["timing"])

    def test_every_vocal_fx_in_vocal_mode(self):
        """A dry vocal with each effect: the result is the effected voice, the dry vocal beside it is the same every time."""
        song = self.song_file(tone_song(band_level=0), "dry.wav")
        base, _ = self.run_job(request(mode="vocal"), audio=song, index=False)
        with open(base["files"]["vocal.wav"], "rb") as f:
            dry_bytes = f.read()
        ref = va.measure(np.repeat(va.decode(base["files"]["vocal.wav"])[0], 2, axis=1), SR)["I"]
        for name in vocalfx.PRESETS:
            out, said = self.run_job(request(mode="vocal", vocal_fx=name), audio=song, index=False)
            files = out["files"]
            self.assertEqual((files["mix.wav"], files["mix.mp3"]), (files["vocal_fx.wav"], files["vocal_fx.mp3"]), name)
            with open(files["vocal.wav"], "rb") as f:
                self.assertEqual(f.read(), dry_bytes, name)
            y, _ = va.decode(files["vocal_fx.wav"])
            self.assertTrue(np.isfinite(y).all(), name)
            self.assertEqual(y.shape[1], 2, name)
            self.assertAlmostEqual(va.measure(y, SR)["I"], ref, delta=0.7, msg=name)
            self.assertEqual(out["report"]["master"], out["report"]["vocal_fx"]["master"], name)
            self.assertEqual(said, ["Singing it in your voice", "Adding the vocal effect", "Finishing your vocal"], name)

    def test_vocal_fx_replaces_the_put_back_room(self):
        out, _ = self.run_job(request(options=ROUND1, vocal_fx="plate"))
        placed = out["report"]["placed"]
        self.assertEqual(placed.get("room"), "replaced by the vocal effect")
        self.assertNotIn("room_db", placed)  # one space on the voice, not two

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


class InputLevelTests(unittest.TestCase):
    def test_mono_keeps_its_level_and_stereo_is_unchanged(self):
        work = tempfile.mkdtemp()
        try:
            sr = voice_pipeline.SR
            t = np.arange(sr * 3) / sr
            tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
            mono = os.path.join(work, 'mono.wav')
            stereo = os.path.join(work, 'stereo.wav')
            va.encode(mono, tone[:, None], sr, bits=32)
            va.encode(stereo, np.stack([tone, tone], axis=1), sr, bits=32)
            m = voice_pipeline.load_input(mono)
            s = voice_pipeline.load_input(stereo)
            self.assertEqual(m.shape[1], 2)
            self.assertAlmostEqual(float(np.abs(m).max()), 0.3, delta=0.01, msg='mono is not 3 dB down')
            self.assertAlmostEqual(float(np.abs(s).max()), 0.3, delta=0.01)
            self.assertAlmostEqual(voice_pipeline.rms_db(m), voice_pipeline.rms_db(s), delta=0.1)
        finally:
            shutil.rmtree(work, ignore_errors=True)


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


def click_track(bpm, seconds=12.0, sr=SR, seed=4):
    """Kick-like noise bursts on every beat (accented each bar), a quieter hat between, over a soft tone."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    x = 0.05 * np.sin(2 * np.pi * 220 * np.arange(n) / sr)
    beat, k, t = 60.0 / bpm, 0, 0.0
    burst = int(0.01 * sr)
    while t + beat / 2 < seconds - 0.1:
        i, j = int(t * sr), int((t + beat / 2) * sr)
        x[i:i + burst] += (1.0 if k % 4 == 0 else 0.6) * rng.standard_normal(burst) * np.exp(-np.arange(burst) / 200)
        x[j:j + burst // 3] += 0.2 * rng.standard_normal(burst // 3)
        t, k = t + beat, k + 1
    return (0.5 * x / np.abs(x).max()).astype(np.float32)


def schroeder_rt60(ir, fc, sr=SR):
    """RT60 of one channel in the octave around fc, from the -5 to -25 dB slope of its backward-integrated energy."""
    n = len(ir)
    nf = 4 * n
    f = np.fft.rfftfreq(nf, 1 / sr)
    w = np.exp(-0.5 * (np.log2(np.maximum(f, 1) / fc) / 0.25) ** 2)
    b = np.fft.irfft(np.fft.rfft(ir, nf) * w, nf)[:n + n // 2]
    e = np.cumsum((b ** 2)[::-1])[::-1]
    db = 10 * np.log10(e / e[0] + 1e-30)
    return (np.argmax(db < -25) - np.argmax(db < -5)) / sr * 3


class VocalFxTests(unittest.TestCase):
    def test_tempo_from_a_steady_beat(self):
        for bpm in (90, 100, 128):
            found = vocalfx.detect_tempo(click_track(bpm), SR)
            self.assertIsNotNone(found, bpm)
            self.assertLess(abs(found["bpm"] - bpm) / bpm, 0.015, (bpm, found))
        rng = np.random.default_rng(9)
        t = np.arange(SR * 12) / SR
        for what, x in (("silence", np.zeros(SR * 12)), ("tone", 0.2 * np.sin(2 * np.pi * 300 * t)),
                        ("noise", 0.1 * rng.standard_normal(SR * 12)), ("short", click_track(100, seconds=5))):
            self.assertIsNone(vocalfx.detect_tempo(np.asarray(x, np.float32), SR), what)

    def test_delay_times_follow_the_tempo(self):
        self.assertAlmostEqual(vocalfx.delay_seconds("dotted_eighth", 120), 0.375)
        self.assertAlmostEqual(vocalfx.delay_seconds("dotted_eighth", None), 0.375)  # no beat: 120 BPM
        self.assertAlmostEqual(vocalfx.delay_seconds("dotted_eighth", 60), 0.375)  # 0.75 s halved: still on the beat
        self.assertAlmostEqual(vocalfx.delay_seconds("dotted_eighth", 180), 0.25)
        self.assertAlmostEqual(vocalfx.delay_seconds("quarter", 100), 0.6)
        self.assertAlmostEqual(vocalfx.delay_seconds("quarter", 200), 0.3)  # 0.3 s is in range
        self.assertAlmostEqual(vocalfx.delay_seconds("quarter", 240), 0.5)  # 0.25 s doubled
        self.assertAlmostEqual(vocalfx.delay_seconds("sixteenth", 120), 0.125)
        self.assertAlmostEqual(vocalfx.delay_seconds("sixteenth", 70), 0.110)  # 214 ms is no slapback: the classic 110 ms

    def test_echo_repeats_ping_pong_and_darken(self):
        d = 0.375
        ir, info = vocalfx.delay_ir(SR, d, feedback=0.38, first_db=-9.0, lp_hz=4200, hp_hz=180, pingpong=True)
        self.assertGreaterEqual(info["repeats"], 4)
        step = int(round(d * SR))
        levels, centroids = [], []
        f = np.fft.rfftfreq(8192, 1 / SR)
        at_1k = int(np.argmin(np.abs(f - 1000)))
        for k in range(1, info["repeats"] + 1):
            seg = ir[k * step:k * step + 4096]
            e = (seg.astype(np.float64) ** 2).sum(0)
            louder = 0 if k % 2 else 1
            self.assertGreater(e[louder], 8 * e[1 - louder], k)  # left, right, left...
            spec = np.abs(np.fft.rfft(seg[:, louder], 8192))
            levels.append(20 * np.log10(spec[at_1k]))  # the repeat's level where a voice lives
            centroids.append((f * spec ** 2).sum() / (spec ** 2).sum())
        self.assertTrue(all(b < a for a, b in zip(levels, levels[1:])), levels)
        self.assertTrue(all(b < a for a, b in zip(centroids, centroids[1:])), centroids)
        self.assertLess(float(np.abs(ir[:step - 50]).max()), 1e-6)  # nothing before the first repeat
        self.assertAlmostEqual(levels[0], -9.0, delta=0.3)  # the first repeat 9 dB down
        self.assertAlmostEqual(levels[1] - levels[0], 20 * np.log10(0.38), delta=0.3)  # then 0.38 of the one before
        slap, info = vocalfx.delay_ir(SR, 0.11, feedback=0.0, first_db=-10.0, lp_hz=4500, hp_hz=250, pingpong=False)
        self.assertEqual(info["repeats"], 1)

    def test_reverbs_decay_as_named(self):
        for name in ("studio", "plate", "hall", "dreamy"):
            rv = vocalfx.PRESETS[name]["reverb"]
            ir = vocalfx.reverb_ir(SR, **rv)
            self.assertTrue(np.array_equal(ir, vocalfx.reverb_ir(SR, **rv)), name)  # same seed, same space
            self.assertAlmostEqual(schroeder_rt60(ir[:, 0], 1000), rv["rt60"], delta=0.15 * rv["rt60"], msg=name)
            self.assertLess(schroeder_rt60(ir[:, 0], 8000), schroeder_rt60(ir[:, 0], 1000), name)  # highs die first
            pre = int(SR * rv["predelay_ms"] / 1000)
            self.assertLess(float(np.abs(ir[:pre]).max()), 1e-6, name)
            self.assertLess(abs(float(np.corrcoef(ir[:, 0], ir[:, 1])[0, 1])), 0.35, name)  # a wide, stereo space
            self.assertTrue(np.allclose((ir.astype(np.float64) ** 2).sum(0), 1.0, atol=1e-3), name)

    def test_compressor_evens_loud_and_soft_phrases(self):
        t = np.arange(SR * 4) / SR
        tone = np.sin(2 * np.pi * 220 * t)
        loud = (t % 1.0) < 0.5
        x = (tone * np.where(loud, 0.5, 0.1)).astype(np.float32)
        y, info = vocalfx.compress(x, SR)
        self.assertTrue(info["applied"])
        core_loud = loud & ((t % 1.0) > 0.2)
        core_soft = ~loud & ((t % 1.0) > 0.7)
        before = voice_pipeline.rms_db(x[core_loud]) - voice_pipeline.rms_db(x[core_soft])
        after = voice_pipeline.rms_db(y[core_loud]) - voice_pipeline.rms_db(y[core_soft])
        self.assertGreater(before - after, 2.5)  # the loud phrases came down
        self.assertLess(before - after, 6.0)  # gently
        self.assertAlmostEqual(voice_pipeline.rms_db(y[core_soft]), voice_pipeline.rms_db(x[core_soft]), delta=0.3)

    def test_de_esser_takes_only_the_loud_hiss_down(self):
        rng = np.random.default_rng(6)
        n = SR * 4
        t = np.arange(n) / SR
        tone = (0.3 * sum(np.sin(2 * np.pi * 220 * k * t) / k for k in range(1, 6))).astype(np.float32)
        spec = np.fft.rfft(rng.standard_normal(n))
        spec[np.fft.rfftfreq(n, 1 / SR) < 6000] = 0
        hiss = (np.fft.irfft(spec, n) / np.std(np.fft.irfft(spec, n)) * 0.25).astype(np.float32)
        s_mask = (t % 1.0) > 0.8
        x = np.where(s_mask, hiss, tone).astype(np.float32)
        y, info = vocalfx.deess(x, SR)
        self.assertTrue(info["applied"])
        core_s = s_mask & ((t % 1.0) > 0.85) & ((t % 1.0) < 0.97)
        drop = voice_pipeline.rms_db(x[core_s]) - voice_pipeline.rms_db(y[core_s])
        self.assertGreater(drop, 2.0)
        self.assertLessEqual(drop, 5.5)
        vowels = ~s_mask & ((t % 1.0) > 0.1) & ((t % 1.0) < 0.7)
        self.assertLess(float(np.abs(y[vowels] - x[vowels]).max()), 2e-3)  # the vowels untouched

    def test_ducking_clears_the_way_while_singing(self):
        rng = np.random.default_rng(8)
        n = SR * 4
        wet = (0.1 * rng.standard_normal((n, 2))).astype(np.float32)
        dry = np.zeros(n, np.float32)
        dry[: n // 2] = 0.3 * np.sin(2 * np.pi * 220 * np.arange(n // 2) / SR)
        y = vocalfx.duck(wet, dry, SR, 6.0)
        sung = voice_pipeline.rms_db(y[SR // 2:SR * 3 // 2]) - voice_pipeline.rms_db(wet[SR // 2:SR * 3 // 2])
        gap = voice_pipeline.rms_db(y[SR * 3:]) - voice_pipeline.rms_db(wet[SR * 3:])
        self.assertAlmostEqual(sung, -6.0, delta=0.5)
        self.assertAlmostEqual(gap, 0.0, delta=0.2)

    def test_every_preset_is_stereo_finite_and_starts_with_the_voice(self):
        t = np.arange(SR * 3) / SR
        voice = (0.3 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 1.5 * t) > 0)).astype(np.float32)
        for name in vocalfx.PRESETS:
            y, info = vocalfx.apply(voice, SR, name, {"bpm": 100.0, "confidence": 0.9})
            self.assertEqual(y.shape[1], 2, name)
            self.assertGreaterEqual(len(y), len(voice), name)
            self.assertTrue(np.isfinite(y).all(), name)
            self.assertEqual(info["preset"], name)
            if "delay" in vocalfx.PRESETS[name]:
                self.assertEqual(info["tempo"], {"bpm": 100.0, "confidence": 0.9, "source": "detected"})
            first = y[:SR // 10].mean(1)  # the first 100 ms: the polished voice itself, no effect yet
            self.assertGreater(float(np.corrcoef(first, voice[:SR // 10])[0, 1]), 0.9, name)

    def test_wet_levels_are_per_channel_against_the_voice(self):
        """A stereo effect at -12 dB sits 12 dB under the voice on each side (the first cut left every reverb 3 dB under)."""
        rng = np.random.default_rng(3)
        ref = (0.2 * rng.standard_normal(SR)).astype(np.float32)
        wet = (0.05 * rng.standard_normal((SR * 2, 2))).astype(np.float32)
        y = vocalfx.at_level(wet, ref, -12.0)
        per_side = 10 * np.log10(vocalfx.energy(y) / 2 / vocalfx.energy(ref))
        self.assertAlmostEqual(per_side, -12.0, delta=0.01)
        self.assertAlmostEqual(10 * np.log10(vocalfx.energy(vocalfx.at_level(ref[:SR // 2], ref, -6.0)) / vocalfx.energy(ref)),
                               -6.0, delta=0.01)  # mono against mono: as before

    def test_reverbs_sit_at_their_numbers(self):
        """Studio, Plate and Hall on a held note: the reverb alone (the effected voice minus the polished voice) sits at its
        wet_db against the polished voice on each side, less the ducking while the note sounds, never more."""
        t = np.arange(SR * 3) / SR
        voice = (0.3 * sum(np.sin(2 * np.pi * 220 * k * t) / k for k in range(1, 5))).astype(np.float32)
        pol, _ = vocalfx.polish(voice, SR)
        levels = {}
        for name in ("studio", "plate", "hall"):
            rv = vocalfx.PRESETS[name]["reverb"]
            y, _ = vocalfx.apply(voice, SR, name)
            effect = y - np.pad(np.repeat(pol[:, None], 2, axis=1), ((0, len(y) - len(pol)), (0, 0)))
            levels[name] = 10 * np.log10(vocalfx.energy(effect) / 2 / vocalfx.energy(pol))
            self.assertGreaterEqual(levels[name], rv["wet_db"] - rv["duck_db"] - 0.5, name)
            self.assertLessEqual(levels[name], rv["wet_db"] + 0.5, name)
        self.assertLess(levels["studio"] + 8, min(levels["plate"], levels["hall"]))  # the glue is far under the rooms


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
            self.assertIsNone(out["vocal_fx"])
            self.assertNotIn("vocal_fx_url", out)
            self.assertIn("vocal-fx", out["features"])
            fx = voice_handler.handler({"input": {"mode": "vocal", "audio_key": "voice/abcdefgh12/in.wav", "model_key": MODEL,
                                                  "output_prefix": "voice/testjob0002", "vocal_fx": "studio"}})
            self.assertNotIn("error", fx, fx)
            self.assertEqual((fx["key"], fx["vocal_key"], fx["vocal_fx_key"], fx["vocal_fx_wav_key"]),
                             ("voice/testjob0002/vocal_fx.mp3", "voice/testjob0002/vocal.mp3", "voice/testjob0002/vocal_fx.mp3",
                              "voice/testjob0002/vocal_fx.wav"))  # vocal mode: the result is the effected voice
            self.assertEqual(sorted(k for k, _, _ in s3.uploads if k.startswith("voice/testjob0002/")),
                             ["voice/testjob0002/report.json", "voice/testjob0002/vocal.mp3", "voice/testjob0002/vocal.wav",
                              "voice/testjob0002/vocal_fx.mp3", "voice/testjob0002/vocal_fx.wav"])
            self.assertEqual((fx["vocal_fx"]["preset"], fx["vocal_fx"]["label"]), ("studio", "Studio polish"))
            self.assertTrue(fx["vocal_fx_url"].startswith("https://bucket.example/voice/testjob0002/vocal_fx.mp3"))
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
