"""CPU tests of the voice worker: numpy + ffmpeg, no torch, no GPU, no bucket, no network.

The separators and RVC are stand-ins written into a temporary folder laid out like the RVC repo (tools/pymss/cli.py, infer/cli.py),
so voice_pipeline.Runner starts real subprocesses with the real command lines, and each stand-in records the arguments and
environment it was given. The bucket is a fake with the three boto3 calls the worker makes.
Run: python -m unittest -v test_voice.py   (ffmpeg on PATH or VP_FFMPEG)"""
import json
import io
import os
import shutil
import sys
import tempfile
import textwrap
import time
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import voice_models  # noqa: E402
import voice_pipeline  # noqa: E402
import vocalfx  # noqa: E402
import voice_request  # noqa: E402
import voice_train  # noqa: E402
import vpaudio as va  # noqa: E402
import hardware  # noqa: E402

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
        elif what == "denoise":  # aufr33's denoiser: "dry" is the clean voice
            out("dry", x * 0.8); out("other", x * 0.2)
''')

# Stand-ins for RVC's training scripts (train/preprocess.py, train/dataset/extract_f0.py, train/dataset/extract_hubert_feature.py,
# train/train.py, train/train_index.py): each records how it was started and leaves the files the real one leaves.
# FAKE_TRAIN (JSON) can break a step: {"preprocess": "none", "f0": "fail" | "fail_cuda", "hubert": "fail_half",
# "train": "fail" | "no_model", "index": "fail"}.
TRAIN_STANDIN = textwrap.dedent('''
    import glob, json, os, sys
    WHO = %r
    a = sys.argv[1:]
    plan = json.loads(os.environ.get("FAKE_TRAIN", "{}"))
    with open(os.environ["FAKE_LOG"], "a") as f:
        f.write(json.dumps({"tool": WHO, "argv": a, "cwd": os.getcwd(), "PYTHONPATH": os.environ.get("PYTHONPATH"),
                            "PYTHONSAFEPATH": os.environ.get("PYTHONSAFEPATH"), "RVC_CUDA_GRAPH": os.environ.get("RVC_CUDA_GRAPH"),
                            "workers": os.environ.get("VOICE_TRAIN_LOADER_WORKERS")}) + "\\n")
    def put(path, data=b"x"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
    if WHO == "preprocess":
        inp, exp = a[0], a[3]
        for i, name in enumerate(sorted(os.listdir(inp))):
            for k in range(0 if plan.get("preprocess") == "none" else 3):
                put(f"{exp}/0_gt_wavs/{i}_{k}.wav"); put(f"{exp}/1_16k_wavs/{i}_{k}.wav")
    elif WHO == "f0":
        exp = a[4] if a[0] == "cuda" else a[1]
        if plan.get("f0") == "fail" or (plan.get("f0") == "fail_cuda" and a[0] == "cuda"):
            sys.exit(2)
        for p in sorted(glob.glob(f"{exp}/1_16k_wavs/*.wav")):
            put(f"{exp}/2a_f0/{os.path.basename(p)}.npy"); put(f"{exp}/2b-f0nsf/{os.path.basename(p)}.npy")
    elif WHO == "hubert":
        exp = a[3]
        if plan.get("hubert") == "fail_half" and a[-1] == "True":
            sys.exit(2)
        for p in sorted(glob.glob(f"{exp}/1_16k_wavs/*.wav")):
            put(f"{exp}/3_feature768/{os.path.basename(p)[:-4]}.npy")
    elif WHO == "train":
        opt = {a[i]: a[i + 1] for i in range(0, len(a) - 1, 2)}
        if plan.get("train") == "fail":
            sys.exit(1)
        lines = [f"2026-11-18 10:00:00,000\\t{opt['-e']}\\tINFO\\t====> Epoch: {i} [2026-11-18 10:00:00] | (0:00:01.000000)"
                 for i in range(1, int(opt["-te"]) + 1)]
        put(f"logs/{opt['-e']}/train.log", "\\n".join(lines).encode())
        if plan.get("train") != "no_model":
            put(f"assets/weights/{opt['-e']}.pth", f"model after {opt['-te']} rounds at batch {opt['-bs']}".encode())
    elif WHO == "index":
        if plan.get("index") == "fail":
            sys.exit(1)
        put(f"logs/{a[0]}/added_IVF7_Flat_nprobe_1_{a[0]}_{a[1]}.index", b"index-bytes")
''')
TRAIN_SCRIPTS = {"preprocess": "train/preprocess.py", "f0": "train/dataset/extract_f0.py",
                 "hubert": "train/dataset/extract_hubert_feature.py", "train": "train/train.py", "index": "train/train_index.py"}
MUTE_FILES = ("0_gt_wavs/mute48k.wav", "3_feature768/mute.npy", "2a_f0/mute.wav.npy", "2b-f0nsf/mute.wav.npy")
CONFIG_48K = {"train": {"epochs": 20000, "batch_size": 4}, "data": {"sampling_rate": 48000}, "model": {"spk_embed_dim": 109}}

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
                LEAD("aufr33"): "karaoke", voice_request.DEREVERB_MODEL: "dereverb", voice_request.DENOISE_MODEL: "denoise"}
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
        # Nov 2026: three semitones is a new key, and nobody asked for the music to follow, so the worker says so (and only that)
        self.assertEqual(r["notes"], ["The voice was moved 3 semitones up and the music was not, so the voice may be out of key "
                                      "with the music."])
        self.assertEqual((r["pitch"]["transposed_song"], r["pitch"]["song_shift"]), (False, 0))
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
        self.assertIn("up to ten minutes", caught.exception.args[0])  # six until Nov 2026
        self.assertEqual(voice_request.MAX_SECONDS, 10 * 60 + 5)


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


class HardwareTests(unittest.TestCase):
    def test_same_major_cubin_runs_on_higher_minor(self):
        self.assertTrue(hardware.supports_device(8, 9, ["sm_80", "sm_86"]))
        self.assertTrue(hardware.supports_device(8, 6, ["sm_86"]))
        self.assertTrue(hardware.supports_device(8, 9, ["sm_89"]))

    def test_higher_minor_cubin_needs_a_lower_compatible_target(self):
        self.assertFalse(hardware.supports_device(8, 6, ["sm_89"]))
        self.assertTrue(hardware.supports_device(8, 6, ["sm_89", "sm_80"]))

    def test_new_major_is_not_proved_by_older_cubins(self):
        self.assertFalse(hardware.supports_device(12, 0, ["sm_80", "sm_86", "sm_89"]))
        self.assertFalse(hardware.supports_device(9, 0, ["sm_86"]))
        self.assertTrue(hardware.supports_device(12, 0, ["sm_120"]))

    def test_missing_malformed_and_ptx_only_information_is_not_support(self):
        for archs in ([], None, "sm_86", [None, 86], ["sm_"], ["sm_86extra"], ["sm_90a"], ["compute_86"]):
            with self.subTest(archs=archs):
                self.assertFalse(hardware.supports_device(8, 9, archs))
        for major, minor in ((None, 9), (8, None), (True, 9), (8, 10), (0, 9), (8, -1)):
            with self.subTest(device=(major, minor)):
                self.assertFalse(hardware.supports_device(major, minor, ["sm_86"]))

    def test_worker_probe_uses_cubin_compatibility_without_launching_kernels(self):
        import voice_handler
        cuda = types.SimpleNamespace(is_available=lambda: True, get_device_capability=lambda device: (8, 9),
                                     get_device_name=lambda device: "NVIDIA GeForce RTX 4090",
                                     get_arch_list=lambda: ["sm_80", "sm_86"])
        output = io.StringIO()
        with mock.patch.dict(sys.modules, {"torch": types.SimpleNamespace(cuda=cuda)}), \
                mock.patch.object(sys, "argv", ["probe", HERE]), mock.patch.object(sys, "path", list(sys.path)), \
                redirect_stdout(output):
            exec(voice_handler.GPU_PROBE, {})
        report = json.loads(output.getvalue())
        self.assertTrue(report["supported"])
        self.assertEqual(report["capability"], "sm_89")
        self.assertEqual(report["compiled_archs"], ["sm_80", "sm_86"])

    def test_unconfirmed_gpu_is_refused_before_recording_or_model_fetch(self):
        import voice_handler
        request = {"input": {"mode": "vocal", "audio_key": "voice/abcdefgh12/in.wav", "model_key": MODEL}}
        for gpu in ({"cuda": True, "supported": False, "name": "NVIDIA GeForce RTX 5090", "capability": "sm_120"},
                    {"cuda": True, "supported": None}, {"cuda": False}, {"cuda": None}):
            with self.subTest(gpu=gpu), mock.patch.object(voice_handler, "GPU", dict(gpu)), \
                    mock.patch.object(voice_handler, "storage") as storage, \
                    mock.patch.object(voice_models, "fetch") as fetch_model, \
                    mock.patch.object(voice_pipeline, "run") as inference:
                result = voice_handler.handler(request)
                self.assertIn("CUDA support", result["error"])
                self.assertNotIn("before the RTX 50", result["error"])
                storage.assert_not_called()
                fetch_model.assert_not_called()
                inference.assert_not_called()


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
            self.assertIn("compatible compiled CUDA support", refused["error"])
        finally:
            voice_handler.storage, voice_pipeline.Runner = saved[0], saved[1]
            voice_handler.GPU.clear()
            voice_handler.GPU.update(saved[2])
            sys.modules.pop("runpod", None)

    def test_song_mode_returns_the_stems(self):
        import voice_handler
        with open(self.song_file(), "rb") as f:
            s3 = FakeS3({"voice/abcdefgh12/in.wav": f.read(), MODEL: b"m", INDEX: b"i"})
        sys.modules["runpod"] = types.SimpleNamespace(serverless=types.SimpleNamespace(progress_update=lambda job, text: None))
        runner = self.runner()
        with mock.patch.object(voice_handler, "storage", lambda: (s3, "bucket")), \
                mock.patch.object(voice_pipeline, "Runner", lambda log: runner), \
                mock.patch.dict(voice_handler.GPU, {"cuda": True, "name": "NVIDIA GeForce RTX 4090", "supported": True}, clear=True):
            try:
                out = voice_handler.handler({"input": {"mode": "song", "audio_key": "voice/abcdefgh12/in.wav", "model_key": MODEL,
                                                       "output_prefix": "voice/testjob0003"}})
            finally:
                sys.modules.pop("runpod", None)
        self.assertNotIn("error", out, out)
        for stem in ("instrumental", "backing", "lead_original"):
            self.assertEqual((out[f"{stem}_key"], out[f"{stem}_wav_key"]),
                             (f"voice/testjob0003/{stem}.mp3", f"voice/testjob0003/{stem}.wav"))
            self.assertTrue(out[f"{stem}_url"].startswith(f"https://bucket.example/voice/testjob0003/{stem}.mp3"))
            self.assertTrue(out[f"{stem}_wav_url"].startswith(f"https://bucket.example/voice/testjob0003/{stem}.wav"))
        self.assertEqual(sorted(k.split("/")[-1] for k, _, _ in s3.uploads),
                         sorted(["mix.mp3", "mix.wav", "vocal.mp3", "vocal.wav", "report.json", "instrumental.mp3",
                                 "instrumental.wav", "backing.mp3", "backing.wav", "lead_original.mp3", "lead_original.wav"]))
        self.assertEqual((out["key"], out["vocal_key"]), ("voice/testjob0003/mix.mp3", "voice/testjob0003/vocal.mp3"))  # as before
        self.assertEqual(out["pitch"]["transposed_song"], False)
        for name in ("stems", "backing-modes", "formant", "dereverb-strength", "denoise", "pitch-fit", "transpose", "train"):
            self.assertIn(name, out["features"])
        self.assertEqual({k: out["settings"][k] for k in voice_request.EXTRA_DEFAULTS}, voice_request.EXTRA_DEFAULTS)


class NewRequestTests(unittest.TestCase):
    """The Nov 2026 fields of a song or vocal request: off by default, each one checked, each refusal a sentence."""

    def test_defaults_do_nothing(self):
        r = request()
        self.assertEqual(r["extra"], {"dereverb_strength": 1.0, "denoise": 0.0, "backing": "keep", "formant": 0.0})
        self.assertEqual(r["extra"], voice_request.EXTRA_DEFAULTS)
        self.assertIs(r["transpose_song"], False)
        self.assertEqual(sorted(r["options"]), sorted(voice_request.DEFAULTS))  # "options" is still the eleven settings

    def test_each_new_option(self):
        r = request(options={"dereverb": True, "dereverb_strength": 0.4, "denoise": 0.5, "backing": "convert", "formant": -3})
        self.assertEqual(r["extra"], {"dereverb_strength": 0.4, "denoise": 0.5, "backing": "convert", "formant": -3.0})
        self.assertEqual(request(options={"backing": "remove"})["extra"]["backing"], "remove")
        self.assertEqual(request(options={"formant": 6})["extra"]["formant"], 6.0)
        self.assertEqual((request(pitch="fit")["pitch"], request(pitch="fit")["transpose_song"]), ("fit", True))
        self.assertIs(request(pitch="fit", transpose_song=False)["transpose_song"], True)  # fit always moves the music
        self.assertIs(request(pitch=2, transpose_song=True)["transpose_song"], True)
        self.assertIs(request(pitch=2, transpose_song="off")["transpose_song"], False)
        self.assertEqual(request(options={"f0_method": "pm"})["options"]["f0_method"], "pm")

    def test_refusals_are_sentences(self):
        bad = [dict(options={"dereverb_strength": 2}), dict(options={"dereverb_strength": "half"}), dict(options={"denoise": -0.1}),
               dict(options={"denoise": True}), dict(options={"backing": "mute"}), dict(options={"backing": 1}),
               dict(options={"formant": 7}), dict(options={"formant": -6.5}), dict(options={"formant": "dark"}),
               dict(transpose_song="maybe"), dict(pitch="best"), dict(pitch="FIT"), dict(pitch=["fit"])]
        for over in bad:
            with self.assertRaises(ValueError, msg=over) as caught:
                request(**over)
            self.assertTrue(caught.exception.args[0].endswith("."), over)
        with self.assertRaises(ValueError) as caught:
            request(options={"backing": "mute"})
        self.assertEqual(caught.exception.args[0], "Backing vocals must be keep, convert or remove.")


def train_request(**over):
    raw = {"mode": "train", "owner": "u123", "voice_id": "voice-abc12345",
           "sample_keys": ["voice-samples/u123/voice-abc12345/one.wav", "voice-samples/u123/voice-abc12345/two.mp3"]}
    raw.update(over)
    return voice_request.parse(raw, resolve=False)


class TrainRequestTests(unittest.TestCase):
    def test_a_train_request(self):
        r = train_request()
        self.assertEqual(r["options"], {"epochs": "auto", "isolate": False, "denoise": 0.0, "dereverb": False})
        self.assertEqual((r["mode"], r["owner"], r["voice_id"], len(r["sample_keys"])), ("train", "u123", "voice-abc12345", 2))
        self.assertEqual(r["output_prefix"], "voice-models/u123/voice-abc12345")  # the only place a train job writes
        r = train_request(options={"epochs": 250, "isolate": "on", "denoise": 0.3, "dereverb": True})
        self.assertEqual(r["options"], {"epochs": 250, "isolate": True, "denoise": 0.3, "dereverb": True})
        self.assertEqual(train_request(options={"epochs": 20})["options"]["epochs"], 20)
        self.assertEqual(train_request(options={"epochs": 600.0})["options"]["epochs"], 600)
        keys = [f"voice-samples/u123/voice-abc12345/sub/take{i}.wav" for i in range(60)]
        self.assertEqual(train_request(sample_keys=keys)["sample_keys"], keys)  # sixty is allowed, and so is a folder inside
        self.assertEqual(train_request(sample_keys=keys[:3] + keys[:1])["sample_keys"], keys[:3])  # a repeat counts once

    def test_recordings_must_be_in_the_voices_own_folder(self):
        sentence = "One of the recordings is not in this voice's own folder."
        for key in ("voice-samples/u999/voice-abc12345/one.wav", "voice-samples/u123/voice-other999/one.wav",
                    "voice-samples/u123/voice-abc12345/../voice-other999/one.wav", "voice-samples/u123/voice-abc12345/",
                    "voice-samples/u123/voice-abc12345x/one.wav", "voice-models/u123/voice-abc12345/model.pth",
                    "voice-samples/u123/voice-abc12345//one.wav", "/voice-samples/u123/voice-abc12345/one.wav",
                    "voice-samples/u123/voice-abc12345/a b.wav", "yue2/abc/master.wav", 7, None):
            with self.assertRaises(ValueError, msg=key) as caught:
                train_request(sample_keys=["voice-samples/u123/voice-abc12345/ok.wav", key])
            self.assertEqual(caught.exception.args[0], sentence, key)

    def test_refusals_are_sentences(self):
        cases = [
            (dict(owner="../u1"), "The voice's owner is not valid."), (dict(owner=""), "The voice's owner is not valid."),
            (dict(owner=None), "The voice's owner is not valid."), (dict(owner="u" * 65), "The voice's owner is not valid."),
            (dict(voice_id="short"), "The voice id is not valid."), (dict(voice_id="Voice-ABC12345"), "The voice id is not valid."),
            (dict(voice_id="voice/abc12345"), "The voice id is not valid."), (dict(voice_id=12345678), "The voice id is not valid."),
            (dict(sample_keys=[]), "Add at least one recording of the voice."),
            (dict(sample_keys="voice-samples/u123/voice-abc12345/one.wav"), "Add at least one recording of the voice."),
            (dict(sample_keys=[f"voice-samples/u123/voice-abc12345/{i}.wav" for i in range(61)]),
             "That is more than 60 recordings. Use 60 or fewer."),
            (dict(options={"epochs": 19}), "Training rounds must be automatic or a whole number from 20 to 600."),
            (dict(options={"epochs": 601}), "Training rounds must be automatic or a whole number from 20 to 600."),
            (dict(options={"epochs": 99.5}), "Training rounds must be automatic or a whole number from 20 to 600."),
            (dict(options={"epochs": "many"}), "Training rounds must be automatic or a whole number from 20 to 600."),
            (dict(options={"epochs": True}), "Training rounds must be automatic or a whole number from 20 to 600."),
            (dict(options={"isolate": "maybe"}), "The isolate setting must be on or off."),
            (dict(options={"dereverb": 3}), "The dereverb setting must be on or off."),
            (dict(options={"denoise": 2}), "Noise cleanup must be a number from 0 to 1."),
            (dict(options=["epochs"]), "The voice options were not readable."),
        ]
        for over, sentence in cases:
            with self.assertRaises(ValueError, msg=over) as caught:
                train_request(**over)
            self.assertEqual(caught.exception.args[0], sentence, over)


def sung_vowel(f0, seconds=3.0, sr=SR, formants=((730, 90), (1090, 110), (2440, 170))):
    """A sung "ah": a harmonic source with a falling tilt under three formants, with a little vibrato."""
    t = np.arange(int(seconds * sr)) / sr
    phase = 2 * np.pi * np.cumsum(f0 * 2 ** (0.2 * np.sin(2 * np.pi * 5 * t) / 12)) / sr
    x = np.zeros(len(t))
    for k in range(1, int(16000 / f0)):
        f = k * f0
        x += sum(1 / np.sqrt(1 + ((f - fc) / (bw / 2)) ** 2) for fc, bw in formants) / k * np.sin(k * phase)
    return (0.3 * x / np.abs(x).max()).astype(np.float32)


def autocorr_pitch(x, sr=SR, lo=60.0, hi=1200.0):
    """The pitch of a steady note from the highest autocorrelation peak (FFT autocorrelation of the middle second)."""
    seg = np.asarray(x[len(x) // 2 - sr // 2:len(x) // 2 + sr // 2], np.float64)
    seg = seg - seg.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(seg, 4 * len(seg))) ** 2)
    a, b = int(sr / hi), int(sr / lo)
    k = a + int(np.argmax(ac[a:b]))
    y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
    return sr / (k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2))


def centroid(x, sr=SR):
    p = (np.abs(va.stft(x)) ** 2).sum(0)
    return float((p * np.arange(len(p)) * sr / 2048).sum() / p.sum())


class FormantTests(unittest.TestCase):
    def test_zero_is_an_exact_no_op(self):
        x = sung_vowel(220.0)
        self.assertIs(va.formant_shift(x, SR, 0), x)
        self.assertIs(va.formant_shift(x, SR, 0.0), x)

    def test_timbre_moves_and_pitch_does_not(self):
        for f0 in (110.0, 220.0, 440.0, 660.0):
            x = sung_vowel(f0)
            base, pitch = centroid(x), autocorr_pitch(x)
            self.assertAlmostEqual(pitch, f0, delta=0.01 * f0)
            last = None
            for st in (-6, -3, 3, 6):
                y = va.formant_shift(x, SR, st)
                self.assertEqual((y.shape, y.dtype), (x.shape, np.float32), (f0, st))
                self.assertTrue(np.isfinite(y).all(), (f0, st))
                self.assertAlmostEqual(autocorr_pitch(y), pitch, delta=0.01 * pitch, msg=(f0, st))  # the note stays put
                c = centroid(y)
                (self.assertLess if st < 0 else self.assertGreater)(c, base, (f0, st))  # darker below 0, brighter above
                if last is not None and (st > 0) == (last[0] > 0):
                    self.assertGreater(c, last[1], (f0, st))  # and further with a bigger move
                last = (st, c)

    def test_the_envelope_itself_slides(self):
        """The loudest formant region of an "ah" (around 730 to 1090 Hz) moves up with a positive shift and down with a
        negative one, measured on the smoothed long-term spectrum."""
        x = sung_vowel(147.0)

        def peak_hz(y):
            p = np.convolve((np.abs(va.stft(y)) ** 2).sum(0), np.ones(31) / 31, mode="same")
            f = np.arange(len(p)) * SR / 2048
            band = (f > 300) & (f < 2000)
            return float((p[band] * f[band]).sum() / p[band].sum())
        self.assertGreater(peak_hz(va.formant_shift(x, SR, 4)), peak_hz(x) * 1.08)
        self.assertLess(peak_hz(va.formant_shift(x, SR, -4)), peak_hz(x) / 1.08)

    def test_same_voice_same_result(self):
        x = sung_vowel(196.0, seconds=1.5)
        self.assertTrue(np.array_equal(va.formant_shift(x, SR, 2.5), va.formant_shift(x, SR, 2.5)))


class KeyTests(unittest.TestCase):
    def test_fit_picks_the_smallest_move_that_fits(self):
        her = {"p05": 55.0, "p50": 62.0, "p95": 70.0}
        rng = np.random.default_rng(1)
        inside = 62 + 3 * rng.standard_normal(4000).clip(-2, 2)  # 56..68: already fits
        self.assertEqual(va.fit_shift(her, inside)["shift"], 0)
        self.assertIn("already", va.fit_shift(her, inside)["why"])
        high = 72 + 2 * rng.standard_normal(4000).clip(-2, 2)  # 68..76: the top is over by six
        sh = va.fit_shift(her, high)
        self.assertEqual(sh["shift"], -6)  # the smallest move that brings every note inside, not an octave
        self.assertEqual(sh["outside_after"], 0.0)
        self.assertGreater(sh["outside_before"], 0.5)
        self.assertIn("down 6 semitones", sh["why"])
        low = 45 + 2 * rng.standard_normal(4000).clip(-2, 2)  # 41..49: fourteen under the bottom
        self.assertEqual(va.fit_shift(her, low)["shift"], 14)  # more than an octave when it takes that
        wide = np.concatenate([np.full(1000, 50.0), np.full(1000, 74.0)])  # wider than the range: only half can ever fit
        sh = va.fit_shift(her, wide)
        self.assertEqual((sh["shift"], sh["outside_before"], sh["outside_after"]), (-4, 1.0, 0.5))  # -4 fits the top; +5 is further
        one = va.fit_shift(her, np.full(500, 71.0))
        self.assertEqual((one["shift"], "down 1 semitone to" in one["why"]), (-1, True))

    def test_the_music_moves_by_the_key_not_the_octaves(self):
        self.assertEqual([voice_pipeline.key_shift(s) for s in (0, 12, -12, 24, 3, -5, 7, -7, 14, -14, 6, -6, 18, -18, 11, -11)],
                         [0, 0, 0, 0, 3, -5, -5, 5, 2, -2, 6, -6, 6, -6, -1, 1])

    @unittest.skipUnless(va.has_rubberband(), "this ffmpeg has no rubberband filter (the image's build fails without it)")
    def test_transpose_moves_the_notes_and_keeps_the_time(self):
        t = np.arange(SR * 5) / SR
        gate = ((t % 1.0) < 0.6).astype(np.float32)
        tone = (0.3 * sum(np.sin(2 * np.pi * 220 * k * t) / k for k in range(1, 6))).astype(np.float32) * gate
        x = np.stack([tone, 0.5 * tone], 1)
        for st, voice in ((2, False), (-5, False), (3, True)):
            y = va.transpose(x, SR, st, voice=voice)
            self.assertEqual((y.shape, y.dtype), (x.shape, np.float32), st)
            tr = va.pitch_track(y, SR)
            self.assertAlmostEqual(float(np.median(tr["f0"][tr["voiced"]])), 220 * 2 ** (st / 12), delta=220 * 0.01, msg=st)
            lag, peak = va.align_lag(x, y, SR, max_ms=100)
            self.assertLessEqual(abs(lag), SR * 0.003, st)  # lined back up with its source
            self.assertGreater(peak, 0.8, st)
            self.assertAlmostEqual(voice_pipeline.rms_db(y[:, 0]) - voice_pipeline.rms_db(y[:, 1]), 6.02, delta=0.5, msg=st)
        self.assertIs(va.transpose(x, SR, 0), x)

    def test_active_seconds_leave_the_silence_out(self):
        t = np.arange(SR * 10) / SR
        x = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        x[SR * 2:SR * 6] = 0  # four seconds of silence
        x[SR * 8:] *= 0.0005  # and two of something 66 dB down
        self.assertAlmostEqual(va.active_seconds(x, SR), 4.0, delta=0.11)
        self.assertEqual(va.active_seconds(np.zeros(SR, np.float32), SR), 0.0)


class NewPipelineTests(Base):
    """Each Nov 2026 option on a song: what it changes, and that leaving it out changes nothing."""

    def read(self, out, name):
        return va.decode(out["files"][name])[0]

    def lead_db(self, out):
        """The level of the lead that went into RVC: the lead_original stem with the mix's master gain taken back off."""
        return voice_pipeline.rms_db(self.read(out, "lead_original.wav")) - out["report"]["stems"]["gain_db"]

    def test_explicit_defaults_are_byte_identical_and_the_stems_are_there(self):
        song = self.song_file()
        plain, said = self.run_job(request(), audio=song)
        same, _ = self.run_job(request(transpose_song=False, options={"dereverb_strength": 1, "denoise": 0, "backing": "keep",
                                                                      "formant": 0}), audio=song)
        for name in ("mix.wav", "mix.mp3", "vocal.wav", "vocal.mp3", "instrumental.wav", "backing.wav", "lead_original.wav"):
            with open(plain["files"][name], "rb") as a, open(same["files"][name], "rb") as b:
                self.assertEqual(a.read(), b.read(), name)
        r = plain["report"]
        self.assertEqual((r["backing"], r["denoise"], r["pitch"]["transposed_song"], r["pitch"]["song_shift"]),
                         ({"mode": "keep", "status": "kept"}, None, False, 0))
        self.assertEqual(said, ["Splitting the voice from the music", "Finding the lead singer", "Singing it in your voice",
                                "Putting the song back together"])  # no new step unless asked for
        self.assertEqual({k: r["settings"][k] for k in voice_request.EXTRA_DEFAULTS}, voice_request.EXTRA_DEFAULTS)
        # the stems: the band as extracted (the stand-in gives half the mix), the backing as sung (a fifth of the vocals), and
        # the original lead as it went into RVC, each with the mix's own master gain
        mix_in = tone_song()
        g = 10 ** (r["stems"]["gain_db"] / 20)
        self.assertEqual(r["stems"]["gain_db"], r["master"]["gain_db"])
        inst, back, lead = self.read(plain, "instrumental.wav"), self.read(plain, "backing.wav"), self.read(plain, "lead_original.wav")
        self.assertEqual((inst.shape, back.shape, lead.shape), ((len(mix_in), 2), (len(mix_in), 2), (len(mix_in), 1)))
        self.assertLess(float(np.abs(inst - 0.5 * g * mix_in).max()), 2e-3)
        self.assertLess(float(np.abs(back - 0.5 * 0.2 * g * mix_in).max()), 2e-3)
        self.assertLess(float(np.abs(lead[:, 0] - 0.5 * 0.8 * g * mix_in.mean(1)).max()), 2e-3)
        voice_in = va.decode(os.path.join(os.path.dirname(plain["files"]["mix.wav"]), "voice_in.wav"))[0]
        self.assertLess(float(np.abs(lead - g * voice_in).max()), 2e-3)  # exactly what RVC was given
        for stem in ("instrumental", "backing", "lead_original"):
            self.assertTrue(os.path.getsize(plain["files"][f"{stem}.mp3"]) > 1000, stem)

    def test_vocal_mode_has_no_stems(self):
        out, _ = self.run_job(request(mode="vocal"), index=False)
        self.assertEqual(sorted(out["files"]), ["mix.mp3", "mix.wav", "report.json", "vocal.mp3", "vocal.wav"])
        self.assertIsNone(out["report"]["stems"])

    def test_no_lead_split_no_backing_stem(self):
        out, _ = self.run_job(request(options={"lead_split": False}))
        self.assertNotIn("backing.wav", out["files"])
        self.assertIn("instrumental.wav", out["files"])
        self.assertEqual(out["report"]["backing"], {"mode": "keep", "status": "none"})
        self.assertEqual(out["report"]["notes"], [])

    def test_dereverb_strength_scales_the_room(self):
        full, _ = self.run_job(request(options=ROUND1))
        half, said = self.run_job(request(options=dict(ROUND1, dereverb_strength=0.5)))
        none, said0 = self.run_job(request(options=dict(ROUND1, dereverb_strength=0)))
        self.assertNotIn("dereverb_strength", full["report"]["separation"])
        self.assertAlmostEqual(full["report"]["separation"]["reverb_ratio_db"], 20 * np.log10(0.1 / 0.9), delta=0.2)
        sep = half["report"]["separation"]
        self.assertEqual((sep["dereverb"], sep["dereverb_strength"]), ("used", 0.5))
        self.assertAlmostEqual(sep["reverb_ratio_db"], 20 * np.log10(0.05 / 0.95), delta=0.2)  # half the room came off
        self.assertLess(half["report"]["placed"]["room_db"], full["report"]["placed"]["room_db"])  # so less is put back
        self.assertIn("Taking the room off the voice", said)
        self.assertAlmostEqual(self.lead_db(half) - self.lead_db(full), 20 * np.log10(0.95 / 0.9), delta=0.3)  # more of it left on
        self.assertEqual(none["report"]["separation"]["dereverb"], "off")  # none of the room asked off: the model is not run
        self.assertNotIn("Taking the room off the voice", said0)
        self.assertNotIn("room_db", none["report"]["placed"])

    def test_denoise_cleans_the_voice_before_it_is_sung(self):
        out, said = self.run_job(request(options={"denoise": 0.5}))
        self.assertEqual([c["argv"][1] for c in self.calls("pymss")], [EX("hyperace"), LEAD("frazer"), voice_request.DENOISE_MODEL])
        self.assertEqual(said, ["Splitting the voice from the music", "Finding the lead singer", "Cleaning the noise off the voice",
                                "Singing it in your voice", "Putting the song back together"])
        d = out["report"]["denoise"]
        self.assertEqual((d["denoise"], d["amount"]), ("used", 0.5))
        self.assertAlmostEqual(d["removed_db"], 20 * np.log10(0.1), delta=0.3)  # half of the stand-in's 20 percent "noise"
        plain, _ = self.run_job(request())
        self.assertAlmostEqual(self.lead_db(out) - self.lead_db(plain), 20 * np.log10(0.9), delta=0.3)  # what RVC was given
        self.assertEqual(out["report"]["notes"], [])
        self.assertIn("denoise_s", out["report"]["timing"])

    def test_denoise_in_vocal_mode_and_when_it_cannot_run(self):
        out, said = self.run_job(request(mode="vocal", options={"denoise": 1}), index=False)
        self.assertEqual([c["argv"][1] for c in self.calls("pymss")], [voice_request.DENOISE_MODEL])
        self.assertEqual(said, ["Cleaning the noise off the voice", "Singing it in your voice", "Finishing your vocal"])
        self.assertEqual(out["report"]["denoise"]["denoise"], "used")
        self.plan({voice_request.DENOISE_MODEL: "fail"})
        with redirect_stdout(io.StringIO()):
            out, _ = self.run_job(request(mode="vocal", options={"denoise": 1}), index=False)
        self.assertEqual(out["report"]["notes"], ["The noise could not be cleaned off the voice, so it was used as it was."])
        self.assertEqual(out["report"]["denoise"]["denoise"], "failed")
        with open(os.environ["VOICE_SELFTEST"], "w") as f:
            json.dump({"denoise": False}, f)
        before = len(self.calls("pymss"))
        out, said = self.run_job(request(mode="vocal", options={"denoise": 1}), index=False)
        self.assertEqual(len(self.calls("pymss")), before)  # a model that failed the build check is not started
        self.assertEqual(out["report"]["notes"], ["The noise cleanup failed the worker's own check when it was built, so the "
                                                  "voice was used as it was."])
        self.assertNotIn("Cleaning the noise off the voice", said)

    def test_backing_vocals_can_be_resung_or_left_out(self):
        song = self.song_file()
        keep, _ = self.run_job(request(pitch=12), audio=song)
        conv, said = self.run_job(request(pitch=12, options={"backing": "convert"}), audio=song)
        gone, said_gone = self.run_job(request(pitch=12, options={"backing": "remove"}), audio=song)
        rvc = [c for c in self.calls("rvc")]
        self.assertEqual(len(rvc), 4)  # one pass each for keep and remove, two for convert
        passes = [c["argv"] for c in rvc[1:3]]
        self.assertEqual([os.path.basename(a[a.index("--input") + 1]) for a in passes], ["voice_in.wav", "backing_in.wav"])
        self.assertEqual([a[a.index("--pitch") + 1] for a in passes], ["12", "12"])  # the same shift as the lead
        self.assertEqual([a[a.index("--index-rate") + 1] for a in passes], ["0.5", "0.5"])
        self.assertEqual(said, ["Splitting the voice from the music", "Finding the lead singer", "Singing it in your voice",
                                "Singing the backing vocals in your voice", "Putting the song back together"])
        self.assertNotIn("Singing the backing vocals in your voice", said_gone)
        b = conv["report"]["backing"]
        self.assertEqual((b["mode"], b["status"]), ("convert", "converted"))
        self.assertAlmostEqual(b["lag_ms"], 25.0, delta=2.0)  # lined up with the backing it replaces
        self.assertAlmostEqual(b["gain_db"], 6.0, delta=0.7)  # and as loud: the stand-in's 6 dB is made up
        self.assertEqual(gone["report"]["backing"], {"mode": "remove", "status": "left out"})
        back_keep, back_conv, back_gone = (self.read(o, "backing.wav") for o in (keep, conv, gone))
        self.assertEqual((back_keep.shape[1], back_conv.shape[1], back_gone.shape[1]), (2, 1, 2))
        level = [va.measure(np.repeat(b, 2, axis=1) if b.shape[1] == 1 else b, SR)["I"] for b in (back_keep, back_conv, back_gone)]
        gains = [o["report"]["stems"]["gain_db"] for o in (keep, conv, gone)]
        self.assertAlmostEqual(level[1] - gains[1], level[0] - gains[0], delta=0.7)  # the re-sung backing is as loud as the sung
        # left out of the mix, but the person still gets the backing vocals as they were sung
        self.assertLess(float(np.abs(back_gone / 10 ** (gains[2] / 20) - back_keep / 10 ** (gains[0] / 20)).max()), 2e-3)
        mix_keep, mix_conv, mix_gone = (self.read(o, "mix.wav") for o in (keep, conv, gone))
        self.assertGreater(float(np.abs(mix_gone - mix_keep).max()), 1e-3)
        self.assertGreater(float(np.abs(mix_conv - mix_keep).max()), 1e-3)
        for o in (keep, conv, gone):
            self.assertEqual(o["report"]["notes"], [])
        self.assertIn("convert_backing_s", conv["report"]["timing"])

    def test_backing_setting_without_a_lead_split_says_so(self):
        out, _ = self.run_job(request(options={"backing": "convert", "lead_split": False}))
        self.assertEqual(out["report"]["notes"], ["The backing vocals setting had no effect, because the lead singer was not "
                                                  "split from the backing vocals."])
        self.assertEqual(len(self.calls("rvc")), 1)
        self.assertEqual(out["report"]["backing"], {"mode": "convert", "status": "none"})
        self.plan({LEAD("frazer"): "named_weak"})  # the split misfired: the same note, after the split's own
        out, _ = self.run_job(request(options={"backing": "remove"}))
        self.assertEqual(out["report"]["notes"][-1], "The backing vocals setting had no effect, because the lead singer was "
                                                     "not split from the backing vocals.")
        out, _ = self.run_job(request(mode="vocal", options={"backing": "remove"}), index=False)
        self.assertEqual(out["report"]["notes"], ["The backing vocals setting only applies to a song with music."])

    def test_formant_changes_the_voice_and_only_when_asked(self):
        song = self.song_file()
        plain, _ = self.run_job(request(), audio=song)
        dark, _ = self.run_job(request(options={"formant": -4}), audio=song)
        bright, _ = self.run_job(request(options={"formant": 4, "backing": "convert"}), audio=song)
        self.assertNotIn("formant_st", plain["report"]["placed"])
        self.assertEqual((dark["report"]["placed"]["formant_st"], bright["report"]["placed"]["formant_st"]), (-4.0, 4.0))
        v0, v1, v2 = (self.read(o, "vocal.wav")[:, 0] for o in (plain, dark, bright))
        self.assertLess(centroid(v1), centroid(v0))
        self.assertGreater(centroid(v2), centroid(v0))
        for v in (v1, v2):
            self.assertAlmostEqual(autocorr_pitch(v), autocorr_pitch(v0), delta=0.01 * autocorr_pitch(v0))
            self.assertAlmostEqual(va.measure(v, SR)["I"], va.measure(v0, SR)["I"], delta=0.7)  # still as loud as the lead
        self.assertIn("formant_s", dark["report"]["timing"])

    def test_pitch_fit_needs_a_range_and_says_so(self):
        out, _ = self.run_job(request(pitch="fit"))
        p = out["report"]["pitch"]
        self.assertEqual((p["shift"], p["source"], p["transposed_song"]), (0, "fit", False))
        self.assertIn("no voice range", p["why"])
        self.assertEqual(out["report"]["notes"], [])

    @unittest.skipUnless(va.has_rubberband(), "this ffmpeg has no rubberband filter (the image's build fails without it)")
    def test_pitch_fit_moves_the_voice_and_the_music_together(self):
        song = self.song_file(tone_song(f0=466.16, band_level=0.03), "high.wav")  # the melody on A#4 (MIDI 70), a quiet 110 Hz band
        rng = {"p05": 55.0, "p50": 60.0, "p95": 66.0}  # four semitones over the top of this range
        out, said = self.run_job(request(pitch="fit", voice_range=rng), audio=song)
        p = out["report"]["pitch"]
        self.assertEqual((p["shift"], p["source"], p["transposed_song"], p["song_shift"]), (-5, "fit", True, -5))
        self.assertEqual((p["outside_after"], p["outside_before"]), (0.0, 1.0))
        self.assertEqual(self.calls("rvc")[0]["argv"][self.calls("rvc")[0]["argv"].index("--pitch") + 1], "-5")
        self.assertEqual(out["report"]["notes"], [])  # in key, so nothing to warn about
        self.assertIn("transpose_s", out["report"]["timing"])

        def low_peak(x):
            spec = np.abs(np.fft.rfft(x[:, 0] * np.hanning(len(x))))
            f = np.fft.rfftfreq(len(x), 1 / SR)
            band = (f > 60) & (f < 150)
            return float(f[band][np.argmax(spec[band])])
        still, _ = self.run_job(request(pitch=-5, voice_range=rng), audio=song)
        self.assertAlmostEqual(low_peak(va.decode(still["files"]["instrumental.wav"])[0]), 110.0, delta=1.0)
        self.assertAlmostEqual(low_peak(va.decode(out["files"]["instrumental.wav"])[0]), 110 * 2 ** (-5 / 12), delta=1.0)
        self.assertEqual(still["report"]["pitch"]["transposed_song"], False)
        self.assertEqual(still["report"]["notes"], ["The voice was moved 5 semitones down and the music was not, so the voice may "
                                                    "be out of key with the music."])
        back_moved, back_still = va.decode(out["files"]["backing.wav"])[0], va.decode(still["files"]["backing.wav"])[0]
        t_moved, t_still = va.pitch_track(back_moved, SR), va.pitch_track(back_still, SR)
        ratio = float(np.median(t_moved["f0"][t_moved["voiced"]]) / np.median(t_still["f0"][t_still["voiced"]]))
        self.assertAlmostEqual(ratio, 2 ** (-5 / 12), delta=0.01)  # the kept backing vocals moved to the new key too
        self.assertEqual(va.decode(out["files"]["mix.wav"])[0].shape, va.decode(still["files"]["mix.wav"])[0].shape)

    @unittest.skipUnless(va.has_rubberband(), "this ffmpeg has no rubberband filter (the image's build fails without it)")
    def test_transpose_song_with_set_semitones_and_octaves(self):
        out, _ = self.run_job(request(pitch=14, transpose_song=True))
        p = out["report"]["pitch"]
        self.assertEqual((p["shift"], p["transposed_song"], p["song_shift"]), (14, True, 2))  # an octave and two: the band moves two
        self.assertEqual(out["report"]["notes"], [])
        out, _ = self.run_job(request(pitch=-12, transpose_song=True))
        p = out["report"]["pitch"]
        self.assertEqual((p["transposed_song"], p["song_shift"]), (False, 0))  # an octave is the same key: nothing to move
        self.assertNotIn("transpose_s", out["report"]["timing"])
        self.assertEqual(out["report"]["notes"], [])

    def test_no_rubberband_is_said_not_crashed(self):
        with mock.patch.object(va, "has_rubberband", lambda: False):
            out, _ = self.run_job(request(pitch=2, transpose_song=True))
        self.assertEqual(out["report"]["pitch"]["transposed_song"], False)
        self.assertEqual(out["report"]["notes"], ["This worker could not move the music to the new key, so the voice may be out "
                                                  "of key with the music."])

    def test_vocal_mode_is_never_out_of_key(self):
        out, _ = self.run_job(request(mode="vocal", pitch=3), index=False)
        self.assertEqual(out["report"]["notes"], [])
        self.assertEqual(out["report"]["pitch"]["transposed_song"], False)


class TrainBase(Base):
    """A stand-in RVC folder with the training scripts, the 48k config and the silence sample, as the image has them."""
    SAMPLES = {}

    def setUp(self):
        super().setUp()
        for who, rel in TRAIN_SCRIPTS.items():
            path = os.path.join(self.rvc, *rel.split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(TRAIN_STANDIN % who)
        for rel, text in (("configs/v2/48k.json", json.dumps(CONFIG_48K)), ("i18n/locale/en_US.json", "{}"),
                          ("assets/rmvpe/rmvpe.pt", "rmvpe")):
            path = os.path.join(self.rvc, *rel.split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
        for rel in MUTE_FILES:
            path = os.path.join(self.rvc, "logs", "mute", *rel.split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(b"mute")
        os.environ.pop("FAKE_TRAIN", None)

    def trainer(self, device="cuda", loader="upstream"):
        root = tempfile.mkdtemp(prefix="root-", dir=self.td)
        return voice_train.TrainRunner(os.path.join(self.td, "tools.log"), os.path.join(root, "rvc"), env={
            "VOICE_RVC_DIR": self.rvc, "VOICE_RVC_PYTHON": sys.executable, "VOICE_DEVICE": device, "VOICE_TRAIN_LOADER": loader})

    def sample(self, seconds=35.0, name="a.wav", kind="voice"):
        """A recording of a 'voice' (the sung tone on A3), cached across tests; or noise, or silence."""
        key = (seconds, kind)
        if key not in self.SAMPLES:
            if kind == "voice":
                x = tone_song(seconds=seconds, band_level=0)
            elif kind == "song":
                x = tone_song(seconds=seconds)
            else:
                x = np.zeros((int(seconds * SR), 2), np.float32)
            self.SAMPLES[key] = x
        path = os.path.join(self.td, name)
        va.encode(path, self.SAMPLES[key], SR, bits=16)
        return path

    def train(self, samples=None, trainer=None, req=None, **kw):
        work = tempfile.mkdtemp(prefix="work-", dir=self.td)
        said = []
        trainer = trainer or self.trainer()
        samples = samples or [self.sample(name="a.wav"), self.sample(name="b.wav")]
        out = voice_train.run(req or train_request(), samples, work, self.runner(), trainer, progress=said.append, **kw)
        return out, said, trainer


class TrainTests(TrainBase):
    def test_the_whole_flow_with_the_real_command_lines(self):
        trace = {}
        out, said, trainer = self.train(trace=trace)
        r = out["report"]
        self.assertEqual(sorted(out["files"]), ["model.index", "model.pth", "train_report.json"])
        with open(out["files"]["model.pth"], "rb") as f:
            model = f.read()
        self.assertEqual(model, b"model after 300 rounds at batch 8")  # what train.py left in assets/weights/<name>.pth
        with open(out["files"]["model.index"], "rb") as f:
            self.assertEqual(f.read(), b"index-bytes")  # the added_*.index
        import hashlib
        self.assertEqual((r["model_sha256"], r["model_bytes"]), (hashlib.sha256(model).hexdigest(), len(model)))
        self.assertEqual(r["index_sha256"], hashlib.sha256(b"index-bytes").hexdigest())
        self.assertAlmostEqual(r["minutes"], 70 / 60, delta=0.03)  # two 35 second recordings, all of it voice
        self.assertEqual((r["epochs"], r["epochs_asked"], r["batch_size"], r["sample_count"], r["samples_sent"], r["slices"]),
                         (300, "auto", 8, 2, 2, 6))
        self.assertEqual(sorted(r["range"]), ["p05", "p50", "p95"])
        self.assertAlmostEqual(r["range"]["p50"], 57.0, delta=0.3)  # the tone sits on A3 with a little vibrato
        self.assertTrue(r["range"]["p05"] <= r["range"]["p50"] <= r["range"]["p95"])
        self.assertLess(r["range"]["p95"] - r["range"]["p05"], 1.5)
        voice_request.parse({"audio_key": "yue2/a/b.wav", "model_key": MODEL, "voice_range": r["range"]}, resolve=False)
        self.assertEqual(r["notes"], [])
        self.assertEqual((r["check"], r["loader"]), ({"ok": True}, "upstream"))
        self.assertEqual(said, ["Listening to the recordings", "Getting the recordings ready", "Learning the voice",
                                "Building the voice's index"])
        self.assertEqual(trace, {"listen": trace["listen"], "preprocess": 6, "f0": 6, "features": 6, "filelist": 6, "train": True,
                                 "index": True, "check": True})
        with open(out["files"]["train_report.json"]) as f:
            self.assertEqual(json.load(f)["model_sha256"], r["model_sha256"])
        # the command lines, as RVC's WebUI builds them, started in the job's own project root with RVC's code on the path
        calls = {c["tool"]: c for c in self.calls() if c["tool"] in TRAIN_SCRIPTS}
        exp = trainer.exp_dir
        for c in calls.values():
            self.assertEqual((os.path.realpath(c["cwd"]), c["PYTHONPATH"], c["PYTHONSAFEPATH"], c["RVC_CUDA_GRAPH"]),
                             (os.path.realpath(trainer.root), self.rvc, "1", "0"), c["tool"])
        pre = calls["preprocess"]["argv"]
        self.assertEqual((os.path.basename(pre[0]), pre[1], pre[3], pre[4], pre[5]), ("dataset", "48000", exp, "False", "3.7"))
        self.assertEqual(sorted(os.listdir(pre[0])), ["000.wav", "001.wav"])  # the key's own name never reaches the tools
        self.assertEqual(calls["f0"]["argv"], ["cuda", "1", "0", "0", exp, "True"])
        self.assertEqual(calls["hubert"]["argv"], ["cuda:0", "1", "0", exp, "v2", "True"])
        self.assertEqual(calls["train"]["argv"], [
            "-e", "voice", "-sr", "48k", "-f0", "1", "-bs", "8", "-g", "0", "-te", "300", "-se", "301",
            "-pg", os.path.join(self.rvc, "assets", "pretrained_v2", "f0G48k.pth"),
            "-pd", os.path.join(self.rvc, "assets", "pretrained_v2", "f0D48k.pth"), "-l", "1", "-c", "0", "-sw", "0", "-v", "v2"])
        self.assertEqual((calls["train"]["RVC_CUDA_GRAPH"], calls["train"]["workers"]), ("0", None))
        index = calls["index"]["argv"]
        self.assertEqual((index[0], index[1], index[2], index[4]),
                         ("voice", "v2", os.path.join(trainer.root, "assets", "indices"), "single"))
        # the file list and config, as run_train_model writes them: every slice with its four files, speaker 0, silence twice
        with open(os.path.join(exp, "filelist.txt")) as f:
            rows = f.read().split("\n")
        e, root = exp.replace("\\", "/"), trainer.root.replace("\\", "/")
        want = [f"{e}/0_gt_wavs/{n}.wav|{e}/3_feature768/{n}.npy|{e}/2a_f0/{n}.wav.npy|{e}/2b-f0nsf/{n}.wav.npy|0"
                for n in ("0_0", "0_1", "0_2", "1_0", "1_1", "1_2")]
        mute = (f"{root}/logs/mute/0_gt_wavs/mute48k.wav|{root}/logs/mute/3_feature768/mute.npy|"
                f"{root}/logs/mute/2a_f0/mute.wav.npy|{root}/logs/mute/2b-f0nsf/mute.wav.npy|0")
        self.assertEqual(sorted(rows), sorted(want + [mute, mute]))
        for row in rows:
            for path in row.split("|")[:4]:
                self.assertTrue(os.path.exists(path), path)
        with open(os.path.join(exp, "config.json")) as f:
            self.assertEqual(json.load(f), CONFIG_48K)
        # the things the scripts look up from the working folder are there, and nothing was written into the RVC folder
        self.assertTrue(os.path.exists(os.path.join(trainer.root, "i18n", "locale", "en_US.json")))
        self.assertTrue(os.path.exists(os.path.join(trainer.root, "assets", "rmvpe", "rmvpe.pt")))
        self.assertFalse(os.path.exists(os.path.join(self.rvc, "assets", "weights")))
        self.assertEqual(sorted(os.listdir(os.path.join(self.rvc, "logs"))), ["mute"])
        # and the new voice sang a test line through the song jobs' own RVC call, index and all
        check = self.calls("rvc")[-1]["argv"]
        self.assertEqual((check[check.index("--model") + 1], check[check.index("--index") + 1]),
                         (out["files"]["model.pth"], out["files"]["model.index"]))

    def test_under_a_minute_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self.train(samples=[self.sample(20.0, "short.wav")])
        self.assertEqual(caught.exception.args[0], "That is under a minute of clear voice. Add more recordings: ten minutes or "
                                                   "more makes a much better voice.")
        self.assertEqual([c for c in self.calls() if c["tool"] in TRAIN_SCRIPTS], [])  # nothing was started
        quiet = self.sample(35.0, "gappy.wav")
        x = va.decode(quiet)[0].copy()
        x[SR * 5:SR * 30] = 0  # 35 seconds on the clock, 10 of voice
        va.encode(quiet, x, SR, bits=16)
        with self.assertRaises(ValueError) as caught:
            self.train(samples=[quiet, self.sample(35.0, "b.wav")])  # 45 seconds of voice in 70 of recording
        self.assertEqual(caught.exception.args[0], voice_train.TOO_LITTLE)

    def test_only_the_first_hour_is_used(self):
        with mock.patch.object(voice_train, "MAX_TRAIN_SECONDS", 50):  # an "hour" of 50 seconds
            out, _, trainer = self.train(min_seconds=20)
        r = out["report"]
        self.assertAlmostEqual(r["seconds"], 50.0, delta=0.2)
        self.assertEqual(r["notes"], ["There was more than an hour of voice, so only the first hour was used."])
        second = va.decode(os.path.join(os.path.dirname(out["files"]["model.pth"]), "dataset", "001.wav"))[0]
        self.assertAlmostEqual(len(second) / SR, 15.0, delta=0.2)  # the second recording is cut where the hour fills

    def test_recordings_that_cannot_be_read_are_left_out(self):
        junk = os.path.join(self.td, "junk.wav")
        with open(junk, "wb") as f:
            f.write(b"this is not audio")
        out, _, _ = self.train(samples=[self.sample(name="a.wav"), junk, self.sample(name="b.wav"),
                                        self.sample(5.0, "silent.wav", kind="silence")])
        r = out["report"]
        self.assertEqual((r["sample_count"], r["samples_sent"]), (2, 4))
        self.assertEqual(r["notes"], ["2 of the recordings could not be read or were silent, so they were left out."])
        with self.assertRaises(ValueError) as caught:
            self.train(samples=[junk])
        self.assertEqual(caught.exception.args[0], "None of the recordings could be read. Upload them again as WAV, MP3, FLAC "
                                                   "or M4A files.")

    def test_noise_is_not_a_voice(self):
        path = os.path.join(self.td, "noise.wav")
        va.encode(path, 0.1 * np.random.default_rng(4).standard_normal((SR * 70, 1)).astype(np.float32), SR, bits=16)
        with self.assertRaises(ValueError) as caught:
            self.train(samples=[path])
        self.assertEqual(caught.exception.args[0], "The recordings have no clear voice to learn. Use recordings of one person "
                                                   "singing or speaking.")

    def test_epochs_by_minutes_and_batch_by_memory(self):
        self.assertEqual([voice_train.auto_epochs(m) for m in (1, 4.99, 5, 14.9, 15, 29.9, 30, 44.9, 45, 60)],
                         [300, 300, 200, 200, 120, 120, 80, 80, 60, 60])
        for minutes in (1, 5, 10, 15, 30, 45, 60):  # the table keeps a run inside the job on the estimate it is built on
            self.assertLess(voice_train.auto_epochs(minutes) * voice_train.epoch_seconds(minutes), 60 * 60, minutes)
        self.assertEqual([voice_train.batch_size(v, env={}) for v in (None, 48, 24, 23.6, 16, 12, 8)], [8, 8, 8, 8, 6, 4, 2])
        self.assertEqual(voice_train.batch_size(24, env={"VOICE_TRAIN_BATCH": "12"}), 12)

    def test_rounds_asked_for_and_rounds_that_do_not_fit(self):
        out, _, _ = self.train(req=train_request(options={"epochs": 40}), vram_gb=16.0)
        self.assertEqual((out["report"]["epochs"], out["report"]["epochs_asked"], out["report"]["batch_size"]), (40, 40, 6))
        train = [c for c in self.calls() if c["tool"] == "train"][-1]["argv"]
        self.assertEqual((train[train.index("-te") + 1], train[train.index("-se") + 1], train[train.index("-bs") + 1]),
                         ("40", "41", "6"))
        # 70 seconds of voice is about 1.8 s a round on the estimate: with 300 s to spare after the reserve, about 160 rounds fit
        room = voice_train.AFTER_TRAIN_S + 300
        out, _, _ = self.train(req=train_request(options={"epochs": 500}), deadline=time.monotonic() + room)
        r = out["report"]
        self.assertTrue(20 <= r["epochs"] <= 165, r["epochs"])
        self.assertEqual(r["notes"], [f"500 training rounds would not finish in the time a job may run with this much audio, so "
                                      f"the voice was trained for {r['epochs']}."])
        out, _, _ = self.train(deadline=time.monotonic() + room)  # automatic rounds (300 here) are trimmed without a note
        self.assertTrue(20 <= out["report"]["epochs"] <= 165, out["report"]["epochs"])
        self.assertEqual(out["report"]["notes"], [])
        with self.assertRaises(ValueError) as caught:
            self.train(deadline=time.monotonic() + voice_train.AFTER_TRAIN_S + 10)
        self.assertEqual(caught.exception.args[0], "There is not enough time left in this job to learn the voice. Start it "
                                                   "again with less audio.")

    def test_a_run_that_will_not_finish_is_stopped_early(self):
        """Three rounds are timed; when the rest would run past the job's end, training is stopped and the person is told."""
        clock = {"now": 1000.0}
        fake_time = types.SimpleNamespace(monotonic=lambda: clock["now"], strftime=lambda *a: "00:00:00", sleep=lambda s: None)
        said = []

        class SlowTrainer:
            device, log_path, loader_used = "cuda", os.path.join(self.td, "tools.log"), "upstream"
            done = 0

            def preprocess(self, dataset): return 6, 1.0
            def extract_f0(self, cpu=False): return 6, 1.0
            def extract_features(self): return 6, 1.0
            def write_filelist(self): return 6
            def epochs_done(self): return self.done

            def train(self, epochs, batch, limit, tick):
                for self.done in range(1, epochs + 1):
                    clock["now"] += 600.0  # ten minutes a round
                    if tick():
                        return False, 1.0, True
                return True, 1.0, False
        work = tempfile.mkdtemp(prefix="work-", dir=self.td)
        samples = [self.sample(name="a.wav"), self.sample(name="b.wav")]
        with mock.patch.object(voice_train, "time", fake_time):
            with self.assertRaises(ValueError) as caught:
                voice_train.run(train_request(options={"epochs": 30}), samples, work, self.runner(), SlowTrainer(),
                                progress=said.append, deadline=clock["now"] + 7200)
        self.assertEqual(caught.exception.args[0], "Learning this voice was going to take longer than a job may run. Use fewer "
                                                   "training rounds or less audio.")
        self.assertEqual(said[-1], "Learning the voice, epoch 5 of 30")  # stopped after four rounds, not after thirty
        self.assertIn("Learning the voice, epoch 2 of 30", said)

    def test_a_slow_card_gets_the_rounds_that_fit(self):
        """The plan was 200 rounds; the first timed rounds take 40 seconds each, which cannot finish. Training is started once
        more with the rounds that do fit at that pace, and the answer says so."""
        clock = {"now": 1000.0}
        fake_time = types.SimpleNamespace(monotonic=lambda: clock["now"], strftime=lambda *a: "00:00:00", sleep=lambda s: None)
        trainer = self.trainer()
        starts = []

        def paced_train(epochs, batch, limit, tick):
            starts.append(epochs)
            for done in range(1, epochs + 1):
                trainer.epochs_done = lambda done=done: done
                clock["now"] += 40.0
                if tick():
                    return False, 1.0, True
            del trainer.epochs_done
            return voice_train.TrainRunner.train(trainer, epochs, batch, limit, None)
        trainer.train = paced_train
        with mock.patch.object(voice_train, "time", fake_time):
            out, said, _ = self.train(trainer=trainer, req=train_request(options={"epochs": 200}), deadline=clock["now"] + 7200)
        self.assertEqual(starts, [200, 159])  # stopped after four rounds; 6,440 seconds were left, less two rounds to start again
        r = out["report"]
        self.assertEqual((r["epochs"], r["epochs_asked"]), (159, 200))
        self.assertEqual(r["notes"], ["This card learned more slowly than planned, so the voice was trained for 159 rounds instead "
                                      "of 200."])
        with open(out["files"]["model.pth"], "rb") as f:
            self.assertEqual(f.read(), b"model after 159 rounds at batch 8")
        self.assertIn("Learning the voice, epoch 159 of 159", said)
        self.assertLessEqual(clock["now"], 1000.0 + 7200 - voice_train.AFTER_TRAIN_S)  # and it finished with the reserve intact

    def test_progress_names_the_round(self):
        clock = {"now": 50.0}
        fake_time = types.SimpleNamespace(monotonic=lambda: clock["now"], strftime=lambda *a: "00:00:00", sleep=lambda s: None)
        trainer = self.trainer()

        def slow_train(epochs, batch, limit, tick):
            for done in range(0, 4):
                trainer.epochs_done = lambda done=done: done
                for _ in range(4):  # a look every 10 seconds
                    clock["now"] += 10.0
                    tick()
            return voice_train.TrainRunner.train(trainer, epochs, batch, limit, None)
        trainer.train = slow_train
        with mock.patch.object(voice_train, "time", fake_time):
            _, said, _ = self.train(trainer=trainer, req=train_request(options={"epochs": 200}))
        told = [s for s in said if s.startswith("Learning the voice")]
        self.assertEqual(told[0], "Learning the voice")
        self.assertIn("Learning the voice, epoch 2 of 200", told)
        self.assertIn("Learning the voice, epoch 4 of 200", told)
        self.assertLessEqual(len(told), 1 + 160 // voice_train.PROGRESS_EVERY_S + 1)  # one sentence each 30 seconds at most

    def test_the_round_counter_reads_train_py_in_any_language(self):
        trainer = self.trainer()
        self.assertEqual(trainer.epochs_done(), 0)
        log = os.path.join(trainer.exp_dir, "train.log")
        lines = ["2026-11-18 10:00:00,000\tvoice\tINFO\tTrain Epoch: 1 [0%]",
                 "2026-11-18 10:00:01,000\tvoice\tINFO\t[0, 0.0001]",
                 "2026-11-18 10:00:09,000\tvoice\tINFO\t====> Epoch: 1 [2026-11-18 10:00:09] | (0:00:09.123456)",
                 "2026-11-18 10:00:18,000\tvoice\tINFO\t====> 轮次：2 [2026-11-18 10:00:18] | (0:00:09.000001)",
                 "2026-11-18 10:00:19,000\tvoice\tINFO\tTrain Epoch: 3 [10%]"]
        with open(log, "w", encoding="utf8") as f:
            f.write("\n".join(lines))
        self.assertEqual(trainer.epochs_done(), 2)
        # on the CPU the line only reaches the tool output (the child process has no log file): read there too, but only
        # from where this training started, so an earlier start's rounds are not counted again
        with open(trainer.log_path, "w", encoding="utf8") as f:
            f.write("INFO:voice:====> Epoch: 9 [2026-11-18 09:00:00] | (0:00:09.000001)\n")
        trainer.log_from = os.path.getsize(trainer.log_path)
        self.assertEqual(trainer.epochs_done(), 2)
        with open(trainer.log_path, "a", encoding="utf8") as f:
            f.write("INFO:voice:Training epoch: 4 [50%]\nINFO:voice:====> Epoch: 3 [2026-11-18 10:00:27] | (0:00:09.000001)\n")
        self.assertEqual(trainer.epochs_done(), 3)
        os.remove(log)
        self.assertEqual(trainer.epochs_done(), 3)

    def test_small_shared_memory_turns_the_loader_workers_off(self):
        self.assertIsNone(voice_train.loader_plan(8, None))  # size unknown: train.py as it is
        self.assertIsNone(voice_train.loader_plan(8, 12.0))
        self.assertEqual(voice_train.loader_plan(8, 0.064), 0)  # Docker's default 64 MB
        self.assertEqual(voice_train.loader_plan(8, 1.0), 0)
        self.assertIsNone(voice_train.loader_plan(1, 0.5))
        with mock.patch.object(voice_train, "shm_free_gb", lambda: 0.064):
            out, _, trainer = self.train(trainer=self.trainer(loader="auto"))
        call = [c for c in self.calls() if c["tool"] == "train"][-1]
        self.assertEqual((call["workers"], out["report"]["loader"]), ("0", 0))
        self.assertEqual(call["PYTHONPATH"], os.path.join(HERE, "train_site") + os.pathsep + self.rvc)
        self.assertTrue(os.path.exists(os.path.join(HERE, "train_site", "sitecustomize.py")))
        with mock.patch.object(voice_train, "shm_free_gb", lambda: 30.0):
            out, _, _ = self.train(trainer=self.trainer(loader="auto"))
        call = [c for c in self.calls() if c["tool"] == "train"][-1]
        self.assertEqual((call["workers"], call["PYTHONPATH"], out["report"]["loader"]), (None, self.rvc, "upstream"))

    def test_the_loader_patch_itself(self):
        """train_site/sitecustomize.py against a stand-in torch: train.py's loader arguments, held to no workers."""
        import importlib.util
        path = os.path.join(HERE, "train_site", "sitecustomize.py")
        asked = dict(num_workers=4, shuffle=False, pin_memory=True, persistent_workers=True, prefetch_factor=8)

        def loaded(workers):
            """The patch run against a fresh stand-in torch.utils.data -> (its DataLoader class, the arguments it was given)."""
            seen = []

            class DataLoader:
                def __init__(self, *args, **kwargs):
                    seen.append(kwargs)
            data = types.ModuleType("torch.utils.data")
            data.DataLoader = DataLoader
            fake = {"torch": types.ModuleType("torch"), "torch.utils": types.ModuleType("torch.utils"), "torch.utils.data": data}
            env = {} if workers is None else {"VOICE_TRAIN_LOADER_WORKERS": workers}
            with mock.patch.dict(sys.modules, fake), mock.patch.dict(os.environ, env):
                if workers is None:
                    os.environ.pop("VOICE_TRAIN_LOADER_WORKERS", None)
                spec = importlib.util.spec_from_file_location("voice_train_site", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            return data.DataLoader, seen, module
        loader, seen, _keep = loaded("0")
        loader("dataset", **asked)
        loader("dataset", batch_size=1)
        self.assertEqual(seen[0], dict(num_workers=0, shuffle=False, pin_memory=True, persistent_workers=False))
        self.assertEqual(seen[1], {"batch_size": 1})  # a loader that asks for no workers is left alone
        loader, seen, _keep = loaded(None)
        loader("dataset", **asked)
        self.assertEqual(seen[0], asked)  # not switched on: train.py's loader exactly as written

    def test_on_a_cpu_the_cpu_forms_are_used(self):
        out, _, trainer = self.train(trainer=self.trainer(device="cpu"))
        calls = {c["tool"]: c for c in self.calls() if c["tool"] in TRAIN_SCRIPTS}
        self.assertEqual(calls["f0"]["argv"][0::3], ["cpu", "rmvpe"])
        self.assertEqual(calls["f0"]["argv"][1], trainer.exp_dir)
        self.assertEqual(calls["hubert"]["argv"], ["cpu", "1", "0", trainer.exp_dir, "v2", "False"])
        self.assertNotIn("-g", calls["train"]["argv"])

    def test_pitch_on_the_gpu_failing_falls_back_to_the_cpu_form(self):
        os.environ["FAKE_TRAIN"] = json.dumps({"f0": "fail_cuda"})
        with redirect_stdout(io.StringIO()):
            out, _, _ = self.train()
        f0 = [c["argv"][0] for c in self.calls() if c["tool"] == "f0"]
        self.assertEqual(f0, ["cuda", "cpu"])
        self.assertEqual(out["report"]["notes"], ["The fast pitch reader did not finish on this card, so the slower one was used."])
        self.assertEqual(out["report"]["slices"], 6)

    def test_features_in_half_precision_failing_are_made_in_full(self):
        os.environ["FAKE_TRAIN"] = json.dumps({"hubert": "fail_half"})
        with redirect_stdout(io.StringIO()):
            out, _, _ = self.train()
        self.assertEqual([c["argv"][-1] for c in self.calls() if c["tool"] == "hubert"], ["True", "False"])
        self.assertEqual(out["report"]["notes"], ["The fast voice reader did not finish on this card, so the slower one was used."])
        self.assertEqual(out["report"]["slices"], 6)

    def test_a_slice_with_no_pitch_is_not_a_failure(self):
        """RVC skips a slice that has no pitch at all (a breath, an S): that slice is left out, with no fallback and no note."""
        trainer = self.trainer()
        real = trainer.extract_f0

        def one_short(cpu=False):
            count, secs = real(cpu)
            for folder in ("2a_f0", "2b-f0nsf"):
                os.remove(os.path.join(trainer.exp_dir, folder, "1_2.wav.npy"))
            return count - 1, secs
        trainer.extract_f0 = one_short
        out, _, _ = self.train(trainer=trainer)
        self.assertEqual((out["report"]["slices"], out["report"]["notes"]), (5, []))
        self.assertEqual([c["argv"][0] for c in self.calls() if c["tool"] == "f0"], ["cuda"])

    def test_failures_are_named_not_hidden(self):
        for plan, error in (({"preprocess": "none"}, ValueError), ({"f0": "fail"}, RuntimeError), ({"train": "fail"}, RuntimeError),
                            ({"train": "no_model"}, RuntimeError), ({"index": "fail"}, RuntimeError)):
            os.environ["FAKE_TRAIN"] = json.dumps(plan)
            with redirect_stdout(io.StringIO()), self.assertRaises(error, msg=plan) as caught:
                self.train()
            if error is ValueError:
                self.assertEqual(caught.exception.args[0], "The recordings were too quiet or too broken up to learn from. Use "
                                                           "clearer recordings.")

    def test_a_voice_that_cannot_sing_its_test_line_is_kept_with_a_note(self):
        with open(os.path.join(self.rvc, "infer", "cli.py"), "w") as f:
            f.write("import sys\nsys.exit(1)\n")
        with redirect_stdout(io.StringIO()):
            out, _, _ = self.train()
        self.assertEqual(out["report"]["check"], {"ok": False})
        self.assertEqual(out["report"]["notes"], ["The new voice could not sing a test line on this worker. Try it on a song, "
                                                  "and train it again if it does not work."])
        self.assertTrue(os.path.exists(out["files"]["model.pth"]))

    def test_isolate_dereverb_and_denoise_run_once_over_every_recording(self):
        songs = [self.sample(35.0, "a.wav", kind="song"), self.sample(35.0, "b.wav", kind="song")]
        out, said, _ = self.train(samples=songs, req=train_request(options={"isolate": True, "dereverb": True, "denoise": 0.5}))
        self.assertEqual([c["argv"][1] for c in self.calls("pymss")],
                         [EX("hyperace"), LEAD("frazer"), voice_request.DEREVERB_MODEL, voice_request.DENOISE_MODEL])  # one load each
        self.assertEqual(said[:5], ["Listening to the recordings", "Taking the music off the recordings",
                                    "Taking the room off the recordings", "Cleaning the noise off the recordings",
                                    "Getting the recordings ready"])
        self.assertEqual(out["report"]["notes"], [])
        self.assertEqual(out["report"]["sample_count"], 2)
        first = va.decode(os.path.join(os.path.dirname(out["files"]["model.pth"]), "dataset", "000.wav"))[0]
        raw = va.decode(songs[0])[0].mean(1)
        # vocals 0.5 -> lead 0.8 -> dry 0.9 -> half of the denoiser's 0.8: what is learned is the isolated, cleaned voice
        self.assertAlmostEqual(voice_pipeline.rms_db(first) - voice_pipeline.rms_db(raw), 20 * np.log10(0.5 * 0.8 * 0.9 * 0.9), delta=0.3)
        work = os.path.dirname(out["files"]["model.pth"])
        self.assertEqual([d for d in os.listdir(work) if d.endswith(("_in", "_out")) and not d.startswith("check")], [])  # cleared

    def test_isolate_uses_the_fallback_and_leaves_out_what_has_no_voice(self):
        self.plan({EX("hyperace"): "fail"})
        songs = [self.sample(35.0, "a.wav", kind="song"), self.sample(35.0, "b.wav", kind="song")]
        with redirect_stdout(io.StringIO()):
            out, _, _ = self.train(samples=songs, req=train_request(options={"isolate": True}))
        self.assertEqual([c["argv"][1] for c in self.calls("pymss")], [EX("hyperace"), EX("bs_roformer"), LEAD("frazer")])
        self.assertEqual(out["report"]["sample_count"], 2)
        self.plan({EX("hyperace"): "silent", EX("bs_roformer"): "silent"})
        with redirect_stdout(io.StringIO()), self.assertRaises(ValueError) as caught:
            self.train(samples=songs, req=train_request(options={"isolate": True}))
        self.assertEqual(caught.exception.args[0], voice_train.TOO_LITTLE)

    def test_cleanup_that_cannot_run_keeps_the_recordings(self):
        self.plan({voice_request.DENOISE_MODEL: "fail", voice_request.DEREVERB_MODEL: "fail"})
        with redirect_stdout(io.StringIO()):
            out, _, _ = self.train(req=train_request(options={"dereverb": True, "denoise": 1}))
        self.assertEqual(out["report"]["notes"], ["The room could not be taken off every recording, so some were used with their echo.",
                                                  "The noise could not be cleaned off every recording, so some were used as they were."])
        self.assertEqual(out["report"]["sample_count"], 2)


class TrainHandlerTests(TrainBase):
    OWNER, VOICE = "u123", "voice-abc12345"

    def setUp(self):
        super().setUp()
        import voice_handler
        self.vh = voice_handler
        self.keys = [f"voice-samples/{self.OWNER}/{self.VOICE}/take one.wav".replace(" ", "_"),
                     f"voice-samples/{self.OWNER}/{self.VOICE}/b/second.MP3"]
        objects = {}
        for key in self.keys:
            with open(self.sample(name="s.wav"), "rb") as f:
                objects[key] = f.read()
        self.s3 = FakeS3(objects)
        self.progress = []
        sys.modules["runpod"] = types.SimpleNamespace(
            serverless=types.SimpleNamespace(progress_update=lambda job, text: self.progress.append(text)))
        runner, real = self.runner(), voice_train.TrainRunner
        env = {"VOICE_RVC_DIR": self.rvc, "VOICE_RVC_PYTHON": sys.executable, "VOICE_DEVICE": "cuda", "VOICE_TRAIN_LOADER": "upstream"}
        self.roots = []

        def make_trainer(log, root):
            self.roots.append(root)
            return real(log, root, env=env)
        self.patches = [mock.patch.object(voice_handler, "storage", lambda: (self.s3, "bucket")),
                        mock.patch.object(voice_pipeline, "Runner", lambda log: runner),
                        mock.patch.object(voice_train, "TrainRunner", make_trainer),
                        mock.patch.dict(voice_handler.GPU, {"cuda": True, "name": "NVIDIA GeForce RTX 4090", "supported": True,
                                                            "capability": "sm_89", "vram_gb": 24.0}, clear=True)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        sys.modules.pop("runpod", None)
        super().tearDown()

    def job(self, **over):
        body = {"mode": "train", "owner": self.OWNER, "voice_id": self.VOICE, "sample_keys": list(self.keys)}
        body.update(over)
        return {"id": "job-1", "input": body}

    def test_a_train_job_answers_with_the_voice_and_writes_three_files(self):
        out = self.vh.handler(self.job())
        self.assertNotIn("error", out, out)
        prefix = f"voice-models/{self.OWNER}/{self.VOICE}"
        self.assertEqual(set(out), {"engine", "mode", "model_key", "model_sha256", "index_key", "index_sha256", "report_key", "range",
                                    "minutes", "epochs", "sample_count", "gpu", "timing", "worker_notes", "processing_ms", "features"})
        self.assertEqual((out["engine"], out["mode"], out["model_key"], out["index_key"], out["report_key"]),
                         ("rvc", "train", f"{prefix}/model.pth", f"{prefix}/model.index", f"{prefix}/train_report.json"))
        import hashlib
        self.assertEqual(out["model_sha256"], hashlib.sha256(b"model after 300 rounds at batch 8").hexdigest())
        self.assertEqual(out["index_sha256"], hashlib.sha256(b"index-bytes").hexdigest())
        self.assertEqual(sorted(out["range"]), ["p05", "p50", "p95"])
        self.assertAlmostEqual(out["minutes"], 70 / 60, delta=0.03)
        self.assertEqual((out["epochs"], out["sample_count"], out["gpu"], out["worker_notes"]),
                         (300, 2, "NVIDIA GeForce RTX 4090", []))
        self.assertIn("train", out["features"])
        for key in ("download_s", "listen_s", "train_s", "index_s", "upload_s"):
            self.assertIn(key, out["timing"])
        # exactly three files, all in the voice's own folder; the recordings are read and nothing else is touched
        self.assertEqual(sorted(self.s3.uploads), sorted([
            (f"{prefix}/model.pth", "application/octet-stream", len(b"model after 300 rounds at batch 8")),
            (f"{prefix}/model.index", "application/octet-stream", len(b"index-bytes")),
            (f"{prefix}/train_report.json", "application/json", self.s3.uploads[-1][2])]))
        self.assertEqual(self.s3.downloads, self.keys)
        self.assertEqual(self.progress, ["Fetching the recordings", "Listening to the recordings", "Getting the recordings ready",
                                         "Learning the voice", "Building the voice's index", "Saving the voice"])
        self.assertFalse(os.path.exists(self.roots[0]))  # the job's project root, slices and checkpoints went with its temp folder
        fetched = [c for c in self.calls() if c["tool"] == "preprocess"][0]["argv"][0]
        self.assertFalse(os.path.exists(fetched))  # and so did the recordings
        self.assertEqual(json.dumps(out), json.dumps(json.loads(json.dumps(out))))  # plain JSON all the way down

    def test_requests_that_reach_outside_the_voices_folder_start_nothing(self):
        for over in (dict(sample_keys=[self.keys[0], "voice-samples/u999/voice-abc12345/one.wav"]),
                     dict(sample_keys=[self.keys[0], f"voice-samples/{self.OWNER}/{self.VOICE}/../../u999/x/one.wav"]),
                     dict(owner="u123/../u999"), dict(voice_id="../voice-other"), dict(voice_id="voice-abc12345/x")):
            out = self.vh.handler(self.job(**over))
            self.assertEqual(set(out), {"error"}, over)
            self.assertTrue(out["error"].endswith("."), over)
        self.assertEqual((self.s3.heads, self.s3.downloads, self.s3.uploads, self.calls()), ([], [], [], []))

    def test_recordings_missing_or_too_large(self):
        out = self.vh.handler(self.job(sample_keys=self.keys + [f"voice-samples/{self.OWNER}/{self.VOICE}/gone.wav"]))
        self.assertEqual(out, {"error": "One of the recordings is no longer in storage. Upload it again."})
        with mock.patch.object(voice_request, "MAX_SAMPLE_BYTES", 1000):
            out = self.vh.handler(self.job())
        self.assertEqual(out, {"error": "One of the recordings is too large. Use files up to 256 megabytes each."})
        with mock.patch.object(voice_request, "MAX_SAMPLES_BYTES", 4 * 1000 * 1000):
            out = self.vh.handler(self.job())
        self.assertEqual(out, {"error": "The recordings are too large together. Use up to 2 gigabytes in all."})
        self.assertEqual((self.s3.downloads, self.s3.uploads), ([], []))  # sizes are checked before anything is fetched

    def test_too_little_voice_is_the_plain_sentence_and_nothing_is_saved(self):
        with open(self.sample(20.0, "short.wav"), "rb") as f:
            self.s3.objects[self.keys[0]] = f.read()
        out = self.vh.handler(self.job(sample_keys=self.keys[:1]))
        self.assertEqual(out, {"error": "That is under a minute of clear voice. Add more recordings: ten minutes or more makes "
                                        "a much better voice."})
        self.assertEqual(self.s3.uploads, [])

    def test_an_image_whose_training_proof_failed_refuses(self):
        with open(os.environ["VOICE_SELFTEST"], "w") as f:
            json.dump({"train": {"ok": False, "error": "RuntimeError: RVC training failed"}}, f)
        out = self.vh.handler(self.job())
        self.assertEqual(out, {"error": "This worker cannot train voices. Its own training check failed when it was built."})
        self.assertEqual((self.s3.heads, self.s3.downloads, self.s3.uploads), ([], [], []))
        with open(os.environ["VOICE_SELFTEST"], "w") as f:
            json.dump({"train": {"ok": True}}, f)
        self.assertNotIn("error", self.vh.handler(self.job()))

    def test_a_broken_run_is_a_sentence_and_nothing_is_saved(self):
        os.environ["FAKE_TRAIN"] = json.dumps({"train": "fail"})
        with redirect_stdout(io.StringIO()):
            out = self.vh.handler(self.job())
        self.assertEqual(out, {"error": "The voice could not be trained (RuntimeError). The recordings are kept; nothing was retried."})
        self.assertEqual(self.s3.uploads, [])

    def test_an_unconfirmed_card_refuses_before_any_download(self):
        with mock.patch.dict(self.vh.GPU, {"cuda": True, "supported": False, "name": "NVIDIA GeForce RTX 5090"}, clear=True):
            out = self.vh.handler(self.job())
        self.assertIn("compatible compiled CUDA support", out["error"])
        self.assertEqual((self.s3.heads, self.s3.downloads), ([], []))


if __name__ == "__main__":
    unittest.main()
