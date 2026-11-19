"""Make a new voice: request mode "train" (voice_request.parse_train). Recordings of one voice in, an RVC v2 48 kHz model with
pitch (the .pth the song jobs load) and its retrieval index out. The steps are the ones RVC's own WebUI runs for "one-click
training" at the pinned commit (webui.py: preprocess_dataset, extract_f0_feature, run_train_model, train_index), with the same
scripts and the same arguments, and the tools passed in like voice_pipeline.Runner, so the CPU tests run the whole flow with
stand-ins.

  1. listen     every recording decoded to 44.1 kHz; with "isolate" the vocal extractor (and its fallback) and the lead split run
                on all of them in one go (one model load for the lot); "dereverb" and "denoise" the same way
  2. measure    the seconds that are not silence. Under a minute is refused; over an hour, only the first hour is used
  3. range      the voice's own low, middle and high notes (p05, p50, p95 as MIDI numbers) from its pitch across the recordings:
                the voice_range a song job needs for pitch "auto" and "fit"
  4. prepare    train/preprocess.py (slices of 3.7 s at 48 kHz and 16 kHz), train/dataset/extract_f0.py (RMVPE on the GPU),
                train/dataset/extract_hubert_feature.py (v2, 768 wide), then the file list and config.json written exactly as
                run_train_model writes them: speaker 0 and the two rows of RVC's own silence sample (logs/mute)
  5. learn      train/train.py from the pretrained f0G48k / f0D48k, for the rounds asked for or auto_epochs()
  6. index      train/train_index.py
  7. check      the new model sings four seconds of its own recordings through the same infer/cli.py the song jobs use; a
                failure is a worker note, not a lost voice
Outputs: model.pth (what train.py saves at the end as assets/weights/<name>.pth), model.index (the added_*.index) and
train_report.json. Nothing else leaves the job.

RVC's scripts write to ./logs/<name> and ./assets/weights/<name>.pth under the folder they are started in, and train.py caches
a spectrogram beside every wav it reads, the silence sample's included. So each job gets a throwaway project root inside its
temp folder (TrainRunner): its own logs/ and assets/weights/, a copy of logs/mute, and links to the two things the scripts look
up from the working folder (i18n/ and assets/rmvpe/). The code itself is imported from the real RVC folder (PYTHONPATH), and the
recordings, the slices, the features and the checkpoints all vanish with the job's temp folder, whatever happens."""
import glob
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import voice_models  # noqa: E402
import voice_pipeline as vp  # noqa: E402
import vpaudio as va  # noqa: E402
from voice_request import (DEFAULTS, DENOISE_MODEL, DEREVERB_MODEL, EPOCHS_MIN, EXTRACTORS, LEAD_MODELS,  # noqa: E402
                           MAX_TRAIN_SECONDS, MIN_TRAIN_SECONDS)

SR = vp.SR
EXP = "voice"  # the experiment's name inside the job's own project root; it never leaves the worker
# "auto" rounds by minutes of voice, so a run stays roughly 15 to 40 minutes on a 24 GB card at batch 8. A round passes over
# every slice once, so more audio needs fewer rounds for the same number of steps:
#   under 5 minutes   300 rounds      15 to under 30   120 rounds
#   5 to under 15     200 rounds      30 to under 45    80 rounds      45 and over   60 rounds
EPOCH_TABLE = ((5, 300), (15, 200), (30, 120), (45, 80), (None, 60))
# An estimate, to be replaced by measured numbers after the first real jobs: about 0.3 s a step at batch 8 on a 3090-class card,
# 2.2 steps for each minute of voice, plus a second of overhead a round. Only used to refuse or trim a run that cannot finish.
EPOCH_BASE_S, EPOCH_S_PER_MINUTE = 1.0, 0.7
MAX_RAW_SECONDS = 75 * 60  # recordings decoded at most: the hour that is used plus room for the silence in it
AFTER_TRAIN_S = 600  # kept back for the index, the check and the upload
PROGRESS_EVERY_S = 30
EPOCH_LINE = re.compile(r"(\d+) \[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\] \| \(")  # train.py's end-of-round line in any language
TOO_LITTLE = "That is under a minute of clear voice. Add more recordings: ten minutes or more makes a much better voice."


def auto_epochs(minutes):
    for under, epochs in EPOCH_TABLE:
        if under is None or minutes < under:
            return epochs


def epoch_seconds(minutes):
    return EPOCH_BASE_S + EPOCH_S_PER_MINUTE * minutes


def batch_size(vram_gb, env=os.environ):
    """8 on a 24 GB card (RVC's own default would be 12; 8 leaves room). Smaller cards get less; an unknown card gets 8,
    because the endpoint only offers 24 GB cards and up."""
    if env.get("VOICE_TRAIN_BATCH"):
        return max(1, int(env["VOICE_TRAIN_BATCH"]))
    if vram_gb is None or vram_gb >= 20:
        return 8
    return 6 if vram_gb >= 15 else (4 if vram_gb >= 10 else 2)


def shm_free_gb(path="/dev/shm"):
    try:
        s = os.statvfs(path)
        return s.f_bavail * s.f_frsize / 1e9
    except (OSError, AttributeError):
        return None


def loader_plan(batch, free_gb):
    """train.py's data loader hands batches between processes through /dev/shm: 4 workers, each 8 batches ahead, about 3.4 MB
    a slice, so about 110 MB times the batch size at once. A container with a small /dev/shm (Docker's default is 64 MB) kills
    the workers with a bus error. -> None (twice that much is free, or the size is unknown: train.py exactly as it is), or 0:
    no workers (train_site/sitecustomize.py), so the slices are loaded in the training process itself and no shared memory
    is used at all. A slice loads in a few milliseconds against a third of a second for the step, so little is lost."""
    if free_gb is None or free_gb >= 2 * 0.0034 * batch * 4 * 8:
        return None
    return 0


class TrainRunner:
    """The real RVC training scripts in the RVC venv, started in a throwaway project root (see the top of this file)."""

    def __init__(self, log_path, root, env=os.environ):
        self.rvc = env.get("VOICE_RVC_DIR", "/opt/rvc")
        self.python = env.get("VOICE_RVC_PYTHON", "/opt/rvc-venv/bin/python")
        self.device = env.get("VOICE_DEVICE", "cuda")
        self.loader = env.get("VOICE_TRAIN_LOADER", "auto")  # "auto", "upstream" (never touch train.py's loader) or a number
        self.log_path = log_path
        self.root = root
        self.exp_dir = os.path.join(root, "logs", EXP)
        self.cpus = max(1, min(8, os.cpu_count() or 2))
        self.loader_used = None
        self.log_from = 0
        for d in (self.exp_dir, os.path.join(root, "assets", "weights"), os.path.join(root, "assets", "indices")):
            os.makedirs(d, exist_ok=True)
        for rel in ("i18n", os.path.join("assets", "rmvpe")):  # looked up from the working folder by the scripts
            src, dst = os.path.join(self.rvc, rel), os.path.join(root, rel)
            if os.path.isdir(src) and not os.path.exists(dst):
                try:
                    os.symlink(src, dst, target_is_directory=True)
                except OSError:
                    shutil.copytree(src, dst)
        mute = os.path.join(self.rvc, "logs", "mute")
        if os.path.isdir(mute) and not os.path.exists(os.path.join(root, "logs", "mute")):
            shutil.copytree(mute, os.path.join(root, "logs", "mute"))

    def _script(self, *parts):
        return os.path.join(self.rvc, *parts)

    def _run(self, cmd, timeout, extra_env=None, tick=None):
        """-> (ok, seconds, stopped). tick() is called every few seconds while the tool runs; returning True stops it."""
        began = time.monotonic()
        # RVC_CUDA_GRAPH=0 is what every script inherits from the WebUI (webui.py sets it unless opted in): no CUDA graph
        # capture in RMVPE, HuBERT or training, and no capture probe in each of train.py's loader workers
        env = dict(os.environ, PYTHONSAFEPATH="1", PYTHONPATH=self.rvc, RVC_CUDA_GRAPH="0")
        env.update(extra_env or {})
        stopped = False
        with open(self.log_path, "a", encoding="utf-8", errors="replace") as log:
            log.write(f"\n=== {time.strftime('%H:%M:%S')} {' '.join(map(str, cmd[:6]))}\n")
            log.flush()
            group = {"start_new_session": True} if os.name == "posix" else {}
            proc = subprocess.Popen(list(map(str, cmd)), cwd=self.root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, **group)
            while True:
                try:
                    code = proc.wait(timeout=3)
                    break
                except subprocess.TimeoutExpired:
                    late = time.monotonic() - began > timeout
                    stopped = bool(tick and tick())
                    if late or stopped:
                        self._kill(proc)
                        code = -9
                        log.write("TIMED OUT\n" if late else "STOPPED\n")
                        break
            self._kill(proc)  # train.py leaves through os._exit; its loader workers must not outlive it
        return code == 0, round(time.monotonic() - began, 1), stopped

    @staticmethod
    def _kill(proc):
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            elif proc.poll() is None:
                proc.kill()
        except (OSError, ProcessLookupError):
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass

    def count(self, folder, pattern="*"):
        return len(glob.glob(os.path.join(self.exp_dir, folder, pattern)))

    def preprocess(self, samples_dir, timeout=1800):
        """webui.run_preprocess_dataset: slices of 3.7 s (0.3 s overlap) at 48 kHz and at 16 kHz. -> (slices, seconds)."""
        ok, secs, _ = self._run([self.python, self._script("train", "preprocess.py"), samples_dir, 48000, self.cpus, self.exp_dir,
                                 "False", "3.7"], timeout)
        return (self.count("0_gt_wavs", "*.wav") if ok else 0), secs

    def extract_f0(self, cpu=False, timeout=1800):
        """webui.run_extract_f0_feature, RMVPE: on the GPU ("cuda", one part, card 0, half precision as the WebUI picks for
        every card this worker runs on) or in CPU processes. -> (pitch files, seconds)."""
        script = self._script("train", "dataset", "extract_f0.py")
        if cpu or self.device != "cuda":
            cmd = [self.python, script, "cpu", self.exp_dir, self.cpus, "rmvpe"]
        else:
            cmd = [self.python, script, "cuda", 1, 0, 0, self.exp_dir, "True"]
        ok, secs, _ = self._run(cmd, timeout)
        return (min(self.count("2a_f0", "*.npy"), self.count("2b-f0nsf", "*.npy")) if ok else 0), secs

    def extract_features(self, half=True, timeout=1800):
        """webui.run_extract_f0_feature, HuBERT v2 (the script finds the GPU itself; half precision there, as the WebUI picks,
        unless half is False). -> (feature files, seconds)."""
        cuda = self.device == "cuda"
        ok, secs, _ = self._run([self.python, self._script("train", "dataset", "extract_hubert_feature.py"),
                                 "cuda:0" if cuda else "cpu", 1, 0, self.exp_dir, "v2", "True" if (cuda and half) else "False"],
                                timeout)
        return (self.count("3_feature768", "*.npy") if ok else 0), secs

    def write_filelist(self):
        """webui.run_train_model's file list and config for one voice with pitch at 48 kHz, v2: every slice that has all four
        of its files, speaker 0, then RVC's silence sample twice. -> the number of slices listed."""
        e = self.exp_dir.replace("\\", "/")
        root = self.root.replace("\\", "/")
        folders = ("0_gt_wavs", "3_feature768", "2a_f0", "2b-f0nsf")
        names = None
        for folder in folders:
            here = {name.split(".")[0] for name in os.listdir(os.path.join(self.exp_dir, folder))}
            names = here if names is None else names & here
        rows = [f"{e}/0_gt_wavs/{n}.wav|{e}/3_feature768/{n}.npy|{e}/2a_f0/{n}.wav.npy|{e}/2b-f0nsf/{n}.wav.npy|0"
                for n in sorted(names)]
        if not os.path.exists(os.path.join(self.root, "logs", "mute", "0_gt_wavs", "mute48k.wav")):
            raise RuntimeError("RVC's silence sample (logs/mute) is not in the image")
        mute = (f"{root}/logs/mute/0_gt_wavs/mute48k.wav|{root}/logs/mute/3_feature768/mute.npy|"
                f"{root}/logs/mute/2a_f0/mute.wav.npy|{root}/logs/mute/2b-f0nsf/mute.wav.npy|0")
        listed = rows + [mute, mute]
        random.Random(0).shuffle(listed)
        with open(os.path.join(self.exp_dir, "filelist.txt"), "w", encoding="utf8", newline="\n") as f:
            f.write("\n".join(listed))
        with open(self._script("configs", "v2", "48k.json"), encoding="utf8") as f:
            config = json.load(f)
        config.pop("speaker_info", None)
        with open(os.path.join(self.exp_dir, "config.json"), "w", encoding="utf8", newline="\n") as f:
            json.dump(config, f, ensure_ascii=False, indent=4, sort_keys=True)
            f.write("\n")
        return len(rows)

    def train(self, epochs, batch, timeout, tick=None):
        """webui.run_train_model's command: 48k, with pitch, v2, from the pretrained generator and discriminator, only the
        latest checkpoint and no weights along the way. The checkpoint interval is one past the last round, so the half
        gigabyte of resume files is never written: the job cannot resume, and train.py saves the small model itself after the
        last round (train/process_ckpt.savee -> assets/weights/<name>.pth). -> (ok, seconds, stopped by tick)."""
        cmd = [self.python, self._script("train", "train.py"), "-e", EXP, "-sr", "48k", "-f0", 1, "-bs", batch]
        if self.device == "cuda":
            cmd += ["-g", 0]
        cmd += ["-te", epochs, "-se", epochs + 1, "-pg", self._script("assets", "pretrained_v2", "f0G48k.pth"),
                "-pd", self._script("assets", "pretrained_v2", "f0D48k.pth"), "-l", 1, "-c", 0, "-sw", 0, "-v", "v2"]
        env = {}
        workers = self.loader
        if workers == "auto":
            workers = loader_plan(batch, shm_free_gb())
        self.loader_used = "upstream" if workers in (None, "upstream") else int(workers)
        if self.loader_used != "upstream":
            env.update({"VOICE_TRAIN_LOADER_WORKERS": str(self.loader_used),
                        "PYTHONPATH": os.path.join(HERE, "train_site") + os.pathsep + self.rvc})
        for stale in (os.path.join(self.exp_dir, "train.log"), self.model_path()):  # a second start counts its own rounds
            if os.path.exists(stale):
                os.remove(stale)
        self.log_from = os.path.getsize(self.log_path) if os.path.exists(self.log_path) else 0
        ok, secs, stopped = self._run(cmd, timeout, env, tick)
        return ok and os.path.exists(self.model_path()), secs, stopped

    def epochs_done(self):
        """The last finished round, from train.py's end-of-round line (0 before the first, or when nothing readable is there).
        On one GPU train.py writes that line to its own train.log; on the CPU the training runs in a child process whose log
        lines only reach the tool output (the image's build proved the line is read from there), so both are looked at: the
        tool output only from where this training started."""
        found = []
        for path, start in ((os.path.join(self.exp_dir, "train.log"), 0), (self.log_path, self.log_from)):
            try:
                with open(path, "rb") as f:
                    f.seek(start)
                    found += EPOCH_LINE.findall(f.read().decode("utf-8", "replace"))
            except OSError:
                pass
        return max(map(int, found)) if found else 0

    def index(self, timeout=1800):
        """webui.run_train_index for one voice. -> (the added_*.index path or None, seconds)."""
        ok, secs, _ = self._run([self.python, self._script("train", "train_index.py"), EXP, "v2",
                                 os.path.join(self.root, "assets", "indices"), self.cpus, "single"], timeout)
        return (self.index_path() if ok else None), secs

    def model_path(self):
        return os.path.join(self.root, "assets", "weights", f"{EXP}.pth")

    def index_path(self):
        found = [p for p in glob.glob(os.path.join(self.exp_dir, "added_*.index")) if os.path.isfile(p)]
        return max(found, key=os.path.getmtime) if found else None


def some(count, rest):
    """"One of the recordings ..." or "3 of the recordings ..."."""
    return f"{'One' if count == 1 else count} of the recordings {rest}"


def batch_stems(runner, model, items, work, tag, instr, timeout=3000):
    """One pymss call over every recording (one model load for the lot): items is a list of [n, 2] arrays or None.
    -> a list with each recording's <instr> stem ([n, 2]) or None, and the folders are cleared again."""
    src, out = os.path.join(work, f"{tag}_in"), os.path.join(work, f"{tag}_out")
    os.makedirs(src, exist_ok=True)
    for i, x in enumerate(items):
        if x is not None:
            va.encode(os.path.join(src, f"s{i:03d}.wav"), x, SR, bits=32)
    ok, _ = runner.pymss(model, src, out, timeout=timeout)
    stems = []
    for i, x in enumerate(items):
        found = vp.stem_file(out, f"s{i:03d}", instr) if (ok and x is not None) else None
        stems.append(vp.fit(va.decode(found, sr=SR, channels=2)[0], len(x)) if found else None)
    if not ok:
        print(f"training separation failed: {tag}\n{vp.log_tail(runner.log_path)}", flush=True)
    shutil.rmtree(src, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    return stems


def isolate(voices, work, runner, notes):
    """Recordings with music under the voice: the song jobs' default chain on every one of them, the vocal extractor (then its
    fallback for the ones it could not split) and the lead split, kept where the lead holds at least a quarter of the vocal
    energy. A recording with no voice to find is left out (None)."""
    vocals = [None] * len(voices)
    for name in (DEFAULTS["extractor"], DEFAULTS["fallback"]):
        todo = [x if vocals[i] is None else None for i, x in enumerate(voices)]
        if not vp.extractor_ready(name) or all(x is None for x in todo):
            continue
        for i, y in enumerate(batch_stems(runner, EXTRACTORS[name]["model"], todo, work, f"sep_{name}", "vocals")):
            if y is not None and vp.rms_db(y) > vp.rms_db(voices[i]) - 35:
                vocals[i] = y
    lost = sum(1 for x, v in zip(voices, vocals) if x is not None and v is None)
    if lost:
        notes.append(some(lost, "had no voice that could be split from its music, so "
                          + ("it was" if lost == 1 else "they were") + " left out."))
    lead_model = LEAD_MODELS[DEFAULTS["lead_model"]]
    if lead_model["lead_stem"] and any(v is not None for v in vocals):
        leads = batch_stems(runner, lead_model["model"], vocals, work, "lead", lead_model["lead_stem"])
        for i, y in enumerate(leads):
            if y is not None and 10 ** ((vp.rms_db(y) - vp.rms_db(vocals[i])) / 10) >= 0.25:
                vocals[i] = y
    return vocals


def clean(voices, work, runner, model, instr, amount, tag, sentence, notes):
    """The dereverb or denoise model over every recording, blended in by amount; one that fails keeps the recording as it was."""
    stems = batch_stems(runner, model, voices, work, tag, instr)
    failed = sum(1 for x, y in zip(voices, stems) if x is not None and y is None)
    if failed:
        notes.append(sentence)
    return [x if (x is None or y is None) else (x * np.float32(1 - amount) + y * np.float32(amount)).astype(np.float32)
            for x, y in zip(voices, stems)]


def listen(req, samples, work, runner, notes, min_seconds=MIN_TRAIN_SECONDS, progress=lambda text: None):
    """Steps 1 to 3 -> (the mono voices that are used, seconds of voice in them, the range, pitch statistics)."""
    opts = req["options"]
    voices, raw_s, unread = [], 0.0, 0
    for path in samples:
        x = None
        if raw_s < MAX_RAW_SECONDS:  # past that, the rest is simply not needed
            try:
                x = vp.load_input(path, MAX_RAW_SECONDS - raw_s)
            except RuntimeError:
                x = None
            if x is not None and (len(x) < SR or vp.rms_db(x) < -70):
                x = None
            unread += x is None
        raw_s += 0 if x is None else len(x) / SR
        voices.append(x)
    if all(x is None for x in voices):
        raise ValueError("None of the recordings could be read. Upload them again as WAV, MP3, FLAC or M4A files.")
    if unread:
        notes.append(some(unread, "could not be read or was silent, so it was left out." if unread == 1
                          else "could not be read or were silent, so they were left out."))
    if opts["isolate"]:
        progress("Taking the music off the recordings")
        voices = isolate(voices, work, runner, notes)
    if opts["dereverb"]:
        progress("Taking the room off the recordings")
        voices = clean(voices, work, runner, DEREVERB_MODEL, "noreverb", 1.0, "dry",
                       "The room could not be taken off every recording, so some were used with their echo.", notes)
    if opts["denoise"] > 0 and not vp.denoise_ready():
        notes.append("The noise cleanup failed the worker's own check when it was built, so the recordings were used as they were.")
    elif opts["denoise"] > 0:
        progress("Cleaning the noise off the recordings")
        voices = clean(voices, work, runner, DENOISE_MODEL, "dry", opts["denoise"], "clean",
                       "The noise could not be cleaned off every recording, so some were used as they were.", notes)
    used, total, f0 = [], 0.0, []
    for x in voices:
        if x is None or total >= MAX_TRAIN_SECONDS:
            continue
        m = x.mean(1).astype(np.float32)
        mask, block = va.active_blocks(m, SR)
        if total + mask.sum() * block / SR > MAX_TRAIN_SECONDS:  # the hour is full part way through this recording
            keep = int(np.searchsorted(np.cumsum(mask), (MAX_TRAIN_SECONDS - total) * SR / block)) + 1
            m, mask = m[:keep * block], mask[:keep]
        if mask.sum() * block / SR < 0.5:
            continue
        total += float(mask.sum()) * block / SR
        used.append(m)
        tr = va.pitch_track(m, SR)
        f0.append(tr["f0"][tr["voiced"]])
    if total >= MAX_TRAIN_SECONDS or raw_s >= MAX_RAW_SECONDS:
        notes.append("There was more than an hour of voice, so only the first hour was used.")
    if total < min_seconds:
        raise ValueError(TOO_LITTLE)
    f0 = np.concatenate(f0) if f0 else np.zeros(0)
    stats = va.pitch_stats({"f0": f0, "voiced": np.ones(len(f0), bool), "hop_s": va.YIN_HOP / va.YIN_SR})
    if "p50" not in stats:
        raise ValueError("The recordings have no clear voice to learn. Use recordings of one person singing or speaking.")
    return used, total, {k: stats[k] for k in ("p05", "p50", "p95")}, stats


def run(req, samples, work, runner, trainer, progress=lambda text: None, vram_gb=None, deadline=None, trace=None,
        min_seconds=MIN_TRAIN_SECONDS, batch=None):
    """samples: the recordings as local files, in the request's order. runner: voice_pipeline.Runner (the separators and the
    final check); trainer: TrainRunner. deadline: time.monotonic() by which the whole job must be done. trace: a dict that
    is filled in as each step finishes (the build self-test reads it after a failure).
    -> {"files": {"model.pth", "model.index", "train_report.json"}, "report": {...}}. ValueError carries a sentence."""
    notes, timing = [], {}
    trace = {} if trace is None else trace
    opts = req["options"]
    progress("Listening to the recordings")
    began = time.monotonic()
    voices, seconds, voice_range, stats = listen(req, samples, work, runner, notes, min_seconds, progress)
    minutes = seconds / 60
    timing["listen_s"] = round(time.monotonic() - began, 1)
    trace["listen"] = round(seconds, 1)
    progress("Getting the recordings ready")
    dataset = os.path.join(work, "dataset")
    os.makedirs(dataset, exist_ok=True)
    for i, x in enumerate(voices):
        va.encode(os.path.join(dataset, f"{i:03d}.wav"), x, SR, bits=32)
    slices, timing["slice_s"] = trainer.preprocess(dataset)
    trace["preprocess"] = slices
    if not slices:
        print(f"training preprocess made no slices\n{vp.log_tail(trainer.log_path)}", flush=True)
        raise ValueError("The recordings were too quiet or too broken up to learn from. Use clearer recordings.")
    pitched, timing["pitch_s"] = trainer.extract_f0()
    # A slice with no pitch at all (a breath, an S) is skipped by RVC and is no failure. Nothing at all means the GPU form
    # itself failed: the CPU form, which the image's build proved, does the work (and skips whatever is already there).
    if not pitched and trainer.device == "cuda":
        print(f"training f0 on the GPU made none of {slices}\n{vp.log_tail(trainer.log_path)}", flush=True)
        pitched, again = trainer.extract_f0(cpu=True)
        timing["pitch_s"] = round(timing["pitch_s"] + again, 1)
        notes.append("The fast pitch reader did not finish on this card, so the slower one was used.")
    trace["f0"] = pitched
    featured, timing["features_s"] = trainer.extract_features()
    if not featured and trainer.device == "cuda":  # the same safety net: once more in full precision
        print(f"training features in half precision made none\n{vp.log_tail(trainer.log_path)}", flush=True)
        featured, again = trainer.extract_features(half=False)
        timing["features_s"] = round(timing["features_s"] + again, 1)
        notes.append("The fast voice reader did not finish on this card, so the slower one was used.")
    trace["features"] = featured
    rows = trainer.write_filelist() if (pitched and featured) else 0
    trace["filelist"] = rows
    if not rows:
        print(f"training features: {pitched} pitch, {featured} hubert\n{vp.log_tail(trainer.log_path)}", flush=True)
        raise RuntimeError("no training rows")
    batch = batch or batch_size(vram_gb)
    asked = opts["epochs"]
    epochs = auto_epochs(minutes) if asked == "auto" else int(asked)
    if deadline is not None:
        fits = int((deadline - time.monotonic() - AFTER_TRAIN_S) / epoch_seconds(minutes))
        if fits < EPOCHS_MIN:
            raise ValueError("There is not enough time left in this job to learn the voice. Start it again with less audio.")
        if epochs > fits:
            if asked != "auto":
                notes.append(f"{epochs} training rounds would not finish in the time a job may run with this much audio, "
                             f"so the voice was trained for {fits}.")
            epochs = fits
    progress("Learning the voice")
    watch = {}

    def tick():
        done, now = trainer.epochs_done(), time.monotonic()
        if done > watch["epoch"]:
            if watch["t0"] is None:
                watch["t0"], watch["e0"] = now, done
            watch["epoch"] = done
        if now - watch["said"] >= PROGRESS_EVERY_S:
            watch["said"] = now
            progress(f"Learning the voice, epoch {min(done + 1, epochs)} of {epochs}" if done else "Learning the voice")
        if deadline is not None and watch["t0"] is not None and done - watch["e0"] >= 3:
            watch["pace"] = (now - watch["t0"]) / (done - watch["e0"])  # the real pace, once three rounds have been timed
            return now + watch["pace"] * (epochs - done) > deadline - AFTER_TRAIN_S
        return False
    # The rounds were planned on an estimate of this card's pace. When the first timed rounds show the plan cannot finish
    # inside the job, training is stopped there and started once more with the rounds that do fit at the measured pace (two
    # rounds' worth is set aside for starting again), instead of running out the clock with nothing to show for it.
    timing["train_s"] = 0.0
    for attempt in (1, 2):
        watch.update({"said": time.monotonic(), "epoch": 0, "t0": None, "e0": 0, "pace": None})
        limit = 7200 if deadline is None else max(60.0, deadline - time.monotonic() - AFTER_TRAIN_S / 2)
        ok, secs, stopped = trainer.train(epochs, batch, limit, tick)
        timing["train_s"] = round(timing["train_s"] + secs, 1)
        if not stopped:
            break
        fits = int((deadline - time.monotonic() - AFTER_TRAIN_S) / watch["pace"]) - 2
        if attempt == 2 or fits < EPOCHS_MIN:
            trace["train"] = False
            raise ValueError("Learning this voice was going to take longer than a job may run. Use fewer training rounds or "
                             "less audio.")
        notes.append(f"This card learned more slowly than planned, so the voice was trained for {fits} rounds instead of {epochs}.")
        epochs = fits
        progress("Learning the voice")
    trace["train"] = bool(ok)
    if not ok:
        print(f"training failed\n{vp.log_tail(trainer.log_path, 4000)}", flush=True)
        raise RuntimeError("RVC training failed")
    progress("Building the voice's index")
    index_path, timing["index_s"] = trainer.index()
    trace["index"] = bool(index_path)
    if not index_path:
        print(f"training index failed\n{vp.log_tail(trainer.log_path, 4000)}", flush=True)
        raise RuntimeError("RVC index failed")
    files = {name: os.path.join(work, name) for name in ("model.pth", "model.index", "train_report.json")}
    shutil.copyfile(trainer.model_path(), files["model.pth"])
    shutil.copyfile(index_path, files["model.index"])
    check = sing_check(voices, files, work, runner, notes)
    timing["check_s"] = check.pop("seconds")
    trace["check"] = check["ok"]
    report = {
        "engine": "rvc", "mode": "train", "version": "v2", "sample_rate": "48k", "f0": True, "voice_id": req.get("voice_id"),
        "minutes": round(minutes, 2), "seconds": round(seconds, 1), "epochs": epochs, "epochs_asked": asked, "batch_size": batch,
        "sample_count": len(voices), "samples_sent": len(samples), "slices": rows, "range": voice_range,
        "range_notes": stats.get("notes"), "voiced_s": stats.get("voiced_s"),
        "model_sha256": voice_models.sha256_file(files["model.pth"]), "model_bytes": os.path.getsize(files["model.pth"]),
        "index_sha256": voice_models.sha256_file(files["model.index"]), "index_bytes": os.path.getsize(files["model.index"]),
        "options": opts, "loader": trainer.loader_used, "shm_free_gb": shm_free_gb(), "check": check,
        "notes": list(dict.fromkeys(notes)), "timing": timing,
    }
    with open(files["train_report.json"], "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1)
    return {"files": files, "report": report}


def sing_check(voices, files, work, runner, notes):
    """The new voice re-sings four seconds of its own recordings through the song jobs' own RVC call, index and all. It proves
    the files load and sing before anybody pays for a song with them. A failure does not throw the voice away: it becomes a
    worker note, because the model may still be good and the check itself may be what failed."""
    began = time.monotonic()
    src, dst = os.path.join(work, "check_in.wav"), os.path.join(work, "check_out.wav")
    clip = max(voices, key=len)[: 4 * SR]
    va.encode(src, clip, SR, bits=32)
    opts = {**DEFAULTS, "index_rate": 0.5}
    try:
        ok, _ = runner.rvc_convert(files["model.pth"], files["model.index"], src, dst, 0, opts, timeout=600)
        ok = ok and os.path.exists(dst) and abs(va.probe(dst)[2] - len(clip) / SR) < 0.5
    except (OSError, RuntimeError):
        ok = False
    if not ok:
        print(f"training check failed\n{vp.log_tail(runner.log_path)}", flush=True)
        notes.append("The new voice could not sing a test line on this worker. Try it on a song, and train it again if it does not work.")
    return {"ok": bool(ok), "seconds": round(time.monotonic() - began, 1)}
