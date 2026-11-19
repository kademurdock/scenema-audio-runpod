"""Build step (RVC venv): download every pretrained file the voice worker needs, at pinned revisions, and record what landed.

  HF lj1995/VoiceConversionWebUI @ HF_RVC_REV   hubert_base/{config.json, preprocessor_config.json, pytorch_model.bin} and rmvpe.pt
                                                 into /opt/rvc/assets (RVC's own layout: assets/hubert_base, assets/rmvpe);
                                                 for training (voice_train.py) the v2 48 kHz pretrained generator and
                                                 discriminator with pitch, pretrained_v2/f0G48k.pth and f0D48k.pth, into
                                                 assets/pretrained_v2, and mute.zip, RVC's silence sample that every training
                                                 file list ends with, unpacked into /opt/rvc/logs/mute (the RVC repo itself no
                                                 longer carries it). These three are checked against the sha256 recorded here.
  HF baicai1145/pymss @ PYMSS_ENDPOINT's commit  the baked separators (voice_request.EXTRACTORS and LEAD_MODELS marked baked,
                                                 the dereverb model and the denoise model) through pymss's own downloader; each
                                                 file's size is checked against pymss's catalog
Writes /opt/voice/models.json (path, bytes, sha256 of every file) so the image says exactly which weights it carries.
Voice models are NOT here: they are personal, live in the private bucket and are fetched per job (voice_models.py)."""
import hashlib
import json
import os
import subprocess
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from voice_request import DENOISE_MODEL, DEREVERB_MODEL, EXTRACTORS, LEAD_MODELS  # noqa: E402

RVC = os.environ.get("VOICE_RVC_DIR", "/opt/rvc")
MODELS = os.environ.get("PYMSS_MODEL_DIR", "/opt/pymss_models")
ENDPOINT = os.environ["PYMSS_ENDPOINT"]
HF_RVC, HF_RVC_REV = "lj1995/VoiceConversionWebUI", "e6d0c1a17da07c33557852f9dfa2bd44cc75737d"
# The training files, with the sha256 the pinned revision lists for each (its LFS object ids, read Nov 18 2026).
TRAIN_FILES = {
    "pretrained_v2/f0G48k.pth": "b5d51f589cc3632d4eae36a315b4179397695042edc01d15312e1bddc2b764a4",
    "pretrained_v2/f0D48k.pth": "2269b73c7a4cf34da09aea99274dabf99b2ddb8a42cbfb065fb3c0aa9a2fc748",
}
MUTE_ZIP_SHA = "ee948e85213e4ed2f2ba2f8dfcee810bfd0b63131d91450e920bbe1cbd0321d0"
MUTE_FILES = ("0_gt_wavs/mute48k.wav", "3_feature768/mute.npy", "2a_f0/mute.wav.npy", "2b-f0nsf/mute.wav.npy")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked(path, want):
    got = sha256(path)
    if got != want:
        raise SystemExit(f"{path}: sha256 {got}, expected {want}")
    return path


def main():
    from huggingface_hub import hf_hub_download
    record = {"hf_rvc": f"{HF_RVC}@{HF_RVC_REV}", "pymss_endpoint": ENDPOINT, "files": []}
    assets = os.path.join(RVC, "assets")
    got = [hf_hub_download(repo_id=HF_RVC, filename=f, revision=HF_RVC_REV, local_dir=assets)
           for f in ("hubert_base/config.json", "hubert_base/preprocessor_config.json", "hubert_base/pytorch_model.bin")]
    got.append(hf_hub_download(repo_id=HF_RVC, filename="rmvpe.pt", revision=HF_RVC_REV, local_dir=os.path.join(assets, "rmvpe")))
    for name, want in TRAIN_FILES.items():
        got.append(checked(hf_hub_download(repo_id=HF_RVC, filename=name, revision=HF_RVC_REV, local_dir=assets), want))
    mute_zip = checked(hf_hub_download(repo_id=HF_RVC, filename="mute.zip", revision=HF_RVC_REV, local_dir="/tmp/rvc-mute"),
                       MUTE_ZIP_SHA)
    logs = os.path.join(RVC, "logs")
    os.makedirs(logs, exist_ok=True)
    with zipfile.ZipFile(mute_zip) as z:
        for member in z.namelist():  # every entry stays under logs/mute/
            if not member.startswith("mute/") or ".." in member or member.startswith("/"):
                raise SystemExit(f"mute.zip holds an unexpected path: {member}")
        z.extractall(logs)
    os.remove(mute_zip)
    got += [os.path.join(logs, "mute", f) for f in MUTE_FILES]
    catalog = json.load(open(os.path.join(RVC, "tools", "pymss", "resources", "model_catalog.json"), encoding="utf-8"))
    catalog = {m["name"]: m for m in (catalog["models"] if isinstance(catalog, dict) else catalog)}
    wanted = [e["model"] for e in EXTRACTORS.values() if e["baked"]] + [m["model"] for m in LEAD_MODELS.values() if m["baked"]]
    wanted += [DEREVERB_MODEL, DENOISE_MODEL]
    for name in wanted:
        entry = catalog[name]
        subprocess.run([sys.executable, "-m", "tools.pymss.cli", "download", name, "--model-dir", MODELS, "--endpoint", ENDPOINT],
                       cwd=RVC, check=True, stdin=subprocess.DEVNULL)
        ckpt = os.path.join(MODELS, entry["relpath"])
        size = os.path.getsize(ckpt)
        if entry.get("size_bytes") and size != int(entry["size_bytes"]):
            raise SystemExit(f"{name}: {size} bytes, the catalog says {entry['size_bytes']}")
        got.append(ckpt)
        if entry.get("config_relpath"):
            got.append(os.path.join(MODELS, entry["config_relpath"]))
    for path in got:
        record["files"].append({"path": path, "bytes": os.path.getsize(path), "sha256": sha256(path)})
        print(f"  {os.path.getsize(path) / 1e6:9.1f} MB  {path}", flush=True)
    os.makedirs("/opt/voice", exist_ok=True)
    with open("/opt/voice/models.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, indent=1)
    print(f"BASE MODELS OK: {len(got)} files", flush=True)


if __name__ == "__main__":
    main()
