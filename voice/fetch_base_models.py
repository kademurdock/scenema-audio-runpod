"""Build step (RVC venv): download every pretrained file the voice worker needs, at pinned revisions, and record what landed.

  HF lj1995/VoiceConversionWebUI @ HF_RVC_REV   hubert_base/{config.json, preprocessor_config.json, pytorch_model.bin} and rmvpe.pt
                                                 into /opt/rvc/assets (RVC's own layout: assets/hubert_base, assets/rmvpe)
  HF baicai1145/pymss @ PYMSS_ENDPOINT's commit  the baked separators (voice_request.EXTRACTORS marked baked, the karaoke lead
                                                 model and the dereverb model) through pymss's own downloader; each file's size is
                                                 checked against pymss's catalog
Writes /opt/voice/models.json (path, bytes, sha256 of every file) so the image says exactly which weights it carries.
Voice models are NOT here: they are personal, live in the private bucket and are fetched per job (voice_models.py)."""
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from voice_request import DEREVERB_MODEL, EXTRACTORS, LEAD_MODEL  # noqa: E402

RVC = os.environ.get("VOICE_RVC_DIR", "/opt/rvc")
MODELS = os.environ.get("PYMSS_MODEL_DIR", "/opt/pymss_models")
ENDPOINT = os.environ["PYMSS_ENDPOINT"]
HF_RVC, HF_RVC_REV = "lj1995/VoiceConversionWebUI", "e6d0c1a17da07c33557852f9dfa2bd44cc75737d"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    from huggingface_hub import hf_hub_download
    record = {"hf_rvc": f"{HF_RVC}@{HF_RVC_REV}", "pymss_endpoint": ENDPOINT, "files": []}
    assets = os.path.join(RVC, "assets")
    got = [hf_hub_download(repo_id=HF_RVC, filename=f, revision=HF_RVC_REV, local_dir=assets)
           for f in ("hubert_base/config.json", "hubert_base/preprocessor_config.json", "hubert_base/pytorch_model.bin")]
    got.append(hf_hub_download(repo_id=HF_RVC, filename="rmvpe.pt", revision=HF_RVC_REV, local_dir=os.path.join(assets, "rmvpe")))
    catalog = json.load(open(os.path.join(RVC, "tools", "pymss", "resources", "model_catalog.json"), encoding="utf-8"))
    catalog = {m["name"]: m for m in (catalog["models"] if isinstance(catalog, dict) else catalog)}
    wanted = [e["model"] for e in EXTRACTORS.values() if e["baked"]] + [LEAD_MODEL, DEREVERB_MODEL]
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
