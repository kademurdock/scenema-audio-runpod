"""Seed-VC for the voice worker (engine "seedvc"): one recording re-sung in the voice of a reference recording, no training.

Runs in its own venv (/opt/seedvc-venv: Python 3.10, torch 2.4.0, Seed-VC's own pins) from the pinned Seed-VC folder
(Plachtaa/seed-vc @ 51383ef, GPL-3.0, upstream archived), which is the current directory and on PYTHONPATH. Seed-VC is run as a
separate program, never imported by the worker. The weights were fetched when the image was built (seedvc_fetch.py) and the worker
runs this with HF_HUB_OFFLINE=1, so a job never downloads anything.

It calls Seed-VC's own inference.main with the settings of the Nov 19 2026 talk test: the 44 kHz singing model with pitch
conditioning (--f0-condition True), no automatic pitch match (the worker's octave rule chose --shift already), length 1.0, fp16 on
a GPU. Seed-VC names its output itself, so it writes into a temp folder and the one file it makes is moved to --output.

  python seedvc_run.py --ref ref.wav --source in.wav --output out.wav --shift 0 --steps 50 --cfg 0.7 --device cuda"""
import argparse
import glob
import os
import shutil
import sys
import tempfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--shift", type=int, default=0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=0.7)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    sys.path.insert(0, os.getcwd())
    import torch
    import inference as sv  # Seed-VC's inference.py (it sets its own HF cache under ./checkpoints when imported)
    if a.device == "cpu" or not torch.cuda.is_available():
        sv.device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as td:
        args = argparse.Namespace(source=a.source, target=a.ref, output=td, diffusion_steps=a.steps, length_adjust=1.0,
                                  inference_cfg_rate=a.cfg, f0_condition=True, auto_f0_adjust=False, semi_tone_shift=a.shift,
                                  checkpoint=None, config=None, fp16=sv.device.type == "cuda")
        sv.main(args)
        made = glob.glob(os.path.join(td, "*.wav"))
        if len(made) != 1:
            sys.exit(f"Seed-VC wrote {len(made)} files instead of one")
        shutil.move(made[0], a.output)
    print("seedvc ok", a.output, flush=True)


if __name__ == "__main__":
    main()
