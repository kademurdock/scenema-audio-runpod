"""Isolated SheetSage2 environment: source recording to an ABC score.

Usage: cover.py <24 kHz wav> <output dir> [melody|full]
melody (the default, and the only call before Part 295) keeps the tune without
chord symbols; full is SheetSage2's full transcription, melody plus chords."""
import json
import os
import sys
from pathlib import Path
from transformers import AutoModel

task = sys.argv[3] if len(sys.argv) > 3 else 'melody'
if task not in ('melody', 'full'):
    raise SystemExit('task must be melody or full')
model = AutoModel.from_pretrained(os.environ['YUE_SCORE_DIR'],
    base_model_path=os.environ['YUE_MERT_DIR'], trust_remote_code=True,
    local_files_only=True).eval().to('cuda')
result = model.transcribe(sys.argv[1], output_dir=sys.argv[2], melody_only=task == 'melody')
if result.get('abc_error') or not result.get('abc'):
    raise RuntimeError('The recording did not produce a usable melody score.')
Path(sys.argv[2], 'warnings.json').write_text(json.dumps(result.get('warnings', [])))
try:
    import torch
    Path(sys.argv[2], 'memory.json').write_text(json.dumps(
        {'peak_allocated_gib': round(torch.cuda.max_memory_allocated() / 2**30, 2)}))
except Exception:
    pass
