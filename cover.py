"""Isolated SheetSage2 environment: source recording to melody ABC."""
import json
import os
import sys
from pathlib import Path
from transformers import AutoModel

model = AutoModel.from_pretrained(os.environ['YUE_SCORE_DIR'],
    base_model_path=os.environ['YUE_MERT_DIR'], trust_remote_code=True,
    local_files_only=True).eval().to('cuda')
result = model.transcribe(sys.argv[1], output_dir=sys.argv[2], melody_only=True)
if result.get('abc_error') or not result.get('abc'):
    raise RuntimeError('The recording did not produce a usable melody score.')
Path(sys.argv[2], 'warnings.json').write_text(json.dumps(result.get('warnings', [])))
