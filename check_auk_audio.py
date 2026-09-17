"""CPU-only deployment gate for AuK's lazily imported reference-audio path."""
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from auk.infer.infer_auk import AukInfer
from qwen_omni_utils import process_mm_info


with tempfile.TemporaryDirectory() as directory:
    audio = Path(directory) / "reference.wav"
    sf.write(audio, np.zeros(24000, dtype=np.float32), 24000)
    inference = object.__new__(AukInfer)
    inference.target_sample_rate = 24000
    waveform, rms = inference._load_audio(str(audio))
    assert waveform.shape == (1, 24000)
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Read this synthetic check."},
        {"type": "audio", "audio": str(audio)},
    ]}]
    audios, _, _ = process_mm_info(messages, use_audio_in_video=True)
    assert audios is not None and len(audios) == 1
    assert len(audios[0]) > 0
print("AuK reference audio decoded and preprocessed on CPU; no inference charged.")
