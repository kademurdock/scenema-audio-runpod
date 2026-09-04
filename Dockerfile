# Scenema Audio on RunPod Serverless — Kade-AI (Part 119.9, Sep 3 2026)
# Upstream: github.com/ScenemaAI/scenema-audio @ 3e3d403a (MIT code; weights LTX-2 Community License).
# This image carries NO model weights. They download once onto the RunPod network volume
# (MODEL_DIR=/runpod-volume/scenema) — Gemma 3 12B from unsloth's ungated mirror, so no HF token.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# ffmpeg from apt (Ubuntu 24.04 ships 6.1): upstream's BtbN static-build URL 404s as of Sep 3 2026.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip git curl wget xz-utils gcc libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir "torch==2.7.1" "torchaudio==2.7.1" --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir \
    "numpy==2.2.6" "transformers==4.57.6" "accelerate==1.13.0" "safetensors==0.7.0" "sentencepiece==0.2.1" \
    "ltx-core @ git+https://github.com/Lightricks/LTX-2.git@41d924371612b692c0fd1e4d9d94c3dfb3c02cb3#subdirectory=packages/ltx-core" \
    "ltx-pipelines @ git+https://github.com/Lightricks/LTX-2.git@41d924371612b692c0fd1e4d9d94c3dfb3c02cb3#subdirectory=packages/ltx-pipelines"

ARG SAGE_WHEEL_URL=https://huggingface.co/ScenemaAI/scenema-audio/resolve/main/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl
RUN pip install --no-cache-dir "${SAGE_WHEEL_URL}" 2>/dev/null || true

# Upstream repo, pinned
ARG UPSTREAM_SHA=3e3d403ac7e1
RUN git clone https://github.com/ScenemaAI/scenema-audio.git /app/upstream \
    && cd /app/upstream && git checkout ${UPSTREAM_SHA}

RUN git clone --depth 1 https://github.com/Plachtaa/seed-vc.git /app/seed-vc \
    && cd /app/seed-vc && pip install --no-cache-dir \
    "scipy==1.13.1" "librosa==0.10.2" "huggingface-hub==0.36.2" "munch==4.0.0" "einops==0.8.0" \
    "descript-audio-codec==1.0.0" "pydub==0.25.1" "soundfile==0.12.1" \
    "hydra-core==1.3.2" "pyyaml==6.0.3" "python-dotenv==1.2.2" "diffusers==0.37.1" \
    "onnxruntime==1.25.0" "funasr==1.3.1"

RUN git clone --depth 1 https://github.com/kijai/ComfyUI-MelBandRoFormer /app/melband_roformer_node
RUN pip install --no-cache-dir "rotary-embedding-torch==0.8.9" "beartype==0.22.9"

RUN pip install --no-cache-dir "fastapi==0.136.1" "uvicorn[standard]==0.46.0" "httpx==0.28.1" \
    "psutil==7.2.2" "bitsandbytes==0.49.2" "faster-whisper==1.2.1" "ctranslate2==4.7.1" \
    "runpod>=1.7,<2" "boto3>=1.35,<2" "requests>=2.32,<3"

RUN pip install --no-cache-dir "kokoro==0.9.4" \
    && python3 -c "from kokoro import KPipeline; KPipeline(lang_code='a')"

# Small models baked (<1 GB), same as upstream
RUN mkdir -p /app/models && python3 -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download('ScenemaAI/scenema-audio', 'scenema-audio-vae-encoder.safetensors', local_dir='/app/models')"
RUN wget -q -O /app/models/MelBandRoformer_fp16.safetensors \
    https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/main/MelBandRoformer_fp16.safetensors

COPY handler.py /app/handler.py
COPY start.sh /app/start.sh
COPY patch_gemma_int8.py /app/patch_gemma_int8.py
RUN chmod +x /app/start.sh && python3 /app/patch_gemma_int8.py /app/upstream/src/audio_core/engine.py

RUN PYTHONPATH=/app/upstream/src python3 -c "\
import torch; print('torch', torch.__version__, torch.version.cuda); \
from common.handlers.base import ProcessJob; from audio_core.compiler import compile_prompt; print('upstream imports OK')"

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/upstream/src
ENV MODEL_DIR=/runpod-volume/scenema
ENV HF_HUB_CACHE=/runpod-volume/scenema/hf_cache
# Full precision by default (her word: as HQ as it can be). 48 GB cards.
ENV AUDIO_CKPT=/runpod-volume/scenema/scenema-audio-transformer.safetensors
ENV VAE_ENCODER_CKPT=/app/models/scenema-audio-vae-encoder.safetensors
ENV PIPELINE_CKPT=/runpod-volume/scenema/scenema-audio-pipeline.safetensors
ENV GEMMA_ROOT=/runpod-volume/scenema/gemma-3-12b-it
ENV GEMMA_REPO=unsloth/gemma-3-12b-it
ENV GEMMA_QUANTIZE=
ENV MELBAND_MODEL_PATH=/app/models/MelBandRoformer_fp16.safetensors
ENV MELBAND_NODE_PATH=/app/melband_roformer_node
ENV SEEDVC_PATH=/app/seed-vc
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD ["/app/start.sh"]
