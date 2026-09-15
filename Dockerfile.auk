FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip git ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
WORKDIR /app
ENV PYTHONPATH=/app/upstream/src
RUN git clone https://github.com/Tencent-Hunyuan/AuK /app/upstream && cd /app/upstream && git checkout e1c935e81e356c87419d9509d7f8a4091457bdca && pip install --no-cache-dir . "transformers==4.57.6" "huggingface-hub==0.36.2" "runpod>=1.7,<2" "boto3>=1.35,<2" "soundfile>=0.12,<1" "requests>=2.32,<3"
RUN python3 -c "from auk.infer.infer_auk import AukInfer"
# Download during the free public image build, never on a billable GPU.
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('tencent/AuK', revision='790742b71a4430120daf2b2099192abae449eb9f', local_dir='/models/AuK', allow_patterns=['*.safetensors','*.yaml','LICENSE']); snapshot_download('Qwen/Qwen2.5-Omni-3B', revision='f75b40e3da2003cdd6e1829b1f420ca70797c34e', local_dir='/models/Qwen', allow_patterns=['*.safetensors','*.json','*.txt','LICENSE'])"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COPY auk_contract.py auk_handler.py /app/
RUN python3 -c "from auk.infer.infer_auk import AukInfer; import auk_handler"
CMD ["python3", "-u", "/app/auk_handler.py"]
