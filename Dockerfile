FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies for building flash-attention and PDF processing
RUN apt-get update && apt-get install -y \
    git \
    ninja-build \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages in correct order (torch first, then flash-attn)
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir torch==2.4.0 && \
    pip install --no-cache-dir transformers accelerate pillow huggingface_hub protobuf "numpy<2.0" runpod pdf2image && \
    pip install --no-cache-dir flash-attn --no-build-isolation

# Create models directory and download model during build
RUN mkdir -p /models/hf && \
    python -c "import os; from huggingface_hub import snapshot_download; \
    os.environ['HF_HOME'] = '/models/hf'; \
    os.environ['TRANSFORMERS_CACHE'] = '/models/hf'; \
    os.environ['HF_HUB_CACHE'] = '/models/hf'; \
    snapshot_download(repo_id='reducto/RolmOCR', local_dir='/models/hf/reducto/RolmOCR', local_dir_use_symlinks=False, resume_download=True); \
    print('Model download complete!')"

# Copy handler
COPY handler.py .

# Set environment variables for optimal H200 performance
ENV CUDA_LAUNCH_BLOCKING=0
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8

# Set HuggingFace offline mode for runtime
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HOME=/models/hf
ENV TRANSFORMERS_CACHE=/models/hf
ENV HF_HUB_CACHE=/models/hf

# Expose RunPod handler
CMD ["python", "-u", "handler.py"]
