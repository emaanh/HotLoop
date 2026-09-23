# Image for the local backend (GPU VM with Docker + NVIDIA Container Toolkit).
# Mirrors the Modal gpu_image in hotloop/backends/modal_app.py.
#   docker build -f docker/Dockerfile -t hotloop:latest .
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates cuda-nsight-compute-12-8 cuda-nsight-systems-12-8 \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
    && uv python install 3.12 && uv venv /opt/venv --python 3.12
ENV PATH=/opt/venv/bin:$PATH VIRTUAL_ENV=/opt/venv
RUN uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 \
    && uv pip install numpy ninja transformers huggingface_hub accelerate
RUN useradd -m -u 1500 solver \
    && ln -sf $(ls -d /opt/nvidia/nsight-compute/*/ | head -1)ncu /usr/local/bin/ncu \
    && ln -sf $(ls -d /opt/nvidia/nsight-systems/*/ | head -1)bin/nsys /usr/local/bin/nsys \
    && chmod 755 /opt/venv -R
COPY hotloop /opt/hotloop_src/hotloop
ENV PYTHONPATH=/opt/hotloop_src
WORKDIR /workspace
