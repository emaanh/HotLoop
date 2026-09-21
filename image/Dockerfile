# HotLoop benchmark image: what the agent's sandbox and the trusted evaluator both run in.
# Pinned to what the task set was certified under (driver 570 -> CUDA 12.8 -> torch cu128).
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:/root/.local/bin:$PATH VIRTUAL_ENV=/opt/venv
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git build-essential cmake ninja-build vim-tiny less procps \
    && (apt-get install -y --no-install-recommends cuda-nsight-systems-12-8 || true) \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && uv venv --python 3.12 /opt/venv \
    && uv pip install --python /opt/venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128 \
    && uv pip install --python /opt/venv/bin/python numpy "pydantic>=2.7" pybind11 ninja pytest

# Only the contracts and the evaluator go in: never taskgen, never private/ (D-16).
COPY packages/schemas /opt/hotloop/packages/schemas
COPY packages/evaluator /opt/hotloop/packages/evaluator
RUN uv pip install --python /opt/venv/bin/python --no-deps \
        /opt/hotloop/packages/schemas /opt/hotloop/packages/evaluator \
    && python -c "import torch, triton, hotloop_evaluator; print(torch.__version__, triton.__version__)"
WORKDIR /workspace
