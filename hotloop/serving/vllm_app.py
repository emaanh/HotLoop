"""Self-hosted open-weight agents: one vLLM server per model variant, on Modal.

    modal deploy -m hotloop.serving.vllm_app      # or: hotloop serve-deploy
    hotloop serve-prefetch                        # download weights into the volume once

Each variant is its own web endpoint (OpenAI-compatible, API-key protected) and is
capped at one container, so every agent episode pointed at it lands on the same
server and vLLM batches them together.
"""

import subprocess

import modal

APP_NAME = "hotloop-serve"
VLLM_VERSION = "0.30.0"
PORT = 8000
SERVED_NAME = "qwen3-coder-next"

# Quantization study: same model, same GPU, same server; only the weights differ.
VARIANTS = {
    "bf16": "Qwen/Qwen3-Coder-Next",
    "fp8": "Qwen/Qwen3-Coder-Next-FP8",
    "nvfp4": "RedHatAI/Qwen3-Coder-Next-NVFP4",
}
GPU = "B200"  # 192 GB: fits BF16 on one card, and runs FP8/NVFP4 natively

# The official vLLM image: ships the Blackwell kernels (FlashInfer, DeepGEMM, CUTLASS)
# prebuilt, so startup doesn't JIT-compile them (which is slow and failed on B200).
image = (
    modal.Image.from_registry(f"vllm/vllm-openai:v{VLLM_VERSION}")
    .entrypoint([])
    .run_commands("ln -sf $(command -v python3) /usr/local/bin/python")  # the image only ships `python3`
    .pip_install("huggingface_hub[hf_transfer]")
    # Anything that still compiles at startup (torch.compile, leftover JIT) is cached on
    # the volume, so only the first start of each variant pays for it.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/models",
          "VLLM_CACHE_ROOT": "/models/vllm-cache", "FLASHINFER_WORKSPACE_BASE": "/models/flashinfer-cache",
          "MAX_JOBS": "6"})
)

app = modal.App(APP_NAME)
models_vol = modal.Volume.from_name("hotloop-models", create_if_missing=True)
secrets = [modal.Secret.from_name("hotloop-keys"), modal.Secret.from_name("hotloop-vllm")]


@app.function(image=image, secrets=secrets, volumes={"/models": models_vol}, timeout=4 * 3600)
def prefetch(variant: str) -> str:
    from huggingface_hub import snapshot_download

    path = snapshot_download(VARIANTS[variant])
    models_vol.commit()
    return path


def _serve(variant: str):
    cmd = [
        "vllm", "serve", VARIANTS[variant],
        "--served-model-name", SERVED_NAME,
        "--host", "0.0.0.0", "--port", str(PORT),
        "--max-model-len", "262144",          # the model's native context
        "--gpu-memory-utilization", "0.92",
        "--max-num-seqs", "32",
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
        "--enable-prefix-caching",             # agents resend their whole history every turn
        "--api-key", "$VLLM_API_KEY",
    ]
    subprocess.Popen(" ".join(cmd), shell=True)


SERVER = dict(image=image, gpu=GPU, cpu=16, memory=65536, secrets=secrets, volumes={"/models": models_vol},
              timeout=6 * 3600, scaledown_window=15 * 60, max_containers=1)


@app.function(**SERVER)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=45 * 60)
def serve_bf16():
    _serve("bf16")


@app.function(**SERVER)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=45 * 60)
def serve_fp8():
    _serve("fp8")


@app.function(**SERVER)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=45 * 60)
def serve_nvfp4():
    _serve("nvfp4")
