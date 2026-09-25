"""Self-hosted open-weight agents: one vLLM server per model variant, on Modal.

    hotloop serve-deploy                          # deploy (after code changes)
    hotloop serve-prefetch --variants fp8         # download weights into the volume once
    hotloop serve-url fp8                         # the variant's OpenAI-compatible base URL

Each variant is its own web endpoint (OpenAI-compatible, API-key protected) and is
capped at one container, so every agent episode pointed at it lands on the same
server and vLLM batches them together.
"""

import subprocess

import modal

APP_NAME = "hotloop-serve"
VLLM_VERSION = "0.30.0"
PORT = 8000

_QWEN_ARGS = ["--max-model-len", "262144", "--gpu-memory-utilization", "0.92", "--max-num-seqs", "32",
              "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder"]
# GLM-5.2 flags follow NVIDIA's model card for the NVFP4 checkpoint (TP4 instead of 8 to
# halve the cost: 465 GB of weights is ~116 GB per B200, leaving room for an fp8 KV cache).
_GLM52_ARGS = ["--tensor-parallel-size", "4", "--enable-expert-parallel", "--trust-remote-code",
               "--reasoning-parser", "glm45", "--tool-call-parser", "glm47", "--enable-auto-tool-choice",
               "--kv-cache-dtype", "fp8_e4m3", "--max-model-len", "262144", "--gpu-memory-utilization", "0.90",
               "--max-num-seqs", "32"]

VARIANTS = {
    # Quantization study: same model, same single B200, same server; only the weights differ.
    "bf16": {"repo": "Qwen/Qwen3-Coder-Next", "served": "qwen3-coder-next", "gpu": "B200", "args": _QWEN_ARGS},
    "fp8": {"repo": "Qwen/Qwen3-Coder-Next-FP8", "served": "qwen3-coder-next", "gpu": "B200", "args": _QWEN_ARGS},
    "nvfp4": {"repo": "RedHatAI/Qwen3-Coder-Next-NVFP4", "served": "qwen3-coder-next", "gpu": "B200",
              "args": _QWEN_ARGS},
    # Showcase: a frontier-class open model.
    "glm52_nvfp4": {"repo": "nvidia/GLM-5.2-NVFP4", "served": "glm-5.2", "gpu": "B200:4", "args": _GLM52_ARGS},
}

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

    path = snapshot_download(VARIANTS[variant]["repo"])
    models_vol.commit()
    return path


def _serve(variant: str):
    v = VARIANTS[variant]
    cmd = ["vllm", "serve", v["repo"], "--served-model-name", v["served"], "--host", "0.0.0.0",
           "--port", str(PORT), *v["args"],
           "--enable-prefix-caching",  # agents resend their whole history every turn
           "--api-key", "$VLLM_API_KEY"]
    subprocess.Popen(" ".join(cmd), shell=True)


def _server_kwargs(variant: str) -> dict:
    gpu = VARIANTS[variant]["gpu"]
    n = int(gpu.split(":")[1]) if ":" in gpu else 1
    return dict(image=image, gpu=gpu, cpu=16 * n, memory=65536 * n, secrets=secrets,
                volumes={"/models": models_vol}, timeout=6 * 3600, scaledown_window=15 * 60, max_containers=1)


@app.function(**_server_kwargs("bf16"))
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=45 * 60)
def serve_bf16():
    _serve("bf16")


@app.function(**_server_kwargs("fp8"))
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=45 * 60)
def serve_fp8():
    _serve("fp8")


@app.function(**_server_kwargs("nvfp4"))
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=45 * 60)
def serve_nvfp4():
    _serve("nvfp4")


@app.function(**_server_kwargs("glm52_nvfp4"))
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=60 * 60)
def serve_glm52_nvfp4():
    _serve("glm52_nvfp4")
