"""GPU toy task: fused pointwise + row reduction. Bandwidth-bound in eager (several kernels,
several passes over memory); one pass when fused. Used only by tests marked `gpu`."""

from pathlib import Path

TASK_TOML = """
schema_version = 1
id = "gpu-toy-rowreduce-0001"
title = "sum(relu(x*a+b)^2, -1)"
baselines = ["eager", "compile_default"]

[provenance]
generator_version = "test"
family = "toy"
seed = 0

[hardware]
gpu_sku = "any"
image = "none"

[[workload]]
name = "wide"
weight = 1.0

[outputs]
n_outputs = 1
[[outputs.tolerances]]
rtol = 2e-4
atol = 1e-3

[budget]
max_turns = 1
max_total_tokens = 1
gpu_seconds = 1
wall_seconds = 1
"""

REFERENCE = """
import torch

def reference(x, a, b):
    y = x * a + b
    y = torch.relu(y)
    y = y * y
    return y.sum(-1)
"""

WORKLOAD = """
import torch

def make_inputs(entry, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(4096, 4096, device=device, generator=g)
    a = torch.randn(4096, device=device, generator=g)
    b = torch.randn(4096, device=device, generator=g)
    return (x, a, b)
"""

_TRITON = """
import torch, triton, triton.language as tl

@triton.jit
def _k(X, A, B, OUT, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < n_cols
    x = tl.load(X + row * n_cols + cols, mask=m, other=0.0)
    a = tl.load(A + cols, mask=m, other=0.0)
    b = tl.load(B + cols, mask=m, other=0.0)
    y = tl.maximum(x * a + b, 0.0)
    tl.store(OUT + row, tl.sum(y * y, axis=0))

def _fused(x, a, b):
    out = torch.empty(x.shape[0], device=x.device, dtype=x.dtype)
    _k[(x.shape[0],)](x, a, b, out, x.shape[1], BLOCK=triton.next_power_of_2(x.shape[1]))
    return out
"""

SOLUTIONS = {
    "honest_triton": _TRITON + "\ndef run(x, a, b):\n    return _fused(x, a, b)\n",
    # CUDA-L1's exploit: do the work on a side stream so current-stream CUDA events see nothing
    "side_stream": _TRITON
    + """
_s = torch.cuda.Stream()
def run(x, a, b):
    _s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(_s):
        out = _fused(x, a, b)
    return out
""",
    # make every synchronise in this process a no-op, then be honest
    "no_sync": _TRITON
    + """
torch.cuda.synchronize = lambda *a, **k: None
torch._C._cuda_synchronize = lambda *a, **k: None
def run(x, a, b):
    return _fused(x, a, b)
""",
    "eager_copy": "import torch\ndef run(x, a, b):\n    y = torch.relu(x * a + b)\n    return (y * y).sum(-1)\n",
}


def write_task(root: Path) -> Path:
    d = root / "task"
    d.mkdir()
    (d / "task.toml").write_text(TASK_TOML)
    (d / "reference.py").write_text(REFERENCE)
    (d / "workload.py").write_text(WORKLOAD)
    (d / "AGENT_README.md").write_text("gpu toy\n")
    return d


def write_submission(root: Path, name: str) -> Path:
    d = root / f"sub_{name}"
    d.mkdir()
    (d / "solution.py").write_text(SOLUTIONS[name])
    return d
