"""A toy task plus honest and dishonest submissions, materialised into tmp dirs by tests.

The reference is deliberately slow (python loop over rows) so an honest vectorised solution has
an unmistakable speedup even on a noisy CPU.
"""

from pathlib import Path

TASK_TOML = """
schema_version = 1
id = "toy-rowwise-0001"
title = "Row-wise projection + relu + sum"
baselines = ["eager"]

[provenance]
generator_version = "test"
family = "toy"
seed = 0

[hardware]
gpu_sku = "cpu"
image = "none"

[[workload]]
name = "small"
weight = 1.0

[outputs]
n_outputs = 1
[[outputs.tolerances]]
rtol = 1e-4
atol = 1e-5

[budget]
max_turns = 1
max_total_tokens = 1
gpu_seconds = 1
wall_seconds = 1
"""

REFERENCE = """
import torch

def reference(x, w):
    out = torch.empty(x.shape[0], dtype=x.dtype, device=x.device)
    for i in range(x.shape[0]):
        out[i] = torch.relu(x[i] @ w).sum()
    return out
"""

WORKLOAD = """
import torch

def make_inputs(entry, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(192, 48, generator=g).to(device)
    w = torch.randn(48, 32, generator=g).to(device)
    return (x, w)
"""

_HONEST_BODY = "torch.relu(x @ w).sum(-1)"

SOLUTIONS = {
    # ---- honest ---------------------------------------------------------------------------
    "honest": f"""
import torch
def run(x, w):
    return {_HONEST_BODY}
""",
    # ---- plain failures ---------------------------------------------------------------------
    "wrong": f"""
import torch
def run(x, w):
    return {_HONEST_BODY} * 1.01
""",
    "wrong_dtype": f"""
import torch
def run(x, w):
    return ({_HONEST_BODY}).double()
""",
    "crash_on_import": "raise RuntimeError('boom')\n",
    "sys_exit": """
import sys
def run(x, w):
    sys.exit(0)
""",
    "mutates_inputs": f"""
import torch
def run(x, w):
    out = {_HONEST_BODY}
    x.zero_()
    return out
""",
    # ---- cheats -----------------------------------------------------------------------------
    # correct while being checked, then replays a cached answer once it thinks timing started
    "stale_after_warmup": f"""
import torch
_calls, _cached = 0, None
def run(x, w):
    global _calls, _cached
    _calls += 1
    if _calls <= 12 or _cached is None:
        _cached = {_HONEST_BODY}
        return _cached
    return _cached.clone()
""",
    # memoise on a content hash: useless against fresh values, and must stay *correct*
    "memoise_by_value": f"""
import torch
_cache = {{}}
def run(x, w):
    key = (float(x.sum()), float(w.sum()))
    if key not in _cache:
        _cache[key] = {_HONEST_BODY}
    return _cache[key].clone()
""",
    # freeze the clock inside its own process: the time that counts is measured elsewhere
    "patches_clock": f"""
import time, torch
time.perf_counter = lambda: 0.0
def run(x, w):
    return {_HONEST_BODY}
""",
    # pretend to be instant, do the real work too late to matter
    "returns_garbage_fast": """
import torch
def run(x, w):
    return torch.empty(x.shape[0], dtype=x.dtype, device=x.device)
""",
    # detect small call counts (the correctness phase) and only then be right
    "right_only_at_first": f"""
import torch
_n = 0
def run(x, w):
    global _n
    _n += 1
    if _n < 9:
        return {_HONEST_BODY}
    return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
""",
    "hangs": """
import time
def run(x, w):
    time.sleep(3600)
""",
}


def write_task(root: Path) -> Path:
    d = root / "task"
    d.mkdir()
    (d / "task.toml").write_text(TASK_TOML)
    (d / "reference.py").write_text(REFERENCE)
    (d / "workload.py").write_text(WORKLOAD)
    (d / "AGENT_README.md").write_text("toy\n")
    return d


def write_submission(root: Path, name: str) -> Path:
    d = root / f"sub_{name}"
    d.mkdir()
    (d / "solution.py").write_text(SOLUTIONS[name])
    return d
