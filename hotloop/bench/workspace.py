"""The agent-facing workspace: TASK.md and the starter solution.

The task statement and rules belong to the benchmark, so every agent reads the
same TASK.md regardless of how it is driven.
"""

import json
import os

from hotloop.interface import Environment

RULES = """## Rules (enforced automatically; violations score 0)
- `solution.py` must define `solution(*inputs)` taking the same positional inputs as `reference` and returning the \
same outputs (same count, dtypes and shapes).
- Write your own kernels: Triton, or CUDA C++ via `torch.utils.cpp_extension.load_inline`. Inside `solution()`, \
PyTorch may only allocate or reinterpret memory (torch.empty/zeros, .view, .permute, .transpose, ...). No torch \
math, copies, dtype casts, torch.compile, cuBLAS/cuDNN, or other library kernels.
- Allowed imports: torch, triton, math, numpy, functools, itertools, typing, dataclasses, collections, operator, enum.
- `solution()` is captured into a CUDA graph for timing: launch kernels on the current stream (C++: \
`at::cuda::getCurrentCUDAStream()`), no host synchronization (.item(), .cpu(), printing tensors), no host memory.
- Scoring uses hidden shapes (B and S below take other values, including non-powers-of-two) and fresh random \
inputs, including inputs with large outliers. Do not hardcode the public shape.
- Accuracy tolerance is derived from PyTorch's own low-precision error against an fp64 reference (a few times \
that error is allowed).
- Score = speedup over torch.compile of the reference, and fraction of the speed-of-light bound.
- Every `bench` run in which all public shapes are correct saves a snapshot of solution.py. If the final \
solution.py fails scoring (e.g. you were mid-edit when time ran out), the latest snapshot is scored instead.

## Tools on this machine (no internet access)
- `python -m hotloop.harness.bench [solution.py]` - correctness, timing, speed-of-light and a per-kernel time \
breakdown on every public shape, with the same checks as final scoring (1-3 min; the first run also compiles the \
torch.compile baselines).
- `ncu --clock-control none ...` (Nsight Compute: per-kernel metrics) and `nsys profile ...` + `nsys stats` \
(Nsight Systems: timeline) are installed.
"""


def _fmt(dims) -> str:
    return "[" + ", ".join(map(str, dims)) + "]"


def render_task_md(task: dict, meta: dict, reference: str, workdir: str, gpu: str, minutes: float) -> str:
    public = task.get("public_shapes") or [meta["shape_id"]]
    sym = task.get("symbolic_shapes")
    if sym:
        inputs = "\n".join(f"  {i['name']}: {i['dtype']}{_fmt(s['shape'])}  (public: {_fmt(i['shape'])}, {i['kind']})"
                           for i, s in zip(meta["inputs"], sym["inputs"]))
        outputs = "\n".join(f"  {o.get('name', '')}{' (updated in place)' if o.get('in_place') else ''}: "
                            f"{o['dtype']}{_fmt(s['shape'])}  (public: {_fmt(o['shape'])})"
                            for o, s in zip(meta["outputs"], sym["outputs"]))
        legend = {"B": "batch size", "S": "prompt length", "T": "KV-cache length (tokens already cached + 1)"}
        names = sym.get("dims", ["B", "S"])
        shape_note = (", ".join(f"{n} = {legend.get(n, n)}" for n in names)
                      + " (vary in hidden tests); other dims are fixed.\n")
        in_place = [o["name"] for o in meta["outputs"] if o.get("in_place")]
        if in_place:
            shape_note += (f"The reference updates {', '.join(in_place)} in place (it writes the new token into the "
                           "cache); your solution must make the same in-place update, which is checked like an "
                           "output. Return only the reference's return values.\n")
    else:
        inputs = "\n".join(f"  {i['name']}: {i['dtype']}{_fmt(i['shape'])} ({i['kind']})" for i in meta["inputs"])
        outputs = "\n".join(f"  {o['dtype']}{_fmt(o['shape'])}" for o in meta["outputs"])
        shape_note = ""
    return f"""# Task: {task['task_id']}
Module `{task['module_path']}` ({task['module_class']}) from {task['model_id']}, {task.get('phase', 'prefill')} phase.
GPU: {gpu}. Wall-clock budget: {minutes:g} minutes. Workspace: {workdir}

Write the fastest correct implementation of `reference` below as custom GPU kernels in {workdir}/solution.py.
Whatever is in that file when the budget ends (or when you stop) is scored.

Public shapes (all checked by `bench`): {', '.join(public)}. Hidden shapes are different values of the same dims.
Reference for {meta['shape_id']} (each public shape has its own copy at {workdir}/task/shapes/<shape>/reference.py;
they differ only in the sizes baked into reshapes):
```python
{reference.strip()}
```
{shape_note}Inputs:
{inputs}
Outputs:
{outputs}

{RULES}"""


def starter(task: dict, meta: dict) -> str:
    args = ", ".join(i["name"] for i in meta["inputs"])
    return f'''"""Solution for {task['task_id']}. See task/shapes/*/reference.py for the exact semantics."""
import torch
import triton
import triton.language as tl


def solution({args}):
    raise NotImplementedError
'''


def prepare_workspace(env: Environment, gpu: str, minutes: float) -> None:
    """Called by a backend after it copied the public task directory to <workdir>/task."""
    wd = env.workdir
    res = env.exec(f"ls {wd}/task/shapes", timeout=60)
    if res.exit_code != 0:
        raise RuntimeError(f"task files missing in workspace: {res.output}")
    task = json.loads(env.read_text(os.path.join(wd, "task", "task.json")))
    sid = (task.get("public_shapes") or res.output.split())[0]
    meta = json.loads(env.read_text(os.path.join(wd, "task", "shapes", sid, "meta.json")))
    ref = env.read_text(os.path.join(wd, "task", "shapes", sid, "reference.py"))
    env.write_text(os.path.join(wd, "TASK.md"), render_task_md(task, meta, ref, wd, gpu, minutes))
    env.write_text(os.path.join(wd, "solution.py"), starter(task, meta))
