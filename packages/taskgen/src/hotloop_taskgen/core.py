"""Emit immutable task packages (schemas.task). Families produce TaskDrafts; this module turns a
draft into a directory, asks the evaluator (as a subprocess, never an import) to calibrate
tolerances, and seals it with a content hash."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hotloop_schemas import PUBLIC_FILES, TaskSpec, load_task_spec, public_content_hash

GENERATOR_VERSION = "0.1.0"
DEFAULT_BASELINES = [
    "eager",
    "compile_default",
    "compile_max_autotune_no_cudagraphs",
    "compile_reduce_overhead",
]
DEFAULT_BUDGET = {
    "max_turns": 100,
    "max_total_tokens": 1_500_000,
    "gpu_seconds": 1800.0,
    "wall_seconds": 5400.0,
}


@dataclass
class TaskDraft:
    id: str
    title: str
    family: str
    regime: str
    family_params: dict[str, Any]
    reference_src: str
    workload_src: str
    description: str  # what the program computes and what the workload looks like, for the agent
    n_outputs: int = 1
    sibling_group: str | None = None
    control: bool = False  # no-headroom control (D-24): the compiler is expected to be optimal
    seed: int = 0
    entries: list[dict] = field(default_factory=lambda: [{"name": "main", "weight": 1.0}])


AGENT_README = """# Task: {title}

Make `reference(*inputs)` in `reference.py` faster on this machine's GPU, without changing what it
computes.

## What you are given
- `reference.py` - the PyTorch reference. It defines the semantics and is the correctness oracle.
- `workload.py` - `make_inputs(entry, seed, device)` builds the inputs you will be timed on.
  Shapes, dtypes and the *distribution* of the data are fixed by this file; the values are not.
- `task.toml` - metadata: output tolerances, the automatic baselines you are scored against.

{description}

## What you must deliver
A file `solution.py` in the submission directory `{submission_dir}` exposing

    def run(*inputs): ...   # same arguments as reference(), same outputs (shape, dtype, device)

Anything is allowed inside: Triton, CUDA/C++ extensions, torch.compile, CUDA graphs, cuBLAS calls,
algebraic rewrites in plain PyTorch. Extra files next to `solution.py` are fine. Rules of the road:
- do not mutate the inputs;
- every returned tensor must stay valid after later calls to `run` (clone if you reuse buffers);
- a returned tensor must be safe to read on the current CUDA stream.

## How you are scored
Score = speedup of `run` over the **fastest automatic baseline** - the best of eager PyTorch and
`torch.compile` in several modes ({baselines}) - measured interleaved with it on the same GPU.
Beating eager is not the goal; beating the compiler is. A score of 1.0 means "no better than the
compiler". Any incorrect output, at any point, scores 0 - including outputs produced while being
timed. Final scoring happens on a fresh machine of the same type with inputs drawn from hidden
seeds, so nothing keyed on particular values or addresses will help.

Check yourself at any time:

    hotloop-eval --task {task_dir} --submission {submission_dir} --quick

It reports correctness and your speedup with a confidence interval, using the same code path as
final scoring. You have a real GPU, a shell, compilers and profilers; use them however you like.
"""


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {_toml_value(x)}" for k, x in v.items()) + " }"
    raise TypeError(f"cannot serialise {type(v).__name__} to TOML")


def render_task_toml(draft: TaskDraft, tolerances: list[dict], hardware: dict, content_hash=None):
    lines = [
        "schema_version = 1",
        f"id = {_toml_value(draft.id)}",
        f"title = {_toml_value(draft.title)}",
    ]
    if content_hash:
        lines.append(f"content_hash = {_toml_value(content_hash)}")
    lines.append(f"baselines = {_toml_value(DEFAULT_BASELINES)}")
    lines += ["", "[provenance]", f"generator_version = {_toml_value(GENERATOR_VERSION)}"]
    lines += [f"family = {_toml_value(draft.family)}", f"seed = {draft.seed}"]
    if draft.sibling_group:
        lines.append(f"sibling_group = {_toml_value(draft.sibling_group)}")
    params = {"regime": draft.regime, "control": draft.control, **draft.family_params}
    lines.append(f"family_params = {_toml_value(params)}")
    lines += ["", "[hardware]"] + [f"{k} = {_toml_value(v)}" for k, v in hardware.items()]
    for e in draft.entries:
        lines += ["", "[[workload]]"] + [f"{k} = {_toml_value(v)}" for k, v in e.items()]
    lines += ["", "[outputs]", f"n_outputs = {draft.n_outputs}"]
    for t in tolerances:
        lines += ["[[outputs.tolerances]]"] + [f"{k} = {_toml_value(v)}" for k, v in t.items()]
    lines += ["", "[budget]"] + [f"{k} = {_toml_value(v)}" for k, v in DEFAULT_BUDGET.items()]
    return "\n".join(lines) + "\n"


def emit(
    draft: TaskDraft,
    out_root: str | Path,
    hardware: dict,
    *,
    calibrate: bool = True,
    device: str = "cuda",
    submission_dir: str = "/workspace/submission",
    task_dir_in_sandbox: str = "/workspace/task",
) -> Path:
    d = Path(out_root) / draft.id
    d.mkdir(parents=True, exist_ok=False)
    (d / "reference.py").write_text(draft.reference_src)
    (d / "workload.py").write_text(draft.workload_src)
    (d / "AGENT_README.md").write_text(
        AGENT_README.format(
            title=draft.title,
            description=draft.description.strip(),
            baselines=", ".join(DEFAULT_BASELINES),
            submission_dir=submission_dir,
            task_dir=task_dir_in_sandbox,
        )
    )
    placeholder = [{"rtol": 1e-3, "atol": 1e-5}] * draft.n_outputs
    (d / "task.toml").write_text(render_task_toml(draft, placeholder, hardware))
    tolerances = placeholder
    if calibrate:
        out = (
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "hotloop_evaluator.cli_calibrate",
                    "--task",
                    str(d),
                    "--device",
                    device,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
            .splitlines()[-1]
        )
        per_entry = json.loads(out)
        # one contract per task: take the loosest calibrated tolerance across entries
        tolerances = [
            {
                "rtol": max(per_entry[e][i]["rtol"] for e in per_entry),
                "atol": max(per_entry[e][i]["atol"] for e in per_entry),
            }
            for i in range(draft.n_outputs)
        ]
        (d / "task.toml").write_text(render_task_toml(draft, tolerances, hardware))
    h = public_content_hash(d)
    (d / "task.toml").write_text(render_task_toml(draft, tolerances, hardware, content_hash=h))
    spec: TaskSpec = load_task_spec(d)
    assert spec.content_hash == public_content_hash(d), "content hash must be self-consistent"
    assert all((d / f).is_file() for f in PUBLIC_FILES)
    return d
