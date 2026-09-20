"""Task package contract.

A task package is an immutable directory emitted by taskgen:

    <task_id>/
      task.toml          TaskSpec (this module)
      reference.py       def reference(*inputs) -> Tensor | tuple[Tensor, ...]
      workload.py        def make_inputs(entry: str, seed: int, device: str) -> tuple[Tensor, ...]
      AGENT_README.md    what the agent is told
      private/           never mounted into the agent sandbox (strategies, certificates)

The submission contract is the mirror image: a directory containing `solution.py` exposing
`run(*inputs)` with the same signature and output contract as `reference`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1

PUBLIC_FILES = ("task.toml", "reference.py", "workload.py", "AGENT_README.md")
PRIVATE_DIR = "private"
SUBMISSION_ENTRYPOINT = "solution.py"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Provenance(_Strict):
    """How the task was generated; enough to regenerate it bit-for-bit."""

    generator_version: str
    family: str
    family_params: dict[str, Any] = Field(default_factory=dict)
    seed: int
    sibling_group: str | None = Field(
        default=None,
        description="Tasks sharing a program but straddling a certified regime boundary share "
        "a group id. Used for transfer matrices / regime regret.",
    )


class Hardware(_Strict):
    gpu_sku: str = Field(
        description="e.g. 'A100-SXM4-40GB'; the same program on another SKU is another task"
    )
    image: str = Field(description="Container image digest the task was certified under")
    min_gpu_memory_gib: float | None = None


class WorkloadEntry(_Strict):
    """One point of the workload distribution. Shapes/layout/data distribution live in
    workload.py; this is the declarative index the evaluator iterates over."""

    name: str
    weight: float = Field(gt=0)
    description: str = ""


class Tolerance(_Strict):
    """Per-output closeness, calibrated at certification time from an fp64 re-run of the
    reference (EVAL.md). Applied as |cand - ref| <= atol + rtol * |ref|."""

    rtol: float = Field(ge=0)
    atol: float = Field(ge=0)
    equal_nan: bool = False


class OutputContract(_Strict):
    n_outputs: int = Field(ge=1)
    tolerances: list[Tolerance]
    check_strides: bool = False
    inputs_may_be_mutated: bool = False
    deterministic: bool = True
    memory_cap_bytes: int | None = Field(
        default=None, description="Peak allocated bytes allowed; None = unconstrained"
    )

    @model_validator(mode="after")
    def _one_tolerance_per_output(self) -> OutputContract:
        if len(self.tolerances) != self.n_outputs:
            raise ValueError("need exactly one Tolerance per output")
        return self


class Budget(_Strict):
    """Agent-side limits. Deliberately not wall-clock-primary (D-6): wall-clock couples score to
    model serving throughput. `wall_seconds` is a safety cap only."""

    max_turns: int = Field(gt=0)
    max_total_tokens: int = Field(gt=0)
    gpu_seconds: float = Field(gt=0)
    wall_seconds: float = Field(gt=0)


class TaskSpec(_Strict):
    schema_version: Literal[1] = SCHEMA_VERSION
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_\-\.]{2,95}$")
    title: str
    provenance: Provenance
    hardware: Hardware
    workload: list[WorkloadEntry] = Field(min_length=1)
    outputs: OutputContract
    budget: Budget
    baselines: list[str] = Field(
        default_factory=lambda: ["eager", "compile_default", "compile_max_autotune"],
        description="Automatic baselines the evaluator measures; score is vs the fastest (D-4)",
    )
    content_hash: str | None = Field(
        default=None, description="sha256 over public files, filled by taskgen at emit time"
    )

    @model_validator(mode="after")
    def _unique_entries(self) -> TaskSpec:
        names = [w.name for w in self.workload]
        if len(set(names)) != len(names):
            raise ValueError("workload entry names must be unique")
        return self
