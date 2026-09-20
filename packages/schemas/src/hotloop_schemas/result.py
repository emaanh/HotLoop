"""Evaluator output contract: result.json.

Everything downstream (analysis, leaderboards, the quantization study) reads only this and
trajectory.jsonl. Every number carries its measurement conditions (EnvFingerprint).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1

Status = Literal[
    "ok",  # correct everywhere, timing conclusive
    "incorrect",  # any correctness/contract failure -> score 0 (D-10)
    "error",  # solution failed to import/build/run
    "inconclusive",  # timing CI too wide or throttling detected -> re-queue, no score
    "flagged",  # anti-cheat tripped -> score 0, needs audit
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TimingStats(_Strict):
    n: int
    median_s: float
    iqr_s: float
    mean_s: float
    min_s: float


class RatioCI(_Strict):
    """Bootstrap CI on t_baseline / t_candidate from interleaved blocks."""

    point: float
    lo: float
    hi: float
    level: float = 0.95


class CorrectnessDetail(_Strict):
    passed: bool
    n_trials: int
    max_abs_err: float | None = None
    max_rel_err: float | None = None
    failure: str | None = None


class EntryResult(_Strict):
    entry: str
    weight: float
    correctness: CorrectnessDetail
    baselines: dict[str, TimingStats] = Field(default_factory=dict)
    best_baseline: str | None = None
    candidate: TimingStats | None = None
    speedup: RatioCI | None = None
    peak_memory_bytes: int | None = None


class GpuState(_Strict):
    clocks_locked: bool | None = None
    sm_clock_mhz: int | None = None
    mem_clock_mhz: int | None = None
    temperature_c: int | None = None
    throttle_reasons: list[str] = Field(default_factory=list)


class EnvFingerprint(_Strict):
    gpu_name: str | None = None
    gpu_uuid: str | None = None
    vbios: str | None = None
    driver: str | None = None
    cuda: str | None = None
    torch: str | None = None
    triton: str | None = None
    python: str | None = None
    cpu_model: str | None = None
    image: str | None = None
    provider: str | None = None
    ecc: bool | None = None
    gpu_before: GpuState | None = None
    gpu_after: GpuState | None = None


class Flag(_Strict):
    """An anti-cheat or hygiene signal. `fatal` flags force status='flagged'."""

    code: str
    detail: str = ""
    fatal: bool = False


class Result(_Strict):
    schema_version: Literal[1] = SCHEMA_VERSION
    task_id: str
    task_hash: str | None = None
    submission_hash: str | None = None
    evaluator_version: str
    status: Status
    score: float = Field(
        ge=0, description="weighted geomean speedup vs best baseline; 0 unless status=='ok'"
    )
    entries: list[EntryResult] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)
    env: EnvFingerprint = Field(default_factory=EnvFingerprint)
    build_seconds: float | None = None
    eval_seconds: float | None = None
    error: str | None = None
