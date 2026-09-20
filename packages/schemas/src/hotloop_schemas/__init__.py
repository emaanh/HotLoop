"""On-disk contracts. The only package other HotLoop components may share."""

from hotloop_schemas.io import dump_json_schemas, load_task_spec, public_content_hash
from hotloop_schemas.result import (
    CorrectnessDetail,
    EntryResult,
    EnvFingerprint,
    Flag,
    GpuState,
    RatioCI,
    Result,
    TimingStats,
)
from hotloop_schemas.task import (
    PRIVATE_DIR,
    PUBLIC_FILES,
    SUBMISSION_ENTRYPOINT,
    Budget,
    Hardware,
    OutputContract,
    Provenance,
    TaskSpec,
    Tolerance,
    WorkloadEntry,
)
from hotloop_schemas.trajectory import TrajectoryEvent

__all__ = [
    "PRIVATE_DIR",
    "PUBLIC_FILES",
    "SUBMISSION_ENTRYPOINT",
    "Budget",
    "CorrectnessDetail",
    "EntryResult",
    "EnvFingerprint",
    "Flag",
    "GpuState",
    "Hardware",
    "OutputContract",
    "Provenance",
    "RatioCI",
    "Result",
    "TaskSpec",
    "TimingStats",
    "Tolerance",
    "TrajectoryEvent",
    "WorkloadEntry",
    "dump_json_schemas",
    "load_task_spec",
    "public_content_hash",
]
