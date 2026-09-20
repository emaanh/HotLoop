"""Runner output contract: trajectory.jsonl, one TrajectoryEvent per line.

Raw model completions are always kept (D-12): tool-call / format failures are scored by us,
not inferred from whatever a serving engine's parser let through.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1

EventType = Literal[
    "run_start",  # payload: task_id, agent, model, budget, env fingerprint, seed
    "model_request",  # payload: n_messages, prompt_tokens (if known)
    "model_response",  # payload: raw completion, usage, latency_s, finish_reason
    "parse_error",  # payload: why the completion could not be turned into an action
    "command",  # payload: cmd, timeout_s
    "observation",  # payload: exit_code, stdout/stderr (possibly truncated), duration_s, gpu_seconds
    "check",  # payload: summarised `hotloop check` result the agent just saw
    "egress_blocked",  # payload: destination (network policy hit)
    "budget",  # payload: which limit, used, limit
    "run_end",  # payload: reason (submitted | budget | error), totals
]


class TrajectoryEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = SCHEMA_VERSION
    run_id: str
    seq: int = Field(ge=0)
    ts: float = Field(description="unix seconds")
    turn: int = Field(ge=0)
    type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)
