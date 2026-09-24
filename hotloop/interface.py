"""The contract between the benchmark and agents.

This is the only hotloop module an agent may import. An agent receives an
`Environment` (a GPU machine with the task workspace), the task text (the
contents of TASK.md), and a `Budget`. It leaves its answer in
`<env.workdir>/solution.py` and returns an `AgentResult`. The benchmark never
imports agents, and agents never import the benchmark's scoring or backends.
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ExecResult:
    output: str       # combined stdout + stderr
    exit_code: int
    timed_out: bool = False


class BudgetExceeded(RuntimeError):
    """Raised by the environment once the wall-clock budget is spent."""


@runtime_checkable
class Environment(Protocol):
    """A GPU machine holding the task workspace. No network access."""

    workdir: str   # contains TASK.md, task/ (public files) and solution.py
    gpu: str       # e.g. "L4", "H100"

    def exec(self, command: str, timeout: int = 600) -> ExecResult:
        """Run a shell command with `workdir` as the current directory."""
        ...

    def read_text(self, path: str) -> str: ...

    def write_text(self, path: str, content: str) -> None: ...

    def seconds_left(self) -> float: ...


@dataclass
class Budget:
    minutes: float
    deadline: float   # unix time; the environment refuses work after this
    # Agents should append their transcript events here as they happen: if an agent
    # overruns the deadline, the benchmark still has the record up to that point.
    events: list = field(default_factory=list)


@dataclass
class AgentResult:
    stop_reason: str                                  # "submitted", "budget", "error: ...", ...
    transcript: list = field(default_factory=list)    # JSON-serialisable events
    usage: dict = field(default_factory=dict)         # tokens, cost, ... (agent-defined)
    metadata: dict = field(default_factory=dict)      # model name, settings, ...


class Agent(Protocol):
    name: str

    def run(self, env: Environment, task: str, budget: Budget) -> AgentResult: ...
