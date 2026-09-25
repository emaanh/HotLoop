"""What a compute backend must provide.

A backend owns *where* things run (Modal, a GPU VM with Docker, ...). The
benchmark logic (tasks, scoring, budgets) and the agents are the same on every
backend.
"""

from typing import Protocol

from hotloop.interface import Environment


class BackendEnvironment(Environment, Protocol):
    def close(self) -> None: ...


class Backend(Protocol):
    name: str

    # --- agent episodes ----------------------------------------------------------
    def open_environment(self, task_id: str, gpu: str, minutes: float, options: dict | None = None) -> BackendEnvironment:
        """A fresh GPU machine with no network, the public task at <workdir>/task,
        and TASK.md + solution.py prepared (see bench.workspace.prepare_workspace)."""
        ...

    def score(self, task_id: str, solution_src: str, gpu: str, hidden: bool = True) -> dict:
        """Score a solution in a fresh, isolated process/container. Hidden shapes stay there."""
        ...

    # --- task set ----------------------------------------------------------------
    def list_tasks(self, gpu: str | None = None, kept_only: bool = True) -> list[str]: ...

    def select_models(self, n: int) -> list[dict]: ...

    def generate(self, model_ids: list[str], phases: tuple[str, ...] = ("prefill", "decode")) -> dict: ...

    def filter(self, gpu: str, task_ids: list[str] | None = None) -> dict: ...
