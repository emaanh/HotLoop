"""Run one agent on one task: open environment -> agent -> collect solution -> score.

The wall-clock budget is enforced here, not by the agent: the environment the
agent sees refuses work after the deadline, and the solution is collected when
the agent returns or the deadline (plus a grace period) passes.
"""

import json
import os
import threading
import time
import traceback

from hotloop.interface import AgentResult, Budget, BudgetExceeded, Environment, ExecResult

GRACE_S = 60


class BudgetedEnvironment:
    """Wraps a backend environment; clips command timeouts and stops at the deadline."""

    def __init__(self, env: Environment, deadline: float):
        self._env = env
        self.deadline = deadline
        self.workdir = env.workdir
        self.gpu = env.gpu

    def seconds_left(self) -> float:
        return max(0.0, self.deadline - time.time())

    def _check(self):
        if self.seconds_left() <= 0:
            raise BudgetExceeded("wall-clock budget exhausted")

    def exec(self, command: str, timeout: int = 600) -> ExecResult:
        self._check()
        return self._env.exec(command, timeout=max(1, int(min(timeout, self.seconds_left()))))

    def read_text(self, path: str) -> str:
        return self._env.read_text(path)

    def write_text(self, path: str, content: str) -> None:
        self._check()
        self._env.write_text(path, content)


def run_episode(backend, agent, task_id: str, gpu: str, minutes: float, score: bool = True, log=print,
                options: dict | None = None) -> dict:
    t_open = time.time()
    env = backend.open_environment(task_id, gpu=gpu, minutes=minutes, options=options)
    t0 = time.time()  # the budget starts once the workspace is ready, not while a GPU is being provisioned
    deadline = t0 + minutes * 60
    result: list[AgentResult] = []
    budget = Budget(minutes=minutes, deadline=deadline)
    try:
        task_md = env.read_text(os.path.join(env.workdir, "TASK.md"))
        benv = BudgetedEnvironment(env, deadline)

        def target():
            try:
                result.append(agent.run(benv, task_md, budget))
            except BudgetExceeded:
                result.append(AgentResult(stop_reason="budget"))
            except Exception as e:
                result.append(AgentResult(stop_reason=f"error: {e!r}"[:300],
                                          metadata={"traceback": traceback.format_exc(limit=6)}))

        th = threading.Thread(target=target, daemon=True)
        th.start()
        th.join(timeout=max(0.0, deadline - time.time()) + GRACE_S)
        if th.is_alive():
            log("[episode] agent did not return before the deadline; collecting solution anyway")
            result.append(AgentResult(stop_reason="deadline", transcript=list(budget.events)))
        solution, snapshot, infra_error = None, None, None
        try:
            solution = env.read_text(os.path.join(env.workdir, "solution.py"))
        except Exception as e:
            infra_error = f"could not read solution.py: {e!r}"[:300]
            log(f"[episode] {infra_error}")
        try:
            latest = env.exec("ls -1t .hotloop/passing/*.py 2>/dev/null | head -1", timeout=60).output.strip()
            if latest:
                snapshot = env.read_text(latest if latest.startswith("/") else os.path.join(env.workdir, latest))
        except Exception as e:
            log(f"[episode] could not read snapshots: {e!r}")
    finally:
        env.close()

    res = result[0]
    if not res.transcript and budget.events:  # agent failed before returning its own copy
        res.transcript = list(budget.events)
    if solution is None and snapshot is None:
        res.stop_reason = "infra_error"  # nothing retrievable: re-run, don't score as 0
    record = {
        "run_id": f"{time.strftime('%Y%m%d-%H%M%S')}_{agent.name}_{task_id}",
        "task_id": task_id, "gpu": gpu, "backend": backend.name, "agent": agent.name,
        "agent_metadata": res.metadata, "stop_reason": res.stop_reason, "usage": res.usage,
        "task_options": options or {}, "budget_minutes": minutes, "agent_minutes": round((min(time.time(), deadline) - t0) / 60, 1),
        "startup_minutes": round((t0 - t_open) / 60, 1),
        "solution": solution or "", "snapshot": snapshot, "infra_error": infra_error,
    }
    provider_error = str(res.stop_reason).startswith("error: api")  # excluded from results anyway
    if score and res.stop_reason != "infra_error" and not provider_error:
        record["eval"], record["scored"] = score_with_fallback(backend, task_id, gpu, solution, snapshot)
    return {"record": record, "transcript": res.transcript}


def score_with_fallback(backend, task_id: str, gpu: str, final: str | None, snapshot: str | None):
    """Score the final solution; if it fails, score the latest snapshot that passed `bench`."""
    ev = backend.score(task_id, final, gpu=gpu, hidden=True) if final else None
    if ev is not None and ev.get("ok"):
        return ev, "final"
    if snapshot and snapshot != final:
        snap_ev = backend.score(task_id, snapshot, gpu=gpu, hidden=True)
        if snap_ev.get("ok") or ev is None:
            snap_ev["final_eval_summary"] = None if ev is None else {"error": ev.get("error"), "violations": ev.get("violations")}
            return snap_ev, "snapshot"
    return ev, "final"


def save_episode(ep: dict, root: str) -> str:
    rec = ep["record"]
    d = os.path.join(root, "episodes", rec["run_id"])
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "result.json"), "w") as f:
        json.dump({k: v for k, v in rec.items() if k not in ("solution", "snapshot")}, f, indent=1, default=str)
    with open(os.path.join(d, "solution.py"), "w") as f:
        f.write(rec["solution"])
    if rec.get("snapshot"):
        with open(os.path.join(d, "snapshot.py"), "w") as f:
            f.write(rec["snapshot"])
    with open(os.path.join(d, "transcript.json"), "w") as f:
        json.dump(ep["transcript"], f, indent=1, default=str)
    return d
