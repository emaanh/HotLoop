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


def run_episode(backend, agent, task_id: str, gpu: str, minutes: float, score: bool = True, log=print) -> dict:
    t0 = time.time()
    env = backend.open_environment(task_id, gpu=gpu, minutes=minutes)
    deadline = time.time() + minutes * 60
    result: list[AgentResult] = []
    try:
        task_md = env.read_text(os.path.join(env.workdir, "TASK.md"))
        benv = BudgetedEnvironment(env, deadline)

        def target():
            try:
                result.append(agent.run(benv, task_md, Budget(minutes=minutes, deadline=deadline)))
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
            result.append(AgentResult(stop_reason="deadline"))
        try:
            solution = env.read_text(os.path.join(env.workdir, "solution.py"))
        except Exception as e:
            solution = ""
            log(f"[episode] could not read solution: {e!r}")
    finally:
        env.close()

    res = result[0]
    record = {
        "run_id": f"{time.strftime('%Y%m%d-%H%M%S')}_{agent.name}_{task_id}",
        "task_id": task_id, "gpu": gpu, "backend": backend.name, "agent": agent.name,
        "agent_metadata": res.metadata, "stop_reason": res.stop_reason, "usage": res.usage,
        "budget_minutes": minutes, "agent_minutes": round((min(time.time(), deadline) - t0) / 60, 1),
        "solution": solution,
    }
    if score:
        record["eval"] = backend.score(task_id, solution, gpu=gpu, hidden=True)
    return {"record": record, "transcript": res.transcript}


def save_episode(ep: dict, root: str) -> str:
    rec = ep["record"]
    d = os.path.join(root, "episodes", rec["run_id"])
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "result.json"), "w") as f:
        json.dump({k: v for k, v in rec.items() if k != "solution"}, f, indent=1, default=str)
    with open(os.path.join(d, "solution.py"), "w") as f:
        f.write(rec["solution"])
    with open(os.path.join(d, "transcript.json"), "w") as f:
        json.dump(ep["transcript"], f, indent=1, default=str)
    return d
