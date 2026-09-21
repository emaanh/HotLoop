"""Minimal bash-tool agent on top of mini-swe-agent (D-18).

    hotloop-agent --session session.json --task-readme AGENT_README.md --model openai/<name> --out DIR

The agent loop runs on the operator's machine; only commands cross into the sandbox, through the
argv prefix the runner handed out in the session file. This package knows nothing about scoring.

What we add to mini-swe-agent:
  * `SessionEnvironment`  - executes through the runner's exec prefix, per-command durations.
  * `BudgetedAgent`       - turn and completion-token budgets (D-6), remaining budget shown to the
                            model every turn, and our own `trajectory.jsonl` with the raw
                            response of every model call (D-12).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import LimitsExceeded, Submitted
from pydantic import BaseModel

from hotloop_schemas import TrajectoryEvent

SUBMIT_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

SYSTEM = """You are an expert GPU performance engineer working autonomously in a Linux sandbox that has a \
real, exclusive NVIDIA GPU. You act by calling the `bash` tool. There is no human to ask; decide for yourself \
what to inspect, measure, build, keep or abandon."""

INSTANCE = """{{ task }}

## Your environment
- Commands run in a Docker container via `bash -lc`, working directory `/workspace`. **Each command runs in a
  fresh shell**: `cd` and environment variables do not persist between commands. Files do.
- `/workspace/task` is read-only. Write your solution to `/workspace/submission/solution.py`.
- There is no network access. Python 3.12, PyTorch, Triton, nvcc, gcc, cmake and ninja are installed.
- A command is killed after {{ command_timeout }} seconds. Long output is truncated; redirect to a file if needed.
- Budget: {{ max_turns }} turns and {{ max_completion_tokens }} generated tokens in total. What remains is shown
  after every command. When the budget runs out, whatever is in `/workspace/submission` is scored as is - so
  keep a correct solution in place at all times once you have one.

## Finishing
When you are done, run exactly: `echo {{ sentinel }}`
After that you cannot continue. Only the contents of `/workspace/submission` count."""

OBSERVATION = """{% if output.exception_info -%}
<exception>{{output.exception_info}}</exception>
{% endif -%}
<returncode>{{output.returncode}}</returncode>
{% if output.output | length < 12000 -%}
<output>
{{ output.output -}}
</output>
{%- else -%}
<warning>Output too long; showing head and tail. Redirect to a file and inspect selectively if you need more.</warning>
<output_head>
{{ output.output[:6000] }}
</output_head>
<elided_chars>{{ output.output | length - 12000 }} characters elided</elided_chars>
<output_tail>
{{ output.output[-6000:] }}
</output_tail>
{%- endif %}
<budget>turns left: {{ turns_left }}, generated tokens left: {{ tokens_left }}, last command took {{ output.duration_s }}s</budget>"""


class SessionEnvConfig(BaseModel):
    exec_prefix: list[str]
    exec_remote_template: str | None = None  # None -> run `prefix + [command]` directly (tests)
    timeout: int = 900


class SessionEnvironment:
    def __init__(self, **kwargs):
        self.config = SessionEnvConfig(**kwargs)

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return {"command_timeout": self.config.timeout, **kwargs}

    def serialize(self) -> dict:
        return {"info": {"config": {"environment": {"timeout": self.config.timeout}}}}

    def _argv(self, command: str) -> list[str]:
        if self.config.exec_remote_template is None:
            return [*self.config.exec_prefix, command]
        remote = self.config.exec_remote_template.format(
            timeout=self.config.timeout, command=shlex.quote(command)
        )
        return [*self.config.exec_prefix, remote]

    def execute(self, action: dict, cwd: str = "") -> dict[str, Any]:
        command = action.get("command", "")
        t0 = time.time()
        try:
            r = subprocess.run(
                self._argv(command), text=True, encoding="utf-8", errors="replace",
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=self.config.timeout + 60, check=False,
            )  # fmt: skip
            out = {"output": r.stdout, "returncode": r.returncode, "exception_info": ""}
            if r.returncode in (124, 137):
                out["exception_info"] = (
                    f"command exceeded the {self.config.timeout}s limit and was killed"
                )
        except subprocess.TimeoutExpired as e:
            raw = (
                e.output.decode("utf-8", "replace")
                if isinstance(e.output, bytes)
                else (e.output or "")
            )
            out = {"output": raw, "returncode": -1, "exception_info": "command timed out"}
        out["duration_s"] = round(time.time() - t0, 2)
        lines = out["output"].lstrip().splitlines()
        if lines and lines[0].strip() == SUBMIT_SENTINEL and out["returncode"] == 0:
            raise Submitted({"role": "exit", "content": "submitted",
                             "extra": {"exit_status": "Submitted", "submission": ""}})  # fmt: skip
        return out


class BudgetedConfig(AgentConfig):
    max_turns: int = 100
    max_completion_tokens: int = 400_000
    events_path: Path | None = None
    run_id: str = "run"


def _usage(message: dict) -> dict:
    u = (message.get("usage") or message.get("extra", {}).get("response", {}).get("usage")) or {}
    out_tok = u.get("output_tokens", u.get("completion_tokens", 0)) or 0
    in_tok = u.get("input_tokens", u.get("prompt_tokens", 0)) or 0
    details = u.get("input_tokens_details") or u.get("prompt_tokens_details") or {}
    return {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cached_tokens": (details or {}).get("cached_tokens", 0) or 0,
    }


class BudgetedAgent(DefaultAgent):
    def __init__(self, model, env, **kwargs):
        super().__init__(model, env, config_class=BudgetedConfig, **kwargs)
        self.tokens = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
        self._seq = 0
        self._event("run_start", {"model": getattr(model.config, "model_name", "?"),
                                  "max_turns": self.config.max_turns,
                                  "max_completion_tokens": self.config.max_completion_tokens})  # fmt: skip

    def _event(self, type_: str, payload: dict) -> None:
        if not self.config.events_path:
            return
        ev = TrajectoryEvent(run_id=self.config.run_id, seq=self._seq, ts=time.time(),
                             turn=self.n_calls, type=type_, payload=payload)  # fmt: skip
        self._seq += 1
        self.config.events_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.events_path, "a") as f:
            f.write(ev.model_dump_json() + "\n")

    def get_template_vars(self, **kwargs) -> dict:
        return super().get_template_vars(
            sentinel=SUBMIT_SENTINEL,
            turns_left=max(self.config.max_turns - self.n_calls, 0),
            tokens_left=max(self.config.max_completion_tokens - self.tokens["output_tokens"], 0),
            **kwargs,
        )

    def query(self) -> dict:
        over_turns = self.n_calls >= self.config.max_turns
        over_tokens = self.tokens["output_tokens"] >= self.config.max_completion_tokens
        if over_turns or over_tokens:
            which = "turns" if over_turns else "completion_tokens"
            self._event("budget", {"limit": which, **self.tokens, "turns": self.n_calls})
            raise LimitsExceeded({"role": "exit", "content": f"budget exhausted: {which}",
                                  "extra": {"exit_status": "BudgetExhausted", "submission": ""}})  # fmt: skip
        self._event("model_request", {"n_messages": len(self.messages)})
        t0 = time.time()
        try:
            message = super().query()
        except Exception as e:
            raw = [m.get("extra", {}).get("response") for m in getattr(e, "messages", [])]
            self._event("parse_error", {"error": f"{type(e).__name__}: {e}"[:2000], "raw": raw})
            raise
        use = _usage(message)
        for k, v in use.items():
            self.tokens[k] += v
        raw = {k: v for k, v in message.items() if k != "extra"}
        self._event("model_response", {"latency_s": round(time.time() - t0, 2), "usage": use,
                                       "actions": message.get("extra", {}).get("actions", []),
                                       "raw": raw})  # fmt: skip
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        outputs = []
        for action in actions:
            self._event("command", {"cmd": action.get("command", "")})
            try:
                out = self.env.execute(action)
            except Submitted:
                self._event(
                    "run_end", {"reason": "submitted", **self.tokens, "turns": self.n_calls}
                )
                raise
            self._event("observation", {"returncode": out["returncode"], "duration_s": out.get("duration_s"),
                                        "output": out["output"][-20000:], "exception": out["exception_info"]})  # fmt: skip
            outputs.append(out)
        return self.add_messages(
            *self.model.format_observation_messages(message, outputs, self.get_template_vars())
        )


def build_model(model_name: str, reasoning_effort: str | None):
    from minisweagent.models.litellm_response_model import LitellmResponseModel

    kwargs: dict[str, Any] = {"drop_params": True}
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}
    return LitellmResponseModel(
        model_name=model_name, model_kwargs=kwargs, cost_tracking="ignore_errors",
        observation_template=OBSERVATION,
    )  # fmt: skip


def _load_env_file() -> None:
    for parent in [Path.cwd(), *Path.cwd().parents]:
        env = parent / ".env"
        if env.is_file():
            for line in env.read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    if v.strip():
                        os.environ.setdefault(k.strip(), v.strip())
            return


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hotloop-agent", description=__doc__)
    p.add_argument("--session", required=True, type=Path)
    p.add_argument("--task-readme", required=True, type=Path)
    p.add_argument("--model", required=True, help="litellm name, e.g. openai/gpt-5.5")
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-turns", type=int, default=None)
    p.add_argument("--max-completion-tokens", type=int, default=400_000)
    p.add_argument("--command-timeout", type=int, default=900)
    a = p.parse_args(argv)
    _load_env_file()
    session = json.loads(a.session.read_text())
    env = SessionEnvironment(exec_prefix=session["exec_prefix"],
                             exec_remote_template=session.get("exec_remote_template"),
                             timeout=a.command_timeout)  # fmt: skip
    agent = BudgetedAgent(
        build_model(a.model, a.reasoning_effort), env,
        system_template=SYSTEM, instance_template=INSTANCE, step_limit=0, cost_limit=0,
        max_turns=a.max_turns or session["budget"]["max_turns"],
        max_completion_tokens=a.max_completion_tokens,
        wall_time_limit_seconds=int(session["budget"]["wall_seconds"]),
        output_path=a.out / "mini_trajectory.json", events_path=a.out / "trajectory.jsonl",
        run_id=session["run_id"],
    )  # fmt: skip
    info = agent.run(task=a.task_readme.read_text())
    (a.out / "agent_summary.json").write_text(json.dumps(
        {"exit_status": info.get("exit_status"), "turns": agent.n_calls, **agent.tokens,
         "model": a.model, "reasoning_effort": a.reasoning_effort}, indent=1))  # fmt: skip
    print(
        json.dumps({"exit_status": info.get("exit_status"), "turns": agent.n_calls, **agent.tokens})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
