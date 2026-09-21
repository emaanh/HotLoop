"""The adapter around mini-swe-agent, driven by a scripted model and a local shell."""

import json

from hotloop_agents.mini import INSTANCE, SUBMIT_SENTINEL, SYSTEM, BudgetedAgent, SessionEnvironment
from hotloop_schemas import TrajectoryEvent


class ScriptedModel:
    """Implements mini-swe-agent's Model protocol; replays a list of bash commands."""

    class _Cfg:
        model_name = "scripted"

    def __init__(self, commands, out_tokens=100):
        self.config, self.commands, self.out_tokens, self.i = (
            self._Cfg(),
            list(commands),
            out_tokens,
            0,
        )

    def query(self, messages, **kw):
        cmd = self.commands[min(self.i, len(self.commands) - 1)]
        self.i += 1
        return {"role": "assistant", "content": "", "usage": {"input_tokens": 1000, "output_tokens": self.out_tokens},
                "extra": {"actions": [{"command": cmd}], "cost": 0.0}}  # fmt: skip

    def format_message(self, **kw):
        return dict(kw)

    def format_observation_messages(self, message, outputs, template_vars=None):
        return [{"role": "user", "content": json.dumps({"rc": o["returncode"], "out": o["output"],
                 "turns_left": template_vars["turns_left"]})} for o in outputs]  # fmt: skip

    def get_template_vars(self, **kw):
        return {}

    def serialize(self):
        return {}


def make(tmp_path, commands, **over):
    env = SessionEnvironment(exec_prefix=["bash", "-lc"], timeout=20)
    kwargs = dict(system_template=SYSTEM, instance_template=INSTANCE, step_limit=0, cost_limit=0,
                  max_turns=10, max_completion_tokens=10_000, output_path=tmp_path / "mini.json",
                  events_path=tmp_path / "trajectory.jsonl", run_id="t1")  # fmt: skip
    return BudgetedAgent(
        ScriptedModel(commands, over.pop("out_tokens", 100)), env, **{**kwargs, **over}
    )


def observations(agent):
    return [
        json.loads(m["content"])
        for m in agent.messages
        if str(m.get("content", "")).startswith("{")
    ]


def events(tmp_path):
    return [
        TrajectoryEvent.model_validate_json(x)
        for x in (tmp_path / "trajectory.jsonl").read_text().splitlines()
    ]


def test_runs_commands_then_submits(tmp_path):
    agent = make(
        tmp_path,
        [f"echo hi > {tmp_path}/f.txt", f"cat {tmp_path}/f.txt", f"echo {SUBMIT_SENTINEL}"],
    )
    info = agent.run(task="do the thing")
    assert info["exit_status"] == "Submitted" and agent.n_calls == 3
    assert "hi" in observations(agent)[1]["out"]
    ev = events(tmp_path)
    assert [e.type for e in ev][:4] == ["run_start", "model_request", "model_response", "command"]
    assert ev[-1].type == "run_end" and ev[-1].payload["reason"] == "submitted"
    assert [e.seq for e in ev] == list(range(len(ev)))
    assert agent.tokens["output_tokens"] == 300 and agent.tokens["input_tokens"] == 3000
    first = agent.messages[1]["content"]
    assert "do the thing" in first and SUBMIT_SENTINEL in first and "10 turns" in first


def test_turn_budget_stops_the_run(tmp_path):
    agent = make(tmp_path, ["true"], max_turns=3)
    assert agent.run(task="x")["exit_status"] == "BudgetExhausted" and agent.n_calls == 3
    assert any(e.type == "budget" and e.payload["limit"] == "turns" for e in events(tmp_path))


def test_token_budget_stops_the_run(tmp_path):
    agent = make(tmp_path, ["true"], max_completion_tokens=250, out_tokens=100)
    assert agent.run(task="x")["exit_status"] == "BudgetExhausted" and agent.n_calls == 3


def test_remaining_budget_is_shown_each_turn(tmp_path):
    agent = make(tmp_path, ["true", "true", f"echo {SUBMIT_SENTINEL}"], max_turns=5)
    agent.run(task="x")
    shown = [o["turns_left"] for o in observations(agent)]
    assert shown == [4, 3]


def test_remote_template_quotes_the_command():
    env = SessionEnvironment(exec_prefix=["ssh", "h"], timeout=30,
                             exec_remote_template="sudo docker exec c timeout -k 5 {timeout} bash -lc {command}")  # fmt: skip
    argv = env._argv("echo 'a b' && ls; rm -rf \"$X\"")
    assert argv[:2] == ["ssh", "h"] and len(argv) == 3
    assert argv[2].startswith("sudo docker exec c timeout -k 5 30 bash -lc '")
    import shlex

    assert shlex.split(argv[2])[-1] == "echo 'a b' && ls; rm -rf \"$X\""


def test_failing_and_slow_commands_are_observations_not_crashes(tmp_path):
    agent = make(tmp_path, ["exit 7", f"echo {SUBMIT_SENTINEL}"])
    agent.run(task="x")
    obs = [e for e in events(tmp_path) if e.type == "observation"]
    assert obs[0].payload["returncode"] == 7 and obs[0].payload["duration_s"] is not None


def test_time_limit_exit_still_leaves_a_run_end_event(tmp_path):
    agent = make(tmp_path, ["sleep 1.2"], wall_time_limit_seconds=1)
    assert agent.run(task="x")["exit_status"] == "TimeExceeded"
    assert (
        events(tmp_path)[-1].type == "run_end"
        and events(tmp_path)[-1].payload["reason"] == "TimeExceeded"
    )


def test_api_failure_mid_run_is_an_infra_failure_not_a_result(tmp_path, monkeypatch):
    """The quota incident: the API died mid-trajectory and the half-done work got scored."""
    from hotloop_agents import mini

    class Dies(ScriptedModel):
        def query(self, messages, **kw):
            if self.i >= 2:
                raise RuntimeError("insufficient_quota")
            return super().query(messages, **kw)

    monkeypatch.setattr(mini, "build_model", lambda *a, **k: Dies(["true"]))
    session = tmp_path / "session.json"
    session.write_text(json.dumps({"run_id": "t", "exec_prefix": ["bash", "-lc"], "exec_remote_template": None,
                                   "budget": {"max_turns": 10, "wall_seconds": 60}}))  # fmt: skip
    readme = tmp_path / "README.md"
    readme.write_text("task")
    rc = mini.main(
        [
            "--session",
            str(session),
            "--task-readme",
            str(readme),
            "--model",
            "x",
            "--out",
            str(tmp_path / "o"),
        ]
    )
    assert rc == mini.INFRA_FAILURE
    assert not (tmp_path / "o" / "agent_summary.json").exists()
    last = [json.loads(x) for x in (tmp_path / "o" / "trajectory.jsonl").read_text().splitlines()][
        -1
    ]
    assert last["type"] == "run_end" and last["payload"]["reason"] == "infra_failure"
