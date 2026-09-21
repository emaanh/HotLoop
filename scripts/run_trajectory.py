"""One trajectory, end to end, by gluing the component CLIs (they share no code):

    hotloop-run start  ->  hotloop-agent  ->  hotloop-run finish

usage: python scripts/run_trajectory.py --host IP --task tasks/<id> --model openai/gpt-5.5 \
           --rep 0 --out runs/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def sh(*argv: str, check=True) -> int:
    print("+", " ".join(argv), flush=True)
    return subprocess.run(argv, check=check).returncode


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", required=True)
    p.add_argument("--task", required=True, type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--evals", type=int, default=3)
    p.add_argument("--max-turns", type=int, default=None)
    p.add_argument("--max-completion-tokens", type=int, default=400_000)
    a = p.parse_args()

    tag = a.model.split("/")[-1].replace(".", "_")
    run_id = f"{a.task.name}--{tag}--r{a.rep}"
    out = a.out / run_id
    if (out / "summary.json").exists():
        print(f"{run_id}: already done, skipping")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    session = out / "session.json"
    t0 = time.time()
    sh("hotloop-run", "start", "--host", a.host, "--task", str(a.task), "--run-id", run_id,
       "--session-out", str(session))  # fmt: skip
    agent_cmd = ["hotloop-agent", "--session", str(session), "--task-readme",
                 str(a.task / "AGENT_README.md"), "--model", a.model, "--out", str(out),
                 "--max-completion-tokens", str(a.max_completion_tokens)]  # fmt: skip
    if a.reasoning_effort:
        agent_cmd += ["--reasoning-effort", a.reasoning_effort]
    if a.max_turns:
        agent_cmd += ["--max-turns", str(a.max_turns)]
    agent_rc = sh(*agent_cmd, check=False)
    t_agent = time.time() - t0
    if agent_rc != 0:  # API/transport died (possibly mid-run): not a result. Tear down, keep the
        # partial artefacts aside for forensics, and leave no summary.json so it is retried.
        sh(
            "hotloop-run",
            "finish",
            "--session",
            str(session),
            "--out",
            str(out),
            "--evals",
            "0",
            check=False,
        )
        out.rename(out.with_name(f"{out.name}__crashed_{int(time.time())}"))
        print(f"{run_id}: INFRA FAILURE (agent rc={agent_rc}), not scored, will be retried")
        return 3
    sh(
        "hotloop-run",
        "finish",
        "--session",
        str(session),
        "--out",
        str(out),
        "--evals",
        str(a.evals),
    )

    results = [json.loads(f.read_text()) for f in sorted(out.glob("result_*.json"))]
    agent = (
        json.loads((out / "agent_summary.json").read_text())
        if (out / "agent_summary.json").exists()
        else {}
    )
    summary = {
        "run_id": run_id, "task": a.task.name, "model": a.model, "rep": a.rep,
        "agent_rc": agent_rc, "agent_seconds": round(t_agent), "total_seconds": round(time.time() - t0),
        "agent": agent, "statuses": [r["status"] for r in results], "scores": [r["score"] for r in results],
    }  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
