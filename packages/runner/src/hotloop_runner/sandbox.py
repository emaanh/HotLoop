"""Sandbox lifecycle on a remote Docker host with an NVIDIA GPU, reached over SSH (D-13).

    hotloop-run prepare  --host IP                     build the image, lock clocks (once per VM)
    hotloop-run start    --host IP --task DIR --run-id ID   -> prints session JSON (incl. exec_prefix)
    hotloop-run finish   --session FILE --out DIR [--evals 3]  collect submission, evaluate, tear down

Trust boundaries
  * Only a task's PUBLIC files are ever copied to the host. `private/` never leaves the operator.
  * The agent container has **no network** (`--network none`), a read-only task mount and one
    writable submission dir. API keys never reach the host: the agent loop runs on the operator's
    machine and only *commands* cross the SSH connection.
  * Scoring runs in a *fresh* container from the same image with its own read-only copy of the
    submission, `--network none`, capabilities dropped - not in the container the agent touched.
  * Every command is wrapped in an in-container `timeout -k`, because killing a local
    `docker exec`/ssh client does not kill the process inside the container.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

from hotloop_schemas import PUBLIC_FILES, Result, load_task_spec

IMAGE = "hotloop-bench:0.1"
REMOTE_ROOT = "/home/ubuntu/hotloop-runs"
SSH_OPTS = [
    "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20",
    "-o", "ServerAliveInterval=30", "-o", "ControlMaster=auto", "-o", "ControlPersist=600",
    "-o", "ControlPath=~/.ssh/hotloop-%C",
]  # fmt: skip
REPO = Path(__file__).resolve().parents[4]


def ssh(host: str, command: str, *, timeout: float | None = None, check=True) -> str:
    r = subprocess.run(
        ["ssh", *SSH_OPTS, f"ubuntu@{host}", command],
        capture_output=True, text=True, timeout=timeout, check=False,
    )  # fmt: skip
    if check and r.returncode != 0:
        raise RuntimeError(f"ssh {host}: {command[:120]!r} -> {r.returncode}\n{r.stderr[-2000:]}")
    return r.stdout


def rsync(host: str, src: str, dst: str, *extra: str) -> None:
    subprocess.run(
        ["rsync", "-az", "-e", "ssh " + " ".join(SSH_OPTS), *extra, src, f"ubuntu@{host}:{dst}"],
        check=True, capture_output=True,
    )  # fmt: skip


def prepare(host: str) -> None:
    ssh(host, f"mkdir -p {REMOTE_ROOT}/build/packages")
    for pkg in ("schemas", "evaluator"):
        rsync(host, str(REPO / "packages" / pkg), f"{REMOTE_ROOT}/build/packages/",
              "--delete", "--exclude", "__pycache__", "--exclude", "tests")  # fmt: skip
    rsync(host, str(REPO / "image") + "/", f"{REMOTE_ROOT}/build/")
    print(
        ssh(
            host,
            f"cd {REMOTE_ROOT}/build && sudo docker build -q -t {IMAGE} . 2>&1 | tail -3",
            timeout=3600,
        )
    )
    out = ssh(
        host,
        "MAX=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1); "
        'sudo nvidia-smi -pm 1 >/dev/null; sudo nvidia-smi -lgc "$MAX,$MAX" | head -1',
    )
    print(out.strip())


def start(host: str, task_dir: Path, run_id: str) -> dict:
    spec = load_task_spec(task_dir)
    base = f"{REMOTE_ROOT}/{run_id}"
    ssh(
        host,
        f"rm -rf {base} && mkdir -p {base}/task {base}/submission && chmod 777 {base}/submission",
    )
    for name in PUBLIC_FILES:  # and nothing else - never private/
        rsync(host, str(task_dir / name), f"{base}/task/{name}")
    name = f"hotloop-agent-{run_id}"
    ssh(
        host,
        f"sudo docker rm -f {name} >/dev/null 2>&1; sudo docker run -d --name {name} --gpus all "
        f"--network none --shm-size 8g -e HOTLOOP_CLOCKS_LOCKED=1 -e HOTLOOP_PROVIDER=lambda "
        f"-e HOTLOOP_IMAGE={IMAGE} -v {base}/task:/workspace/task:ro "
        f"-v {base}/submission:/workspace/submission -w /workspace {IMAGE} sleep infinity",
    )
    session = {
        "run_id": run_id, "host": host, "container": name, "remote_base": base,
        "task_id": spec.id, "task_dir": str(task_dir), "started": time.time(),
        "budget": spec.budget.model_dump(),
        # argv prefix; the agent side appends: <timeout seconds> <command string>
        "exec_prefix": ["ssh", *SSH_OPTS, f"ubuntu@{host}"],
        "exec_remote_template": f"sudo docker exec -w /workspace {name} timeout -k 5 {{timeout}} bash -lc {{command}}",
    }  # fmt: skip
    return session


def finish(session: dict, out_dir: Path, evals: int) -> list[Result]:
    host, base, name = session["host"], session["remote_base"], session["container"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ssh(host, f"sudo docker rm -f {name} >/dev/null 2>&1 || true", check=False)
    # the agent may have left root-owned files; snapshot what it produced
    # The scoring container drops all capabilities, so its root cannot bypass file permissions:
    # make the submission world-readable and the results dir world-writable.
    ssh(
        host,
        f"sudo chown -R ubuntu:ubuntu {base}/submission; chmod -R a+rX {base}/submission; "
        f"mkdir -p {base}/results; chmod 777 {base}/results",
    )
    subprocess.run(
        ["rsync", "-az", "-e", "ssh " + " ".join(SSH_OPTS), "--max-size=50m",
         f"ubuntu@{host}:{base}/submission/", str(out_dir / "submission") + "/"],
        check=False, capture_output=True,
    )  # fmt: skip
    results = []
    for i in range(evals):
        ssh(
            host,
            f"sudo docker run --rm --gpus all --network none --shm-size 8g --cap-drop ALL "
            f"--security-opt no-new-privileges -e HOTLOOP_CLOCKS_LOCKED=1 -e HOTLOOP_PROVIDER=lambda "
            f"-e HOTLOOP_IMAGE={IMAGE} -v {base}/task:/task:ro -v {base}/submission:/submission_src:ro "
            f"-v {base}/results:/results {IMAGE} bash -lc "
            + shlex.quote(
                "cp -r /submission_src /tmp/submission && hotloop-eval --task /task "
                f"--submission /tmp/submission --out /results/result_{i}.json"
            ),
            timeout=3 * 3600, check=False,
        )  # fmt: skip
        text = ssh(host, f"cat {base}/results/result_{i}.json 2>/dev/null || true", check=False)
        if text.strip():
            (out_dir / f"result_{i}.json").write_text(text)
            results.append(Result.model_validate_json(text))
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hotloop-run", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("prepare")
    sp.add_argument("--host", required=True)
    sp = sub.add_parser("start")
    sp.add_argument("--host", required=True)
    sp.add_argument("--task", required=True, type=Path)
    sp.add_argument("--run-id", required=True)
    sp.add_argument("--session-out", required=True, type=Path)
    sp = sub.add_parser("finish")
    sp.add_argument("--session", required=True, type=Path)
    sp.add_argument("--out", required=True, type=Path)
    sp.add_argument("--evals", type=int, default=3)
    a = p.parse_args(argv)
    if a.cmd == "prepare":
        prepare(a.host)
    elif a.cmd == "start":
        session = start(a.host, a.task, a.run_id)
        a.session_out.parent.mkdir(parents=True, exist_ok=True)
        a.session_out.write_text(json.dumps(session, indent=1))
        print(a.session_out)
    elif a.cmd == "finish":
        results = finish(json.loads(a.session.read_text()), a.out, a.evals)
        for r in results:
            print(f"{r.task_id}: {r.status} score={r.score:.3f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
