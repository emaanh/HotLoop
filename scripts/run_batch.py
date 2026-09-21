"""Run (task x rep) trajectories across several GPU hosts, one trajectory per host at a time.

Resumable: a run directory with summary.json is skipped. Order is rep-major, so a partial batch
still covers every task.

usage: python scripts/run_batch.py --hosts ip1,ip2 --tasks benchmark/dev-v0 --reps 3 \
           --model openai/gpt-5.5 --reasoning-effort medium --out runs/m3
"""

from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


def worker(host: str, jobs: queue.Queue, a, log_dir: Path) -> None:
    while True:
        try:
            task, rep = jobs.get_nowait()
        except queue.Empty:
            return
        cmd = [sys.executable, "scripts/run_trajectory.py", "--host", host, "--task", str(task),
               "--model", a.model, "--rep", str(rep), "--out", str(a.out), "--evals", str(a.evals),
               "--max-completion-tokens", str(a.max_completion_tokens)]  # fmt: skip
        if a.reasoning_effort:
            cmd += ["--reasoning-effort", a.reasoning_effort]
        t0 = time.time()
        with open(log_dir / f"{task.name}--r{rep}.log", "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
        print(f"[{time.strftime('%H:%M:%S')}] {host} {task.name} r{rep} rc={rc} {time.time() - t0:.0f}s "
              f"({jobs.qsize()} left)", flush=True)  # fmt: skip


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hosts", required=True)
    p.add_argument("--tasks", required=True, type=Path)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--model", required=True)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--out", type=Path, default=Path("runs/m3"))
    p.add_argument("--evals", type=int, default=3)
    p.add_argument("--max-completion-tokens", type=int, default=400_000)
    a = p.parse_args()
    tasks = sorted(d for d in a.tasks.iterdir() if (d / "task.toml").is_file())
    jobs: queue.Queue = queue.Queue()
    for rep in range(a.reps):
        for t in tasks:
            jobs.put((t, rep))
    log_dir = a.out / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    threads = [
        threading.Thread(target=worker, args=(h, jobs, a, log_dir)) for h in a.hosts.split(",")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("BATCH DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
