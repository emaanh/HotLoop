"""Run (task x rep) trajectories across GPU hosts, one trajectory per host at a time.

Owns the whole lifecycle so that no instance can outlive the batch:
  * `--launch N` launches N instances, builds the image on each, and **terminates them in a
    `finally`** - on success, on crash, on Ctrl-C, and when the model becomes unreachable.
  * `--deadline-hours H` is a hard stop: at H hours everything launched here is terminated and the
    process exits, whatever state the batch is in.
  * Run it under `caffeinate -i` on a laptop: the agent loop is local, a sleeping machine stalls
    the batch while the instances keep billing.
Resumable: a run directory with summary.json is skipped. Order is rep-major, so a partial batch
still covers every task.

usage: caffeinate -i python scripts/run_batch.py --launch 4 --deadline-hours 7 \
           --tasks benchmark/dev-v0 --reps 3 --model openai/gpt-5.5 --out runs/m3
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ABORT = threading.Event()
LAUNCHED: list[str] = []


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def vm(*args: str) -> str:
    return subprocess.run(
        ["hotloop-vm", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def terminate_launched(why: str) -> None:
    if not LAUNCHED:
        return
    log(f"terminating {len(LAUNCHED)} instance(s): {why}")
    for attempt in range(5):
        try:
            vm("terminate", *LAUNCHED)
            log("terminate request accepted")
            return
        except subprocess.CalledProcessError as e:
            log(f"terminate attempt {attempt + 1} failed: {e.stderr[-200:]}")
            time.sleep(15)
    log("!!! COULD NOT TERMINATE - check cloud.lambda.ai/instances NOW: " + " ".join(LAUNCHED))


def launch_and_prepare(n: int, instance_type: str) -> list[str]:
    for i in range(n):
        try:
            LAUNCHED.append(vm("launch", instance_type, "--name", f"hotloop-batch-{i}"))
        except subprocess.CalledProcessError as e:
            log(f"launch {i} failed (capacity?): {e.stderr[-160:]}")
    if not LAUNCHED:
        raise SystemExit("no instances could be launched")
    hosts = [vm("wait", iid) for iid in LAUNCHED]
    log(f"instances up: {hosts}")
    ready: list[str] = []

    def prep(host: str) -> None:
        for _ in range(8):  # sshd comes up a little after the API says 'active'
            if (
                subprocess.run(
                    ["hotloop-run", "prepare", "--host", host], capture_output=True, check=False
                ).returncode
                == 0
            ):
                ready.append(host)
                return
            time.sleep(20)
        log(f"prepare failed on {host}; not using it")

    threads = [threading.Thread(target=prep, args=(h,)) for h in hosts]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if not ready:
        raise SystemExit("no host could be prepared")
    log(f"prepared: {ready}")
    return ready


def worker(host: str, jobs: queue.Queue, a, log_dir: Path) -> None:
    while not ABORT.is_set():
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
        run_log = log_dir / f"{task.name}--r{rep}.log"
        with open(run_log, "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False).returncode
        if rc == 3:
            ABORT.set()
            log(f"ABORTING BATCH: model unreachable (see {run_log})")
        log(f"{host} {task.name} r{rep} rc={rc} {time.time() - t0:.0f}s ({jobs.qsize()} left)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--hosts", help="comma-separated, already prepared; NOT terminated by this script"
    )
    g.add_argument(
        "--launch", type=int, help="launch this many instances and terminate them at the end"
    )
    p.add_argument("--instance-type", default="gpu_1x_a100_sxm4")
    p.add_argument("--deadline-hours", type=float, default=8.0)
    p.add_argument("--tasks", required=True, type=Path)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--model", required=True)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--out", type=Path, default=Path("runs/m3"))
    p.add_argument("--evals", type=int, default=3)
    p.add_argument("--max-completion-tokens", type=int, default=400_000)
    a = p.parse_args()

    def deadline() -> None:
        log(f"DEADLINE of {a.deadline_hours}h reached")
        terminate_launched("deadline")
        os._exit(4)

    timer = threading.Timer(a.deadline_hours * 3600, deadline)
    timer.daemon = True
    timer.start()
    try:
        hosts = a.hosts.split(",") if a.hosts else launch_and_prepare(a.launch, a.instance_type)
        tasks = sorted(d for d in a.tasks.iterdir() if (d / "task.toml").is_file())
        jobs: queue.Queue = queue.Queue()
        for rep in range(a.reps):
            for t in tasks:
                if not list(a.out.glob(f"{t.name}--*--r{rep}/summary.json")):
                    jobs.put((t, rep))
        log(f"{jobs.qsize()} trajectories to run on {len(hosts)} host(s)")
        log_dir = a.out / "_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        threads = [threading.Thread(target=worker, args=(h, jobs, a, log_dir)) for h in hosts]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        terminate_launched("batch finished" if not ABORT.is_set() else "batch aborted")
    log("BATCH ABORTED" if ABORT.is_set() else "BATCH DONE")
    return 3 if ABORT.is_set() else 0


if __name__ == "__main__":
    sys.exit(main())
