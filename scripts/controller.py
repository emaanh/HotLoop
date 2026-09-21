"""Run a batch from an always-on **controller** VM instead of the operator's laptop (D-30, D-31).

The controller holds the API keys and drives the bench VMs; bench VMs (where untrusted agents have
a shell) still never see a key. The operator's machine can sleep or disconnect at any time:

  * the batch's own wall-clock deadline terminates the bench instances;
  * an independent dead-man process on the controller terminates **every** instance, itself
    included, at a hard cap - even if the batch process hangs or dies;
  * results are synced every 5 minutes to a persistent Lambda filesystem, so they outlive the VMs.

    controller.py up --batch-args "..."   launch controller, bootstrap, start the batch
    controller.py status                   tail the batch log, list instances
    controller.py fetch                    copy results back into runs/
    controller.py down                     fetch, then terminate everything
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STATE = REPO / "runs" / "controller.json"
FS_NAME, FS_REGION = "hotloop-results", "us-east-1"
SSH = [
    "ssh",
    "-o",
    "StrictHostKeyChecking=accept-new",
    "-o",
    "ConnectTimeout=20",
    "-o",
    "ServerAliveInterval=30",
]


def vm(*args: str) -> str:
    return subprocess.run(
        ["hotloop-vm", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def ssh(host: str, cmd: str, check=True, capture=True) -> str:
    r = subprocess.run(
        [*SSH, f"ubuntu@{host}", cmd], capture_output=capture, text=True, check=False
    )
    if check and r.returncode != 0:
        raise SystemExit(
            f"remote command failed ({r.returncode}): {cmd[:100]}\n{(r.stderr or '')[-1500:]}"
        )
    return r.stdout if capture else ""


def rsync_to(host: str, *paths: str, dest: str, extra: tuple[str, ...] = ()) -> None:
    subprocess.run(
        ["rsync", "-az", "-e", " ".join(SSH), *extra, *paths, f"ubuntu@{host}:{dest}"], check=True
    )


def up(a) -> None:
    if STATE.exists():
        raise SystemExit(f"{STATE} exists: a controller may already be up (run `status` or `down`)")
    iid = vm("launch", a.instance_type, "--region", FS_REGION, "--name", "hotloop-controller",
             "--file-system", FS_NAME)  # fmt: skip
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"id": iid, "launched": time.time()}))
    host = vm("wait", iid)
    key_name = f"hotloop-controller-{iid[:8]}"
    STATE.write_text(
        json.dumps({"id": iid, "host": host, "key_name": key_name, "launched": time.time()})
    )
    for _ in range(20):
        if (
            subprocess.run(
                [*SSH, f"ubuntu@{host}", "true"], capture_output=True, check=False
            ).returncode
            == 0
        ):
            break
        time.sleep(10)
    print(f"controller {iid[:8]} at {host}", flush=True)

    ssh(host, "mkdir -p ~/hotloop/runs")
    rsync_to(host, *(str(REPO / p) for p in ("packages", "scripts", "benchmark", "image", "pyproject.toml",
                                              "uv.lock", ".python-version")), dest="~/hotloop/",
             extra=("--delete", "--exclude", "__pycache__", "--exclude", ".venv"))  # fmt: skip
    if (REPO / a.out).is_dir():  # completed runs, so the batch resumes instead of repeating them
        rsync_to(host, str(REPO / a.out) + "/", dest=f"~/hotloop/{a.out}/",
                 extra=("--exclude", "_*", "--exclude", "*__crashed_*"))  # fmt: skip
    rsync_to(host, str(REPO / ".env"), dest="~/hotloop/.env")
    ssh(host, "chmod 600 ~/hotloop/.env")

    res = f"/lambda/nfs/{FS_NAME}/{Path(a.out).name}"
    cap_h = a.deadline_hours + a.grace_hours
    if a.drill:  # exercise the dead-man alone: no batch, fires after 3 minutes
        cap_h = 0.05
    batch = (f"python scripts/run_batch.py --launch {a.launch} --ssh-key {key_name} "
             f"--deadline-hours {a.deadline_hours} --out {a.out} {a.batch_args}")  # fmt: skip
    script = f"""set -e
command -v ~/.local/bin/uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH=$HOME/.local/bin:$PATH
cd ~/hotloop
[ -d ~/cvenv ] || uv venv -q --python 3.12 ~/cvenv
. ~/cvenv/bin/activate
uv pip install -q -e packages/schemas -e packages/runner -e packages/agents
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -q -t ed25519 -N '' -f ~/.ssh/id_ed25519
hotloop-vm ssh-key-add {key_name} ~/.ssh/id_ed25519.pub >/dev/null
mkdir -p {res}
cat > ~/batch.sh <<'EOS'
#!/usr/bin/env bash
export PATH=$HOME/.local/bin:$PATH MSWEA_SILENT_STARTUP=1
. ~/cvenv/bin/activate; cd ~/hotloop
( while true; do sleep 300; rsync -a {a.out}/ {res}/ 2>/dev/null; cp ~/batch.log {res}/ 2>/dev/null; done ) &
SYNC=$!
{batch} > ~/batch.log 2>&1; echo $? > ~/batch.rc
kill $SYNC; rsync -a {a.out}/ {res}/; cp ~/batch.log ~/batch.rc {res}/; touch ~/DONE
EOS
cat > ~/deadman.sh <<'EOS'
#!/usr/bin/env bash
# Independent of the batch process: at the hard cap, save results and terminate EVERYTHING.
export PATH=$HOME/.local/bin:$PATH
. ~/cvenv/bin/activate; cd ~/hotloop
sleep {int(cap_h * 3600)}
rsync -a {a.out}/ {res}/ 2>/dev/null; cp ~/batch.log {res}/ 2>/dev/null
echo "deadman fired $(date -u)" >> {res}/deadman.log
for i in 1 2 3 4 5; do hotloop-vm terminate-all && break; sleep 30; done
EOS
chmod +x ~/batch.sh ~/deadman.sh
nohup ~/deadman.sh > ~/deadman.out 2>&1 &
{"" if a.drill else "nohup ~/batch.sh > /dev/null 2>&1 &"}
echo started
"""
    print(ssh(host, script).strip()[-300:], flush=True)
    print(f"batch deadline {a.deadline_hours} h; dead-man terminates everything at {cap_h} h "
          f"({time.strftime('%a %H:%M', time.localtime(time.time() + cap_h * 3600))} local)")  # fmt: skip


def state() -> dict:
    if not STATE.exists():
        raise SystemExit("no controller state; nothing is up (check `hotloop-vm list`)")
    return json.loads(STATE.read_text())


def status(_a) -> None:
    s = state()
    print(ssh(s["host"], "ls ~/DONE 2>/dev/null && echo '== batch finished, rc:' $(cat ~/batch.rc); "
                         "tail -n 12 ~/batch.log | cut -c1-160", check=False))  # fmt: skip
    print(vm("list"))


def fetch(a) -> None:
    s = state()
    subprocess.run(["rsync", "-az", "-e", " ".join(SSH), f"ubuntu@{s['host']}:~/hotloop/{a.out}/",
                    str(REPO / a.out) + "/"], check=True)  # fmt: skip
    subprocess.run(["rsync", "-az", "-e", " ".join(SSH), f"ubuntu@{s['host']}:~/batch.log",
                    str(REPO / a.out / "controller_batch.log")], check=False)  # fmt: skip
    print("fetched into", a.out)


def down(a) -> None:
    try:
        fetch(a)
    finally:
        print(vm("terminate-all") or "terminate-all requested")
        STATE.unlink(missing_ok=True)


def wait(a) -> None:
    """Block until the batch finishes, then fetch results and terminate everything."""
    s = state()
    while True:
        r = subprocess.run(
            [*SSH, f"ubuntu@{s['host']}", "test -f ~/DONE"], capture_output=True, check=False
        )
        if r.returncode == 0:
            break
        time.sleep(60)
    down(a)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("up")
    u.add_argument("--instance-type", default="gpu_1x_a10")
    u.add_argument("--launch", type=int, default=4)
    u.add_argument("--deadline-hours", type=float, default=7.0)
    u.add_argument("--grace-hours", type=float, default=3.0)
    u.add_argument("--batch-args", default="", help="remaining run_batch.py arguments")
    u.add_argument("--drill", action="store_true", help="no batch; dead-man fires after 3 min")
    for name in ("status", "fetch", "down", "wait"):
        sub.add_parser(name)
    for sp in sub.choices.values():
        sp.add_argument("--out", default="runs/m3")
    a = p.parse_args()
    {"up": up, "status": status, "fetch": fetch, "down": down, "wait": wait}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
