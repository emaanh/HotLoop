"""Thin Lambda Cloud client (D-13, D-20). stdlib only.

The API key stays on the operator's machine: nothing here ever copies it to a VM, because bench
VMs host untrusted agents. Consequence: there is no on-box auto-terminate, so every launch is
recorded in a local ledger and `reap` exists to kill anything older than a deadline.

    hotloop-vm types | launch <type> [--region R] [--name N] | list | wait <id> | terminate <id>...
               | terminate-all | reap --older-than-hours H
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://cloud.lambda.ai/api/v1"
LEDGER = Path(os.environ.get("HOTLOOP_VM_LEDGER", Path.home() / ".hotloop" / "vm_ledger.jsonl"))


def _key() -> str:
    key = os.environ.get("LAMBDA_API_KEY")
    if not key:
        for parent in [Path.cwd(), *Path.cwd().parents]:
            env = parent / ".env"
            if env.is_file():
                for line in env.read_text().splitlines():
                    if line.startswith("LAMBDA_API_KEY="):
                        key = line.split("=", 1)[1].strip()
                break
    if not key:
        raise SystemExit("LAMBDA_API_KEY not set (env or .env)")
    return key


def _request(method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        API + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Bearer {_key()}",
            "Content-Type": "application/json",
            # Lambda's Cloudflare front rejects urllib's default agent (403, error 1010)
            "User-Agent": "hotloop-vm/0.1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raise SystemExit(
            f"Lambda API {method} {path} -> {e.code}: {e.read().decode()[:500]}"
        ) from e


def _ledger(event: str, **fields) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"ts": time.time(), "event": event, **fields}) + "\n")


def instance_types() -> dict:
    return _request("GET", "/instance-types")["data"]


def instances() -> list[dict]:
    return _request("GET", "/instances")["data"]


def launch(
    type_name: str,
    region: str | None,
    name: str,
    ssh_key: str,
    file_systems: list[str] | None = None,
) -> str:
    if region is None:
        regions = instance_types()[type_name]["regions_with_capacity_available"]
        if not regions:
            raise SystemExit(f"no capacity for {type_name} in any region right now")
        region = regions[0]["name"]
    data = _request(
        "POST",
        "/instance-operations/launch",
        {
            "region_name": region,
            "instance_type_name": type_name,
            "ssh_key_names": [ssh_key],
            "quantity": 1,
            "name": name,
            **({"file_system_names": file_systems} if file_systems else {}),
        },
    )["data"]
    (iid,) = data["instance_ids"]
    price = instance_types()[type_name]["instance_type"]["price_cents_per_hour"] / 100
    _ledger("launch", id=iid, type=type_name, region=region, name=name, usd_per_hour=price)
    return iid


def add_ssh_key(name: str, public_key: str) -> str:
    return _request("POST", "/ssh-keys", {"name": name, "public_key": public_key.strip()})["data"][
        "id"
    ]


def wait_active(iid: str, timeout_s: float = 900) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        inst = _request("GET", f"/instances/{iid}")["data"]
        if inst["status"] == "active" and inst.get("ip"):
            return inst
        if inst["status"] in ("terminated", "terminating", "unhealthy"):
            raise SystemExit(f"instance {iid} is {inst['status']}")
        time.sleep(10)
    raise SystemExit(f"instance {iid} not active after {timeout_s:.0f}s")


def terminate(ids: list[str]) -> None:
    if not ids:
        return
    _request("POST", "/instance-operations/terminate", {"instance_ids": ids})
    for iid in ids:
        _ledger("terminate", id=iid)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hotloop-vm", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("types")
    lp = sub.add_parser("launch")
    lp.add_argument("type")
    lp.add_argument("--region")
    lp.add_argument("--name", default="hotloop")
    lp.add_argument("--ssh-key", default="emaan-macbook-hotloop")
    lp.add_argument("--file-system", action="append", default=None, help="persistent FS name(s)")
    kp = sub.add_parser("ssh-key-add")
    kp.add_argument("name")
    kp.add_argument("public_key_file", type=Path)
    sub.add_parser("list")
    wp = sub.add_parser("wait")
    wp.add_argument("id")
    tp = sub.add_parser("terminate")
    tp.add_argument("ids", nargs="+")
    sub.add_parser("terminate-all")
    rp = sub.add_parser("reap")
    rp.add_argument("--older-than-hours", type=float, required=True)
    a = p.parse_args(argv)

    if a.cmd == "types":
        for name, v in sorted(
            instance_types().items(), key=lambda kv: kv[1]["instance_type"]["price_cents_per_hour"]
        ):
            regions = [r["name"] for r in v["regions_with_capacity_available"]]
            cents = v["instance_type"]["price_cents_per_hour"]
            print(f"${cents / 100:6.2f}/h  {name:30s} {', '.join(regions) or '-'}")
    elif a.cmd == "launch":
        print(launch(a.type, a.region, a.name, a.ssh_key, a.file_system))
    elif a.cmd == "ssh-key-add":
        print(add_ssh_key(a.name, a.public_key_file.read_text()))
    elif a.cmd == "list":
        for i in instances():
            print(i["id"], i["status"], i.get("ip"), i["instance_type"]["name"], i.get("name"))
    elif a.cmd == "wait":
        inst = wait_active(a.id)
        print(inst["ip"])
    elif a.cmd == "terminate":
        terminate(a.ids)
    elif a.cmd == "terminate-all":
        terminate([i["id"] for i in instances()])
    elif a.cmd == "reap":
        launched = {}
        if LEDGER.exists():
            for line in LEDGER.read_text().splitlines():
                rec = json.loads(line)
                if rec["event"] == "launch":
                    launched[rec["id"]] = rec["ts"]
        cutoff = time.time() - a.older_than_hours * 3600
        old = [i["id"] for i in instances() if launched.get(i["id"], 0) < cutoff]
        terminate(old)
        print(f"reaped {len(old)}: {old}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
