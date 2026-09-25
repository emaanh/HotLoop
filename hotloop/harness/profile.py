"""Agent-facing profiler: Nsight Compute on the solution's own kernels, condensed.

    python -m hotloop.harness.profile [solution.py] [--shape SID] [--full]

Runs the solution on one public shape under `ncu`, capturing only the kernels
launched inside `solution()` (after warm-up), and prints per kernel: time,
memory and compute throughput against peak, occupancy, launch configuration, and
ncu's own optimization hints, plus a one-line verdict on what bounds it.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

WORKDIR = os.environ.get("HOTLOOP_WORKDIR", "/workspace")
TASK_DIR = os.environ.get("HOTLOOP_TASK_DIR", os.path.join(WORKDIR, "task"))
SECTIONS = ["SpeedOfLight", "Occupancy", "LaunchStats", "MemoryWorkloadAnalysis", "WarpStateStats"]
KEEP_METRICS = (
    "Duration", "Memory Throughput", "DRAM Throughput", "Compute (SM) Throughput", "L1/TEX Hit Rate",
    "L2 Hit Rate", "Achieved Occupancy", "Theoretical Occupancy", "Block Limit Registers",
    "Block Limit Shared Mem", "Registers Per Thread", "Block Size", "Grid Size", "Waves Per SM",
    "Warp Cycles Per Issued Instruction", "Mem Busy", "Max Bandwidth",
)
MAX_KERNELS = 6

DRIVER = r'''
import importlib.util, json, sys, torch
sys.path.insert(0, sys.argv[3])
from hotloop.harness.inputs import make_inputs
from hotloop.harness.task import load_shape
shape = load_shape(sys.argv[2])
spec = importlib.util.spec_from_file_location("solution", sys.argv[1])
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
inputs = make_inputs(shape.meta, shape.exact, 1)
with torch.no_grad():
    for _ in range(2):                      # warm-up: JIT compilation, autotuning
        mod.solution(*[t.clone() for t in inputs])
    torch.cuda.synchronize()
    ins = [t.clone() for t in inputs]
    torch.cuda.synchronize()
    torch.cuda.profiler.start()             # ncu captures only this call
    mod.solution(*ins)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
'''


def _condense(text: str) -> str:
    kernels, cur = [], None
    for line in text.splitlines():
        m = re.match(r"^  (\S.*?) \((\d+, \d+, \d+)\)x\((\d+, \d+, \d+)\)", line)
        if m:
            cur = {"name": m.group(1), "grid": m.group(2), "block": m.group(3), "metrics": {}, "hints": []}
            kernels.append(cur)
            continue
        if cur is None:
            continue
        mm = re.match(r"^\s{4}(\S.*?)\s{2,}(\S+)\s{2,}(\S+)\s*$", line)
        if mm and mm.group(1).strip() in KEEP_METRICS:
            cur["metrics"][mm.group(1).strip()] = f"{mm.group(3)} {mm.group(2)}"
        elif re.match(r"^\s{4}(OPT|WRN)\s", line) and "could not be found" not in line:
            cur["hints"].append(line.strip()[4:].strip())
        elif cur["hints"] and re.match(r"^\s{10,}\S", line) and len(cur["hints"][-1]) < 400:
            cur["hints"][-1] += " " + line.strip()   # wrapped continuation of a hint
    if not kernels:
        return "(no kernels captured: does solution() launch GPU kernels?)\n" + text[-2000:]

    def dur_us(k):
        v = k["metrics"].get("Duration", "0 us").split()
        x = float(v[0].replace(",", "")) if v and re.match(r"[\d.,]+$", v[0]) else 0.0
        return x / 1e3 if len(v) > 1 and v[1] == "ns" else x * 1e3 if len(v) > 1 and v[1] == "ms" else x

    kernels.sort(key=dur_us, reverse=True)
    out = []
    for k in kernels[:MAX_KERNELS]:
        mt = k["metrics"]
        mem = float(mt.get("Memory Throughput", "0 %").split()[0].replace(",", "") or 0)
        comp = float(mt.get("Compute (SM) Throughput", "0 %").split()[0].replace(",", "") or 0)
        verdict = ("memory-bound" if mem > comp * 1.2 else "compute-bound" if comp > mem * 1.2 else "balanced")
        if max(mem, comp) < 40:
            verdict = f"latency/occupancy-bound (memory {mem:.0f}%, compute {comp:.0f}% of peak)"
        out.append(f"== {k['name'][:120]}  grid=({k['grid']}) block=({k['block']})")
        out.append(f"   verdict: {verdict}")
        for key in KEEP_METRICS:
            if key in mt:
                out.append(f"   {key:<36} {mt[key]}")
        for h in k["hints"][:4]:
            out.append(f"   hint: {h[:300]}")
    if len(kernels) > MAX_KERNELS:
        out.append(f"({len(kernels) - MAX_KERNELS} smaller kernels omitted)")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("solution", nargs="?", default=os.path.join(WORKDIR, "solution.py"))
    ap.add_argument("--shape", help="public shape id (default: the first)")
    ap.add_argument("--full", action="store_true", help="print ncu's full report instead of the summary")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        sys.exit("profile: Nsight Compute needs an NVIDIA GPU (on Apple GPUs use Xcode Instruments / xctrace).")
    with open(os.path.join(TASK_DIR, "task.json")) as f:
        task = json.load(f)
    sid = args.shape or task["public_shapes"][0]
    shape_dir = os.path.join(TASK_DIR, "shapes", sid)
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(DRIVER)
        driver = f.name
    cmd = ["ncu", "--clock-control", "none", "--profile-from-start", "off", "--target-processes", "all"]
    cmd += ["--set", "full"] if args.full else sum((["--section", s] for s in SECTIONS), [])
    cmd += [sys.executable, driver, os.path.abspath(args.solution), shape_dir, pkg_root]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    text = p.stdout + ("\n" + p.stderr if p.returncode else "")
    print(f"ncu profile of {os.path.basename(args.solution)} on shape {sid} (one call to solution(), after warm-up)\n")
    print(text[-20000:] if args.full else _condense(text))
    if p.returncode:
        print(f"\n[ncu exited with {p.returncode}]")


if __name__ == "__main__":
    main()
