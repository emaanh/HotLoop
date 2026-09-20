"""Day-1 GPU box probe (ACCESS.md). Standalone: needs only torch + numpy on the box.

Measures what gate G0 cares about, per workload class:
  * cv_block      - run-to-run CV of block times (absolute noise)
  * null ratio    - A-vs-A interleaved paired ratio: should be 1.0; its CI width is our noise floor
  * planted ratio - a known 1.10x difference: is it recovered, does the CI exclude 1.0?
Classes: compute-bound (matmul), bandwidth-bound (pointwise chain), launch-bound (many tiny kernels).

usage: python probe_gpu.py --seconds 60 --tag unlocked > probe_unlocked.json
"""

import argparse
import json
import platform
import subprocess
import time

import numpy as np
import torch


def smi(q):
    try:
        return subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}"


def paired_ratio(b, c, n_boot=4000, seed=0):
    logr = np.log(np.asarray(b) / np.asarray(c))
    rng = np.random.default_rng(seed)
    boots = np.median(logr[rng.integers(0, logr.size, (n_boot, logr.size))], axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    point = float(np.exp(np.median(logr)))
    return {"point": point, "lo": float(np.exp(lo)), "hi": float(np.exp(hi)),
            "rel_halfwidth": float((np.exp(hi) - np.exp(lo)) / (2 * point)), "n_pairs": int(logr.size)}


def make_workloads(dev):
    a = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    x = torch.randn(64_000_000, device=dev, dtype=torch.float16)
    tiny = [torch.randn(64, device=dev) for _ in range(8)]

    def compute(reps=10):
        y = a
        for _ in range(reps):
            y = a @ y
        return y

    def bandwidth(reps=10):
        y = x
        for _ in range(reps):
            y = torch.nn.functional.gelu(y * 1.0001 + 0.5)
        return y

    def launch(reps=1000):
        t = tiny[0]
        for i in range(reps):
            t = t + tiny[i & 7]
        return t

    return {"compute": (compute, 10, 11), "bandwidth": (bandwidth, 10, 11), "launch": (launch, 1000, 1100)}


def timed(fn, reps):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn(reps)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def study(fn, base_reps, planted_reps, seconds):
    for _ in range(5):
        timed(fn, base_reps)
    a_blocks, a2_blocks, p_blocks = [], [], []
    t_end = time.perf_counter() + seconds
    i = 0
    while time.perf_counter() < t_end:
        order = [("a", base_reps), ("a2", base_reps), ("p", planted_reps)]
        order = order[i % 3:] + order[: i % 3]  # rotate order so position effects cancel
        got = {name: timed(fn, reps) for name, reps in order}
        a_blocks.append(got["a"]); a2_blocks.append(got["a2"]); p_blocks.append(got["p"])
        i += 1
    a = np.asarray(a_blocks)
    thirds = np.array_split(a, 3)
    return {
        "n_blocks": int(a.size),
        "median_ms": float(np.median(a) * 1e3),
        "cv_block": float(a.std() / a.mean()),
        "p99_over_median": float(np.quantile(a, 0.99) / np.median(a)),
        "drift_last_over_first_third": float(np.median(thirds[-1]) / np.median(thirds[0])),
        "null_ratio": paired_ratio(a_blocks, a2_blocks),
        "planted_1p10_ratio": paired_ratio(p_blocks, a_blocks),  # p is ~1.10x slower => ratio ~1.10
        "null_ratio_first30": paired_ratio(a_blocks[:30], a2_blocks[:30]),
        "planted_first30": paired_ratio(p_blocks[:30], a_blocks[:30]),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dev = "cuda"
    out = {
        "tag": args.tag,
        "gpu": smi("name,uuid,driver_version,vbios_version,pstate,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,clocks_throttle_reasons.active,power.draw"),
        "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
        "cpu": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), None),
        "results": {},
    }
    for name, (fn, base, planted) in make_workloads(dev).items():
        out["results"][name] = study(fn, base, planted, args.seconds)
        out["results"][name]["temp_after"] = smi("temperature.gpu,clocks.sm,clocks_throttle_reasons.active")
    print(json.dumps(out, indent=2))
