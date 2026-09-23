"""Agent-facing benchmark on the public shape: same checks as final scoring.

    python -m hotloop.harness.bench [path/to/solution.py]
"""

import json
import os
import sys
import time

WORKDIR = os.environ.get("HOTLOOP_WORKDIR", "/workspace")
TASK_DIR = os.environ.get("HOTLOOP_TASK_DIR", os.path.join(WORKDIR, "task"))
CACHE_DIR = "/tmp/hotloop_cache"


def report(r: dict) -> str:
    lines = []
    if r["violations"]:
        lines.append("RULE VIOLATIONS (score 0):")
        lines += [f"  - {v}" for v in r["violations"]]
    if r["error"]:
        lines.append(f"ERROR: {r['error']}")
    for sid, s in r["shapes"].items():
        lines.append(f"shape {sid}: {'CORRECT' if s['correct'] else 'INCORRECT'}")
        if s.get("reason"):
            lines.append("  " + s["reason"].strip().replace("\n", "\n  "))
        if "time_ms" in s:
            lines.append(f"  your time      {s['time_ms'] * 1e3:9.1f} us")
            lines.append(f"  torch.compile  {s['compile_ms'] * 1e3:9.1f} us   (baseline)")
            lines.append(f"  eager pytorch  {s['eager_ms'] * 1e3:9.1f} us   (for reference)")
        if "sol_ms" in s:
            lines.append(f"  speed-of-light {s['sol_ms'] * 1e3:9.1f} us  "
                         f"({s['flops'] / 1e9:.2f} GFLOP, {s['bytes'] / 1e6:.2f} MB, {s['dtype']})")
            lines.append(f"  speedup vs torch.compile: {s['speedup']:.3f}x   fraction of speed-of-light: {100 * s['sol_frac']:.1f}%")
            lines.append(f"  max relative L2 error: {s['max_rel_err']:.2e}")
        if s.get("kernels"):
            lines.append("  GPU time per kernel in the timed graph (profiled, one replay):")
            lines += [f"    {t:9.1f} us  {n[:110]}" for n, t in s["kernels"][:10]]
    sc = r.get("score", {})
    lines.append(f"SCORE (public shape only): correct={sc.get('correct')} "
                 f"speedup={sc.get('speedup_geomean', 0):.3f}x sol={100 * sc.get('sol_frac_geomean', 0):.1f}%  [{r.get('wall_s')}s]")
    lines.append("Final scoring uses hidden shapes (other batch sizes / sequence lengths) and fresh random inputs.")
    return "\n".join(lines)


def main():
    from hotloop.harness.evaluate import evaluate
    from hotloop.harness.service import get_peaks

    sol_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(WORKDIR, "solution.py")
    with open(os.path.join(TASK_DIR, "task.json")) as f:
        task = json.load(f)
    sdir = os.path.join(TASK_DIR, "shapes")
    shape_dirs = [os.path.join(sdir, s) for s in sorted(os.listdir(sdir))]
    peaks = get_peaks(CACHE_DIR)
    cache_path = os.path.join(CACHE_DIR, "baselines.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    with open(sol_path) as f:
        src = f.read()
    r = evaluate(task["task_id"], shape_dirs, src, hidden=False, peaks=peaks, baseline_cache=cache,
                 timed_per_block=5, correct_seeds=1)
    with open(cache_path, "w") as f:
        json.dump(cache, f)
    print(report(r))
    if r["ok"]:
        # Best-submission fallback: scored if the final solution.py fails.
        snap_dir = os.path.join(WORKDIR, ".hotloop", "passing")
        os.makedirs(snap_dir, exist_ok=True)
        path = os.path.join(snap_dir, f"{time.strftime('%Y%m%d-%H%M%S')}.py")
        with open(path, "w") as f:
            f.write(src)
        print(f"All public shapes correct: snapshot saved to {path}")


if __name__ == "__main__":
    main()
