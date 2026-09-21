"""Aggregate an M3 batch: outcomes against certificates, and mechanical behaviour metrics.

Reads only on-disk artefacts: runs/<batch>/*/{summary.json,result_*.json,trajectory.jsonl} and a
certification summary (private). Writes <batch>/analysis.json and prints a markdown report.

usage: python scripts/analyze_m3.py --runs runs/m3 --cert private/cert/m2-dev-v0/summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

# litellm list prices for gpt-5.5 at the time of the run ($/token); unverified against the vendor
PRICE = {"input": 5e-6, "cached": 5e-7, "output": 3e-5}
SCORE_RE = re.compile(r"main: ([0-9.]+)x \[")


def geomean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0


def trajectory_metrics(path: Path) -> dict:
    ev = [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []
    cmds = [e["payload"]["cmd"] for e in ev if e["type"] == "command"]
    obs = [e["payload"] for e in ev if e["type"] == "observation"]
    checks, first_write, looked_at_data_first = [], None, False
    for i, c in enumerate(cmds):
        if (
            first_write is None
            and "solution.py" in c
            and any(k in c for k in ("cat >", "write_text", "tee "))
        ):
            first_write = i
        if first_write is None and any(
            k in c for k in ("make_inputs", "lengths(", "DIMS", ".shape")
        ):
            looked_at_data_first = True
    for c, o in zip(cmds, obs, strict=False):
        if "hotloop-eval" in c:
            m = SCORE_RE.findall(o.get("output", ""))
            checks.append(float(m[-1]) if m else 0.0)
    usage = defaultdict(int)
    for e in ev:
        if e["type"] == "model_response":
            for k, v in e["payload"]["usage"].items():
                usage[k] += v
    uncached = usage["input_tokens"] - usage["cached_tokens"]
    blob = "\n".join(cmds)
    return {
        "turns": max((e["turn"] for e in ev), default=0),
        "n_commands": len(cmds),
        "n_failed_commands": sum(1 for o in obs if o["returncode"] != 0),
        "n_self_checks": len(checks),
        "best_self_check": max(checks, default=0.0),
        "last_self_check": checks[-1] if checks else 0.0,
        "looked_at_data_before_first_solution": looked_at_data_first,
        "used_profiler": bool(re.search(r"nsys|torch\.profiler|profile\(|proton|nvprof", blob)),
        "used_triton": "triton" in blob,
        "used_cuda_ext": bool(re.search(r"cpp_extension|\.cu\b|nvcc", blob)),
        "used_torch_compile": "torch.compile" in blob,
        "used_cuda_graphs": bool(re.search(r"CUDAGraph|cudagraph|reduce-overhead", blob)),
        "command_seconds": round(sum(o.get("duration_s") or 0 for o in obs)),
        "model_seconds": round(
            sum(e["payload"]["latency_s"] for e in ev if e["type"] == "model_response")
        ),
        "tokens": dict(usage),
        "api_usd": round(
            uncached * PRICE["input"]
            + usage["cached_tokens"] * PRICE["cached"]
            + usage["output_tokens"] * PRICE["output"],
            2,
        ),  # fmt: skip
        "end": next(
            (
                e["payload"].get("reason") or e["payload"].get("limit")
                for e in reversed(ev)
                if e["type"] in ("run_end", "budget")
            ),
            "?",
        ),  # fmt: skip
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True, type=Path)
    p.add_argument("--cert", type=Path, default=None)
    a = p.parse_args()
    cert = json.loads(a.cert.read_text())["tasks"] if a.cert and a.cert.is_file() else {}

    per_task: dict[str, list[dict]] = defaultdict(list)
    for d in sorted(x for x in a.runs.iterdir() if (x / "summary.json").is_file()):
        s = json.loads((d / "summary.json").read_text())
        results = [json.loads(f.read_text()) for f in sorted(d.glob("result_*.json"))]
        ok = [r["score"] for r in results if r["status"] == "ok"]
        run = {
            "run": d.name,
            "rep": s["rep"],
            "statuses": [r["status"] for r in results],
            "score": geomean(ok) if len(ok) == len(results) and ok else 0.0,
            "eval_scores": [round(r["score"], 3) for r in results],
            "flags": sorted({f["code"] for r in results for f in r["flags"]}),
            "failure": next(
                (
                    r["error"] or r["entries"][0]["correctness"]["failure"]
                    for r in results
                    if r["status"] != "ok" and (r["error"] or r["entries"])
                ),
                None,
            ),  # fmt: skip
            **trajectory_metrics(d / "trajectory.jsonl"),
        }
        per_task[s["task"]].append(run)

    rows, out = [], {}
    for task, runs in sorted(per_task.items()):
        c = cert.get(task, {})
        best_known = max([c.get("headroom", 0.0)] + [r["score"] for r in runs])
        scores = [r["score"] for r in runs]
        kind = (
            "control"
            if c.get("control")
            else (
                "diagnostic" if c.get("lazy_best", 0) > 0.9 * c.get("headroom", 1e9) else "headroom"
            )
        )
        out[task] = {"kind": kind, "certified_best": c.get("headroom"), "lazy_best": c.get("lazy_best"),
                     "best_known_now": best_known, "runs": runs}  # fmt: skip
        closed = [
            math.log(max(s, 1e-9)) / math.log(best_known) if best_known > 1.05 and s > 0 else None
            for s in scores
        ]
        rows.append((task, kind, c.get("headroom", 0), c.get("lazy_best", 0), scores, closed, runs))
    (a.runs / "analysis.json").write_text(json.dumps(out, indent=1))

    print(
        "| task | kind | certified best | lazy best | agent score per rep | headroom closed | turns | self-checks | $ |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for task, kind, best, lazy, scores, closed, runs in rows:
        print(f"| {task.replace('-s0', '')} | {kind} | {best:.2f} | {lazy:.2f} | "
              + " / ".join(f"{s:.2f}" if s else f"0 ({r['statuses'][0] if r['statuses'] else 'none'})" for s, r in zip(scores, runs, strict=True))
              + " | " + " / ".join("-" if c is None else f"{c:.2f}" for c in closed)
              + " | " + "/".join(str(r["turns"]) for r in runs)
              + " | " + "/".join(str(r["n_self_checks"]) for r in runs)
              + " | " + "/".join(f"{r['api_usd']:.2f}" for r in runs) + " |")  # fmt: skip
    allruns = [r for _, _, _, _, _, _, rs in rows for r in rs]
    n = len(allruns) or 1
    print(f"\nruns: {len(allruns)}   total API ${sum(r['api_usd'] for r in allruns):.2f}   "
          f"ended: { {k: sum(1 for r in allruns if r['end'] == k) for k in sorted({r['end'] for r in allruns})} }")  # fmt: skip
    for k in ("looked_at_data_before_first_solution", "used_profiler", "used_triton", "used_cuda_ext",
              "used_torch_compile", "used_cuda_graphs"):  # fmt: skip
        print(f"  {k}: {sum(1 for r in allruns if r[k])}/{n}")
    regress = [
        r for r in allruns if r["best_self_check"] > 0 and r["score"] < 0.9 * r["best_self_check"]
    ]
    print(f"  final official score < 0.9 x best self-check seen: {len(regress)}/{n}  "
          + ", ".join(f"{r['run'].split('--')[0]} r{r['rep']} ({r['best_self_check']:.2f}->{r['score']:.2f})" for r in regress))  # fmt: skip
    audit = [r for r in allruns if r["score"] > 2]
    print(f"  runs scoring > 2x (to be read by hand, G2e): {len(audit)}")
    bad = [r for r in allruns if r["score"] == 0]
    for r in bad:
        print(f"  ZERO: {r['run']}: {r['statuses']} {str(r['failure'])[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
