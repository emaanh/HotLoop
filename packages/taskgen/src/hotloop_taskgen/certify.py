"""Certification (DESIGN 3.3): run hidden reference strategies and the lazy battery through the
real evaluator, as ordinary submissions, and record what they score.

    hotloop-certify --tasks DIR --strategies DIR --out DIR [--runs 1]

Strategies live outside the public repo (D-16): `<strategies>/<family>/<name>/solution.py`.
The evaluator is only ever invoked as a subprocess. Output: one result.json per
(task, strategy, run) plus `summary.json` with, per task: best strategy, headroom over the best
automatic baseline, what the lazy battery reaches, and per sibling group the regime-regret matrix.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from hotloop_schemas import Result, load_task_spec

# Certification sweeps run ~10 submissions per task: trade a little precision for a lot of time.
# Headroom/regret effects of interest are >= 1.25x, far above a 2% CI.
EVAL_FLAGS = ["--rel-halfwidth", "0.02", "--max-pairs", "30", "--max-calls", "64"]

LAZY = {
    "lazy_compile_default": "run = torch.compile(reference)",
    "lazy_compile_max_autotune": "run = torch.compile(reference, mode='max-autotune-no-cudagraphs')",
    "lazy_compile_cudagraphs": (
        "_g = torch.compile(reference, mode='reduce-overhead')\n\n\n"
        "def run(*a):\n    torch.compiler.cudagraph_mark_step_begin()\n"
        "    out = _g(*a)\n    return out.clone() if isinstance(out, torch.Tensor) else tuple(o.clone() for o in out)"
    ),
}
LAZY_BY_MEMBER = {"chain": {"lazy_multi_dot": "def run(*m):\n    return torch.linalg.multi_dot(m)"}}


def _lazy_submissions(task_dir: Path, root: Path) -> dict[str, Path]:
    spec = load_task_spec(task_dir)
    bodies = dict(LAZY)
    bodies.update(LAZY_BY_MEMBER.get(spec.provenance.family_params.get("member"), {}))
    out = {}
    for name, body in bodies.items():
        d = root / name
        d.mkdir(parents=True)
        shutil.copy(task_dir / "reference.py", d / "reference.py")
        (d / "solution.py").write_text(
            f"import torch\nfrom reference import reference  # noqa: F401\n\n{body}\n"
        )
        out[name] = d
    return out


def _evaluate(task: Path, submission: Path, out: Path, device: str) -> Result:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "hotloop_evaluator.cli",
            "--task",
            str(task),
            "--submission",
            str(submission),
            "--out",
            str(out),
            "--device",
            device,
            *EVAL_FLAGS,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if not out.is_file():
        return Result(
            task_id=task.name, evaluator_version="?", status="error", score=0.0, error="no result"
        )
    return Result.model_validate_json(out.read_text())


def _geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0


def certify(tasks_dir: Path, strategies_dir: Path, out_dir: Path, runs: int, device: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"tasks": {}, "groups": {}}
    for task in sorted(p for p in tasks_dir.iterdir() if (p / "task.toml").is_file()):
        spec = load_task_spec(task)
        fam_dir = strategies_dir / spec.provenance.family
        subs = (
            {p.name: p for p in sorted(fam_dir.iterdir()) if (p / "solution.py").is_file()}
            if fam_dir.is_dir()
            else {}
        )
        with tempfile.TemporaryDirectory() as tmp:
            subs.update(_lazy_submissions(task, Path(tmp)))
            scores, status, baselines = {}, {}, {}
            for name, sub in subs.items():
                rs = [
                    _evaluate(task, sub, out_dir / f"{spec.id}__{name}__run{i}.json", device)
                    for i in range(runs)
                ]
                ok = [r for r in rs if r.status == "ok"]
                scores[name] = _geomean([r.score for r in ok]) if len(ok) == runs else 0.0
                status[name] = sorted({r.status for r in rs})
                for r in ok:
                    for e in r.entries:
                        baselines[e.best_baseline] = baselines.get(e.best_baseline, 0) + 1
                print(f"{spec.id:52s} {name:28s} {scores[name]:8.3f}  {status[name]}", flush=True)
        lazy = {k: v for k, v in scores.items() if k.startswith("lazy_")}
        real = {k: v for k, v in scores.items() if not k.startswith("lazy_")}
        best = max(real, key=real.get) if real else None
        summary["tasks"][spec.id] = {
            "family": spec.provenance.family,
            "regime": spec.provenance.family_params.get("regime"),
            "control": spec.provenance.family_params.get("control"),
            "sibling_group": spec.provenance.sibling_group,
            "scores": scores,
            "status": status,
            "best_baseline_votes": baselines,
            "best_strategy": best,
            "headroom": real.get(best, 0.0) if best else 0.0,
            "lazy_best": max(lazy.values(), default=0.0),
        }
    groups: dict[str, list[str]] = {}
    for tid, t in summary["tasks"].items():
        groups.setdefault(t["sibling_group"], []).append(tid)
    for g, tids in groups.items():
        reg = {}
        for a in tids:
            sa = summary["tasks"][a]["best_strategy"]
            for b in tids:
                tb = summary["tasks"][b]
                got = tb["scores"].get(sa, 0.0)
                reg.setdefault(a, {})[b] = (tb["headroom"] / got) if got > 0 else None
        summary["groups"][g] = {"regret": reg}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hotloop-certify", description=__doc__)
    p.add_argument("--tasks", required=True, type=Path)
    p.add_argument("--strategies", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    certify(a.tasks, a.strategies, a.out, a.runs, a.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
