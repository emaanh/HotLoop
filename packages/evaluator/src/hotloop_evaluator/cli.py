"""hotloop-eval --task <task_pkg> --submission <dir> --out result.json"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hotloop_evaluator.driver import EvalConfig, evaluate


def calibrate_main(argv: list[str] | None = None) -> int:
    """hotloop-calibrate --task <task_pkg> [--seeds N]: print calibrated tolerances as JSON.

    Runs the reference natively and in float64 on fresh inputs (correctness.calibrate_tolerances).
    Taskgen calls this as a subprocess at emit time; it never imports the evaluator.
    """
    import json

    from hotloop_evaluator.correctness import calibrate_tolerances
    from hotloop_evaluator.driver import _load_workload
    from hotloop_evaluator.worker import _import_from
    from hotloop_schemas import load_task_spec

    p = argparse.ArgumentParser(prog="hotloop-calibrate", description=calibrate_main.__doc__)
    p.add_argument("--task", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seeds", type=int, default=4)
    args = p.parse_args(argv)
    spec = load_task_spec(args.task)
    make_inputs = _load_workload(args.task)
    reference = _import_from(args.task / "reference.py", "hotloop_reference").reference
    out = {}
    for entry in spec.workload:
        sets = [
            make_inputs(entry.name, 1_000_003 * (i + 1), args.device) for i in range(args.seeds)
        ]
        out[entry.name] = [t.model_dump() for t in calibrate_tolerances(reference, sets)]
    print(json.dumps(out))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hotloop-eval", description=__doc__)
    p.add_argument("--task", required=True, type=Path)
    p.add_argument("--submission", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None, help="result.json path (default: stdout)")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--seed", type=int, default=None, help="fix the base seed (public/self-check mode)"
    )
    p.add_argument("--quick", action="store_true", help="fewer pairs; for the agent's own checks")
    args = p.parse_args(argv)

    cfg = EvalConfig(device=args.device, seed=args.seed)
    if args.quick:
        cfg.correctness_trials = 2
        cfg.stopping = type(cfg.stopping)(target_rel_halfwidth=0.03, min_pairs=5, max_pairs=30)
    result = evaluate(args.task, args.submission, cfg)
    text = result.model_dump_json(indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    else:
        print(text)
    summary = f"{result.task_id}: {result.status} score={result.score:.3f}"
    for e in result.entries:
        if e.speedup:
            summary += f"\n  {e.entry}: {e.speedup.point:.3f}x [{e.speedup.lo:.3f}, {e.speedup.hi:.3f}] vs {e.best_baseline}"
        elif not e.correctness.passed:
            summary += f"\n  {e.entry}: {e.correctness.failure}"
    if result.error:
        summary += f"\n  error: {result.error.splitlines()[-1] if result.error else ''}"
    print(summary, file=sys.stderr)
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
