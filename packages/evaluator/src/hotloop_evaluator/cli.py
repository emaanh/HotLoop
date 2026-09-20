"""hotloop-eval --task <task_pkg> --submission <dir> --out result.json"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hotloop_evaluator.driver import EvalConfig, evaluate


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
