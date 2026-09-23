"""Exploit suite: every exploit must score 0, positive controls must pass.

Each file in exploits/<suite>/ declares `EXPECT pass|fail|violation` in its docstring.
"""

import glob
import os
import re


def load_suite(suite_dir: str, only: list[str] | None = None) -> list[tuple[str, str, str]]:
    cases = []
    for f in sorted(glob.glob(os.path.join(suite_dir, "*.py"))):
        name = os.path.basename(f)[:-3]
        if only and name not in only:
            continue
        src = open(f).read()
        cases.append((name, re.search(r"EXPECT (\w+)", src).group(1), src))
    return cases


def outcome(r) -> tuple[str, str]:
    if isinstance(r, Exception):
        return "error", repr(r)[:120]
    if r["ok"]:
        return "pass", ""
    got = "violation" if r["violations"] else "fail"
    reasons = r["violations"] or ([r["error"]] if r["error"] else
                                  [f"{k}: {v['reason']}" for k, v in r["shapes"].items() if v.get("reason")])
    return got, (reasons[0] if reasons else "").replace("\n", " ")[-110:]


def run_selftest(backend, task_id: str, suite_dir: str, gpu: str, only: list[str] | None = None, log=print) -> bool:
    cases = load_suite(suite_dir, only)
    # Positive controls first, so baselines are cached before the parallel batch.
    first = [c for c in cases if c[1] == "pass"]
    rest = [c for c in cases if c[1] != "pass"]
    results = {n: backend.score(task_id, src, gpu=gpu, hidden=True) for n, _, src in first}
    if hasattr(backend, "score_many"):
        batch = backend.score_many([(task_id, src) for _, _, src in rest], gpu=gpu)
    else:
        batch = []
        for _, _, src in rest:
            try:
                batch.append(backend.score(task_id, src, gpu=gpu, hidden=True))
            except Exception as e:
                batch.append(e)
    results.update({n: r for (n, _, _), r in zip(rest, batch)})

    bad = 0
    log(f"\n  {'case':20} {'expect':10} {'got':10} {'speedup':>8} {'SOL%':>6}  detail")
    for name, expect, _ in cases:
        r = results[name]
        got, detail = outcome(r)
        sc = r.get("score", {}) if isinstance(r, dict) else {}
        ok = (got == "pass") == (expect == "pass") and (expect != "violation" or got == "violation")
        bad += not ok
        log(f"{'✓' if ok else '✗'} {name:20} {expect:10} {got:10} {sc.get('speedup_geomean', 0):8.2f} "
            f"{100 * sc.get('sol_frac_geomean', 0):5.1f}%  {detail}")
    log(f"\n{len(cases) - bad}/{len(cases)} behaved as expected")
    return bad == 0
