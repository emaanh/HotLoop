"""Task-set maintenance on a Store (used by every backend, inside its GPU machine).

    python -m hotloop.gen.cli select --n 8
    python -m hotloop.gen.cli trace MODEL_ID [MODEL_ID ...]
    python -m hotloop.gen.cli filter [--tasks a,b]
    python -m hotloop.gen.cli annotate
"""

import argparse
import json
import os
import subprocess
import sys

from hotloop.bench.store import Store


def select(n: int) -> list[dict]:
    from hotloop.gen.models import select_models
    return select_models(n)


def trace(store: Store, model_id: str, phases: tuple[str, ...] = ("prefill", "decode")) -> dict:
    from hotloop.gen.trace import trace_model, write_task
    tasks = trace_model(model_id, phases=tuple(phases))
    for t in tasks.values():
        write_task(t, store.tasks, store.hidden)
    return {tid: {"class": t["task"]["module_class"], "shapes": list(t["shapes"])} for tid, t in tasks.items()}


def filter_tasks(store: Store, task_ids: list[str] | None = None, part: str | None = None) -> dict:
    """Writes results to filters_<gpu>.json, or filters_<gpu>__<part>.json when run in
    parallel chunks (Store.filter_results merges all of them)."""
    from hotloop.gen.filters import filter_shape
    from hotloop.harness.service import get_peaks
    from hotloop.harness.task import load_task

    peaks = get_peaks(store.cache_dir)
    path = store.filter_path(peaks["gpu"])
    if part:
        path = path.replace(".json", f"__{part}.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    for tid in task_ids or store.task_ids():
        # One process per task: a CUDA error in one task must not poison the next.
        p = subprocess.run([sys.executable, "-m", "hotloop.gen.cli", "filter-one", tid],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1800)
        try:
            results[tid] = next(json.loads(l[len("HOTLOOP_RESULT "):]) for l in reversed(p.stdout.splitlines())
                                if l.startswith("HOTLOOP_RESULT "))
        except StopIteration:
            results[tid] = {"keep": False, "drop_reasons": [f"filter error: {p.stdout.strip().splitlines()[-1][:300] if p.stdout.strip() else p.returncode}"]}
        print(tid, "KEEP" if results[tid]["keep"] else results[tid]["drop_reasons"], flush=True)
    os.makedirs(store.cache_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=1)
    return results


def filter_one(store: Store, task_id: str) -> dict:
    from hotloop.gen.filters import filter_shape
    from hotloop.harness.service import get_peaks
    from hotloop.harness.task import load_task

    _, shapes = load_task(store.task_dir(task_id))
    try:
        return filter_shape(shapes[0], get_peaks(store.cache_dir))
    except Exception as e:
        return {"keep": False, "drop_reasons": [f"filter error: {type(e).__name__}: {e}"[:300]]}


def annotate(store: Store) -> int:
    """Backfill symbolic shapes into public task.json files."""
    from hotloop.gen.trace import symbolic_shapes

    for tid in store.task_ids():
        metas = []
        for root in (store.tasks, store.hidden):
            d = os.path.join(root, tid, "shapes")
            for sid in sorted(os.listdir(d)) if os.path.isdir(d) else []:
                with open(os.path.join(d, sid, "meta.json")) as f:
                    metas.append(json.load(f))
        path = os.path.join(store.task_dir(tid), "task.json")
        with open(path) as f:
            task = json.load(f)
        task["symbolic_shapes"] = symbolic_shapes(metas)
        with open(path, "w") as f:
            json.dump(task, f, indent=2)
    return len(store.task_ids())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select").add_argument("--n", type=int, default=8)
    p = sub.add_parser("trace")
    p.add_argument("model_ids", nargs="+")
    p.add_argument("--phases", default="prefill,decode")
    sub.add_parser("filter").add_argument("--tasks", default="")
    sub.add_parser("filter-one").add_argument("task_id")
    sub.add_parser("annotate")
    args = ap.parse_args()
    store = Store.default()
    if args.cmd == "select":
        out = select(args.n)
    elif args.cmd == "trace":
        out = {m: trace(store, m, tuple(args.phases.split(","))) for m in args.model_ids}
    elif args.cmd == "filter":
        out = filter_tasks(store, args.tasks.split(",") if args.tasks else None)
    elif args.cmd == "filter-one":
        out = filter_one(store, args.task_id)
    else:
        out = annotate(store)
    print("HOTLOOP_RESULT " + json.dumps(out, default=str))


if __name__ == "__main__":
    main()
