"""Scoring against a Store: cached device peaks and baselines, then evaluate()."""

import json
import os

from hotloop.bench.store import Store, gpu_key


def get_peaks(cache_dir: str | None = None) -> dict:
    from hotloop.harness import roofline
    from hotloop.harness.device import get_device

    path = os.path.join(cache_dir, f"peaks_{gpu_key(get_device().device_name())}.json") if cache_dir else None
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    peaks = roofline.measure_peaks()
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(peaks, f)
    return peaks


def run_eval(store: Store, task_id: str, solution_src: str, hidden: bool = True, **kw) -> dict:
    from hotloop.harness.evaluate import evaluate

    peaks = get_peaks(store.cache_dir)
    cache_path = os.path.join(store.cache_dir, f"baselines_{gpu_key(peaks['gpu'])}.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    n_before = len(cache)
    result = evaluate(task_id, store.shape_dirs(task_id, hidden), solution_src, hidden=hidden,
                      peaks=peaks, baseline_cache=cache, **kw)
    if len(cache) > n_before:
        # Re-read before writing so parallel evals don't drop each other's entries.
        latest = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
        latest.update(cache)
        with open(cache_path, "w") as f:
            json.dump(latest, f, indent=1)
    return result
