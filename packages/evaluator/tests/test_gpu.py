"""Evaluator on a real GPU: CUDA IPC path, real timing, GPU-only exploits. Skipped without CUDA."""

import json
import os

import pytest
import torch
from fixtures.gpu_toy import write_submission, write_task

from hotloop_evaluator.driver import EvalConfig, evaluate
from hotloop_evaluator.stats import StoppingRule

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

CFG = {"stopping": StoppingRule(target_rel_halfwidth=0.01, min_pairs=8, max_pairs=40)}
REPORT = os.environ.get("HOTLOOP_GPU_REPORT")


@pytest.fixture(scope="module")
def task(tmp_path_factory):
    return write_task(tmp_path_factory.mktemp("gputoy"))


@pytest.fixture(scope="module")
def results(task, tmp_path_factory):
    root, out = tmp_path_factory.mktemp("subs"), {}
    for name in ("honest_triton", "side_stream", "no_sync", "eager_copy"):
        out[name] = evaluate(task, write_submission(root, name), EvalConfig(**CFG))
    # same honest solution, but the worker's own synchronise is switched off entirely
    nosync_root = root / "nosync"
    nosync_root.mkdir()
    os.environ["HOTLOOP_TEST_DISABLE_WORKER_SYNC"] = "1"
    try:
        out["honest_triton_worker_never_syncs"] = evaluate(
            task, write_submission(nosync_root, "honest_triton"), EvalConfig(**CFG)
        )
    finally:
        del os.environ["HOTLOOP_TEST_DISABLE_WORKER_SYNC"]
    if REPORT:
        with open(REPORT, "w") as f:
            json.dump({k: json.loads(v.model_dump_json()) for k, v in out.items()}, f, indent=1)
    return out


def _speedup(r):
    return r.entries[0].speedup


def test_honest_triton_is_ok_with_tight_ci(results):
    r = results["honest_triton"]
    assert r.status == "ok", r.error or r.entries[0].correctness.failure
    e = r.entries[0]
    assert set(e.baselines) == {"eager", "compile_default"}
    assert (e.speedup.hi - e.speedup.lo) / e.speedup.point < 0.05
    assert r.env.gpu_name and r.env.driver and r.env.gpu_before.sm_clock_mhz


def test_strongest_baseline_is_used(results):
    """D-4: the fused Triton kernel beats eager by a lot, but is scored against the compiler."""
    e = results["honest_triton"].entries[0]
    assert e.best_baseline == "compile_default"
    assert e.baselines["eager"].median_s > 1.5 * e.baselines["compile_default"].median_s


def test_copying_the_reference_scores_about_one(results):
    r = results["eager_copy"]
    assert r.status == "ok" and r.score < 1.0  # it is eager, the baseline is the compiler


def test_side_stream_gains_nothing(results):
    honest, cheat = results["honest_triton"], results["side_stream"]
    assert cheat.status in ("ok", "incorrect")
    if cheat.status == "ok":
        assert cheat.score < honest.score * 1.05


def test_rebinding_synchronize_is_flagged_fatal(results):
    r = results["no_sync"]
    assert r.status == "flagged" and r.score == 0
    assert any(f.code == "sync_tampered" and f.fatal for f in r.flags)


def test_no_gain_is_possible_even_if_worker_never_syncs(results):
    """The defence that matters, independent of the worker-side guard: with *no* synchronise in
    the worker at all, the run is either caught (the driver copies a result buffer the queued
    work has not reached yet, finds the NaN poison, and fails it) or timed correctly (the copy
    queued behind the work). It must never come out faster than the honest measurement."""
    honest, broken = results["honest_triton"], results["honest_triton_worker_never_syncs"]
    assert broken.status in ("incorrect", "ok")
    if broken.status == "ok":
        assert broken.score < honest.score * 1.05
    else:
        assert broken.score == 0
