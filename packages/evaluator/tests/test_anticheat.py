"""End-to-end evaluator runs on CPU: an honest solution scores, every null agent scores 0.

This is the regression suite EVAL.md promises. Each cheat here corresponds to an exploit seen in
the wild (ECOSYSTEM.md): stale/memoised outputs in the timed phase, clock patching, behaving
only while being checked, garbage-fast returns, input mutation, process-killing submissions.
GPU-only exploits (side streams, async return) are covered by tests marked `gpu`.
"""

import pytest
from fixtures.toy import write_submission, write_task

from hotloop_evaluator.driver import EvalConfig, evaluate
from hotloop_evaluator.stats import StoppingRule

FAST = {
    "device": "cpu",
    "correctness_trials": 3,
    "target_block_s": 0.02,
    "baseline_probe_blocks": 3,
    "stopping": StoppingRule(target_rel_halfwidth=0.2, min_pairs=6, max_pairs=10),
    # CPU timing is noisy; this suite is about verdicts, not precision
    "inconclusive_rel_halfwidth": 10.0,
    "load_timeout_s": 120,
    "call_timeout_s": 20,
}


@pytest.fixture(scope="module")
def task(tmp_path_factory):
    return write_task(tmp_path_factory.mktemp("toy"))


def run(task, tmp_path, name, **over):
    return evaluate(task, write_submission(tmp_path, name), EvalConfig(**{**FAST, **over}))


def test_honest_solution_scores(task, tmp_path):
    r = run(task, tmp_path, "honest")
    assert r.status == "ok", r.error or r.entries[0].correctness.failure
    assert r.score > 3, "vectorised solution must clearly beat the python-loop reference"
    e = r.entries[0]
    assert e.speedup.lo > 1 and e.best_baseline == "eager" and e.correctness.n_trials == 3
    assert r.eval_seed is not None and r.task_hash and r.env.torch


def test_fresh_secret_seed_each_evaluation(task, tmp_path):
    a = run(task, tmp_path, "honest")
    b = evaluate(task, tmp_path / "sub_honest", EvalConfig(**FAST))
    assert a.eval_seed != b.eval_seed


@pytest.mark.parametrize("name", ["wrong", "wrong_dtype", "mutates_inputs", "returns_garbage_fast"])
def test_incorrect_submissions_score_zero(task, tmp_path, name):
    r = run(task, tmp_path, name)
    assert r.status == "incorrect" and r.score == 0
    assert r.entries[0].speedup is None, "no timing is reported for an incorrect submission"


@pytest.mark.parametrize("name", ["stale_after_warmup", "right_only_at_first"])
def test_correct_only_while_checked_is_caught_in_timed_phase(task, tmp_path, name):
    r = run(task, tmp_path, name)
    assert r.status == "incorrect" and r.score == 0
    assert any(f.code == "timed_phase_mismatch" for f in r.flags)
    assert "timed-phase output wrong" in r.entries[0].correctness.failure


def test_value_memoisation_gains_nothing(task, tmp_path):
    """Fresh values every call: the cache never hits, so it is merely a slower honest solution."""
    honest = run(task, tmp_path, "honest")
    memo = run(task, tmp_path, "memoise_by_value")
    assert memo.status == "ok"
    assert memo.score < honest.score * 1.5


def test_clock_patching_does_not_move_the_score(task, tmp_path):
    honest = run(task, tmp_path, "honest")
    patched = run(task, tmp_path, "patches_clock")
    assert patched.status == "ok"
    assert patched.score < honest.score * 2, "driver-side timing must be immune"
    assert any(f.code == "clock_disagreement" for f in patched.flags)


@pytest.mark.parametrize("name", ["crash_on_import", "sys_exit"])
def test_crashing_submissions_are_errors(task, tmp_path, name):
    r = run(task, tmp_path, name)
    assert r.status == "error" and r.score == 0 and r.error


def test_hanging_submission_times_out(task, tmp_path):
    r = run(task, tmp_path, "hangs", call_timeout_s=3)
    assert r.status == "error" and "timed out" in r.error


def test_tampered_task_package_is_refused(task, tmp_path):
    import shutil

    from hotloop_schemas import public_content_hash

    t = tmp_path / "task_copy"
    shutil.copytree(task, t)
    h = public_content_hash(t)
    toml = (
        (t / "task.toml")
        .read_text()
        .replace("schema_version = 1", f'schema_version = 1\ncontent_hash = "{h}"')
    )
    (t / "task.toml").write_text(toml)
    (t / "reference.py").write_text((t / "reference.py").read_text() + "\n# edited\n")
    r = evaluate(t, write_submission(tmp_path, "honest"), EvalConfig(**FAST))
    assert r.status == "error" and "content_hash" in r.error
