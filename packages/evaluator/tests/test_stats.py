"""Statistical behaviour of the timing pipeline on synthetic noise.

These encode gate G0's requirements (PLAN.md M1) as properties of the estimator, so that if the
real hardware meets the assumed noise level, the conclusions follow.
"""

import numpy as np
import pytest

from hotloop_evaluator.stats import StoppingRule, paired_ratio, weighted_geomean


def _blocks(rng, n, t, cv, drift=None):
    """Lognormal multiplicative noise with optional shared drift per pair."""
    sigma = np.sqrt(np.log1p(cv**2))
    x = t * rng.lognormal(-(sigma**2) / 2, sigma, size=n)
    return x * (drift if drift is not None else 1.0)


def test_recovers_true_speedup_and_covers():
    rng = np.random.default_rng(1)
    true, hits, trials = 1.37, 0, 300
    for i in range(trials):
        b = _blocks(rng, 30, 1e-3 * true, 0.03)
        c = _blocks(rng, 30, 1e-3, 0.03)
        r = paired_ratio(b, c, seed=i, n_boot=1000)
        hits += r.lo <= true <= r.hi
    assert hits / trials > 0.90  # nominal 0.95; bootstrap-of-median is slightly liberal


def test_shared_drift_cancels():
    """A 30% slow ramp over the session (thermal/clock drift) must not bias the ratio."""
    rng = np.random.default_rng(2)
    drift = np.linspace(1.0, 1.3, 40)
    b = _blocks(rng, 40, 2e-3, 0.01, drift)
    c = _blocks(rng, 40, 1e-3, 0.01, drift)
    r = paired_ratio(b, c)
    assert r.lo <= 2.0 <= r.hi
    assert r.rel_halfwidth < 0.02


def test_outlier_blocks_do_not_move_the_estimate():
    rng = np.random.default_rng(3)
    b = _blocks(rng, 40, 1.5e-3, 0.02)
    c = _blocks(rng, 40, 1e-3, 0.02)
    c[[3, 17]] *= 25  # host hiccups
    r = paired_ratio(b, c)
    assert abs(r.point - 1.5) / 1.5 < 0.03


@pytest.mark.parametrize("cv,pairs", [(0.03, 30), (0.08, 120)])
def test_g0_planted_10pct_improvement_is_detected(cv, pairs):
    """G0: a planted 1.10x is detected (CI excludes 1.0) >= 95% of the time, at the noise
    levels G0 allows for device-bound (3%) and launch-bound (8%) work."""
    rng = np.random.default_rng(4)
    detected, trials = 0, 300
    for i in range(trials):
        b = _blocks(rng, pairs, 1.10e-3, cv)
        c = _blocks(rng, pairs, 1.00e-3, cv)
        r = paired_ratio(b, c, seed=i, n_boot=1000)
        detected += r.lo > 1.0
    assert detected / trials >= 0.95


def test_no_false_speedups_under_the_null():
    rng = np.random.default_rng(5)
    false_pos, trials = 0, 400
    for i in range(trials):
        b = _blocks(rng, 30, 1e-3, 0.03)
        c = _blocks(rng, 30, 1e-3, 0.03)
        false_pos += paired_ratio(b, c, seed=i, n_boot=1000).excludes(1.0)
    assert false_pos / trials < 0.10


def test_stopping_rule():
    rule = StoppingRule(target_rel_halfwidth=0.01, min_pairs=8, max_pairs=50)
    rng = np.random.default_rng(6)
    tight = paired_ratio(_blocks(rng, 20, 2e-3, 0.002), _blocks(rng, 20, 1e-3, 0.002))
    loose = paired_ratio(_blocks(rng, 20, 2e-3, 0.2), _blocks(rng, 20, 1e-3, 0.2))
    assert rule.decide(None, 0) == "continue"
    assert rule.decide(tight, 4) == "continue"  # below min_pairs even if tight
    assert rule.decide(tight, 20) == "converged"
    assert rule.decide(loose, 20) == "continue"
    assert rule.decide(loose, 50) == "exhausted"


def test_weighted_geomean():
    assert weighted_geomean([2.0, 0.5], [1, 1]) == pytest.approx(1.0)
    assert weighted_geomean([4.0, 1.0], [3, 1]) == pytest.approx(4**0.75)
    with pytest.raises(ValueError):
        weighted_geomean([1.0, 0.0], [1, 1])


def test_input_validation():
    with pytest.raises(ValueError):
        paired_ratio([1.0, 1.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        paired_ratio([1.0, 1.0, 1.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        paired_ratio([1.0, -1.0, 1.0], [1.0, 1.0, 1.0])
