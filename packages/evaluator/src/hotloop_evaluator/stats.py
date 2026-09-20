"""Timing statistics. Pure numpy, no GPU: everything here is testable on synthetic noise.

Measurement model (EVAL.md): baseline and candidate are timed in *interleaved blocks* so slow
drift (clocks, temperature, neighbours) hits both sides of a pair equally and cancels in the
ratio. A block's time is the median of its calls (robust to a stray host hiccup); a pair's
speedup is t_baseline / t_candidate; the reported speedup is the geometric median of pair
speedups with a bootstrap CI over pairs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Ratio:
    point: float
    lo: float
    hi: float
    level: float
    n_pairs: int

    @property
    def rel_halfwidth(self) -> float:
        return (self.hi - self.lo) / (2 * self.point)

    def excludes(self, value: float = 1.0) -> bool:
        return self.lo > value or self.hi < value


def block_time(call_times_s: Sequence[float]) -> float:
    if len(call_times_s) == 0:
        raise ValueError("empty block")
    return float(np.median(np.asarray(call_times_s, dtype=np.float64)))


def paired_ratio(
    baseline_blocks_s: Sequence[float],
    candidate_blocks_s: Sequence[float],
    *,
    level: float = 0.95,
    n_boot: int = 4000,
    seed: int = 0,
) -> Ratio:
    """Speedup = t_baseline / t_candidate from paired interleaved blocks."""
    b = np.asarray(baseline_blocks_s, dtype=np.float64)
    c = np.asarray(candidate_blocks_s, dtype=np.float64)
    if b.shape != c.shape or b.ndim != 1:
        raise ValueError("baseline and candidate blocks must be paired 1-D sequences")
    if b.size < 3:
        raise ValueError("need at least 3 pairs for a CI")
    if np.any(b <= 0) or np.any(c <= 0):
        raise ValueError("non-positive time")
    logr = np.log(b / c)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, logr.size, size=(n_boot, logr.size))
    boots = np.median(logr[idx], axis=1)
    alpha = (1 - level) / 2
    lo, hi = np.quantile(boots, [alpha, 1 - alpha])
    return Ratio(
        point=float(np.exp(np.median(logr))),
        lo=float(np.exp(lo)),
        hi=float(np.exp(hi)),
        level=level,
        n_pairs=int(logr.size),
    )


@dataclass(frozen=True)
class StoppingRule:
    """Adaptive repeats: keep adding pairs until the CI is tight enough or a cap is hit."""

    target_rel_halfwidth: float = 0.01
    min_pairs: int = 8
    max_pairs: int = 200

    def decide(self, ratio: Ratio | None, n_pairs: int) -> str:
        """-> 'continue' | 'converged' | 'exhausted'"""
        if n_pairs < self.min_pairs or ratio is None:
            return "continue"
        if ratio.rel_halfwidth <= self.target_rel_halfwidth:
            return "converged"
        return "exhausted" if n_pairs >= self.max_pairs else "continue"


def weighted_geomean(values: Sequence[float], weights: Sequence[float]) -> float:
    """Task score across workload entries (DESIGN §2)."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.shape != w.shape or v.size == 0:
        raise ValueError("values and weights must be same non-zero length")
    if np.any(v <= 0) or np.any(w <= 0):
        raise ValueError("values and weights must be positive")
    return float(math.exp(float(np.sum(w * np.log(v)) / np.sum(w))))


def summarize(call_times_s: Sequence[float]) -> dict[str, float]:
    a = np.asarray(call_times_s, dtype=np.float64)
    q25, q50, q75 = np.quantile(a, [0.25, 0.5, 0.75])
    return {
        "n": int(a.size),
        "median_s": float(q50),
        "iqr_s": float(q75 - q25),
        "mean_s": float(a.mean()),
        "min_s": float(a.min()),
    }
