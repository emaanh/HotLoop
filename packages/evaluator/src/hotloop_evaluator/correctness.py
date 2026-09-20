"""Correctness: PyTorch reference is the oracle (EVAL.md).

Two jobs:
  * compare()    - candidate outputs vs reference outputs under the task's OutputContract.
  * calibrate()  - derive per-output tolerances at certification time by re-running the
                   reference in float64. The reference's own rounding error |ref - ref64| sets
                   the scale a *legitimate* reimplementation (different reduction order, fused
                   arithmetic) will differ by; tolerance is a safety multiple of that scale,
                   floored at torch's assert_close defaults. This admits re-association and
                   rejects "cast everything to half precision".
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from hotloop_schemas import CorrectnessDetail, OutputContract, Tolerance

# torch.testing.assert_close defaults
_DEFAULT_TOL = {
    torch.float16: (1e-3, 1e-5),
    torch.bfloat16: (1.6e-2, 1e-5),
    torch.float32: (1.3e-6, 1e-5),
    torch.float64: (1e-7, 1e-7),
}


def as_tuple(out) -> tuple[torch.Tensor, ...]:
    if isinstance(out, torch.Tensor):
        return (out,)
    if isinstance(out, (tuple, list)) and all(isinstance(o, torch.Tensor) for o in out):
        return tuple(out)
    raise TypeError(
        f"outputs must be a Tensor or a tuple/list of Tensors, got {type(out).__name__}"
    )


@dataclass
class Comparison:
    passed: bool
    max_abs_err: float = 0.0
    max_rel_err: float = 0.0
    failure: str | None = None


def compare_outputs(candidate, reference, contract: OutputContract) -> Comparison:
    try:
        cand = as_tuple(candidate)
    except TypeError as e:
        return Comparison(False, failure=str(e))
    ref = as_tuple(reference)
    if len(cand) != len(ref):
        return Comparison(False, failure=f"expected {len(ref)} outputs, got {len(cand)}")

    worst_abs = worst_rel = 0.0
    for i, (c, r, tol) in enumerate(zip(cand, ref, contract.tolerances, strict=True)):
        where = f"output[{i}]"
        if c.shape != r.shape:
            return Comparison(False, failure=f"{where}: shape {tuple(c.shape)} != {tuple(r.shape)}")
        if c.dtype != r.dtype:
            return Comparison(False, failure=f"{where}: dtype {c.dtype} != {r.dtype}")
        if c.device != r.device:
            return Comparison(False, failure=f"{where}: device {c.device} != {r.device}")
        if contract.check_strides and c.stride() != r.stride():
            return Comparison(False, failure=f"{where}: stride {c.stride()} != {r.stride()}")
        if c.data_ptr() == r.data_ptr() and c.numel() > 0:
            return Comparison(False, failure=f"{where}: aliases the reference output")

        if not (r.is_floating_point() or r.is_complex()):
            if not torch.equal(c, r):
                return Comparison(False, failure=f"{where}: integer/bool output differs")
            continue

        c64, r64 = c.to(torch.float64), r.to(torch.float64)
        c_nan, r_nan = torch.isnan(c64), torch.isnan(r64)
        if not torch.equal(c_nan, r_nan) or (bool(r_nan.any()) and not tol.equal_nan):
            return Comparison(False, failure=f"{where}: NaN pattern differs from reference")
        c_inf, r_inf = torch.isinf(c64), torch.isinf(r64)
        if not torch.equal(c_inf, r_inf) or not torch.equal(c64[r_inf], r64[r_inf]):
            return Comparison(False, failure=f"{where}: Inf pattern differs from reference")

        finite = ~(r_nan | r_inf)
        if not bool(finite.any()):
            continue
        diff = (c64[finite] - r64[finite]).abs()
        mag = r64[finite].abs()
        worst_abs = max(worst_abs, float(diff.max()))
        worst_rel = max(worst_rel, float((diff / mag.clamp_min(1e-300)).max()))
        bad = diff > (tol.atol + tol.rtol * mag)
        if bool(bad.any()):
            n_bad = int(bad.sum())
            return Comparison(
                False,
                worst_abs,
                worst_rel,
                failure=f"{where}: {n_bad}/{diff.numel()} elements outside tolerance "
                f"(max abs err {float(diff.max()):.3e}, atol {tol.atol:.1e}, rtol {tol.rtol:.1e})",
            )
    return Comparison(True, worst_abs, worst_rel)


def snapshot_inputs(inputs: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    return [x.detach().clone() if isinstance(x, torch.Tensor) else x for x in inputs]


def inputs_unmutated(inputs: Sequence, snapshot: Sequence) -> bool:
    for x, s in zip(inputs, snapshot, strict=True):
        if not isinstance(x, torch.Tensor):
            continue
        if x.shape != s.shape or not torch.equal(torch.nan_to_num(x.detach()), torch.nan_to_num(s)):
            return False
    return True


def fold(trials: Sequence[Comparison]) -> CorrectnessDetail:
    """Collapse per-trial comparisons into the result.json record. Any failure fails (D-10)."""
    first_fail = next((t for t in trials if not t.passed), None)
    return CorrectnessDetail(
        passed=first_fail is None,
        n_trials=len(trials),
        max_abs_err=max((t.max_abs_err for t in trials), default=None),
        max_rel_err=max((t.max_rel_err for t in trials), default=None),
        failure=first_fail.failure if first_fail else None,
    )


def _to_f64(x):
    return x.to(torch.float64) if isinstance(x, torch.Tensor) and x.is_floating_point() else x


def calibrate_tolerances(
    reference: Callable,
    input_sets: Sequence[Sequence[torch.Tensor]],
    *,
    safety: float = 16.0,
) -> list[Tolerance]:
    """Per-output tolerance from the reference's own rounding error against a float64 re-run.

    atol covers the absolute error floor (matters near zero); rtol covers relative error on
    elements of typical magnitude. Both are `safety` x a high quantile of what the reference
    itself exhibits, floored at assert_close defaults for the output dtype.
    """
    abs_q: list[list[float]] = []
    rel_q: list[list[float]] = []
    dtypes: list[torch.dtype] = []
    for inputs in input_sets:
        native = as_tuple(reference(*inputs))
        exact = as_tuple(reference(*[_to_f64(x) for x in inputs]))
        if not abs_q:
            abs_q = [[] for _ in native]
            rel_q = [[] for _ in native]
            dtypes = [o.dtype for o in native]
        for i, (n, e) in enumerate(zip(native, exact, strict=True)):
            if not n.is_floating_point():
                continue
            n64, e64 = n.to(torch.float64).flatten(), e.to(torch.float64).flatten()
            ok = torch.isfinite(n64) & torch.isfinite(e64)
            if not bool(ok.any()):
                continue
            err, mag = (n64[ok] - e64[ok]).abs(), e64[ok].abs()
            abs_q[i].append(float(torch.quantile(_cap(err), 0.999)))
            typical = mag >= mag.median()
            rel_q[i].append(float(torch.quantile(_cap(err[typical] / mag[typical]), 0.99)))

    tols = []
    for i, dt in enumerate(dtypes):
        rtol0, atol0 = _DEFAULT_TOL.get(dt, (0.0, 0.0))
        tols.append(
            Tolerance(
                rtol=max(rtol0, safety * max(rel_q[i], default=0.0)),
                atol=max(atol0, safety * max(abs_q[i], default=0.0)),
            )
        )
    return tols


def _cap(x: torch.Tensor, n: int = 2_000_000) -> torch.Tensor:
    """torch.quantile has an input size limit; subsample deterministically."""
    if x.numel() <= n:
        return x
    step = x.numel() // n + 1
    return x[::step]
