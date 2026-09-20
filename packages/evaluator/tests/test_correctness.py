"""CPU tests of the oracle comparison and tolerance calibration.

The central property: calibrated tolerances accept legitimate reimplementations (different
reduction order, algebraically equal forms) and reject precision dumping.
"""

import torch

from hotloop_evaluator.correctness import (
    calibrate_tolerances,
    compare_outputs,
    fold,
    inputs_unmutated,
    snapshot_inputs,
)
from hotloop_schemas import OutputContract, Tolerance


def reference(x, w):
    """softmax-weighted projection, written naively"""
    return torch.softmax(x @ w, dim=-1).sum(dim=0)


def reordered(x, w):
    """same maths, different association + manual stable softmax, chunked reduction"""
    z = torch.matmul(x, w)
    z = z - z.amax(dim=-1, keepdim=True)
    e = z.exp()
    p = e / e.sum(dim=-1, keepdim=True)
    return sum(chunk.sum(dim=0) for chunk in p.flip(0).chunk(7, dim=0))


def half_dumped(x, w):
    return reference(x.half().float(), w.half().float())


def _inputs(seed, n=512, d=64, k=96):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g), torch.randn(d, k, generator=g) * 0.5


def _contract(tols, **kw):
    return OutputContract(n_outputs=len(tols), tolerances=tols, **kw)


def test_calibration_accepts_reordering_rejects_precision_dumping():
    tols = calibrate_tolerances(reference, [_inputs(s) for s in range(4)])
    contract = _contract(tols)
    for seed in range(100, 110):  # unseen seeds
        x, w = _inputs(seed)
        ref = reference(x, w)
        assert compare_outputs(reordered(x, w), ref, contract).passed
        assert not compare_outputs(half_dumped(x, w), ref, contract).passed


def test_calibrated_tolerance_is_tighter_for_fp32_than_bf16():
    f32 = calibrate_tolerances(reference, [_inputs(0)])[0]
    bf = calibrate_tolerances(reference, [tuple(t.bfloat16() for t in _inputs(0))])[0]
    assert f32.rtol < bf.rtol


def test_contract_violations():
    c = _contract([Tolerance(rtol=1e-5, atol=1e-6)])
    ref = torch.arange(6.0).reshape(2, 3)
    ok = ref.clone()
    assert compare_outputs(ok, ref, c).passed
    assert "shape" in compare_outputs(ok.reshape(3, 2), ref, c).failure
    assert "dtype" in compare_outputs(ok.double(), ref, c).failure
    assert "expected 1 outputs" in compare_outputs((ok, ok), ref, c).failure
    assert "aliases" in compare_outputs(ref, ref, c).failure
    assert "must be a Tensor" in compare_outputs([1.0], ref, c).failure
    nan = ok.clone()
    nan[0, 0] = float("nan")
    assert "NaN" in compare_outputs(nan, ref, c).failure
    inf = ok.clone()
    inf[1, 1] = float("inf")
    assert "Inf" in compare_outputs(inf, ref, c).failure
    off = ok.clone()
    off[1, 2] += 1e-2
    assert "outside tolerance" in compare_outputs(off, ref, c).failure


def test_strides_only_checked_when_declared():
    ref = torch.randn(4, 6)
    cand = ref.t().contiguous().t()  # same values, different strides
    assert compare_outputs(cand, ref, _contract([Tolerance(rtol=0, atol=0)])).passed
    strict = _contract([Tolerance(rtol=0, atol=0)], check_strides=True)
    assert "stride" in compare_outputs(cand, ref, strict).failure


def test_integer_outputs_must_match_exactly():
    c = _contract([Tolerance(rtol=1.0, atol=1.0)])
    ref = torch.tensor([3, 1, 2])
    assert compare_outputs(ref.clone(), ref, c).passed
    assert not compare_outputs(torch.tensor([3, 2, 1]), ref, c).passed


def test_input_mutation_detected():
    x, w = _inputs(0)
    snap = snapshot_inputs((x, w))
    assert inputs_unmutated((x, w), snap)
    x.add_(1)
    assert not inputs_unmutated((x, w), snap)


def test_fold_any_failure_fails():
    c = _contract([Tolerance(rtol=1e-5, atol=1e-6)])
    ref = torch.ones(3)
    trials = [compare_outputs(ref.clone(), ref, c), compare_outputs(ref * 2, ref, c)]
    detail = fold(trials)
    assert not detail.passed and detail.n_trials == 2 and "outside tolerance" in detail.failure
