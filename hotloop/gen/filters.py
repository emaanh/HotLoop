"""Automatic task filters (run per GPU type, results cached).

A task is kept only if:
  - its output depends on its activations and on its weights (not degenerate)
  - its output is not near-constant
  - it runs long enough to time reliably
  - the baseline (torch.compile) is well below the speed-of-light bound (there is headroom)
"""

import torch

from hotloop.harness import numerics, roofline
from hotloop.harness.inputs import make_inputs

MIN_BASELINE_MS = 0.015
MAX_BASELINE_SOL_FRAC = 0.75
SENSITIVITY = 1e-3


def _rel_diff(a: list, b: list) -> float:
    worst = 0.0
    for x, y in zip(a, b):
        if not x.is_floating_point():
            worst = max(worst, float(not torch.equal(x, y)))
            continue
        x, y = x.double(), y.double()
        worst = max(worst, ((x - y).norm() / x.norm().clamp_min(1e-30)).item())
    return worst


def graph_time_ms(fn, inputs, iters: int = 20) -> float:
    flush = torch.empty(4 * torch.cuda.get_device_properties(0).L2_cache_size, dtype=torch.uint8, device="cuda")
    with torch.no_grad():
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fn(*inputs)
            fn(*inputs)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn(*inputs)
    times = []
    for _ in range(iters):
        flush.zero_()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    return sorted(times)[len(times) // 2]


def _resample(meta: dict, exact: dict, base: list, seed: int, kind: str) -> list:
    """Resample only inputs of one kind, keep the rest fixed."""
    fresh = make_inputs(meta, exact, seed)
    return [f if spec["kind"] == kind else b for spec, f, b in zip(meta["inputs"], fresh, base)]


def filter_shape(shape, peaks: dict) -> dict:
    ref = shape.reference()
    meta, exact = shape.meta, shape.exact
    mutated = meta.get("mutated_inputs", [])
    run = lambda ins: numerics.run_native(ref, ins, mutated)
    inputs = make_inputs(meta, exact, 1)
    out = run(inputs)
    kinds = {s["kind"] for s in meta["inputs"]}
    stats = {}
    stats["activation_sensitivity"] = _rel_diff(out, run(_resample(meta, exact, inputs, 2, "activation"))) if "activation" in kinds else None
    stats["param_sensitivity"] = _rel_diff(out, run(_resample(meta, exact, inputs, 3, "param"))) if "param" in kinds else None
    floats = [o.double() for o in out if o.is_floating_point()]
    stats["output_spread"] = min((o.std() / o.abs().mean().clamp_min(1e-30)).item() for o in floats) if floats else 0.0
    stats["eager_ms"] = graph_time_ms(ref, inputs)
    try:
        stats["compile_ms"] = graph_time_ms(torch.compile(ref, dynamic=False), inputs)
    except Exception as e:  # a baseline that fails to compile just doesn't count
        stats["compile_ms"] = None
        stats["compile_error"] = f"{type(e).__name__}: {e}"[:300]
    flops = roofline.count_flops(ref, inputs)
    nbytes = roofline.io_bytes(inputs, out[: len(out) - len(mutated)])
    stats["sol_ms"] = roofline.sol_ms(flops, nbytes, roofline.compute_dtype(inputs), peaks)
    base = stats["compile_ms"] if stats["compile_ms"] is not None else stats["eager_ms"]
    stats["baseline_sol_frac"] = stats["sol_ms"] / base

    reasons = []
    for k in ("activation_sensitivity", "param_sensitivity"):
        if stats[k] is not None and stats[k] < SENSITIVITY:
            reasons.append(f"output ignores its {k.split('_')[0]}s")
    if stats["output_spread"] < SENSITIVITY:
        reasons.append("output is near-constant")
    if base < MIN_BASELINE_MS:
        reasons.append(f"too small to time (baseline {base * 1e3:.1f}us)")
    if stats["baseline_sol_frac"] > MAX_BASELINE_SOL_FRAC:
        reasons.append(f"no headroom (baseline at {100 * stats['baseline_sol_frac']:.0f}% of speed-of-light)")
    stats["keep"] = not reasons
    stats["drop_reasons"] = reasons
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    return stats
