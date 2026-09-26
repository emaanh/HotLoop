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
OVERHEAD_FACTOR = 3.0   # baseline must exceed this multiple of the device's timing floor
_floor_ms = None


def timing_floor_ms() -> float:
    """Time of a trivial operation through the scoring timing path: the per-call overhead
    (tiny on CUDA-graph replay, ~0.2 ms for eager launch+sync on Apple GPUs). Tasks whose
    baseline is near this floor measure launch overhead, not kernel work."""
    global _floor_ms
    if _floor_ms is None:
        from hotloop.harness.device import get_device
        x = torch.zeros(1, device=get_device().torch_device)
        _floor_ms = graph_time_ms(lambda t: t.add_(0), [x], iters=30)
    return _floor_ms
MAX_BASELINE_SOL_FRAC = 0.75
SENSITIVITY = 1e-3


def _rel_diff(a: list, b: list) -> float:
    worst = 0.0
    for x, y in zip(a, b):
        if not x.is_floating_point():
            worst = max(worst, float(not torch.equal(x.cpu(), y.cpu())))
            continue
        x, y = x.cpu().double(), y.cpu().double()  # CPU: Apple GPUs have no fp64
        worst = max(worst, ((x - y).norm() / x.norm().clamp_min(1e-30)).item())
    return worst


def graph_time_ms(fn, inputs, iters: int = 20) -> float:
    """Median time of `fn` on `inputs`, replayed the way scoring does it (CUDA graph on NVIDIA,
    eager on Apple GPUs) with a cache flush before every run."""
    from hotloop.harness.device import get_device

    dev = get_device()
    flush = dev.make_flush()
    with torch.no_grad():
        rep = dev.capture(lambda ins: fn(*ins), inputs, lambda x: x)
        times = []
        for _ in range(iters):
            flush()
            dev.synchronize()
            start, stop = dev.timer()
            start()
            rep.replay()
            times.append(stop())
    return sorted(times)[len(times) // 2]


def _resample(meta: dict, exact: dict, base: list, seed: int, kind: str) -> list:
    """Resample only inputs of one kind, keep the rest fixed."""
    fresh = make_inputs(meta, exact, seed)
    return [f if spec["kind"] == kind else b for spec, f, b in zip(meta["inputs"], fresh, base)]


def filter_shape(shape, peaks: dict) -> dict:
    from hotloop.harness.device import get_device

    dev = get_device()
    ref = shape.reference(dev.torch_device)
    meta, exact = shape.meta, shape.exact
    mutated = meta.get("mutated_inputs", [])
    run = lambda ins: numerics.run_native(ref, ins, mutated)
    inputs = make_inputs(meta, exact, 1)
    out = run(inputs)
    kinds = {s["kind"] for s in meta["inputs"]}
    stats = {}
    stats["activation_sensitivity"] = _rel_diff(out, run(_resample(meta, exact, inputs, 2, "activation"))) if "activation" in kinds else None
    stats["param_sensitivity"] = _rel_diff(out, run(_resample(meta, exact, inputs, 3, "param"))) if "param" in kinds else None
    floats = [o.cpu().double() for o in out if o.is_floating_point()]
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
    stats["timing_floor_ms"] = timing_floor_ms()
    min_ms = max(MIN_BASELINE_MS, OVERHEAD_FACTOR * stats["timing_floor_ms"])
    if base < min_ms:
        reasons.append(f"too small to time (baseline {base * 1e3:.1f}us, device overhead "
                       f"{stats['timing_floor_ms'] * 1e3:.1f}us)")
    if stats["baseline_sol_frac"] > MAX_BASELINE_SOL_FRAC:
        reasons.append(f"no headroom (baseline at {100 * stats['baseline_sol_frac']:.0f}% of speed-of-light)")
    stats["keep"] = not reasons
    stats["drop_reasons"] = reasons
    torch._dynamo.reset()
    dev.empty_cache()
    return stats
