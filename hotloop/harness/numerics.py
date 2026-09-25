"""fp64 reference, tolerance calibration, and output comparison.

The tolerance is not hand-picked: it is derived from how far PyTorch's own
native-dtype reference is from the fp64 reference on the same inputs.
"""

import torch
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_flatten, tree_map

ABS_FACTOR = 8.0   # max-abs error allowed, relative to the reference's own error
REL_FACTOR = 4.0   # relative L2 error allowed, relative to the reference's own error


class Upcast(TorchFunctionMode):
    """Runs a reference with every floating dtype replaced by float64, including
    explicit casts like `.to(torch.bfloat16)` inside the reference."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in (torch.Tensor.float, torch.Tensor.half, torch.Tensor.bfloat16):
            return args[0].double()
        fix = lambda x: torch.float64 if isinstance(x, torch.dtype) and x.is_floating_point else x
        return func(*tree_map(fix, args), **tree_map(fix, kwargs))


def upcast_inputs(inputs):
    return [t.double() if t.is_floating_point() else t for t in inputs]


def flat(x) -> list:
    return tree_flatten(x)[0]


def run_ref64(ref, inputs, mutated=(), device: str | None = None):
    """Outputs of the reference in fp64, followed by any inputs it updates in place. `device`
    is where fp64 runs (the CPU on Apple GPUs, which have no fp64)."""
    ins = upcast_inputs([t.clone().to(device) if device else t.clone() for t in inputs])
    with torch.no_grad(), Upcast():
        return flat(ref(*ins)) + [ins[i] for i in mutated]


def run_native(ref, inputs, mutated=()):
    ins = [t.clone() for t in inputs]
    with torch.no_grad():
        return flat(ref(*ins)) + [ins[i] for i in mutated]


def _f64_device(t: torch.Tensor) -> torch.device:
    """Where fp64 comparisons can run: the tensor's own device, or the CPU for Apple GPUs."""
    return torch.device("cpu") if t.device.type == "mps" else t.device


def _errors(x: torch.Tensor, r: torch.Tensor) -> tuple[float, float]:
    dev = _f64_device(r)
    x, r = x.to(dev), r.to(dev)
    finite = torch.isfinite(r)
    d = (x.double() - r.double())[finite]
    if d.numel() == 0:
        return 0.0, 0.0
    max_abs = d.abs().max().item()
    rel = (d.norm() / r[finite].norm().clamp_min(1e-30)).item()
    return max_abs, rel


def calibrate(native: list, ref64: list) -> list[dict]:
    """Per-output tolerance from the native reference's own error."""
    tols = []
    for n, r in zip(native, ref64):
        if not n.is_floating_point():
            tols.append({"exact": True})
            continue
        e_abs, e_rel = _errors(n, r)
        eps = torch.finfo(n.dtype).eps
        rr = r.to(_f64_device(r))
        finite = rr[torch.isfinite(rr)].double()
        scale = finite.abs().mean().item() if finite.numel() else 1.0
        tols.append({
            "exact": False,
            "abs": ABS_FACTOR * e_abs + eps * scale,
            "rel": REL_FACTOR * e_rel + eps,
            "ref_abs": e_abs, "ref_rel": e_rel,
        })
    return tols


def compare(sol: list, ref: list, native: list, tols: list[dict]) -> dict:
    """Check solution outputs against the reference (fp64 or native)."""
    if len(sol) != len(native):
        return {"ok": False, "reason": f"expected {len(native)} outputs, got {len(sol)}"}
    worst = {"ok": True, "max_abs": 0.0, "rel": 0.0}
    for i, (s, r, n, tol) in enumerate(zip(sol, ref, native, tols)):
        if not isinstance(s, torch.Tensor):
            return {"ok": False, "reason": f"output {i} is {type(s).__name__}, expected tensor"}
        if s.shape != n.shape or s.dtype != n.dtype:
            return {"ok": False, "reason": f"output {i}: got {s.dtype}{list(s.shape)}, expected {n.dtype}{list(n.shape)}"}
        if tol["exact"]:
            if not torch.equal(s.cpu(), n.cpu()):
                return {"ok": False, "reason": f"output {i}: integer/bool output differs"}
            continue
        dev = _f64_device(r)
        s, r = s.to(dev), r.to(dev)
        finite = torch.isfinite(r)
        bad_nonfinite = (~finite & ~((s.double() == r.double()) | (torch.isnan(s) & torch.isnan(r)))).any()
        if bad_nonfinite or (~torch.isfinite(s[finite])).any():
            return {"ok": False, "reason": f"output {i}: inf/nan pattern differs from reference"}
        e_abs, e_rel = _errors(s, r)
        worst["max_abs"], worst["rel"] = max(worst["max_abs"], e_abs), max(worst["rel"], e_rel)
        if e_abs > tol["abs"] or e_rel > tol["rel"]:
            return {**worst, "ok": False, "reason": (
                f"output {i}: max_abs_err={e_abs:.3g} (tol {tol['abs']:.3g}), "
                f"rel_l2_err={e_rel:.3g} (tol {tol['rel']:.3g})")}
    return worst
