"""Generate task inputs from recorded specs.

Each input has a kind:
  exact      - stored tensor, used as-is (ints, masks, buffers)
  param      - weight: random with the recorded mean/std
  activation - random with the recorded mean/std, clamped to the recorded range
Variants stress numerics on the hidden set without changing the math.
"""

import torch

VARIANTS = ("normal", "outliers", "scaled")


def _effective_std(spec) -> float:
    s = spec["stats"]
    std = s["std"]
    if spec["kind"] == "param":
        # Constant-initialised weights (norm scales = 1, biases = 0) would let a
        # kernel ignore them and still pass; real trained weights are never constant.
        std = max(std, 0.1 * abs(s["mean"]), 0.02)
    return max(std, 1e-3)


def make_inputs(meta: dict, exact: dict, seed: int, variant: str = "normal", device: str | None = None) -> list:
    if device is None:
        from hotloop.harness.device import get_device
        device = get_device().torch_device
    g = torch.Generator(device=device).manual_seed(seed)
    out = []
    for spec in meta["inputs"]:
        if spec["kind"] == "exact":
            out.append(exact[spec["name"]].to(device).clone())
            continue
        dtype = getattr(torch, spec["dtype"])
        s = spec["stats"]
        t = torch.randn(spec["shape"], generator=g, device=device, dtype=torch.float32)
        t = t * _effective_std(spec) + s["mean"]
        lo, hi = s["min"], s["max"]
        if spec["kind"] == "activation" and variant == "scaled":
            t, lo, hi = t * 4, lo * 4, hi * 4
        if spec["kind"] == "param":
            lo, hi = min(lo, s["mean"] - 4 * _effective_std(spec)), max(hi, s["mean"] + 4 * _effective_std(spec))
        t = t.clamp(lo, hi) if lo < hi else t
        if spec["kind"] == "activation" and variant == "outliers" and t.numel() > 1:
            # Sparse large-magnitude channels, like real LLM activation outliers.
            mask = torch.rand(t.shape, generator=g, device=device) < 1e-3
            t = torch.where(mask, t.sign() * max(abs(lo), abs(hi), 1.0) * 20, t)
        out.append(t.to(dtype))
    return out
