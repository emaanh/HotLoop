"""Speed-of-light bound per task shape on the current GPU.

Peaks are measured on the device (large GEMM, large copy), so any GPU works
without a spec table. FLOPs are counted from the reference; for attention with
a boolean mask, only unmasked positions count, so skipping masked work is not
mistaken for cheating.
"""

import torch
from torch.utils._python_dispatch import TorchDispatchMode

aten = torch.ops.aten


def measure_peaks() -> dict:
    """Achievable peaks on the current device: FLOP/s per dtype and memory bandwidth in bytes/s."""
    from hotloop.harness.device import get_device

    dev = get_device()
    d = dev.torch_device
    peaks = {"gpu": dev.device_name(), "device": dev.name}
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    # Best over several sizes: power-limited GPUs (e.g. L4) clock down on the largest GEMMs.
    for dtype, sizes in ((torch.bfloat16, (2048, 4096, 8192)), (torch.float16, (2048, 4096, 8192)),
                         (torch.float32, (2048, 4096))):
        best = 0.0
        for n in sizes:
            a = torch.randn(n, n, device=d, dtype=dtype)
            b = torch.randn(n, n, device=d, dtype=dtype)
            best = max(best, 2 * n**3 / (dev.time_ms(lambda: a @ b) * 1e-3))
            del a, b
        peaks[str(dtype).removeprefix("torch.")] = best
    torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    best = 0.0
    for n in (1 << 26, 1 << 28):  # 256 MiB and 1 GiB, both far larger than any on-chip cache
        x = torch.empty(n, device=d, dtype=torch.float32)
        y = torch.empty_like(x)
        best = max(best, 2 * n * 4 / (dev.time_ms(lambda: y.copy_(x)) * 1e-3))
        del x, y
    peaks["bandwidth"] = best
    dev.empty_cache()
    return peaks


class FlopCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.flops = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = func.overloadpacket.__name__
        if func is aten.linear.default:  # some backends (e.g. MPS) run linear as one op
            x, w = args[0], args[1]
            self.flops += 2 * (x.numel() // x.shape[-1]) * w.shape[0] * w.shape[1]
        elif func in (aten.mm.default, aten.addmm.default):
            a, b = (args[1], args[2]) if func is aten.addmm.default else (args[0], args[1])
            self.flops += 2 * a.shape[0] * a.shape[1] * b.shape[1]
        elif func in (aten.bmm.default, aten.baddbmm.default):
            a, b = (args[1], args[2]) if func is aten.baddbmm.default else (args[0], args[1])
            self.flops += 2 * a.shape[0] * a.shape[1] * a.shape[2] * b.shape[2]
        elif name.startswith("_scaled_dot_product") and "backward" not in name:
            q, k, v = args[0], args[1], args[2]
            mask = kwargs.get("attn_mask", kwargs.get("attn_bias"))
            if mask is None and len(args) > 3 and isinstance(args[3], torch.Tensor):
                mask = args[3]
            causal = kwargs.get("is_causal", False) or any(a is True for a in args[3:])
            density = 1.0
            if isinstance(mask, torch.Tensor):
                density = mask.float().mean().item() if mask.dtype == torch.bool else (mask > -1e30).float().mean().item()
            elif causal:
                density = 0.5
            L, S = q.shape[-2], k.shape[-2]
            batch = q.numel() // (L * q.shape[-1])
            self.flops += int(2 * batch * L * S * (q.shape[-1] + v.shape[-1]) * density)
        elif name == "convolution":
            from torch.utils.flop_counter import conv_flop_count
            self.flops += conv_flop_count(args[0].shape, args[1].shape, out.shape, args[6])
        return out


def count_flops(ref, inputs) -> int:
    counter = FlopCounter()
    with torch.no_grad(), counter:
        ref(*[t.clone() for t in inputs])
    return counter.flops


def io_bytes(inputs, outputs) -> int:
    seen, total = set(), 0
    for t in list(inputs) + list(outputs):
        key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
        if key not in seen:
            seen.add(key)
            total += t.numel() * t.element_size()
    return total


def compute_dtype(inputs) -> str:
    floats = [t for t in inputs if t.is_floating_point()]
    if not floats:
        return "float32"
    big = max(floats, key=lambda t: t.numel())
    return {torch.bfloat16: "bfloat16", torch.float16: "float16"}.get(big.dtype, "float32")


def sol_ms(flops: int, nbytes: int, dtype: str, peaks: dict) -> float:
    return max(flops / peaks[dtype], nbytes / peaks["bandwidth"]) * 1e3
