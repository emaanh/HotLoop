"""M2 exploration: do regime flips with real headroom over the compiler exist? (PLAN.md gate G1)

Exploratory, not the evaluator: fixed inputs, simple blocked timing. Its job is to find where in
workload space (a) the argmax strategy changes and (b) the best strategy clearly beats the best
*automatic* baseline. Families that survive become taskgen families certified with the real
evaluator.

usage: python explore_regimes.py --family chain|ragged --out out.json
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import time

import torch

DEV = "cuda"


# --------------------------------------------------------------------------------------- timing
def bench(fn, args, min_time=0.4, max_calls=2000):
    """median seconds per call over blocks; assumes fn already warm."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn(*args)
    torch.cuda.synchronize()
    one = max(time.perf_counter() - t0, 1e-6)
    k = int(min(max_calls, max(1, 0.02 / one)))
    blocks, t_end = [], time.perf_counter() + min_time
    while time.perf_counter() < t_end or len(blocks) < 5:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(k):
            fn(*args)
        torch.cuda.synchronize()
        blocks.append((time.perf_counter() - t0) / k)
    blocks.sort()
    return blocks[len(blocks) // 2]


def warm(fn, args, n=3):
    for _ in range(n):
        out = fn(*args)
    torch.cuda.synchronize()
    return out


def auto_baselines(ref):
    return {
        "eager": ref,
        "compile_default": torch.compile(ref),
        "compile_max_autotune_nocg": torch.compile(ref, mode="max-autotune-no-cudagraphs"),
        "compile_reduce_overhead": torch.compile(ref, mode="reduce-overhead"),
    }


def measure_point(ref, strategies, args, check):
    want = warm(ref, args, 1)
    row = {"auto": {}, "strategies": {}, "errors": {}}
    for group, fns in (("auto", auto_baselines(ref)), ("strategies", strategies)):
        for name, fn in fns.items():
            try:
                torch._dynamo.reset() if group == "auto" else None
                got = warm(fn, args)
                if not check(got, want):
                    row["errors"][name] = "mismatch"
                    continue
                row[group][name] = bench(fn, args)
            except Exception as e:  # noqa: BLE001 - exploration: record and move on
                row["errors"][name] = f"{type(e).__name__}: {str(e)[:160]}"
            finally:
                got = None
                torch.cuda.empty_cache()
    best_auto = min(row["auto"], key=row["auto"].get)
    row["best_auto"], row["best_auto_s"] = best_auto, row["auto"][best_auto]
    if row["strategies"]:
        best = min(row["strategies"], key=row["strategies"].get)
        row["best_strategy"] = best
        row["headroom"] = row["best_auto_s"] / row["strategies"][best]
        row["speedup_vs_auto"] = {k: row["best_auto_s"] / v for k, v in row["strategies"].items()}
    return row


# ------------------------------------------------------------------------ family: matrix chain
def chain_orders(n):
    """all full parenthesisations of M0..M{n-1} as nested tuples"""

    @functools.cache
    def go(i, j):
        if i == j:
            return [i]
        return [(a, b) for s in range(i, j) for a in go(i, s) for b in go(s + 1, j)]

    return go(0, n - 1)


def order_name(o):
    return str(o) if isinstance(o, int) else f"({order_name(o[0])}{order_name(o[1])})"


def order_fn(o):
    def ev(node, ms):
        return ms[node] if isinstance(node, int) else ev(node[0], ms) @ ev(node[1], ms)

    return lambda *ms: ev(o, ms)


def order_flops(o, dims):
    def go(node):
        if isinstance(node, int):
            return dims[node], dims[node + 1], 0
        r1, c1, f1 = go(node[0])
        r2, c2, f2 = go(node[1])
        return r1, c2, f1 + f2 + 2 * r1 * c1 * c2

    return go(o)[2]


CHAIN_GRID = {
    "lowrank_sandwich_narrow_out": [8192, 64, 8192, 64, 16],
    "narrow_in_wide_out": [16, 8192, 64, 8192, 8192],
    "bottleneck_middle": [4096, 4096, 32, 4096, 4096],
    "tall_skinny_alternating": [8192, 128, 4096, 128, 2048],
    "wide_then_vector": [2048, 8192, 8192, 2048, 1],
    "all_square_small": [512, 512, 512, 512, 512],
    "batchy_left_small": [64, 4096, 4096, 4096, 64],
}


def run_chain(dtype=torch.float32):
    out = {}
    orders = chain_orders(4)
    for name, dims in CHAIN_GRID.items():
        g = torch.Generator(device=DEV).manual_seed(0)
        ms = tuple(
            torch.randn(dims[i], dims[i + 1], device=DEV, dtype=dtype, generator=g)
            / dims[i + 1] ** 0.5
            for i in range(4)
        )

        def ref(a, b, c, d):
            return a @ b @ c @ d

        strategies = {order_name(o): order_fn(o) for o in orders}
        strategies["multi_dot"] = lambda *m: torch.linalg.multi_dot(m)
        row = measure_point(
            ref, strategies, ms, lambda g_, w: torch.allclose(g_, w, rtol=2e-2, atol=1e-3)
        )
        row["dims"] = dims
        row["flops"] = {order_name(o): order_flops(o, dims) for o in orders}
        out[name] = row
        print(
            name,
            dims,
            "best_auto",
            row["best_auto"],
            f"{row['best_auto_s'] * 1e3:.3f}ms",
            "best",
            row.get("best_strategy"),
            f"headroom {row.get('headroom', 0):.2f}x",
            row["errors"] or "",
            flush=True,
        )
    return out


# --------------------------------------------------------------------- family: ragged pooling
def make_ragged(S, dist, d, dtype, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if dist["kind"] == "uniform":
        lens = torch.full((S,), dist["len"], dtype=torch.int64)
    elif dist["kind"] == "zipf":
        u = torch.rand(S, generator=g)
        lens = (
            (dist["min"] * (1 - u) ** (-1 / dist["alpha"])).clamp(max=dist["max"]).to(torch.int64)
        )
    elif dist["kind"] == "bimodal":
        lens = torch.where(
            torch.rand(S, generator=g) < dist["p_long"], dist["long"], dist["short"]
        ).to(torch.int64)
    lmax = dist.get("max") or dist.get("long") or dist["len"]
    offsets = torch.zeros(S + 1, dtype=torch.int64)
    offsets[1:] = lens.cumsum(0)
    T = int(offsets[-1])
    gd = torch.Generator(device=DEV).manual_seed(seed)
    v = torch.randn(T, d, device=DEV, dtype=dtype, generator=gd)
    s = torch.randn(T, device=DEV, dtype=dtype, generator=gd)
    return (v, s, offsets.to(DEV)), lmax, T


def ragged_reference(lmax):
    def reference(v, s, offsets):
        lens = offsets[1:] - offsets[:-1]
        idx = torch.arange(lmax, device=v.device)[None, :]
        mask = idx < lens[:, None]
        pos = (offsets[:-1, None] + idx).clamp_max(v.shape[0] - 1)
        sp = s[pos].masked_fill(~mask, float("-inf"))
        w = torch.softmax(sp.float(), dim=-1).to(v.dtype)
        return (w[..., None] * v[pos]).sum(1)

    return reference


def ragged_seg(v, s, offsets):
    S = offsets.numel() - 1
    lens = offsets[1:] - offsets[:-1]
    seg = torch.repeat_interleave(torch.arange(S, device=v.device), lens, output_size=v.shape[0])
    sf = s.float()
    m = torch.full((S,), float("-inf"), device=v.device).scatter_reduce(0, seg, sf, "amax")
    e = torch.exp(sf - m[seg])
    z = torch.zeros(S, device=v.device).index_add_(0, seg, e)
    w = (e / z[seg]).to(v.dtype)
    return torch.zeros(S, v.shape[1], device=v.device, dtype=v.dtype).index_add_(
        0, seg, w[:, None] * v
    )


def build_triton_ragged():
    import triton
    import triton.language as tl

    @triton.jit
    def _pool(V, Sc, OFF, OUT, D: tl.constexpr, BLOCK: tl.constexpr):
        seg = tl.program_id(0)
        start = tl.load(OFF + seg)
        end = tl.load(OFF + seg + 1)
        m = tl.full((), float("-inf"), tl.float32)
        for lo in range(start, end, BLOCK):
            i = lo + tl.arange(0, BLOCK)
            x = tl.load(Sc + i, mask=i < end, other=float("-inf")).to(tl.float32)
            m = tl.maximum(m, tl.max(x, axis=0))
        z = tl.zeros((), tl.float32)
        acc = tl.zeros((D,), tl.float32)
        dcols = tl.arange(0, D)
        for lo in range(start, end, BLOCK):
            i = lo + tl.arange(0, BLOCK)
            ok = i < end
            x = tl.load(Sc + i, mask=ok, other=float("-inf")).to(tl.float32)
            e = tl.exp(x - m)
            z += tl.sum(e, axis=0)
            vals = tl.load(V + i[:, None] * D + dcols[None, :], mask=ok[:, None], other=0.0).to(
                tl.float32
            )
            acc += tl.sum(e[:, None] * vals, axis=0)
        tl.store(OUT + seg * D + dcols, acc / z)

    def run(v, s, offsets, block=64):
        S, D = offsets.numel() - 1, v.shape[1]
        out = torch.empty(S, D, device=v.device, dtype=torch.float32)
        _pool[(S,)](v, s, offsets, out, D=D, BLOCK=block)
        return out.to(v.dtype)

    return {f"triton_jagged_b{b}": functools.partial(run, block=b) for b in (16, 128, 1024)}


def ragged_bucketed(v, s, offsets):
    """sort segments into power-of-two length buckets; dense padded softmax-pool per bucket"""
    S = offsets.numel() - 1
    lens = offsets[1:] - offsets[:-1]
    out = torch.empty(S, v.shape[1], device=v.device, dtype=v.dtype)
    bucket = torch.ceil(torch.log2(lens.clamp_min(1).float())).to(torch.int64)
    for b in bucket.unique().tolist():
        ids = (bucket == b).nonzero().squeeze(1)
        L = 1 << b
        idx = torch.arange(L, device=v.device)[None, :]
        mask = idx < lens[ids, None]
        pos = (offsets[ids, None] + idx).clamp_max(v.shape[0] - 1)
        sp = s[pos].masked_fill(~mask, float("-inf"))
        w = torch.softmax(sp.float(), dim=-1).to(v.dtype)
        out[ids] = (w[..., None] * v[pos]).sum(1)
    return out


RAGGED_GRID = {
    "uniform_dense_len64": dict(S=16384, d=32, dist={"kind": "uniform", "len": 64}),
    "many_tiny_len4": dict(S=262144, d=32, dist={"kind": "uniform", "len": 4}),
    "few_long_len16k": dict(S=64, d=32, dist={"kind": "uniform", "len": 16384}),
    "zipf_heavy_tail": dict(
        S=16384, d=32, dist={"kind": "zipf", "min": 8, "alpha": 1.1, "max": 4096}
    ),
    "zipf_mild_tail": dict(
        S=16384, d=32, dist={"kind": "zipf", "min": 32, "alpha": 3.0, "max": 512}
    ),
    "bimodal_1pct_long": dict(
        S=8192, d=32, dist={"kind": "bimodal", "p_long": 0.01, "long": 4096, "short": 16}
    ),
    "zipf_heavy_tail_wide_d256": dict(
        S=4096, d=256, dist={"kind": "zipf", "min": 8, "alpha": 1.1, "max": 2048}
    ),
}


def run_ragged(dtype=torch.float16):
    out = {}
    tri = build_triton_ragged()
    for name, p in RAGGED_GRID.items():
        args, lmax, T = make_ragged(p["S"], p["dist"], p["d"], dtype)
        strategies = {"seg_scatter": ragged_seg, "bucketed_dense": ragged_bucketed, **tri}
        strategies["seg_scatter_compiled"] = torch.compile(ragged_seg)
        row = measure_point(
            ragged_reference(lmax),
            strategies,
            args,
            lambda g_, w: torch.allclose(g_.float(), w.float(), rtol=2e-2, atol=2e-3),
        )
        row.update(S=p["S"], d=p["d"], T=T, lmax=lmax, pad_waste=p["S"] * lmax / T)
        out[name] = row
        print(
            name,
            f"T={T} lmax={lmax} waste={row['pad_waste']:.1f}x",
            "best_auto",
            row["best_auto"],
            f"{row['best_auto_s'] * 1e3:.3f}ms",
            "best",
            row.get("best_strategy"),
            f"headroom {row.get('headroom', 0):.2f}x",
            row["errors"] or "",
            flush=True,
        )
    return out


# ------------------------------------------------------------------------------ transfer matrix
def transfer(results):
    """regret[A][B] = time of A's best strategy on B / time of B's best strategy on B"""
    pts = [p for p, r in results.items() if r.get("strategies")]
    best = {p: results[p]["best_strategy"] for p in pts}
    reg = {}
    for a, b in itertools.product(pts, pts):
        sb = results[b]["strategies"]
        reg.setdefault(a, {})[b] = (sb[best[a]] / sb[best[b]]) if best[a] in sb else None
    return reg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["chain", "ragged"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    res = run_chain() if a.family == "chain" else run_ragged()
    payload = {
        "family": a.family,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "results": res,
        "regime_regret": transfer(res),
    }
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=1)
