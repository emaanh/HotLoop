"""Family: ragged softmax-pooling over packed segments.

Program. Inputs are *packed*: values `v (T, D)`, scores `s (T,)`, segment `offsets (S+1,)`. For
each segment, softmax the scores and return the score-weighted mean of the values -> `(S, D)`.
The reference is how a modeller writes it: pad every segment to LMAX, mask, softmax, sum.

Why it is a family. The padded reference does `S * LMAX` work for `T` useful elements. When
lengths are uniform that ratio is 1 and the compiler's fused dense kernel is hard to beat
(no-headroom *controls*). When a few segments are long, padding waste is 50-200x and the right
answer depends on *how* the lengths are ragged (D-23): many tiny segments favour element-parallel
scatter/segment ops; longer or heavy-tailed segments favour a segment-parallel jagged kernel,
whose block size is itself regime-dependent.

Shapes never change between seeds (lengths are rescaled to a fixed total T), so compiled
baselines are measured fairly; only the length pattern and values vary.
"""

from __future__ import annotations

import torch

from hotloop_taskgen.core import TaskDraft

FAMILY = "ragged_pool"

REGIMES: dict[str, dict] = {
    # --- headroom regimes -------------------------------------------------------------------
    "tiny_outliers": dict(
        S=262144, D=32, dist={"kind": "bimodal", "p_long": 0.001, "long": 256, "short": 4}
    ),
    "tiny_rare_long_outliers": dict(
        S=262144, D=16, dist={"kind": "bimodal", "p_long": 0.001, "long": 512, "short": 4}
    ),
    "small_outliers": dict(
        S=65536, D=32, dist={"kind": "bimodal", "p_long": 0.002, "long": 1024, "short": 16}
    ),
    "zipf_heavy_tail": dict(
        S=16384, D=32, dist={"kind": "zipf", "min": 8, "alpha": 1.1, "max": 4096}
    ),
    # --- no-headroom controls (padding is free; the compiler should win) -------------------------
    "uniform_len64": dict(S=16384, D=32, dist={"kind": "uniform", "len": 64}, control=True),
    "uniform_tiny_len4": dict(S=262144, D=32, dist={"kind": "uniform", "len": 4}, control=True),
}

REFERENCE = '''import torch

LMAX = {lmax}  # no segment is longer than this


def reference(v, s, offsets):
    """v: (T, D) values, s: (T,) scores, offsets: (S+1,) int64 segment boundaries into T.
    Returns (S, D): for each segment, sum_i softmax(s_seg)_i * v_seg_i."""
    lens = offsets[1:] - offsets[:-1]
    idx = torch.arange(LMAX, device=v.device)[None, :]
    mask = idx < lens[:, None]
    pos = (offsets[:-1, None] + idx).clamp_max(v.shape[0] - 1)
    sp = s[pos].masked_fill(~mask, float("-inf"))
    w = torch.softmax(sp.float(), dim=-1).to(v.dtype)
    return (w[..., None] * v[pos]).sum(1)
'''

WORKLOAD = '''import torch

S, T, D, LMAX = {S}, {T}, {D}, {lmax}
DIST = {dist!r}
DTYPE = torch.float16


def _raw_lengths(g):
    kind = DIST["kind"]
    if kind == "uniform":
        return torch.full((S,), DIST["len"], dtype=torch.int64)
    if kind == "zipf":
        u = torch.rand(S, generator=g, dtype=torch.float64)
        return (DIST["min"] * (1 - u) ** (-1 / DIST["alpha"])).clamp(max=LMAX).to(torch.int64)
    if kind == "bimodal":
        is_long = torch.rand(S, generator=g) < DIST["p_long"]
        return torch.where(is_long, DIST["long"], DIST["short"]).to(torch.int64)
    raise ValueError(kind)


def lengths(seed):
    """Segment lengths for a seed. Always sums to exactly T and never exceeds LMAX, so tensor
    shapes are identical for every seed; only the pattern of lengths changes."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    lens = _raw_lengths(g)
    lens = (lens.double() * (T / int(lens.sum()))).round().long().clamp(1, LMAX)
    for _ in range(256):
        diff = T - int(lens.sum())
        if diff == 0:
            break
        idx = ((lens < LMAX) if diff > 0 else (lens > 1)).nonzero().squeeze(1)
        pick = idx[torch.randperm(idx.numel(), generator=g)[: abs(diff)]]
        lens[pick] += 1 if diff > 0 else -1
    assert int(lens.sum()) == T and int(lens.max()) <= LMAX and int(lens.min()) >= 1
    return lens


def make_inputs(entry, seed, device):
    lens = lengths(seed)
    offsets = torch.zeros(S + 1, dtype=torch.int64)
    offsets[1:] = lens.cumsum(0)
    g = torch.Generator(device=device).manual_seed(seed)
    v = torch.randn(T, D, device=device, dtype=DTYPE, generator=g)
    s = torch.randn(T, device=device, dtype=DTYPE, generator=g)
    return (v, s, offsets.to(device))
'''


def _lmax(dist: dict) -> int:
    return int(dist.get("max") or dist.get("long") or dist["len"])


def _nominal_total(S: int, dist: dict, lmax: int) -> int:
    """T is fixed per task: the total of the seed-0 raw sample, so rescaling is ~identity."""
    g = torch.Generator(device="cpu").manual_seed(0)
    if dist["kind"] == "uniform":
        return S * dist["len"]
    if dist["kind"] == "zipf":
        u = torch.rand(S, generator=g, dtype=torch.float64)
        return int(
            (dist["min"] * (1 - u) ** (-1 / dist["alpha"])).clamp(max=lmax).to(torch.int64).sum()
        )
    is_long = torch.rand(S, generator=g) < dist["p_long"]
    return int(torch.where(is_long, dist["long"], dist["short"]).sum())


def draft(regime: str, seed: int = 0) -> TaskDraft:
    p = REGIMES[regime]
    S, D, dist = p["S"], p["D"], p["dist"]
    lmax = _lmax(dist)
    T = _nominal_total(S, dist, lmax)
    desc = f"""## The program and the workload
Packed ragged data: `v` is `({T}, {D})` float16 values, `s` is `({T},)` float16 scores, and
`offsets` is `({S + 1},)` int64 segment boundaries ({S} segments, none longer than {lmax}).
For every segment the program softmaxes that segment's scores and returns the score-weighted sum
of its values, shape `({S}, {D})`.

Segment lengths follow `{dist}` (see `workload.py::lengths`). The total number of elements is
always exactly {T} and the tensor shapes never change; the pattern of lengths and all values
change with every call. The padded tensor the reference builds has {S * lmax / T:.0f}x as many
slots as there are real elements."""
    return TaskDraft(
        id=f"ragged-pool-{regime.replace('_', '-')}-s{seed}",
        title=f"Ragged softmax-pooling ({regime.replace('_', ' ')})",
        family=FAMILY,
        regime=regime,
        family_params={
            "S": S,
            "T": T,
            "D": D,
            "lmax": lmax,
            "dist": dist,
            "pad_waste": round(S * lmax / T, 2),
        },
        reference_src=REFERENCE.format(lmax=lmax),
        workload_src=WORKLOAD.format(S=S, T=T, D=D, lmax=lmax, dist=dist),
        description=desc,
        sibling_group=f"{FAMILY}-v1",
        control=bool(p.get("control")),
        seed=seed,
    )
