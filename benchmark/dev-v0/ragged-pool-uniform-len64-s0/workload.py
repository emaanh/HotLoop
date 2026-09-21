import torch

S, T, D, LMAX = 16384, 1048576, 32, 64
DIST = {'kind': 'uniform', 'len': 64}
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
