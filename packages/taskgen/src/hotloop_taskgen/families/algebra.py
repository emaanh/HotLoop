"""Family: algebraic structure in matmul chains.

`torch.compile` never re-associates a product of matrices; the cost of `A @ B @ C @ D` ranges
over orders of magnitude with the parenthesisation, and which one is best is decided entirely by
the shapes (D-23: 4.6x-61x headroom, up to 197x cross-regime regret).

Two members:
  * `chain`  - four 2-D matrices. **Diagnostic tier** (D-24): `torch.linalg.multi_dot` solves it,
    so it only measures whether the agent looks at shapes at all.
  * `bchain` - the same idea where the outer factors are per-sample batches and the inner ones
    are shared. `multi_dot` does not apply; a shared sub-product can be formed once for the whole
    batch, and the best order again flips with the shapes.
"""

from __future__ import annotations

from hotloop_taskgen.core import TaskDraft

FAMILY = "algebra"
GAP_CLASS = "G1"  # COMPILER_GAPS.md: numerical contract: re-association changes rounding

REGIMES: dict[str, dict] = {
    "chain_lowrank_sandwich": dict(member="chain", dims=[8192, 64, 8192, 64, 16]),
    "chain_tall_skinny": dict(member="chain", dims=[8192, 128, 4096, 128, 2048]),
    "chain_as_written_is_optimal": dict(
        member="chain", dims=[64, 4096, 4096, 4096, 64], control=True
    ),
    "bchain_vector_on_the_right": dict(member="bchain", batch=64, dims=[512, 512, 512, 512, 1]),
    "bchain_shared_wide_middle": dict(member="bchain", batch=64, dims=[256, 64, 2048, 64, 256]),
    "bchain_vector_on_the_left": dict(
        member="bchain", batch=64, dims=[1, 512, 512, 512, 512], control=True
    ),
}

REFERENCE = {
    "chain": '''import torch


def reference(a, b, c, d):
    """a: (d0, d1), b: (d1, d2), c: (d2, d3), d: (d3, d4) -> (d0, d4)"""
    return a @ b @ c @ d
''',
    "bchain": '''import torch


def reference(x, w1, w2, y):
    """x: (B, d0, d1) per-sample, w1: (d1, d2) shared, w2: (d2, d3) shared, y: (B, d3, d4)
    per-sample -> (B, d0, d4)"""
    return x @ w1 @ w2 @ y
''',
}

WORKLOAD = {
    "chain": """import torch

DIMS = {dims}


def make_inputs(entry, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    return tuple(
        torch.randn(DIMS[i], DIMS[i + 1], device=device, generator=g) / DIMS[i + 1] ** 0.5
        for i in range(4)
    )
""",
    "bchain": """import torch

BATCH = {batch}
DIMS = {dims}


def make_inputs(entry, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)

    def m(*shape):
        return torch.randn(*shape, device=device, generator=g) / shape[-1] ** 0.5

    d = DIMS
    return (m(BATCH, d[0], d[1]), m(d[1], d[2]), m(d[2], d[3]), m(BATCH, d[3], d[4]))
""",
}


def draft(regime: str, seed: int = 0) -> TaskDraft:
    p = REGIMES[regime]
    member, dims = p["member"], p["dims"]
    if member == "chain":
        shapes = ", ".join(f"({dims[i]}, {dims[i + 1]})" for i in range(4))
        what = f"A product of four float32 matrices with shapes {shapes}."
    else:
        b = p["batch"]
        what = (
            f"A batched product `x @ w1 @ w2 @ y`, float32: `x` is `({b}, {dims[0]}, {dims[1]})` and `y` is "
            f"`({b}, {dims[3]}, {dims[4]})` (different for every sample); `w1` `({dims[1]}, {dims[2]})` and "
            f"`w2` `({dims[2]}, {dims[3]})` are shared across the batch."
        )
    desc = f"""## The program and the workload
{what} Values are random and change with every call; shapes are fixed."""
    return TaskDraft(
        id=f"algebra-{regime.replace('_', '-')}-s{seed}",
        title=f"Matmul chain ({regime.replace('_', ' ')})",
        family=FAMILY,
        regime=regime,
        family_params={k: v for k, v in p.items() if k != "control"},
        reference_src=REFERENCE[member],
        workload_src=WORKLOAD[member].format(dims=dims, batch=p.get("batch")),
        description=desc,
        sibling_group=f"{FAMILY}-{member}-v1",
        control=bool(p.get("control")),
        seed=seed,
    )
