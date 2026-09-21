"""Is torch.compile really slower than eager on a batched matmul chain, or is it our harness?

Same function, same shapes, three ways of feeding it:
  A. plain loop, the same input tensors every call          (what a user would benchmark)
  B. fresh input tensors every call, same process            (new addresses, like our evaluator)
  C. input tensors shared from ANOTHER process over CUDA IPC  (exactly like our evaluator)
"""

import time

import torch
import torch.multiprocessing as mp

B, D = 64, [1, 512, 512, 512, 512]


def reference(x, w1, w2, y):
    return x @ w1 @ w2 @ y


def make(seed, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(seed)

    def m(*s):
        return torch.randn(*s, device=dev, generator=g) / s[-1] ** 0.5

    return (m(B, D[0], D[1]), m(D[1], D[2]), m(D[2], D[3]), m(B, D[3], D[4]))


def bench(fn, pools, reps=30):
    for p in pools[:3]:
        fn(*p)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for p in pools:
            fn(*p)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) / len(pools))
    ts.sort()
    return ts[len(ts) // 2] * 1e6


def child(conn):
    fns = {"eager": reference, "compile_default": torch.compile(reference)}
    pools = conn.recv()
    out = {k: bench(f, pools) for k, f in fns.items()}
    conn.send(out)


if __name__ == "__main__":
    print(torch.__version__, torch.cuda.get_device_name(0))
    fns = {"eager": reference, "compile_default": torch.compile(reference)}
    same = [make(0)] * 32
    fresh = [make(i) for i in range(32)]
    for label, pools in (
        ("A same tensors, one process ", same),
        ("B fresh tensors, one process", fresh),
    ):
        r = {k: bench(f, pools) for k, f in fns.items()}
        print(
            f"{label}: eager {r['eager']:7.1f} us   compile {r['compile_default']:7.1f} us   ratio {r['compile_default'] / r['eager']:.2f}"
        )
    ctx = mp.get_context("spawn")
    a, b = ctx.Pipe()
    p = ctx.Process(target=child, args=(b,))
    p.start()
    a.send(fresh)
    r = a.recv()
    p.join()
    print(
        f"C tensors via CUDA IPC       : eager {r['eager']:7.1f} us   compile {r['compile_default']:7.1f} us   ratio {r['compile_default'] / r['eager']:.2f}"
    )


def single_calls(fn, pools, n=40, gap_s=0.003):
    """one call at a time with an idle gap in between, like the evaluator's baseline probe"""
    for p in pools[:3]:
        fn(*p)
    torch.cuda.synchronize()
    ts = []
    for i in range(n):
        time.sleep(gap_s)
        t0 = time.perf_counter()
        fn(*pools[i % len(pools)])
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1e6


def extra():
    fresh = [make(i) for i in range(32)]
    print(
        "\nD. three compiled variants of the SAME function object (as the evaluator's worker does):"
    )
    variants = {
        "eager": reference,
        "compile_default": torch.compile(reference),
        "compile_max_autotune_nocg": torch.compile(reference, mode="max-autotune-no-cudagraphs"),
        "compile_reduce_overhead": torch.compile(reference, mode="reduce-overhead"),
    }
    for _ in range(2):  # warm all of them first, interleaved
        for f in variants.values():
            f(*fresh[0])
    torch.cuda.synchronize()
    for k, f in variants.items():
        print(
            f"   {k:28s} back-to-back {bench(f, fresh):7.1f} us    single calls with idle gaps {single_calls(f, fresh):7.1f} us"
        )
    import torch._dynamo.utils as u

    print("   recompiles/graph breaks:", dict(u.counters.get("recompiles", {})) or "none recorded")


if __name__ == "__main__":
    extra()
