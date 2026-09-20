"""Function server. One per trust domain: a *reference* worker (trusted code: reference.py and
its automatic baselines) and a *candidate* worker (untrusted code: the submission).

The worker never generates inputs, never sees seeds, and never reports the time that counts.
It receives tensors owned by the driver (CUDA IPC / shared memory), runs a named callable over
a block of them, synchronises, and says "done". Outputs are fetched afterwards, outside the
timed region. Separate processes also mean separate allocators: a candidate cannot find the
reference's result in recycled memory (the Sakana CUDA Engineer exploit).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import traceback
from pathlib import Path

COMPILE_VARIANTS = {
    "compile_default": {},
    "compile_reduce_overhead": {"mode": "reduce-overhead"},
    "compile_max_autotune": {"mode": "max-autotune"},
    "compile_max_autotune_no_cudagraphs": {"mode": "max-autotune-no-cudagraphs"},
}


def _import_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sync(device: str) -> None:
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize()


def serve(conn, device: str) -> None:
    """Message loop. Every request gets exactly one reply: ("ok", payload) | ("err", text)."""
    import torch

    fns: dict[str, object] = {}
    pool: list[tuple] = []
    last_outputs: list = []

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        op, *args = msg
        try:
            if op == "load_reference":
                (path, variants) = args
                ref = _import_from(Path(path), "hotloop_reference").reference
                t0 = time.perf_counter()
                for v in variants:
                    if v == "eager":
                        fns[v] = ref
                    elif v in COMPILE_VARIANTS:
                        torch._dynamo.reset()
                        fns[v] = torch.compile(ref, **COMPILE_VARIANTS[v])
                    else:
                        raise ValueError(f"unknown baseline variant {v!r}")
                conn.send(("ok", time.perf_counter() - t0))

            elif op == "load_candidate":
                (submission_dir,) = args
                sub = Path(submission_dir).resolve()
                os.chdir(sub)
                sys.path.insert(0, str(sub))
                t0 = time.perf_counter()
                fns["candidate"] = _import_from(sub / "solution.py", "solution").run
                conn.send(("ok", time.perf_counter() - t0))

            elif op == "set_pool":
                (pool,) = args
                last_outputs = []
                conn.send(("ok", len(pool)))

            elif op == "run_block":
                name, indices = args
                fn = fns[name]
                outs = []
                _sync(device)
                t0 = time.perf_counter()
                for i in indices:
                    outs.append(fn(*pool[i]))
                _sync(device)
                inner = time.perf_counter() - t0
                last_outputs = outs
                # Reply carries no tensors: pickling IPC handles must not sit in the timed path.
                conn.send(("ok", inner))

            elif op == "fetch":
                (j,) = args
                conn.send(("ok", last_outputs[j]))

            elif op == "peak_memory":
                peak = torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0
                if device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                conn.send(("ok", int(peak)))

            elif op == "release":
                pool, last_outputs = [], []
                conn.send(("ok", None))

            elif op == "exit":
                conn.send(("ok", None))
                return
            else:
                raise ValueError(f"unknown op {op!r}")
        except BaseException:  # noqa: BLE001 - untrusted code may raise anything, incl. SystemExit
            try:
                conn.send(("err", traceback.format_exc(limit=8)))
            except OSError:
                return
