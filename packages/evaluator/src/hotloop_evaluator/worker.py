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


class _SyncGuard:
    """Holds the real device-synchronise entry points, captured before any untrusted import.

    Untrusted code shares this process and can rebind `torch.cuda.synchronize` to a no-op so the
    worker says "done" while kernels are still queued. We call the captured originals, and report
    if the public names no longer point at them. This is defence in depth: the driver does not
    rely on it (it consumes an output inside the timed region, see driver.py).
    """

    def __init__(self, device: str):
        import torch

        self.cuda = device.startswith("cuda")
        # Test-only: lets the GPU suite prove the driver-side defence holds with *no* worker
        # sync at all. Read once, before any untrusted import.
        self.disabled = os.environ.get("HOTLOOP_TEST_DISABLE_WORKER_SYNC") == "1"
        self._py = torch.cuda.synchronize
        self._c = getattr(torch._C, "_cuda_synchronize", None)

    def sync(self) -> None:
        if self.cuda and not self.disabled:
            self._py()
            if self._c is not None:
                self._c()

    def tampered(self) -> bool:
        import torch

        return self.cuda and (
            torch.cuda.synchronize is not self._py
            or getattr(torch._C, "_cuda_synchronize", None) is not self._c
        )


def serve(conn, device: str) -> None:
    """Message loop. Every request gets exactly one reply: ("ok", payload) | ("err", text)."""
    import torch

    guard = _SyncGuard(device)
    fns: dict[str, object] = {}
    pool: list[tuple] = []
    last_outputs: list = []
    slots: list[tuple] = []  # driver-owned result buffers, shared once per workload entry

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
                guard.sync()
                t0 = time.perf_counter()
                for i in indices:
                    outs.append(fn(*pool[i]))
                guard.sync()
                inner = time.perf_counter() - t0
                last_outputs = outs
                conn.send(("ok", {"inner_s": inner, "sync_tampered": guard.tampered()}))

            elif op == "set_result_slots":
                (slots,) = args
                conn.send(("ok", len(slots)))

            elif op == "stash":
                # Copy chosen outputs into driver-owned buffers on the *current* stream: an output
                # must be valid for a current-stream consumer, as any PyTorch op's would be. The
                # indices arrive only now, after the block ran, so the submission cannot know
                # which calls get checked. No CUDA-IPC handle is created here (that costs ~350us).
                (js,) = args
                for slot, j in zip(slots, js, strict=False):
                    outs = last_outputs[j]
                    outs = (outs,) if isinstance(outs, torch.Tensor) else tuple(outs)
                    if len(outs) != len(slot):
                        raise ValueError(f"expected {len(slot)} outputs, got {len(outs)}")
                    for buf, out in zip(slot, outs, strict=True):
                        if out.shape != buf.shape or out.dtype != buf.dtype:
                            raise ValueError(
                                f"timed-phase output {tuple(out.shape)}/{out.dtype} does not match "
                                f"checked-phase output {tuple(buf.shape)}/{buf.dtype}"
                            )
                        buf.copy_(out)
                guard.sync()
                conn.send(("ok", None))

            elif op == "fetch":
                (js,) = args
                conn.send(("ok", [last_outputs[j] for j in js]))

            elif op == "peak_memory":
                peak = torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0
                if device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                conn.send(("ok", int(peak)))

            elif op == "release":
                pool, last_outputs = [], []
                conn.send(("ok", None))

            elif op == "release_slots":
                slots = []
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
