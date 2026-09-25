"""Everything hardware-specific in scoring, behind one interface.

    CudaDevice - NVIDIA: CUDA-graph replay, CUDA events, L2 flush, CUPTI kernel list
    MpsDevice  - Apple GPUs (Metal via PyTorch MPS): eager replay, MPS events,
                 system-level-cache flush; no fp64 on the GPU (references run on CPU)

The rest of the harness (correctness, tolerances, roofline math, bans, scoring) is
shared. `get_device()` picks the device: HOTLOOP_DEVICE, else CUDA, else MPS.
"""

import os
import time

import torch


class Replay:
    """A captured implementation: `replay()` runs it on the static inputs, `outputs`
    holds its outputs (the same tensors every replay for graph capture)."""

    def __init__(self, replay_fn, outputs_fn):
        self._replay, self._outputs = replay_fn, outputs_fn

    def replay(self):
        self._replay()

    @property
    def outputs(self) -> list:
        return self._outputs()


class Device:
    name = "base"
    torch_device = "cpu"
    ref64_device = "cpu"          # where the fp64 reference runs
    graph_capture = False         # True: timing replays a captured graph (no CPU code runs)

    def device_name(self) -> str: ...
    def synchronize(self): ...
    def empty_cache(self): ...
    def cache_bytes(self) -> int: ...

    def timer(self):
        """Returns (start, stop) callables; stop() returns elapsed milliseconds."""
        ...

    def make_flush(self):
        """A callable that evicts the on-chip caches between timed runs."""
        buf = torch.empty(4 * self.cache_bytes(), dtype=torch.uint8, device=self.torch_device)
        return buf.zero_

    def capture(self, call, static_in: list, flat) -> Replay: ...

    def profile_kernels(self, replay: Replay, reps: int = 3) -> dict:
        """Per-kernel GPU time and host<->device copies during replay, if the platform exposes them."""
        return {"kernels": [], "kernel_us": [], "host_copies": [], "available": False}

    def time_ms(self, fn, iters: int = 10) -> float:
        fn()
        self.synchronize()
        times = []
        for _ in range(iters):
            start, stop = self.timer()
            start()
            fn()
            times.append(stop())
        return sorted(times)[len(times) // 2]

    def integrity_items(self) -> dict:
        """Functions the timing path relies on; a solution that replaces them is caught."""
        return {"synchronize": type(self).synchronize, "Tensor.copy_": torch.Tensor.copy_,
                "Tensor.zero_": torch.Tensor.zero_, "save": torch.save}


class CudaDevice(Device):
    name = "cuda"
    torch_device = "cuda"
    ref64_device = "cuda"
    graph_capture = True

    def device_name(self) -> str:
        return torch.cuda.get_device_name()

    def synchronize(self):
        torch.cuda.synchronize()

    def empty_cache(self):
        torch.cuda.empty_cache()

    def cache_bytes(self) -> int:
        return torch.cuda.get_device_properties(0).L2_cache_size

    def timer(self):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        def stop():
            e.record()
            e.synchronize()
            torch.cuda.synchronize()
            return s.elapsed_time(e)

        return s.record, stop

    def capture(self, call, static_in, flat) -> Replay:
        # Warm up outside capture (JIT compilation, autotuning, lazy init).
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                call(static_in)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = flat(call(static_in))
        torch.cuda.synchronize()
        return Replay(graph.replay, lambda: static_out)

    def profile_kernels(self, replay, reps: int = 3) -> dict:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(reps):
                replay.replay()
            torch.cuda.synchronize()
        times, copies = {}, set()
        for e in prof.profiler.kineto_results.events():
            if "cuda" not in str(e.device_type()).lower():
                continue
            n = e.name()
            if n.startswith("Memcpy HtoD") or n.startswith("Memcpy DtoH"):
                copies.add(n)
            elif not n.startswith(("Memcpy", "Memset")):
                times[n] = times.get(n, 0.0) + e.duration_ns() / 1e3 / reps
        ranked = sorted(times.items(), key=lambda kv: -kv[1])
        return {"kernels": [n for n, _ in ranked], "kernel_us": [round(t, 2) for _, t in ranked],
                "host_copies": sorted(copies), "available": True}

    def integrity_items(self) -> dict:
        return {**super().integrity_items(), "Event.record": torch.cuda.Event.record,
                "Event.elapsed_time": torch.cuda.Event.elapsed_time,
                "Event.synchronize": torch.cuda.Event.synchronize, "cuda.synchronize": torch.cuda.synchronize,
                "CUDAGraph.replay": torch.cuda.CUDAGraph.replay}


class MpsDevice(Device):
    """Apple GPUs. PyTorch has no graph capture on MPS, so timing replays the
    implementation eagerly on the static inputs. Anti-caching then rests on fresh
    input values every run plus a check of every timed run's output."""

    name = "mps"
    torch_device = "mps"
    ref64_device = "cpu"          # Apple GPUs have no fp64
    graph_capture = False

    def device_name(self) -> str:
        try:
            import subprocess
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True).stdout.strip()
        except Exception:
            chip = "Apple"
        return f"{chip} GPU"

    def synchronize(self):
        torch.mps.synchronize()

    def empty_cache(self):
        torch.mps.empty_cache()

    def cache_bytes(self) -> int:
        # Apple GPUs share a system-level cache (tens of MB); 64 MB covers current chips.
        return 64 << 20

    def timer(self):
        # Host-side timing around full device syncs. (torch.mps.Event.elapsed_time hangs on
        # current PyTorch/macOS.) This includes CPU launch overhead, which solution and
        # baseline pay equally since both run eagerly on MPS.
        t = {}

        def start():
            torch.mps.synchronize()
            t["0"] = time.perf_counter()

        def stop():
            torch.mps.synchronize()
            return (time.perf_counter() - t["0"]) * 1e3

        return start, stop

    def capture(self, call, static_in, flat) -> Replay:
        for _ in range(2):  # warm up (shader compilation, lazy init)
            call(static_in)
        torch.mps.synchronize()
        holder = {"out": flat(call(static_in))}

        def replay():
            holder["out"] = flat(call(static_in))

        return Replay(replay, lambda: holder["out"])

    def integrity_items(self) -> dict:
        return {**super().integrity_items(), "mps.synchronize": torch.mps.synchronize,
                "perf_counter": time.perf_counter}


_DEVICES = {"cuda": CudaDevice, "mps": MpsDevice}
_current: Device | None = None


def get_device() -> Device:
    global _current
    if _current is None:
        name = os.environ.get("HOTLOOP_DEVICE")
        if not name:
            name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None
        if name not in _DEVICES:
            raise RuntimeError("no supported GPU found (need CUDA or Apple MPS); set HOTLOOP_DEVICE")
        _current = _DEVICES[name]()
    return _current
