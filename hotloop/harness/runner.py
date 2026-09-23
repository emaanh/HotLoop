"""Child process that runs one implementation (solution or baseline).

The parent sends JSON commands over a pipe. The runner never sees reference
outputs: it only writes the implementation's outputs to disk, and the parent
checks them in a separate process.

Timing replays a CUDA graph of the implementation:
  - fresh inputs are copied into the graph's input buffers every iteration
  - L2 is flushed before every replay
  - CPU-side tricks (caching, other threads/processes, side streams that never
    rejoin) are not part of the graph, so they produce wrong outputs
"""

import argparse
import importlib.util
import json
import os
import sys
import traceback

import torch

from hotloop.harness import ban
from hotloop.harness.inputs import make_inputs
from hotloop.harness.numerics import flat
from hotloop.harness.task import load_shape


def _drop_privileges():
    if os.getuid() != 0:
        return
    import pwd
    try:
        pw = pwd.getpwnam("solver")
    except KeyError:
        return
    os.setgid(pw.pw_gid)
    os.setuid(pw.pw_uid)


def _integrity_snapshot() -> dict:
    """Identity of the functions the timing and I/O paths rely on."""
    import hotloop.harness.runner as me
    items = {
        "Event.record": torch.cuda.Event.record, "Event.elapsed_time": torch.cuda.Event.elapsed_time,
        "Event.synchronize": torch.cuda.Event.synchronize, "synchronize": torch.cuda.synchronize,
        "CUDAGraph.replay": torch.cuda.CUDAGraph.replay, "Tensor.copy_": torch.Tensor.copy_,
        "Tensor.zero_": torch.Tensor.zero_, "save": torch.save, "make_inputs": make_inputs,
    }
    snap = {k: id(v) for k, v in items.items()}
    for k in dir(me):
        v = getattr(me, k)
        if callable(v) and getattr(v, "__module__", None) == me.__name__:
            snap[f"runner.{k}"] = id(v)
            if hasattr(v, "__code__"):
                snap[f"runner.{k}.code"] = hash(v.__code__.co_code)
    return snap


class Runner:
    def __init__(self, kind: str, workdir: str, solution_path: str | None):
        self.kind = kind  # "solution" | "eager" | "compile"
        self.workdir = workdir
        self.out_dir = os.path.join(workdir, "out", kind)
        os.makedirs(self.out_dir, exist_ok=True)
        self.shapes = {}
        self.snapshot = _integrity_snapshot()
        self.solution = None
        if kind == "solution":
            spec = importlib.util.spec_from_file_location("solution", solution_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if not callable(getattr(mod, "solution", None)):
                raise RuntimeError("solution.py must define a function `solution(*inputs)`")
            self.solution = mod.solution
        l2 = torch.cuda.get_device_properties(0).L2_cache_size
        self.flush_buf = torch.empty(4 * l2, dtype=torch.uint8, device="cuda")

    # --- helpers ---------------------------------------------------------------
    def _call(self, fn, inputs):
        if self.kind == "solution":
            mode = ban.BanMode()
            with torch.no_grad(), mode:
                return fn(*inputs)
        with torch.no_grad():
            return fn(*inputs)

    def _fn_for(self, shape):
        if self.kind == "solution":
            return self.solution
        ref = shape.reference()
        if self.kind == "compile":
            return torch.compile(ref, dynamic=False)
        return ref

    # --- commands --------------------------------------------------------------
    def prepare(self, sid: str, shape_dir: str, seed: int):
        shape = load_shape(shape_dir)
        fn = self._fn_for(shape)
        static_in = make_inputs(shape.meta, shape.exact, seed)
        # Warm up outside capture (JIT compilation, autotuning, lazy init).
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._call(fn, static_in)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        mutated = shape.meta.get("mutated_inputs", [])
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                static_out = self._call(fn, static_in)
        except ban.BanViolation:
            raise
        except Exception as e:
            raise RuntimeError(
                "CUDA graph capture failed. Kernels must launch on the current stream "
                "(e.g. at::cuda::getCurrentCUDAStream() in C++), must not synchronize "
                f"with the host, and must not allocate host memory. Error: {e}") from e
        torch.cuda.synchronize()
        # Inputs the reference updates in place (e.g. a KV cache) are checked like outputs.
        outs = flat(static_out) + [static_in[i] for i in mutated]
        self.shapes[sid] = {"shape": shape, "fn": fn, "graph": graph, "in": static_in, "out": outs, "mutated": mutated}
        return {"n_outputs": len(self.shapes[sid]["out"])}

    def correct(self, sid: str, cases: list[dict]):
        st = self.shapes[sid]
        shape = st["shape"]
        for c in cases:
            inputs = make_inputs(shape.meta, shape.exact, c["seed"], c["variant"])
            out = flat(self._call(st["fn"], inputs)) + [inputs[i] for i in st["mutated"]]
            torch.cuda.synchronize()
            torch.save([o.detach().cpu() if isinstance(o, torch.Tensor) else o for o in out],
                       os.path.join(self.out_dir, f"{sid}__{c['name']}.pt"))
        return {"n": len(cases)}

    def time(self, sid: str, seeds: list[int], save: list[bool]):
        st = self.shapes[sid]
        shape = st["shape"]
        times = []
        for seed, keep in zip(seeds, save):
            new = make_inputs(shape.meta, shape.exact, seed)
            for buf, x in zip(st["in"], new):
                buf.copy_(x)
            del new
            self.flush_buf.zero_()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            st["graph"].replay()
            end.record()
            end.synchronize()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
            if keep:
                torch.save([o.detach().cpu() for o in st["out"]], os.path.join(self.out_dir, f"{sid}__t{seed}.pt"))
        return {"times_ms": times}

    def kernels(self, sid: str):
        """Profile a few replays; report per-kernel GPU time and host<->device copies."""
        from torch.profiler import ProfilerActivity, profile
        st = self.shapes[sid]
        reps = 3
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(reps):
                st["graph"].replay()
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
                "host_copies": sorted(copies)}

    def integrity(self):
        now = _integrity_snapshot()
        changed = sorted(k for k in self.snapshot if now.get(k) != self.snapshot[k])
        return {"changed": changed}

    def release(self, sid: str):
        self.shapes.pop(sid, None)
        torch.cuda.empty_cache()
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--solution")
    ap.add_argument("--cmd-fd", type=int, required=True)
    ap.add_argument("--resp-fd", type=int, required=True)
    args = ap.parse_args()

    cmd_in = os.fdopen(args.cmd_fd, "r")
    resp_out = os.fdopen(args.resp_fd, "w")

    def reply(obj):
        resp_out.write(json.dumps(obj) + "\n")
        resp_out.flush()

    if args.kind == "solution":
        _drop_privileges()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        runner = Runner(args.kind, args.workdir, args.solution)
        reply({"ok": True})
    except Exception:
        reply({"ok": False, "error": traceback.format_exc(limit=8)})
        return
    for line in cmd_in:
        msg = json.loads(line)
        cmd = msg.pop("cmd")
        if cmd == "exit":
            reply({"ok": True})
            break
        try:
            reply({"ok": True, **getattr(runner, cmd)(**msg)})
        except Exception as e:
            violation = isinstance(e, ban.BanViolation) or isinstance(e.__cause__, ban.BanViolation)
            reply({"ok": False, "violation": violation, "error": traceback.format_exc(limit=8)})
    sys.stdout.flush()


if __name__ == "__main__":
    main()
