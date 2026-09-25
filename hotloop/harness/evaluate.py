"""Evaluate a solution on a task: correctness, timing, roofline, score.

Process layout (all on one GPU, never running at the same time):
  parent            - holds seeds, computes references, checks outputs, scores
  solution runner   - loads solution.py as an unprivileged user, writes outputs
  baseline runners  - eager reference and torch.compile of the reference
"""

import json
import math
import os
import random
import secrets
import select
import shutil
import subprocess
import sys
import tempfile
import time

import torch

import hotloop
from hotloop.harness import ban, numerics, roofline
from hotloop.harness.device import get_device
from hotloop.harness.inputs import VARIANTS, make_inputs
from hotloop.harness.task import Shape, load_shape

TIMING_BLOCKS = 3
FLAG_BELOW_SOL = 0.8  # faster than 80% of the speed-of-light bound is physically suspect


class RunnerError(RuntimeError):
    def __init__(self, msg, violation=False):
        super().__init__(msg)
        self.violation = violation


class RunnerProc:
    def __init__(self, kind: str, workdir: str, solution: str | None = None, env_extra: dict | None = None,
                 timeout: float = 600):
        cmd_r, cmd_w = os.pipe()
        resp_r, resp_w = os.pipe()
        env = dict(os.environ)
        env.update({
            "PYTHONPATH": os.path.join(workdir, "pkg"),
            "HOME": os.path.join(workdir, "home"),
            "TRITON_CACHE_DIR": os.path.join(workdir, "home", "triton"),
            "TORCH_EXTENSIONS_DIR": os.path.join(workdir, "home", "ext"),
            "TORCHINDUCTOR_CACHE_DIR": os.path.join(workdir, "home", "inductor"),
            "PYTHONUNBUFFERED": "1",
        })
        env.update(env_extra or {})
        args = [sys.executable, "-m", "hotloop.harness.runner", "--kind", kind, "--workdir", workdir,
                "--cmd-fd", str(cmd_r), "--resp-fd", str(resp_w)]
        if solution:
            args += ["--solution", solution]
        self.log_path = os.path.join(workdir, f"{kind}.log")
        self.log = open(self.log_path, "w")
        self.proc = subprocess.Popen(args, pass_fds=(cmd_r, resp_w), env=env, cwd=workdir,
                                     stdout=self.log, stderr=subprocess.STDOUT)
        os.close(cmd_r)
        os.close(resp_w)
        self.cmd = os.fdopen(cmd_w, "w")
        self.resp = os.fdopen(resp_r, "r")
        self.kind = kind
        self._read(timeout)  # startup (imports the solution, compiles its kernels)

    def _read(self, timeout: float) -> dict:
        ready, _, _ = select.select([self.resp], [], [], timeout)
        if not ready:
            self.kill()
            raise RunnerError(f"{self.kind} runner timed out after {timeout:.0f}s")
        line = self.resp.readline()
        if not line:
            self.proc.wait(timeout=5)
            raise RunnerError(f"{self.kind} runner died (exit {self.proc.returncode}). Log tail:\n{self.log_tail()}")
        msg = json.loads(line)
        if not msg.get("ok"):
            raise RunnerError(msg.get("error", "unknown error"), violation=msg.get("violation", False))
        return msg

    def call(self, cmd: str, timeout: float = 600, **kw) -> dict:
        try:
            self.cmd.write(json.dumps({"cmd": cmd, **kw}) + "\n")
            self.cmd.flush()
        except BrokenPipeError:
            raise RunnerError(f"{self.kind} runner is gone. Log tail:\n{self.log_tail()}")
        return self._read(timeout)

    def log_tail(self, n: int = 3000) -> str:
        self.log.flush()
        with open(self.log_path) as f:
            return f.read()[-n:]

    def close(self):
        try:
            self.call("exit", timeout=30)
        except Exception:
            pass
        self.kill()

    def kill(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()


def _prepare_workdir(solution_src: str, shapes: list[Shape]) -> str:
    workdir = tempfile.mkdtemp(prefix="hotloop_")
    shutil.copytree(os.path.dirname(hotloop.__file__), os.path.join(workdir, "pkg", "hotloop"))
    for sub in ("home", "out", "sol_shapes"):
        os.makedirs(os.path.join(workdir, sub), exist_ok=True)
    with open(os.path.join(workdir, "solution.py"), "w") as f:
        f.write(solution_src)
    # The solution runner only gets input specs, never the reference code.
    for s in shapes:
        d = os.path.join(workdir, "sol_shapes", s.sid)
        os.makedirs(d)
        shutil.copy(os.path.join(s.dir, "meta.json"), d)
        shutil.copy(os.path.join(s.dir, "exact.pt"), d)
    for root, dirs, files in os.walk(workdir):
        os.chmod(root, 0o777)
        for f in files:
            os.chmod(os.path.join(root, f), 0o666)
    return workdir


def _geomean(xs: list[float]) -> float:
    xs = [max(x, 1e-9) for x in xs]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0


def evaluate(
    task_id: str,
    shape_dirs: list[str],
    solution_src: str,
    *,
    hidden: bool,
    peaks: dict,
    baseline_cache: dict | None = None,
    timed_per_block: int = 10,
    correct_seeds: int = 2,
    keep_workdir: bool = False,
) -> dict:
    """Returns a result dict. `baseline_cache` is updated in place with compile timings."""
    t0 = time.time()
    shapes = [load_shape(d) for d in shape_dirs]
    gpu = peaks["gpu"]
    result = {"task_id": task_id, "split": "hidden" if hidden else "public", "gpu": gpu,
              "ok": False, "violations": [], "shapes": {}, "error": None}

    problems = ban.static_check(solution_src)
    if problems:
        result["violations"] = problems
        result["error"] = "static check failed"
        return result

    rng = random.Random(secrets.randbits(64) if hidden else 1234)
    seed = lambda: rng.randrange(1, 2**31)
    workdir = _prepare_workdir(solution_src, shapes)
    baseline_cache = baseline_cache if baseline_cache is not None else {}

    sol = eager = comp = None
    plans = {}
    try:
        try:
            sol = RunnerProc("solution", workdir, os.path.join(workdir, "solution.py"))
        except RunnerError as e:
            log = open(os.path.join(workdir, "solution.log")).read()[-3000:]
            result["error"] = f"solution failed to load:\n{e}\n--- solution output (tail) ---\n{log}"
            return result
        eager = RunnerProc("eager", workdir)
        need_compile = {s.sid for s in shapes if f"{task_id}/{s.sid}/{gpu}" not in baseline_cache}
        if need_compile:
            comp = RunnerProc("compile", workdir, timeout=900)

        for s in shapes:
            info = {"correct": False, "reason": None}
            result["shapes"][s.sid] = info
            cases = [{"name": f"c{i}_{v}", "seed": seed(), "variant": v}
                     for i in range(correct_seeds) for v in VARIANTS]
            cases.append({**cases[0], "name": "repeat"})
            plans[s.sid] = {"cases": cases, "timed": []}
            try:
                sol.call("prepare", sid=s.sid, shape_dir=os.path.join(workdir, "sol_shapes", s.sid), seed=seed())
                sol.call("correct", sid=s.sid, cases=cases)
                k = sol.call("kernels", sid=s.sid)
                denied = ban.denied_kernels(k["kernels"])
                if denied:
                    result["violations"].append(f"{s.sid}: library kernels launched: {denied[:5]}")
                if k["host_copies"]:
                    result["violations"].append(f"{s.sid}: host<->device copies inside the timed graph: {k['host_copies']}")
                info["kernels"] = [[n, t] for n, t in zip(k["kernels"], k["kernel_us"])][:30]
            except RunnerError as e:
                if e.violation:
                    result["violations"].append(f"{s.sid}: {str(e).strip().splitlines()[-1]}")
                info["reason"] = str(e)[-3000:] + "\n--- solution output (tail) ---\n" + sol.log_tail(1500)
                continue

            base_seed = seed()
            eager.call("prepare", sid=s.sid, shape_dir=s.dir, seed=base_seed)
            use_comp = bool(comp and s.sid in need_compile)
            if use_comp:
                try:
                    comp.call("prepare", sid=s.sid, shape_dir=s.dir, seed=base_seed, timeout=900)
                except RunnerError as e:
                    # A reference torch.compile can't handle is simply not a baseline.
                    use_comp = False
                    baseline_cache[f"{task_id}/{s.sid}/{gpu}"] = {"eager_ms": 1.0, "compile_ms": math.inf,
                                                                  "compile_error": str(e)[-300:]}
                    comp.close()
                    comp = RunnerProc("compile", workdir, timeout=900) if len(need_compile) > 1 else None
            # Warm-up replays, then interleaved blocks so drift hits both equally.
            for r in (sol, eager) + ((comp,) if use_comp else ()):
                r.call("time", sid=s.sid, seeds=[seed() for _ in range(3)], save=[False] * 3)
            t_sol, t_eager, t_comp = [], [], []
            for _ in range(TIMING_BLOCKS):
                seeds = [seed() for _ in range(timed_per_block)]
                save = [rng.random() < 0.25 for _ in seeds]
                save[rng.randrange(len(save))] = True
                t_sol += sol.call("time", sid=s.sid, seeds=seeds, save=save)["times_ms"]
                plans[s.sid]["timed"] += [sd for sd, keep in zip(seeds, save) if keep]
                t_eager += eager.call("time", sid=s.sid, seeds=[seed() for _ in seeds], save=[False] * len(seeds))["times_ms"]
                if use_comp:
                    t_comp += comp.call("time", sid=s.sid, seeds=[seed() for _ in seeds], save=[False] * len(seeds))["times_ms"]
            med = lambda xs: sorted(xs)[len(xs) // 2]
            info["time_ms"], info["eager_ms"] = med(t_sol), med(t_eager)
            key = f"{task_id}/{s.sid}/{gpu}"
            if t_comp:
                baseline_cache[key] = {"eager_ms": info["eager_ms"], "compile_ms": med(t_comp)}
            cached = baseline_cache[key]
            # Rescale the cached compile time by today's eager drift.
            info["compile_ms"] = cached["compile_ms"] * info["eager_ms"] / cached["eager_ms"]
            for r in (sol, eager) + ((comp,) if comp else ()):
                r.call("release", sid=s.sid)

        changed = sol.call("integrity")["changed"]
        if changed:
            result["violations"].append(f"solution modified harness functions: {changed}")
    except RunnerError as e:
        result["error"] = str(e)[-4000:]
    finally:
        for r in (sol, eager, comp):
            if r:
                r.close()

    # --- verification in this (trusted) process ---------------------------------
    out_dir = os.path.join(workdir, "out", "solution")
    for s in shapes:
        info = result["shapes"][s.sid]
        if s.sid not in plans or info.get("reason"):
            continue
        dev = get_device()
        ref = s.reference(dev.torch_device)
        ref64_fn = s.reference(dev.ref64_device)
        mutated = s.meta.get("mutated_inputs", [])
        n_returned = len(s.meta["outputs"]) - len(mutated)
        verdicts, timed_tols = [], None
        try:
            for c in plans[s.sid]["cases"]:
                inputs = make_inputs(s.meta, s.exact, c["seed"], c["variant"])
                native = numerics.run_native(ref, inputs, mutated)
                ref64 = numerics.run_ref64(ref64_fn, inputs, mutated, device=dev.ref64_device)
                tols = numerics.calibrate(native, ref64)
                if c["variant"] == "normal" and timed_tols is None:
                    timed_tols = [t if t["exact"] else {**t, "abs": t["abs"] + t["ref_abs"], "rel": t["rel"] + t["ref_rel"]}
                                  for t in tols]
                if info.get("flops") is None:
                    info["flops"] = roofline.count_flops(ref, inputs)
                    # In-place outputs (a KV cache) are read, not rewritten: count them once, as inputs.
                    info["bytes"] = roofline.io_bytes(inputs, native[:n_returned])
                    info["dtype"] = roofline.compute_dtype(inputs)
                got = torch.load(os.path.join(out_dir, f"{s.sid}__{c['name']}.pt"))
                v = numerics.compare(got, ref64, native, tols)
                verdicts.append({"case": c["name"], **v})
                del inputs, native, ref64
            for sd in plans[s.sid]["timed"]:
                inputs = make_inputs(s.meta, s.exact, sd)
                native = numerics.run_native(ref, inputs, mutated)
                got = torch.load(os.path.join(out_dir, f"{s.sid}__t{sd}.pt"))
                v = numerics.compare(got, native, native, timed_tols)
                verdicts.append({"case": f"timed_{sd}", **v})
        except Exception as e:
            info["reason"] = f"verification error: {type(e).__name__}: {e}"
            continue
        dev.empty_cache()
        bad = [v for v in verdicts if not v["ok"]]
        info["correct"] = not bad
        info["max_rel_err"] = max((v.get("rel", 0.0) for v in verdicts), default=0.0)
        if bad:
            info["reason"] = f"{bad[0]['case']}: {bad[0].get('reason')}"
            continue
        if "time_ms" in info:
            info["sol_ms"] = roofline.sol_ms(info["flops"], info["bytes"], info["dtype"], peaks)
            # The baseline is torch.compile (eager only if the reference can't be compiled).
            info["baseline_ms"] = info["compile_ms"] if math.isfinite(info["compile_ms"]) else info["eager_ms"]
            info["speedup"] = info["baseline_ms"] / info["time_ms"]
            info["sol_frac"] = info["sol_ms"] / info["time_ms"]
            info["baseline_sol_frac"] = info["sol_ms"] / info["baseline_ms"]
            if info["time_ms"] < FLAG_BELOW_SOL * info["sol_ms"]:
                info["flagged"] = "faster than the speed-of-light bound; review for cheating or harness bugs"

    shp = list(result["shapes"].values())
    all_ok = bool(shp) and all(i["correct"] and "speedup" in i for i in shp) and not result["violations"] and not result["error"]
    result["ok"] = all_ok
    result["score"] = {
        "correct": all_ok,
        "speedup_geomean": _geomean([i["speedup"] for i in shp]) if all_ok else 0.0,
        "sol_frac_geomean": _geomean([i["sol_frac"] for i in shp]) if all_ok else 0.0,
        "flagged": any(i.get("flagged") for i in shp),
    }
    result["wall_s"] = round(time.time() - t0, 1)
    if keep_workdir:
        result["workdir"] = workdir
    else:
        shutil.rmtree(workdir, ignore_errors=True)
    return result
