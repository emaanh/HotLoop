"""Trusted evaluation driver: (task package, submission dir) -> Result.

Trust model (EVAL.md, D-19)
  * The driver owns inputs, seeds, the clock and the verdict. It never imports the submission.
  * Seeds are drawn fresh per evaluation and expanded with HMAC; nothing is derivable by a solver.
  * Inputs are generated here and handed to workers as shared tensors; every timed call sees
    values it has never seen before, at a new address.
  * Time that counts is measured *here*, around a batched RPC ("run these k calls, synchronise
    the device, say done"), so nothing the submission patches in its own process can touch it.
    k is sized so RPC overhead is negligible. Worker-side time is kept as a tamper diagnostic.
  * The clock does not stop at "done". It stops after the driver has itself fetched and copied
    the block's *last* output plus a random one. A submission that neuters synchronisation in
    its own process gains nothing: work still queued is either waited for inside the timed
    region (the copy queues behind it) or not finished when copied, which fails the check.
    The same consumption is applied to the baseline, so the overhead is symmetric.
  * Baseline and candidate alternate block-by-block on identical inputs (order flipped each
    pair); stats.paired_ratio turns the pairs into a speedup with a CI.
  * One randomly chosen output of *every* timed candidate block is checked against the eager
    reference. A solution cannot be correct in the check phase and sloppy (or asynchronous, or
    memoised) in the timed phase.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import importlib.util
import math
import random
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.multiprocessing as mp

from hotloop_evaluator import __version__
from hotloop_evaluator.correctness import (
    Comparison,
    compare_outputs,
    fold,
    inputs_unmutated,
    snapshot_inputs,
)
from hotloop_evaluator.env import fingerprint, gpu_state
from hotloop_evaluator.stats import StoppingRule, paired_ratio, summarize, weighted_geomean
from hotloop_evaluator.worker import serve
from hotloop_schemas import (
    EntryResult,
    Flag,
    RatioCI,
    Result,
    TaskSpec,
    TimingStats,
    load_task_spec,
    public_content_hash,
)


@dataclass
class EvalConfig:
    device: str = "cuda"
    correctness_trials: int = 5
    target_block_s: float = 0.05
    max_calls_per_block: int = 256
    pool_bytes: int = 1 << 30
    baseline_probe_blocks: int = 5
    stopping: StoppingRule = field(default_factory=StoppingRule)
    inconclusive_rel_halfwidth: float = 0.05
    load_timeout_s: float = 1800.0
    call_timeout_s: float = 600.0
    clock_disagreement: float = 0.25  # flag if worker-reported time < (1 - x) * driver time
    seed: int | None = None  # None -> fresh secret per evaluation


class WorkerError(RuntimeError):
    pass


class Worker:
    def __init__(self, ctx, device: str, name: str):
        self.name = name
        self.device = device
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=serve, args=(child, device), daemon=True, name=name)
        self.proc.start()
        child.close()

    def call(self, *msg, timeout: float):
        self.conn.send(msg)
        if not self.conn.poll(timeout):
            self.kill()
            raise WorkerError(f"{self.name}: timed out after {timeout:.0f}s on {msg[0]}")
        try:
            status, payload = self.conn.recv()
        except (EOFError, ConnectionError) as e:
            raise WorkerError(f"{self.name}: died during {msg[0]}") from e
        if status != "ok":
            raise WorkerError(f"{self.name}: {msg[0]} failed\n{payload}")
        return payload

    def timed_block(self, fn_name: str, indices: list[int], timeout: float, take: list[int]):
        """Run a block and consume outputs `take` inside the timed region.

        -> (driver-side seconds, worker-reported seconds, sync_tampered, copies of taken outputs)
        """
        t0 = time.perf_counter()
        info = self.call("run_block", fn_name, indices, timeout=timeout)
        fetched = self.call("fetch", take, timeout=timeout)
        taken = [_clone(o) for o in fetched]
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        outer = time.perf_counter() - t0
        del fetched
        return outer, info["inner_s"], info["sync_tampered"], taken

    def kill(self):
        if self.proc.is_alive():
            self.proc.kill()

    def close(self):
        if self.proc.is_alive():
            with contextlib.suppress(WorkerError, OSError):
                self.call("exit", timeout=5)
        self.kill()


class Seeds:
    def __init__(self, base: int):
        self.base = base
        self._key = base.to_bytes(8, "big")

    def derive(self, *parts) -> int:
        msg = "|".join(str(p) for p in parts).encode()
        return int.from_bytes(hmac.new(self._key, msg, hashlib.sha256).digest()[:7], "big")


def _load_workload(task_dir: Path):
    spec = importlib.util.spec_from_file_location("hotloop_workload", task_dir / "workload.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_inputs


def _clone(out):
    if isinstance(out, torch.Tensor):
        return out.clone()
    if isinstance(out, (tuple, list)):
        return tuple(_clone(o) for o in out)
    return out


def _nbytes(inputs) -> int:
    return sum(x.numel() * x.element_size() for x in inputs if isinstance(x, torch.Tensor))


def evaluate(
    task_dir: str | Path, submission_dir: str | Path, cfg: EvalConfig | None = None
) -> Result:
    cfg = cfg or EvalConfig()
    task_dir, submission_dir = Path(task_dir).resolve(), Path(submission_dir).resolve()
    spec = load_task_spec(task_dir)
    t_start = time.perf_counter()
    result = Result(
        task_id=spec.id,
        task_hash=public_content_hash(task_dir),
        evaluator_version=__version__,
        status="error",
        score=0.0,
        env=fingerprint(cfg.device),
    )
    result.env.gpu_before = gpu_state(cfg.device)
    if spec.content_hash and spec.content_hash != result.task_hash:
        result.error = "task package does not match its content_hash"
        return result
    if not (submission_dir / "solution.py").is_file():
        result.error = "submission has no solution.py"
        return result

    ctx = mp.get_context("spawn")
    ref = Worker(ctx, cfg.device, "reference")
    cand = Worker(ctx, cfg.device, "candidate")
    try:
        _run(spec, task_dir, submission_dir, cfg, ref, cand, result)
    except WorkerError as e:
        culprit_is_candidate = str(e).startswith("candidate")
        result.status = "error"
        result.score = 0.0
        result.error = str(e)[-4000:]
        if not culprit_is_candidate:
            result.flags.append(Flag(code="reference_worker_failure", detail="evaluator-side"))
    finally:
        ref.close()
        cand.close()
        result.env.gpu_after = gpu_state(cfg.device)
        result.eval_seconds = time.perf_counter() - t_start
    return result


def _run(spec: TaskSpec, task_dir, submission_dir, cfg, ref: Worker, cand: Worker, result: Result):
    make_inputs = _load_workload(task_dir)
    seeds = Seeds(cfg.seed if cfg.seed is not None else secrets.randbits(63))
    rng = random.Random(seeds.derive("spotcheck"))
    result.eval_seed = seeds.base

    variants = ["eager"] + [b for b in spec.baselines if b != "eager"]
    ref.call("load_reference", str(task_dir / "reference.py"), variants, timeout=cfg.load_timeout_s)
    result.build_seconds = cand.call(
        "load_candidate", str(submission_dir), timeout=cfg.load_timeout_s
    )

    def run_once(worker: Worker, fn: str, inputs, timeout=cfg.call_timeout_s):
        worker.call("set_pool", [inputs], timeout=60)
        worker.call("run_block", fn, [0], timeout=timeout)
        return _clone(worker.call("fetch", [0], timeout=60)[0])

    def check(inputs) -> Comparison:
        snap = snapshot_inputs(inputs)
        got = run_once(cand, "candidate", inputs)
        if not spec.outputs.inputs_may_be_mutated and not inputs_unmutated(inputs, snap):
            return Comparison(False, failure="inputs were mutated")
        want = run_once(ref, "eager", inputs)
        return compare_outputs(got, want, spec.outputs)

    for entry in spec.workload:
        er = EntryResult(entry=entry.name, weight=entry.weight, correctness=fold([]))
        result.entries.append(er)

        # ---- correctness on hidden seeds -------------------------------------------------
        trials = [
            check(make_inputs(entry.name, seeds.derive(entry.name, "correct", i), cfg.device))
            for i in range(cfg.correctness_trials)
        ]
        er.correctness = fold(trials)
        if not er.correctness.passed:
            result.status = "incorrect"
            return

        # ---- warm everything (compile/autotune is untimed), pick the strongest baseline ----
        probe = make_inputs(entry.name, seeds.derive(entry.name, "probe"), cfg.device)
        set_bytes = max(_nbytes(probe), 1)
        for w, names in (
            (ref, [v for v in variants if v in spec.baselines]),
            (cand, ["candidate"]),
        ):
            w.call("set_pool", [probe], timeout=60)
            for n in names:
                for _ in range(3):
                    w.call("run_block", n, [0], timeout=cfg.load_timeout_s)

        def per_call(worker, fn, n_blocks):
            return [
                worker.timed_block(fn, [0], cfg.call_timeout_s, take=[0])[0]
                for _ in range(n_blocks)
            ]

        baseline_names = [v for v in variants if v in spec.baselines] or ["eager"]
        for n in baseline_names:
            er.baselines[n] = TimingStats(**summarize(per_call(ref, n, cfg.baseline_probe_blocks)))
        best = min(baseline_names, key=lambda n: er.baselines[n].median_s)
        er.best_baseline = best

        t_call = min(
            er.baselines[best].median_s, summarize(per_call(cand, "candidate", 3))["median_s"]
        )
        k = max(1, min(cfg.max_calls_per_block, math.ceil(cfg.target_block_s / max(t_call, 1e-7))))
        k = max(1, min(k, cfg.pool_bytes // set_bytes))
        ref.call("peak_memory", timeout=60)
        cand.call("peak_memory", timeout=60)

        # ---- paired, interleaved, fresh-valued timing ----------------------------------------
        b_blocks, c_blocks, c_calls = [], [], []
        c_outer_total = c_inner_total = 0.0
        ratio, n_pairs, verdict = None, 0, "continue"
        while verdict == "continue":
            pool = [
                make_inputs(entry.name, seeds.derive(entry.name, "time", n_pairs, i), cfg.device)
                for i in range(k)
            ]
            j = rng.randrange(k)
            snap_j = snapshot_inputs(pool[j])
            for w, fn in ((ref, best), (cand, "candidate")):
                w.call("set_pool", pool, timeout=60)
                w.call("run_block", fn, [0], timeout=cfg.call_timeout_s)  # absorb context switch
            order = [(ref, best), (cand, "candidate")]
            if n_pairs % 2:
                order.reverse()
            take = sorted({k - 1, j})
            times, got = {}, {}
            for w, fn in order:
                outer, inner, tampered, taken = w.timed_block(
                    fn, list(range(k)), cfg.call_timeout_s, take
                )
                times[fn] = (outer, inner)
                if fn == "candidate":
                    got = dict(zip(take, taken, strict=True))
                    if tampered and not any(f.code == "sync_tampered" for f in result.flags):
                        result.flags.append(
                            Flag(
                                code="sync_tampered",
                                detail="device synchronise rebound",
                                fatal=True,
                            )
                        )
                del taken
            c_outer, c_inner = times["candidate"]
            c_outer_total += c_outer
            c_inner_total += c_inner if math.isfinite(c_inner) and c_inner > 0 else 0.0
            if not spec.outputs.inputs_may_be_mutated and not inputs_unmutated(pool[j], snap_j):
                er.correctness = fold(
                    [*trials, Comparison(False, failure="inputs mutated during timing")]
                )
                result.status = "incorrect"
                return
            spot = Comparison(True)
            for idx in take:
                spot = compare_outputs(got[idx], run_once(ref, "eager", pool[idx]), spec.outputs)
                if not spot.passed:
                    break
            del got
            if not spot.passed:
                spot.failure = f"timed-phase output wrong: {spot.failure}"
                er.correctness = fold([*trials, spot])
                result.status = "incorrect"
                result.flags.append(
                    Flag(code="timed_phase_mismatch", detail=entry.name, fatal=True)
                )
                return
            b_blocks.append(times[best][0] / k)
            c_blocks.append(c_outer / k)
            c_calls.append(c_outer / k)
            n_pairs += 1
            ratio = paired_ratio(b_blocks, c_blocks) if n_pairs >= 3 else None
            verdict = cfg.stopping.decide(ratio, n_pairs)
            for w in (ref, cand):
                w.call("release", timeout=60)
            del pool

        # The worker's own clock is only a diagnostic. If it claims far less time than we observed
        # from outside (or an impossible value), something in that process touched the clock.
        if c_inner_total < (1 - cfg.clock_disagreement) * c_outer_total:
            result.flags.append(
                Flag(
                    code="clock_disagreement",
                    detail=f"{entry.name}: worker reported {c_inner_total:.4g}s, "
                    f"driver observed {c_outer_total:.4g}s",
                )
            )
        er.candidate = TimingStats(**summarize(c_calls))
        er.baselines[best] = TimingStats(**summarize(b_blocks))
        er.speedup = RatioCI(point=ratio.point, lo=ratio.lo, hi=ratio.hi, level=ratio.level)
        er.peak_memory_bytes = cand.call("peak_memory", timeout=60) or None
        cap = spec.outputs.memory_cap_bytes
        if cap and er.peak_memory_bytes and er.peak_memory_bytes > cap:
            er.correctness = fold([*trials, Comparison(False, failure="memory cap exceeded")])
            result.status = "incorrect"
            return
        if verdict == "exhausted" and ratio.rel_halfwidth > cfg.inconclusive_rel_halfwidth:
            result.flags.append(
                Flag(code="wide_ci", detail=f"{entry.name}: ±{ratio.rel_halfwidth:.1%}")
            )
            result.status = "inconclusive"

    if any(f.fatal for f in result.flags):
        result.status = "flagged"
    elif result.status != "inconclusive":
        result.status = "ok"
    if result.status == "ok":
        result.score = weighted_geomean(
            [e.speedup.point for e in result.entries], [e.weight for e in result.entries]
        )
