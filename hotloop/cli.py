"""hotloop command line.

    hotloop deploy                                   # (modal) deploy the app; redo after code changes
    hotloop models --n 8                             # which models the top-downloads rule picks
    hotloop gen --n-models 8 | --model-id ID ...     # trace models into tasks
    hotloop filter --gpu L4                          # automatic task filters for a GPU type
    hotloop tasks --gpu L4 [--all]                   # list (kept) tasks
    hotloop score TASK solution.py [--public]        # score any solution file
    hotloop selftest [--task T] [--suite rmsnorm]    # exploit suite
    hotloop run --agent openai -a model=gpt-5.4-mini --task T --minutes 30

Global options: --backend modal|local, --gpu, and for local: --root, --isolation docker|none.
"""

import argparse
import json
import os
import random
import subprocess
import sys

from hotloop import config

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def _backend(args):
    from hotloop.bench.registry import make_backend
    if args.backend == "local":
        return make_backend("local", root=args.root, isolation=args.isolation)
    return make_backend("modal")


def _agent_kwargs(pairs: list[str]) -> dict:
    out = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        out[k] = v
    return out


def _reason_line(reason: str) -> str:
    lines = [l.strip() for l in (reason or "").strip().splitlines() if l.strip()]
    for l in reversed(lines):
        if "Error" in l or "error" in l or l.startswith(("output", "c0_", "c1_", "timed_", "repeat")):
            return l
    return lines[-1] if lines else ""


def _print_eval(ev: dict):
    sc = ev.get("score", {})
    print(f"correct={sc.get('correct')}  speedup={sc.get('speedup_geomean', 0):.3f}x  "
          f"sol={100 * sc.get('sol_frac_geomean', 0):.1f}%  flagged={sc.get('flagged')}")
    for sid, s in ev.get("shapes", {}).items():
        print(f"  {sid:10} correct={s['correct']!s:5} speedup={s.get('speedup', 0):6.3f} "
              f"sol={100 * s.get('sol_frac', 0):5.1f}%  {_reason_line(s.get('reason'))[:140]}")
    if ev.get("violations"):
        print("violations:", ev["violations"])
    if ev.get("error"):
        print("error:", ev["error"][:1500])


def cmd_run(args, backend):
    from hotloop.bench.episode import run_episode, save_episode
    from hotloop.bench.registry import make_agent

    kwargs = _agent_kwargs(args.agent_arg)
    if args.task:
        tasks = args.task
    else:
        kept = backend.list_tasks(args.gpu, kept_only=True)
        if args.phase:
            kept = [t for t in kept if ("__decode__" in t) == (args.phase == "decode")]
        # A seeded random sample, so which tasks get run is not a hand-picked choice.
        tasks = random.Random(args.seed).sample(kept, min(args.limit, len(kept))) if args.sample else kept[: args.limit]
    if not tasks:
        sys.exit("no tasks (run `hotloop filter` first or pass --task)")
    remote = args.backend == "modal" and not args.local_agent
    if remote:
        calls = {t: backend.spawn_episode(args.agent, kwargs, t, args.gpu, args.minutes) for t in tasks}
        for t, c in calls.items():
            print(f"[spawned] {t}  call={c.object_id}")
        for t, c in calls.items():
            rec = c.get()
            print(f"\n=== {t}  ({rec['agent']}, {rec['stop_reason']}, {rec['agent_minutes']} min, run {rec['run_id']})")
            print("usage:", json.dumps(rec["usage"]))
            _print_eval(rec["eval"])
    else:
        agent = make_agent(args.agent, **kwargs)
        for t in tasks:
            ep = run_episode(backend, agent, t, args.gpu, args.minutes)
            d = save_episode(ep, os.path.join(REPO, "runs"))
            rec = ep["record"]
            print(f"\n=== {t}  ({rec['agent']}, {rec['stop_reason']}, {rec['agent_minutes']} min) saved to {d}")
            _print_eval(rec["eval"])


def main():
    # Common options work before or after the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--backend", default=argparse.SUPPRESS, choices=["modal", "local"])
    common.add_argument("--gpu", default=argparse.SUPPRESS)
    common.add_argument("--root", default=argparse.SUPPRESS, help="local backend data dir (default ~/.hotloop)")
    common.add_argument("--isolation", default=argparse.SUPPRESS, choices=["docker", "none"],
                        help="local backend isolation (default docker)")
    ap = argparse.ArgumentParser(prog="hotloop", description=__doc__, parents=[common],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    add = lambda name: sub.add_parser(name, parents=[common])

    add("deploy")
    p = add("models"); p.add_argument("--n", type=int, default=8)
    p = add("gen"); p.add_argument("--n-models", type=int, default=0); p.add_argument("--model-id", nargs="*", default=[])
    p.add_argument("--phases", default="prefill,decode")
    p = add("filter"); p.add_argument("--tasks", default="")
    p = add("tasks"); p.add_argument("--all", action="store_true")
    p = add("score"); p.add_argument("task"); p.add_argument("solution"); p.add_argument("--public", action="store_true")
    p = add("selftest")
    p.add_argument("--task", default="qwen2.5-0.5b-instruct__layers.input_layernorm")
    p.add_argument("--suite", default="rmsnorm"); p.add_argument("--only", default="")
    p = add("run")
    p.add_argument("--agent", default="openai", help="registered name or module:Class")
    p.add_argument("-a", "--agent-arg", action="append", help="agent option key=value (repeatable)")
    p.add_argument("--task", action="append", help="task id (repeatable); default: kept tasks for --gpu")
    p.add_argument("--limit", type=int, default=1, help="with no --task: how many kept tasks to run")
    p.add_argument("--sample", action="store_true", help="with no --task: random sample of kept tasks (see --seed)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--phase", choices=["prefill", "decode"], help="with no --task: only tasks of this phase")
    p.add_argument("--minutes", type=float, default=30)
    p.add_argument("--local-agent", action="store_true", help="(modal) run the agent loop on this machine")
    args = ap.parse_args()
    for k, v in (("backend", "modal"), ("gpu", config.DEV_GPU), ("root", "~/.hotloop"), ("isolation", "docker")):
        if not hasattr(args, k):
            setattr(args, k, v)

    if args.cmd == "deploy":
        if args.backend != "modal":
            sys.exit("deploy is for the modal backend; for local, build docker/Dockerfile")
        sys.exit(subprocess.call(["modal", "deploy", "-m", "hotloop.backends.modal_app"], cwd=REPO))

    backend = _backend(args)
    if args.cmd == "models":
        for m in backend.select_models(args.n):
            print(f"{m['downloads']:>12,}  {m['model_id']}")
    elif args.cmd == "gen":
        ids = args.model_id or [m["model_id"] for m in backend.select_models(args.n_models)]
        print("models:", ids)
        for mid, r in backend.generate(ids, tuple(args.phases.split(","))).items():
            print(f"{mid}: {sorted(r) if 'error' not in r else r['error']}")
    elif args.cmd == "filter":
        res = backend.filter(args.gpu, args.tasks.split(",") if args.tasks else None)
        for tid, s in sorted(res.items()):
            extra = (f"eager={s['eager_ms']:.3f}ms compile={s.get('compile_ms') or 0:.3f}ms SOL={s['sol_ms']:.3f}ms"
                     if "sol_ms" in s else "")
            print(f"{'KEEP' if s['keep'] else 'drop'}  {tid:55} {extra}  {'; '.join(s.get('drop_reasons', []))}")
    elif args.cmd == "tasks":
        for t in backend.list_tasks(args.gpu, kept_only=not args.all):
            print(t)
    elif args.cmd == "score":
        _print_eval(backend.score(args.task, open(args.solution).read(), gpu=args.gpu, hidden=not args.public))
    elif args.cmd == "selftest":
        from hotloop.bench.selftest import run_selftest
        ok = run_selftest(backend, args.task, os.path.join(REPO, "exploits", args.suite), args.gpu,
                          only=args.only.split(",") if args.only else None)
        sys.exit(0 if ok else 1)
    elif args.cmd == "run":
        cmd_run(args, backend)


if __name__ == "__main__":
    main()
