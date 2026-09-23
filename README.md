# HotLoop — agentic GPU kernel optimization benchmark

Agents get a PyTorch reference cut from a real model and must write faster
custom kernels. Tasks are generated automatically, scoring is against the
speed-of-light bound and the best existing baseline, and the harness is built
so that the only way to score is a genuinely fast, correct kernel.

## Layout

```
hotloop/
  interface.py     the agent contract: Environment, Budget, AgentResult, Agent
  bench/           benchmark core (no backend or agent code)
    store.py         task/hidden/runs directory layout
    workspace.py     TASK.md (task statement + rules) and the starter solution
    episode.py       open env -> run agent -> collect solution -> score; enforces the budget
    selftest.py      exploit suite runner
    registry.py      backends and agents by name
  gen/             task generation: model selection, tracing, filters
  harness/         scoring: correctness, timing, roofline, bans (+ `bench` for agents)
  backends/        where things run
    modal_app.py     Modal: images, volumes, deployed functions
    modal_backend.py Modal client: sandboxes, scoring, remote episodes
    local_backend.py a GPU machine you control (Docker, or plain processes)
  agents/          agent adapters; may import only hotloop.interface
    openai_agent.py  reference loop for any OpenAI-compatible endpoint
docker/Dockerfile  image for the local backend
exploits/          solutions that must score 0 (plus positive controls)
```

The benchmark never imports agents and agents never import the benchmark.
An agent gets an `Environment` (GPU machine, no network, workspace with
`TASK.md`, `task/`, `solution.py`), the task text and a `Budget`, and leaves
its answer in `solution.py`. Adding an agent means writing one class; adding a
compute provider means implementing `backends/base.Backend`.

## Pipeline

```
top-N HF models by downloads      (gen/models.py)   one rule, no hand-picked lists
  → trace each model with Dynamo  (gen/trace.py)    per-module FX subgraphs, real value stats
      prefill: one forward pass over a prompt                     varying B (batch), S (prompt length)
      decode:  one token per sequence against a filled KV cache   varying B, T (cache length)
  → standalone reference.py per shape, deduplicated by computation, symbolic shapes
  → filters                       (gen/filters.py)  degenerate? too small? baseline already near SOL?
  → public shape (agents see) + hidden shapes (scoring only, separate storage)
```

## Scoring (`harness/`)

| Step | What happens |
|---|---|
| Static check | import allowlist, no escape hatches (`getattr`, `torch.os`, `ctypes`, ...) |
| Load | solution runs in its own process as an unprivileged user; hidden data unreadable |
| Correctness | hidden shapes × secret seeds × input variants (normal, outliers, scaled) + repeat; checked against an **fp64 reference** in a separate trusted process |
| Tolerance | derived per case from PyTorch's own native-dtype error vs fp64 (no hand-picked atol) |
| Ban | `TorchDispatchMode` allows only allocation/view ops inside `solution()`; kernel-name denylist catches cuBLAS/ATen/Inductor kernels launched from anywhere |
| Timing | solution is captured into a **CUDA graph**; each replay gets fresh inputs, an L2 flush, and its output is checked. Caching, CPU/side-process work and non-rejoined streams produce wrong outputs |
| Baseline | `torch.compile` of the reference, timed interleaved in the same session (eager is reported for reference and used to correct cached baseline times for clock drift) |
| Roofline | peaks measured on the device (any GPU); FLOPs counted from the reference (masked attention counts only unmasked work); bytes = unavoidable I/O |
| Flags | faster than 80% of speed-of-light → flagged for review; harness function tampering → violation |

Score per task: 0 unless every hidden shape passes; otherwise geomean speedup
over `torch.compile` of the reference and geomean fraction of speed-of-light.

## Usage

```bash
uv venv && uv pip install -e ".[agents]"
modal setup                                                   # once
modal secret create hotloop-keys HF_TOKEN=... OPENAI_API_KEY=...
hotloop deploy                                                # after every code change

hotloop models --n 8                                          # which models the rule picks
hotloop gen --n-models 8                                      # trace → tasks
hotloop filter --gpu L4                                       # automatic filters for a GPU type
hotloop tasks --gpu L4                                        # kept tasks
hotloop selftest                                              # exploit suite
hotloop score <task> my_solution.py                           # score any solution
hotloop run -a model=gpt-5.4-mini --task <task> --minutes 30  # one agent episode
hotloop run -a model=gpt-5.4-mini --sample --limit 5 --phase decode   # seeded sample of kept tasks
```

Agent options go through `-a key=value`. For open-weight models served by
vLLM/SGLang: `-a model=<name> -a api=chat -a base_url=http://host:8000/v1 -a effort=none`.
A custom agent: `--agent mypkg.module:MyAgent`.

On Modal, episodes run in a deployed function (API keys stay in Modal and runs
survive local disconnects); results land in the `hotloop-runs` volume.

### Local backend (GPU VM with Docker)

```bash
docker build -f docker/Dockerfile -t hotloop:latest .
sudo nvidia-smi -lgc <mhz>                  # optional: lock clocks for stable timing
hotloop --backend local gen --model-id Qwen/Qwen3-0.6B
hotloop --backend local run -a model=... --task <task>
```

`--isolation none` runs without containers (no network/filesystem isolation;
development only).

## Exploit suite

`exploits/<suite>/*.py` are solutions that each try one known hack (cached
outputs, zeros, stale memory, fp8 precision loss, side streams, torch ops,
ATen from C++, torch.compile, timer monkeypatching, harness imports, capture
detection, hidden-data probing, shape hardcoding). Each file states its
expected outcome; `hotloop selftest` must report all as expected.

## Known limitations (v1)

- Prefill and decode (static KV cache, updated in place and checked) are covered;
  training backward passes and MoE models are next.
- Prefill shapes are a fixed grid; should be sampled from public request traces.
- The solution shares a process with the timing code; tampering is caught by
  identity checks and static rules, not by process isolation.
- The kernel denylist is name-based.
- Modal cannot lock GPU clocks; the local backend can.
