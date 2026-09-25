# HotLoop

HotLoop is a benchmark for AI agents that write GPU kernels. The agent gets
a piece of PyTorch code, a GPU machine, a time budget and no internet, and
has to replace the code with custom kernels that run faster.

The tasks are realistic workloads. They aren't toy problems like "write a
matmul". Each one is a module traced out of a popular Hugging Face model,
at the shapes it actually runs at during inference: prefill over a prompt,
or decode against a filled KV cache. Speedups are measured against
`torch.compile`, the baseline people actually use, and against the
hardware's speed-of-light limit.

The harness also assumes the agent will try to cheat. Returning cached
outputs, calling cuBLAS under the hood, skipping work the checker doesn't
look at, or tuning for the one visible shape all score zero. The only way to
score is a kernel that is actually fast and correct.

## Where tasks come from

There are no hand-written tasks. HotLoop takes the most downloaded models on
Hugging Face, traces them with TorchDynamo, and cuts each module out into a
standalone `reference.py`. It traces two workloads:

- **Prefill**: one forward pass over a prompt, varying batch size and prompt length.
- **Decode**: one new token per sequence against a filled KV cache, varying
  batch size and cache length. The cache is updated in place, and that update
  is checked too.

Identical computations across models are merged, and shapes are kept
symbolic. Filters then throw out tasks that are trivial, too small to time
reliably, or where `torch.compile` is already close to the hardware limit.
Each task has one shape the agent can see and several hidden shapes used only
for scoring, stored where the agent can't read them.

## How a solution is scored

**Correctness.** The solution runs in its own process as an unprivileged
user. It is tested on every hidden shape, with secret seeds and different
input distributions (normal, outliers, large and small values), and compared
to an fp64 reference computed in a separate trusted process. The tolerance
isn't hand-picked: for each case it is set by how far PyTorch's own
native-precision result drifts from fp64.

**No borrowed kernels.** Before anything runs, a static check limits imports
and blocks escape hatches like `getattr`, `torch.os` and `ctypes`. Inside
`solution()`, a `TorchDispatchMode` only
allows allocation and view ops, so the agent can't just call `torch.matmul`.
A second check looks at the names of every kernel launched on the GPU and
rejects cuBLAS, ATen and Inductor kernels, including ones launched from C++.
Ban messages tell the agent exactly what tripped them (for example a hidden
copy from `.reshape()` or `.contiguous()`), so a failed attempt is useful
feedback rather than a dead end.

**Timing.** The solution is captured into a CUDA graph and replayed. Every
replay gets fresh inputs and a flushed L2 cache, and every output is checked.
This means caching results, doing work on the CPU or in another process, or
launching on a side stream that never rejoins all produce wrong answers
instead of fast times.

**Baseline and roofline.** The baseline is `torch.compile` of the reference,
timed in the same session, interleaved with the solution to cancel out clock
drift. The speed-of-light bound uses peak compute and bandwidth measured on the
actual GPU, FLOPs counted from the reference (masked attention only counts
the unmasked part), and the minimum bytes the task has to move. This works on
any NVIDIA GPU.

**Result.** A task scores 0 unless every hidden shape passes. Otherwise it
reports the geomean speedup over `torch.compile` and the geomean fraction of
speed-of-light. Anything above 80% of speed-of-light is flagged for manual
review, and any tampering with harness functions is a violation.

### Apple GPUs (Metal)

The harness runs on Apple GPUs through PyTorch MPS; solutions write Metal kernels
with `torch.mps.compile_shader`. Differences from CUDA: no graph replay (timing is
host-side around device syncs, eager for both solution and baseline), fp64
references run on the CPU, and there is no kernel-level profiler list, so the ban
relies on the op-level check and static rules.

```bash
HOTLOOP_DEVICE=mps hotloop --backend local --isolation none selftest --suite rmsnorm_metal --gpu M5
```

## Exploit suite

`exploits/` holds solutions that each try one known trick: cached outputs,
returning zeros, reading stale memory, fp8 precision loss, side streams,
torch ops, ATen from C++, `torch.compile`, patching the timer, importing the
harness, detecting graph capture, probing hidden data, hardcoding shapes.
Each one states the outcome it should get, and `hotloop selftest` runs them
all. There's also a real Triton kernel as a positive control, to make sure
the checks don't reject honest work.

## Agents and backends

The benchmark and the agents don't import each other. An agent receives an
`Environment` (a GPU machine with `TASK.md`, `task/` and `solution.py` in its
workspace), the task text and a `Budget`, and leaves its answer in
`solution.py`. Adding an agent means writing one class (see
`hotloop/interface.py`). Adding a compute provider means implementing
`backends/base.Backend`.

Right now there's an agent loop for any OpenAI-compatible endpoint, plus
self-hosted serving of open-weight models on vLLM. Runs happen on Modal or
on any GPU machine with Docker.

## Usage

```bash
uv venv && uv pip install -e ".[agents]"
modal setup                                                   # once
modal secret create hotloop-keys HF_TOKEN=... OPENAI_API_KEY=...
hotloop deploy                                                # after every code change

hotloop models --n 8                                          # which models get picked
hotloop gen --n-models 8                                      # trace them into tasks
hotloop filter --gpu L4                                       # run the filters for a GPU type
hotloop tasks --gpu L4                                        # list kept tasks
hotloop selftest                                              # run the exploit suite
hotloop score <task> my_solution.py                           # score any solution
hotloop run -a model=gpt-5.4-mini --task <task> --minutes 30  # one agent episode
hotloop run -a model=gpt-5.4-mini --sample --limit 5 --phase decode   # seeded sample of tasks
```

Agent options are passed with `-a key=value`. For an open-weight model behind
vLLM or SGLang: `-a model=<name> -a api=chat -a base_url=http://host:8000/v1 -a effort=none`.
For your own agent: `--agent mypkg.module:MyAgent`.

On Modal, episodes run inside a deployed function, so API keys stay on Modal
and runs keep going if your laptop disconnects. Results go to the
`hotloop-runs` volume.

### Local backend

```bash
docker build -f docker/Dockerfile -t hotloop:latest .
sudo nvidia-smi -lgc <mhz>                  # optional: lock clocks for stable timing
hotloop --backend local gen --model-id Qwen/Qwen3-0.6B
hotloop --backend local run -a model=... --task <task>
```

`--isolation none` skips containers. It has no network or filesystem
isolation, so only use it for development.

## Limitations

- Only inference (prefill and decode) for now. Training backward passes and
  MoE models are next.
- Prefill shapes come from a fixed grid. They should be sampled from real
  request traces.
- The solution runs in the same process as the timing code. Tampering is
  caught with identity checks and static rules, not process isolation.
- The kernel denylist matches on names.
- Modal can't lock GPU clocks. The local backend can.
