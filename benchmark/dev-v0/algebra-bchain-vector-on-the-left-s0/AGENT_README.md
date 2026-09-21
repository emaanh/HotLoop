# Task: Matmul chain (bchain vector on the left)

Make `reference(*inputs)` in `reference.py` faster on this machine's GPU, without changing what it
computes.

## What you are given
- `reference.py` - the PyTorch reference. It defines the semantics and is the correctness oracle.
- `workload.py` - `make_inputs(entry, seed, device)` builds the inputs you will be timed on.
  Shapes, dtypes and the *distribution* of the data are fixed by this file; the values are not.
- `task.toml` - metadata: output tolerances, the automatic baselines you are scored against.

## The program and the workload
A batched product `x @ w1 @ w2 @ y`, float32: `x` is `(64, 1, 512)` and `y` is `(64, 512, 512)` (different for every sample); `w1` `(512, 512)` and `w2` `(512, 512)` are shared across the batch. Values are random and change with every call; shapes are fixed.

## What you must deliver
A file `solution.py` in the submission directory `/workspace/submission` exposing

    def run(*inputs): ...   # same arguments as reference(), same outputs (shape, dtype, device)

Anything is allowed inside: Triton, CUDA/C++ extensions, torch.compile, CUDA graphs, cuBLAS calls,
algebraic rewrites in plain PyTorch. Extra files next to `solution.py` are fine. Rules of the road:
- do not mutate the inputs;
- every returned tensor must stay valid after later calls to `run` (clone if you reuse buffers);
- a returned tensor must be safe to read on the current CUDA stream.

## How you are scored
Score = speedup of `run` over the **fastest automatic baseline** - the best of eager PyTorch and
`torch.compile` in several modes (eager, compile_default, compile_max_autotune_no_cudagraphs, compile_reduce_overhead) - measured interleaved with it on the same GPU.
Beating eager is not the goal; beating the compiler is. A score of 1.0 means "no better than the
compiler". Any incorrect output, at any point, scores 0 - including outputs produced while being
timed. Final scoring happens on a fresh machine of the same type with inputs drawn from hidden
seeds, so nothing keyed on particular values or addresses will help.

Check yourself at any time:

    hotloop-eval --task /workspace/task --submission /workspace/submission --quick

It reports correctness and your speedup with a confidence interval, using the same code path as
final scoring. You have a real GPU, a shell, compilers and profilers; use them however you like.
