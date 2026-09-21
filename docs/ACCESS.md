# Access, accounts, infra state

Status as of 2026-09-20. ✅ = ready, ⬜ = needed from Emaan, 🔬 = to be probed by Claude once access exists.
Secrets go in `.env` (gitignored) or the provider CLI's own config — never in the repo.

## Local machine (checked)

- ✅ macOS arm64, `uv` 0.11.8, Docker 29.4, `gh` authenticated as `emaanh` (ssh), `~/.ssh/id_ed25519.pub`
- ⬜ no GPU locally (expected) → all GPU work is remote
- ⬜ no cloud GPU CLI/credentials; no HF token; no LLM API keys in env (SkyPilot to be installed via `uv`)

## What I need (ordered by how much it blocks)

| # | What | Why | How (≈ time) | Blocks |
|---|---|---|---|---|
| 1 | ✅ **Lambda Cloud** — `LAMBDA_API_KEY` in `.env` (verified 2026-09-20, HTTP 200, 0 running instances). SSH key `emaan-macbook-hotloop` uploaded via API. | Root GPU VMs for everything (D-13). | done | — |
| 2 | ⬜ (backup, only if Lambda is out of capacity or refuses clock locking) **Crusoe** or **Hyperstack** account + API key | Same role. Crusoe documents clock locking; Hyperstack is cheapest ($2.50 H100 PCIe). Runner is provider-agnostic so this is a drop-in. | 5 min | nothing yet |
| 3 | ✅ **Frontier model key — OpenAI** (`OPENAI_API_KEY` in `.env`, verified 2026-09-20 via `/v1/models`: 132 models incl. gpt-5.x / codex lines; exact model chosen at M3. Anthropic optional) | Frontier reference agent (D-15): proves benchmark is unsaturated & discriminative (gate G2). Also secondary trajectory labelling. | console.anthropic.com (2 min) | M3 |
| 4 | ⬜ **Hugging Face token** (`HF_TOKEN`, read scope; accept licences for gated models when we pick them) | Pull open-weight checkpoints + quantized variants | hf.co/settings/tokens (2 min) | M3/M5 |
| 5 | ⬜ **Budget approval** — see [BUDGET.md](BUDGET.md): honest total ≈ $4,200 (≈ $2,200–2,600 with levers). Staged: **T1 $300** through G1, T2 $600 through G2, T3 sized after pilot | Lets me launch GPU jobs without asking each time. I log spend in `docs/LOG.md`. | just tell me the number | — |
| 6 | ⬜ **Permission allowlist** for this repo so I'm not prompting on every command: `uv`, `uvx`, `sky`, `git` (non-push), `ssh`/`scp` to GPU boxes, `docker build`, `curl` to provider APIs | Long unattended runs | I can write `.claude/settings.json` via `/update-config` once you say OK | convenience |
| 7 | ⬜ (optional) **OpenRouter key** | Cheap 1-day triage of which open-weight model is strong enough *before* renting serving GPUs | openrouter.ai (2 min) | speeds M3 |
| 8 | ⬜ (optional) **Anthropic key** | Second frontier reference point | — | nice-to-have |
| 9 | ✅ GitHub remote exists: `emaanh/HotLoop`, **public**, already `origin` (created by Emaan; empty as of 2026-09-20). Hidden material stays out via gitignored `private/` (D-16). ⬜ Need explicit OK before first push (public = outward-facing). | Backup + GHCR for the pinned benchmark image | say "push ok" | — |

Not needed: Modal (dropped as primary by D-13; possible burst backend later), AWS/GCP (quota
delays), W&B (results are JSONL/parquet), RunPod.

## Infra plan (→ D-13, supersedes D-8)

- **All GPU work on root VMs**, Lambda first. Provisioned/torn down with **SkyPilot** (autostop on
  idle) + an on-box idle watchdog, because Lambda has no stop/timeout and a leaked VM bills forever.
- **Bench VM** (1× H100 or A100 per VM, 4–8 in parallel at peak): clocks locked, persistence mode,
  `NVreg_RestrictProfilingToAdminUsers=0`. Agent runs in Docker `--gpus all` (untrusted); final
  evaluation in a fresh container on the same SKU (trusted). One trajectory per GPU at a time.
- **Serving VM** (separate, 1× H100): vLLM behind an OpenAI-compatible URL, reached over an SSH
  tunnel. Never shares a GPU with benchmarking.
- Runner assumes only "Docker host + NVIDIA GPU over SSH" → any provider is a drop-in.
- Known limits: Lambda capacity sells out; no L40S (hardware axis = A100 vs H100 / A10 / GH200).

## Lambda live snapshot (API, 2026-09-20)

| Type | $/h | Available |
|---|---|---|
| gpu_1x_a10 (24 GB) | 1.29 | us-east-1, us-west-1 |
| **gpu_1x_a100_sxm4 (40 GB)** | **1.99** | asia-south-1, us-east-1, us-west-2 |
| gpu_1x_gh200 (96 GB, aarch64) | 2.29 | — |
| gpu_1x_h100_pcie | 3.29 | — |
| gpu_1x_h100_sxm5 | 4.29 | — |
| gpu_2x_h100_sxm5 | 8.38 | us-south-3 |
| gpu_8x_a100 (40 GB) | 15.92 (1.99/GPU) | us-east-1 |
| gpu_8x_h100_sxm5 | 31.92 (3.99/GPU) | — |
| gpu_1x_b200_sxm6 | 6.99 | — |

Takeaways: single H100s are sold out (capacity risk is real); A100-40GB is cheap and available →
primary bench SKU (D-14); multi-GPU nodes give no per-GPU discount.

## What a Lambda A100 box actually is (probed 2026-09-20, `docs/data/m1/probe_box.txt`)

- KVM VM, AMD EPYC 7J13, 30 vCPU, 216 GiB RAM, Ubuntu 22.04, kernel 6.8. Boot-to-SSH ≈ 3.5 min.
- A100-SXM4-40GB, driver 570.148.08 (→ max CUDA 12.8), ECC on, persistence on.
- ✅ `sudo nvidia-smi -lgc 1410,1410` works. Memory-clock lock unsupported on A100 (HBM; expected).
- ⚠️ `RmProfilingAdminOnly: 1` → hardware counters need root / a privileged container, or a
  modprobe option + reboot. Not needed yet; tracing profilers (torch.profiler, nsys) don't need it.
- ⚠️ Host Python is 3.10 and nsys is not installed; `docker` needs sudo (user not in docker group).
  → the benchmark must bring its own image: Python 3.12, **torch from the cu128 index**
  (PyPI's default torch targets CUDA 13 and refuses driver 570), nsys, nvcc.
- Working stack used for M1: torch 2.11.0+cu128, triton 3.6.0, Python 3.12 via uv.
- API quirk: Lambda's Cloudflare front rejects urllib's default User-Agent (403 / error 1010).

## 🔬 Day-1 probes (done 2026-09-20 for items 1–4; 5–6 need a second box)

1. Do `sudo nvidia-smi -pm 1` and `-lgc <max>,<max>` work? (Unverified for Lambda. If refused → Crusoe.)
2. Can we set `NVreg_RestrictProfilingToAdminUsers=0`? Do `nsys --gpu-metrics-devices` and Nsight
   Compute counters work inside the Docker container?
3. Timing noise, clocks locked vs unlocked: CV of (a) large matmul, (b) bandwidth-bound pointwise,
   (c) launch-bound loop of 1k tiny kernels, over 10 min; across reboots; across two VMs of same SKU.
4. Docker overhead on (c) vs bare host (expected ≈ 0; confirm).
5. CPU model / PCIe topology consistency across Lambda VMs of the same instance type.
6. Boot-to-ready time and how often the instance type is actually available.

## Spend log

| Date | Provider | What | $ |
|---|---|---|---|
| 2026-09-20 | Lambda | `hotloop-probe-1` gpu_1x_a100_sxm4 us-west-2, 23:26Z–23:51Z (24.6 min): day-1 probes + evaluator GPU suite. **Terminated.** | 0.82 |
| 2026-09-20 | Lambda | `hotloop-m2-explore` gpu_1x_a100_sxm4, 23:57Z–00:14Z (16.8 min): cross-box G0 retest + M2 regime exploration. **Terminated.** | 0.56 |
| 2026-09-21 | Lambda | `hotloop-m2-certify` gpu_1x_a100_sxm4, launched 00:50Z: certification, M3 plumbing, pilot trajectory; now M3 batch host #1 — **LIVE** | 1.99/h, running |
| 2026-09-21 | Lambda | `hotloop-m3-2`, `-3`, `-4` gpu_1x_a100_sxm4, launched 02:14Z: M3-lite batch hosts — **LIVE** | 3 × 1.99/h, running |
| | | **Total to date** (from `~/.hotloop/vm_ledger.jsonl`) | **1.37** |
