# Budget

Status: estimate **v2** (2026-09-20), using **live Lambda prices** from the API. **Not yet approved
by Emaan** beyond verbal "sounds good" to staging. Actual spend is logged in ACCESS.md "Spend log".
v1 (same day) assumed H100 bench boxes at $4.20/h and an 8-GPU discount; both were wrong — see D-14.

Prices: bench = 1× A100-40GB $1.99/h. Serving = 2× H100 $8.38/h (only ≥80 GB Hopper option with
capacity today; 1× H100 $3.29–4.29 or GH200 $2.29 if they appear). Per-minute billing. Bench VM is
held for the whole trajectory (~1 h) + ~0.15 h final eval.

| Stage | What | Hours | $ |
|---|---|---|---|
| M1 | Probes + timing-noise study (A100; H100 if obtainable) | ~20 | ~50 |
| M2 (→G1) | 3 families × ~30 points × ~5 strategies/baselines + strategy dev on GPU | ~50 | ~100 |
| M3 (→G2) bench | 72 trajectories × 1.15 h + debug | ~110 | ~220 |
| M3 serving | Qwen BF16 ~15 h on 2× H100 (or ~$20 via OpenRouter if the model is hosted there) | ~15 | ~125 |
| M3 API | 36 frontier trajectories (OpenAI) @ ~$10 guess; pricing unverified | — | ~350 |
| M4 | 5–7 more families + H100 second-SKU sweeps | ~110 | ~300 |
| M5 bench | 360 trajectories × 1.15 h | ~414 | ~825 |
| M5 serving | 5 arms × ~14 h on 2× H100 (halves if 1× H100/GH200 available) | ~70 | ~590 |
| Slop ~15% | idle, mistakes, reruns | | ~380 |
| **Total** | | | **~2,950** |

## Structure of the cost
- Bench-GPU idle during model turns is still the biggest line, but at $1.99/h it's tolerable.
- Serving is now a real cost because only 2× H100 has capacity. Watch for 1× H100 / GH200.
- Core-idea falsification (through G1) ≈ **$150**.

## Levers
0. A 2× H100 box can serve **two arms at once, each TP=1 on its own GPU** (keeps the TP=1 control
   from QUANT_STUDY): M5 serving ≈ $590 → ≈ $300. Default plan.
1. Serving on 1× H100 or GH200 when available: similar saving, simpler.
2. Cap trajectory at 45 min; seeds from pilot power analysis: −25–30% on M5 bench.
3. Hosted inference (OpenRouter) for the BF16 pilot arm in M3: −~$100.
4. Defer H100 second SKU in M4: −~$100.
(Dropped: 8-GPU node discount — doesn't exist on Lambda.)

With 0+2 ≈ **$2,300**.

## Staged approval
| Tranche | Unlocks | Amount |
|---|---|---|
| T1 | M1 + M2, through gate G1 | $300 (expected use ~$150–200) |
| T2 | M3, through gate G2 (only if G1 passes) | $700 |
| T3 | M4 + M5 — sized after pilot variance/power analysis | TBD (~$1,300–2,000) |
