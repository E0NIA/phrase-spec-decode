# How the numbers were measured

Anything that could move a result and is not forced by the hardware is listed here.

## Setup

| | |
|---|---|
| target | `Qwen/Qwen3-4B`, 4,022,468,096 params, 36 layers, d_model 2560, vocab 151,936 |
| baseline comparison | `AngelSlim/Qwen3-4B_eagle3`, 218,245,120 params, run via SafeAILab/EAGLE @ `cb7e0841fe0c` |
| card | one Tesla T4, sm75, 15 GB |
| dtype | fp16 both sides (T4 has no bf16; the EAGLE head ships bf16 and is cast) |
| transformers | 4.53.1 for the EAGLE stage, as its pinned commit requires |
| batch | 1 |
| temperature | 0 (greedy) |
| cap | 2048 new tokens, **EOS-terminated** — DFlash Table 1's protocol |
| chat template | Qwen3 default with `enable_thinking=False`, both sides |
| prompt truncation | 384 tokens, applied to the prompt *text* then re-templated, so the trailing turn marker is never cut |

## Prompts

50 GSM8K test + 50 MATH-500 test + 40 MT-Bench (the authors' own
`eagle/data/mt_bench/question.jsonl`). Drawn once by stage 2, written to
`results/e89_prompts.json` with a sha, and **loaded** by stage 3 rather than redrawn.

That matters. An earlier version re-derived the pool in each script and a single
constant differing between them (`PER_SOURCE`, 20000 against 4000) changed the
training-pool dedup filter, which changed which test prompts survived, which changed
the draw — while an assert built from a hand-typed constant table certified that the
sets matched. The prompt file removes the whole class of error: there is no selection
code left to disagree with anything.

Test prompts are filtered against the training pool by exact normalised match before
selection. Contamination is additionally measured, not assumed — see below.

## The two statistics, and which to quote

**Aggregate speedup** = `Σ baseline_seconds / Σ arm_seconds`. The whole workload's
wall-clock saving. Its interval comes from resampling prompts and recomputing that
ratio.

**Per-prompt speedup** = `mean(baseline_i / arm_i)`. What a typical prompt sees. Its
interval comes from bootstrapping that mean.

These differ whenever generation lengths vary, and MATH-500's vary a great deal. An
earlier report printed the first as its point estimate beside the second's interval and
produced `2.032 with [2.114, 2.280]` — a figure outside its own CI. Simulated: when
per-prompt speedup correlates with answer length, as depth decay guarantees, the
mean-of-ratios interval misses the aggregate **400 times out of 400**. Both are now
printed, each with the bootstrap that belongs to it, and a run aborts if an aggregate
falls outside its own interval.

**For comparing the two methods, read `ms per generated token`.** A speedup is relative
to its own baseline and the two baselines are different loops — ours measures 63.7
ms/token against EAGLE's `naivegenerate` at 59.6, consistently on all three sets, which
inflates our speedup column by ~7%. Milliseconds per token has no baseline in it.

## Pairing

Paired by `(set, prompt index)`. Both runs load the same prompt file, so index *k* is
the same question on both sides, and the paired delta is bootstrapped over prompts.

Pairing is **checked, not assumed**: both sides are greedy at temperature 0 on the same
weights, so baseline token counts must agree per prompt. They agree 50/50 on GSM8K and
40/40 on MT-Bench exactly, and 47/50 on MATH-500 with a mean absolute difference of 8.7
tokens — fp16 tie-breaking between two different greedy implementations, the same effect
the exactness check reports. A set is skipped rather than reported if agreement falls
below 60% or the mean difference exceeds 8% of the answer length.

## Exactness

The arm's output is compared token-by-token against the baseline's. At each divergence,
one fresh **uncached** forward over prompt + common prefix recomputes the target's own
argmax and the top-2 logit margin, which classifies the divergence instead of
attributing it. A divergence with margin ≥ 0.5 (fp16) would be a real correctness
failure.

Measured: **98 divergences across 300 arm runs, 0 confident, worst margin 0.0625.**
Every one is a tie-break.

## Contamination

8-gram overlap between each test prompt and the training pool, as
`|8-grams(test) ∩ 8-grams(train)| / |8-grams(test)|`:

| set | mean | p90 | max | exact duplicates |
|---|---|---|---|---|
| gsm8k-test | 0.003 | 0.000 | 0.083 | 0 |
| math500 | 0.052 | 0.203 | 0.833 | 0 |
| mtbench | 0.003 | 0.000 | 0.125 | 0 |

Thresholds are **swept, not chosen** — any / 0.50 / 0.25 / 0.10 / >0 — because 8-gram
coincidences occur innocently in maths phrasing while 0.833 is memorisation. At the
strictest setting, dropping 8 of 50 MATH-500 prompts, EAGLE-3(60) moves **2.032 →
2.009** and our figure moves **upward**. Both shifts are inside their CIs.

## What is not controlled

- **One card.** Every speedup here is a T4 number. The decode loop is CPU-dispatch
  bound, so a faster card shrinks the GPU work and exposes the Python: the same
  pipeline on an A5000 measured 1.061× with *better* acceptance. `tok/pass` is
  hardware-independent and transfers; the speedups do not.
- **One target model.** Qwen3-4B only.
- **One domain for training.** The head sees distilled maths and nothing else. MT-Bench
  is reported as the out-of-domain bound rather than filtered out.
- **No per-run repeats.** Each arm runs once per prompt. The CIs are over prompts, not
  over repeated timings, so they do not capture thermal drift. Arms are interleaved per
  prompt against a shared baseline to limit it.
