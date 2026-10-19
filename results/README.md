# Results

## The shared prompt set

`e89_prompts.json` — sha `5c3a2f181dd5`, 31 KB. The 50 GSM8K + 50 MATH-500 + 40
MT-Bench prompts both methods were measured on. Stage 2 writes it; stage 3 loads it and
asserts the sha. **This file is why the comparison is on identical prompts** rather
than on two independent draws that a single differing constant could separate.

## Aggregate tables

### Ours — `pipeline/3_bench_ours.py`

| set | n | base ms/tok | mean length | arm | speedup | tok/pass | nodes | fed | banked/fed |
|---|---|---|---|---|---|---|---|---|---|
| gsm8k-test | 50 | 63.7 | 276 | N8 | 1.985 | 2.406 | 10.0 | 12.4 | 0.1136 |
| | | | | N16 | 2.108 | 2.590 | 21.6 | 24.2 | 0.0658 |
| | | | | **N32** | **2.211** | **2.742** | 46.4 | 49.1 | 0.0355 |
| math500 | 50 | 63.8 | 571 | N8 | 1.983 | 2.445 | 10.1 | 12.7 | 0.1141 |
| | | | | **N16** | **2.037** | 2.620 | 21.8 | 24.5 | 0.0661 |
| | | | | N32 | 2.005 | 2.762 | 46.4 | 49.3 | 0.0357 |
| mtbench | 40 | 63.6 | 362 | N8 | 1.341 | 1.571 | 7.9 | 9.5 | 0.0601 |
| | | | | **N16** | **1.402** | 1.672 | 18.6 | 20.3 | 0.0331 |
| | | | | N32 | 1.348 | 1.679 | 42.6 | 44.3 | 0.0153 |

### EAGLE-3 — `pipeline/2_bench_eagle3.py`

| set | n | naive ms/tok | mean length | arm | speedup | tok/pass | ms/tok |
|---|---|---|---|---|---|---|---|
| gsm8k-test | 50 | 59.6 | 276 | tree 60 | **2.451** | **3.811** | 24.3 |
| | | | | tree 16 | 2.418 | 3.294 | 24.5 |
| math500 | 50 | 59.8 | 562 | tree 60 | 2.032 | 3.497 | 28.9 |
| | | | | tree 16 | 2.061 | 3.035 | 27.9 |
| mtbench | 40 | 59.6 | 362 | tree 60 | 2.005 | 3.324 | 29.0 |
| | | | | tree 16 | 1.898 | 2.871 | 29.7 |

DFlash's Table 1 publishes tok/pass 3.77 and 3.52 for trees 60 and 16 on GSM8K and
MATH-500. Measured here: **3.811 and 3.497.**

## Per-prompt records

| file | what |
|---|---|
| `e89_pairs.csv` | 280 rows — EAGLE-3, per prompt: `base_secs`, `arm_secs`, `speedup`, `base_tok`, `arm_tok` |
| `e89_rows.csv` | the aggregate table above, as written by the run |
| `e89_diag.csv` | the run's own invariant checks and their values |
| `e87_pairs.csv` | 120 rows — an **earlier, 12-prompt** run of our head. See the warning below. |
| `e87_arms.csv`, `e87_ladder.csv` | that run's aggregates and its data-scaling ladder |

**`e91_pairs.csv` is absent, and that is a recorded failure rather than an omission.**
The run that produced the "ours" table above saved its diagnostics and its head
checkpoint but not its tables: `flush()` wrote a list that nothing appends to. The
aggregate numbers above were recovered from the run log; the per-prompt records were
not recoverable and the stage has to be re-run to regenerate them. The write function
now asserts, as a post-condition on the filesystem, that every table it should have
written exists with the right row count.

### Warning about `e87_*.csv`

Those numbers (GSM8K 2.341×, MATH-500 2.620× at N16) are on **12 prompts** — indices
0–11 of the 50-prompt pool — not 50. On MATH-500 those 12 average **361 tokens against
571** over the full set, 36% shorter, and speedup falls with generation length, so that
figure is inflated by sample composition. GSM8K's 12 *are* representative (274 against
276 tokens).

Shown independently rather than argued: restricted to the same 12 MATH-500 prompts,
**EAGLE-3(60) scores 2.181 instead of 2.032 on all 50**. The short subset flatters both
methods.

The 12-prompt figures are kept here for provenance and **should not be quoted.** The
50-prompt numbers at the top of this file supersede them.
