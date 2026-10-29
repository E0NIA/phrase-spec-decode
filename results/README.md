# Results

## The shared prompt set

`e89_prompts.json` — sha `5c3a2f181dd5`, 31 KB. The 50 GSM8K + 50 MATH-500 + 40 MT-Bench
prompts every number in this repository was measured on. `pipeline/e107_clean.py` loads
it and asserts the sha rather than redrawing the pool.

That assert is not decoration. An earlier version re-derived the pool in each script,
and one constant differing between them changed the training-pool dedup filter, which
changed which test prompts survived, which changed the draw — while an assert built from
a hand-typed constant table certified that the sets matched. Loading the file removes
the whole class of error: there is no selection code left to disagree with anything.

## Measured rows

One A5000, fp16, batch 1, temperature 0, EOS-terminated with a 2048-token cap. Each file
is one run of `pipeline/e107_clean.py` against one head.

| file | head | draft params | vocabulary |
|---|---|---|---|
| `e107_ours_8k_rows.csv` | v8k | 3,433,792 (0.085% of Qwen3-4B) | 8,000 entries |
| `e107_ours_40k_rows.csv` | post10k | 11,657,792 (0.290%) | 40,000 entries |

Each file holds, per set: the `HF generate()` baseline row, the `run_tree(None)` harness
row where it was timed, and the arm rows.

| column | |
|---|---|
| `arm`, `kind`, `N` | the arm's name, its layout, and the number of candidates per pass |
| `speedup` | `Σ baseline sec / Σ arm sec` over the same prompts, back to back |
| `tok_pass` | generated tokens per verification pass. 1.000 is an arithmetic floor |
| `ms_tok`, `ms_pass` | wall clock per generated token and per verification pass |
| `rows`, `qper` | mean tree rows and mean query positions fed, per pass |
| `base_ms_tok` | the baseline this row's `speedup` divides by |
| `diverge` | prompts whose output differed from `HF generate()`'s, out of `n` |
| `peak_gb` | peak allocated bytes above the pass's own entry watermark |
| `len_delta` | the arm's total generated length against the baseline's |

**`speedup` and `len_delta` are not independent.** The identity that holds on every row
of both files, to five decimal places, is

```
speedup = (base_ms_tok / arm_ms_tok) / (1 + len_delta)
```

so a `speedup` quoted without its `len_delta` is a speed measurement with a length
measurement folded into it. On MATH-500 `len_delta` reaches +2.0 to +2.9% here, which
moves the speedup by that much with no change in per-token cost at all. The mechanism is
fp16 tie-breaking: once a greedy decode diverges from the baseline's on one tie, the two
continuations end at different lengths, which is also why `diverge` ranks by answer
length across the three sets.

## Reading the table in the top-level README

The headline table rescales both heads onto the `transformers==4.53.1` baseline that
EAGLE-3 is measured against — 38.522 / 39.337 / 39.961 ms/tok — rather than the 5.x
baseline recorded in these files. The CSVs carry the raw `base_ms_tok` from their own
process, so the two numbers are both here and neither is hidden.
