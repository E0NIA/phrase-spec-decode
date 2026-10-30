# Phrase-level speculative decoding on a frozen target

Speculative decoders raise throughput by banking more target tokens per forward pass,
and Medusa, EAGLE-3, DFlash and DSpark all do it the same way: more candidates that
survive verification, through deeper and wider trees, extra draft heads and better draft
accuracy. All of them draft in the **target's own vocabulary**, so one node of the
candidate tree is one target token, and that unit goes unquestioned because the
tokenizer makes it look like a constant.

It is not. BPE's pretokenizer splits on whitespace before merges are learned, so no
multi-word string can ever become a single token. In Qwen3-4B, "United States of
America" is four tokens and no adjacent pair of them exists as one entry — which puts a
phrase out of reach of any method that drafts in the target's vocabulary.

**We change the unit.** A small MLP head reads the target's last hidden state and
predicts a **1–3 word phrase as one entry** in a separate draft vocabulary. Greedy
longest-match expansion maps that entry back into the target's own tokens, and N
candidates are verified in **one** forward pass behind a 4-D prefix-tree mask. The
change is orthogonal to tree depth and width and composes with them. The target model
and its tokenizer stay frozen, and verification is greedy-exact — a drafted token is
accepted only if it equals the target's own argmax — so the served output is
bit-identical to greedy decoding by construction, up to fp16 tie-breaking.

This repository holds the two small heads, at **3.43 M** and **11.66 M** parameters
(0.085% and 0.290% of Qwen3-4B), and the benchmark that measures them against one
shared baseline.

## Result

One A5000, fp16, batch 1, temperature 0, EOS-terminated with a 2048-token cap — DFlash
Table 1's protocol. The same 50 GSM8K + 50 MATH-500 + 40 MT-Bench prompts throughout,
loaded from `results/e89_prompts.json` (sha `5c3a2f181dd5`) rather than redrawn. Both
heads are trained on maths distillation only, so **MT-Bench is zero-shot cross-domain
transfer** and is marked as such.

| set | arm | draft params | N | tok/pass | ms/tok | speedup | vs eagle3-16 |
|---|---|---|---|---|---|---|---|
| **GSM8K** | ours 8k | **3.43 M** | 32 | 2.520 | 16.67 | **2.299×** | 92.1% |
| | ours 40k | 11.66 M | 16 | 2.438 | 16.78 | **2.298×** | 92.1% |
| | eagle3-16 | 218 M | — | 3.296 | 15.30 | 2.496× | — |
| | eagle3-60 | 218 M | — | 3.809 | 13.73 | 2.790× | — |
| **MATH-500** | ours 8k | **3.43 M** | 16 | 2.370 | 17.37 | **2.200×** | 88.3% |
| | ours 40k | 11.66 M | 16 | 2.412 | 17.38 | **2.218×** | **89.0%** |
| | eagle3-16 | 218 M | — | 3.067 | 15.98 | 2.493× | — |
| | eagle3-60 | 218 M | — | 3.525 | 13.97 | 2.757× | — |
| **MT-Bench** *(transfer)* | ours 8k | **3.43 M** | 32 | 1.654 | 24.73 | **1.595×** | 72.1% |
| | ours 40k | 11.66 M | 16 | 1.529 | 27.30 | **1.450×** | 65.5% |
| | eagle3-16 | 218 M | — | 2.839 | 17.90 | 2.214× | — |
| | eagle3-60 | 218 M | — | 3.312 | 15.22 | 2.608× | — |

**The claim: a 3.43 M head reaches 92.1% of EAGLE-3 tree-16's GSM8K speedup at 1.6% of
its draft parameters, and 88.3% on MATH-500.** Not that it beats EAGLE-3 — it does not,
on any set. On MT-Bench it is 28–35% behind, and that is the set neither head was
trained for.

### Both methods draft multiple tokens; they get there differently

EAGLE-3 reaches depth by running its draft model **autoregressively, seven times** per
verification pass, building a branching tree of 60 nodes. We reach depth in **one** head
forward, because a vocabulary entry *already is* several tokens — the expansion is a
table lookup, not a forward. The depth is comparable; the draft cost is not.

| | ours 8k | ours 40k | EAGLE-3 (tree 60) |
|---|---|---|---|
| draft parameters | **3,433,792** (0.085%) | 11,657,792 (0.290%) | 218,245,120 (5.426%) |
| drafting vocabulary | its own: 8,000 entries of 1–3 words | its own: 40,000 | 32,000 of the target's own tokens |
| mean target tokens per entry | 2.498 | 3.045 | 1 |
| draft steps per verification pass | **1** | **1** | **7**, autoregressive |
| draft compute per verification pass | 3.4 M parameter-forwards | 11.7 M | ~1,528 M (**445×** the 8k head) |
| training data | ~3 M distilled maths tokens | ~3 M | ShareGPT + UltraChat |

## What the baseline is

Every speedup above divides by **HF `generate()`** — greedy, same card, same prompts,
same `attn_implementation` — not by our own decode loop. The harness loop is timed
beside it on GSM8K inside the same process, and comes in at **+1.17% with 0 of 50
outputs diverging**, so the two denominators are interchangeable on this card.

One wrinkle is load-bearing. EAGLE-3 pins `transformers==4.53.1`, our harness runs on
5.x, and a single process cannot hold both, so HF `generate()` is measured once per
environment: **38.522 / 39.337 / 39.961** ms/tok under 4.53.1 against **36.460 / 36.920
/ 36.626** under 5.x. The library version alone is worth **5.7–9.1%** of a speedup
number. Table 1 therefore divides every row by the 4.53.1 figure EAGLE-3 is itself
measured against:

```
speedup = base_4.53 / (arm_ms_tok × (1 + len_delta))
```

**`len_delta` is not cosmetic.** A speedup defined as `Σ baseline sec / Σ arm sec` is a
speed measurement multiplied by a length measurement: when the arm's greedy decode
diverges from the baseline's on an fp16 tie, the two continuations end at different
lengths, and that lands directly on the speedup. On MATH-500 it reaches +2.0 to +2.9%
here, and on a longer run it has reached +5%. It is a column in every CSV for that
reason and belongs beside any speedup quoted from them.

The EAGLE-3 column was measured through SafeAILab's own code at a pinned commit, on
these same prompts, and reproduces DFlash's published Table 1 within 1–2% —
`tok/pass` 3.809 and 3.525 at tree-60 against their 3.77 and 3.52, 3.296 and 3.067 at
tree-16 against 3.30 and 3.08. That runner is not in this repository; what is here is
the baseline it shares.

## The two heads against each other

`ms/tok` carries no denominator, so it is the clean head-to-head. Each row divides by HF
`generate()` measured in its own process, which is why the two speedup columns should
not be compared directly and the `ms/tok` column should.

| | ours 8k (3.43 M) | ours 40k (11.66 M) | |
|---|---|---|---|
| GSM8K ms/tok | **16.67** (N=32) | 16.78 (N=16) | +0.7% |
| MATH-500 ms/tok | **17.37** (N=16) | 17.38 (N=16) | +0.1% |
| MT-Bench ms/tok | **24.73** (N=32) | 27.30 (N=16) | **+10.4%** |
| top-1 accuracy | **0.511** | 0.441 | |
| peak verifier memory | 0.26 / 0.44 / 0.72 GB | **0.17** / 1.64 / **0.24** GB | |

**A 3.4× parameter cut costs 0.1–0.7% on maths and wins 10.4% on chat.** Because the
draft vocabulary *is* the parameter budget — one output row of 257 parameters per
entry, which is 59.9% of the 8k head and 88.2% of the 40k — cutting it from 40,000 to
8,000 entries is most of the saving. The small head's higher top-1 matters more on chat
than the large vocabulary's longer expansions, and everyday conversation needs a few
thousand word types rather than forty thousand.

The 40k head reaches GSM8K parity at **N=16 against the 8k head's N=32**, so the larger
vocabulary buys its position on half the candidates.

*Open: the 40k head's 1.64 GB MATH-500 peak against the 8k head's 0.44 is unexplained.
It is a single worst-moment watermark over 50 prompts, and the obvious length
explanation does not hold — that run's generations were shorter on average, not longer.
Do not quote it as a memory cost until it is checked.*

## What is actually in the draft vocabulary

Both vocabularies are built from a distilled corpus of the target's own maths
completions, and both are dominated by **formatting rather than English phrases**. The
twelve most frequent entries are identical in the two heads and are markdown and LaTeX
scaffolding ahead of any content word: `' *'`, `'*'`, `':'`, `' ='`, `'}'`, `' \'`,
`' the'`, `'.'`, `'{'`, `' $'`, `'\n$'`, `'\n\n-'`.

| | ours 8k | ours 40k |
|---|---|---|
| entries | 8,000 | 40,000 |
| 1 / 2 / 3 words | 4,003 / 1,579 / 2,418 | 14,800 / 8,455 / 16,745 |
| multi-word entries | 3,997 | 25,200 |
| of those, pure letters + whitespace | **467 (11.7%)** | **4,101 (16.3%)** |
| containing a LaTeX character | 43.9% | 37.1% |
| containing a digit | 55.0% | 54.3% |
| containing a newline | 24.7% | 22.6% |

The genuine English phrases exist and are exactly chain-of-thought boilerplate —
`'We are given'`, `' to find the'`, `' the number of'`, `' Final Answer'`. The word
counts above use the regex the heads were trained with, in which a word is a run of
alphanumerics plus its leading whitespace, or one standalone punctuation character
— so the 8k head's 4,003 one-word entries are exactly `UNI_FRAC` (0.5) x 8,000, and no
entry exceeds `MAX_GRAM = 3`. Because punctuation counts as a word, a
"3-word" entry is usually a LaTeX fragment.

This is a measured explanation for the MT-Bench column rather than an excuse for it:
cross-domain transfer is weak because a third to a half of the phrase budget is spent on
maths formatting.

**Train/test overlap.** Both heads are trained on completions Qwen3-4B generated from
MetaMathQA, OpenMathInstruct-2 and similar pools, several of which derive from the same
problem sources as MATH-500. MATH-500 is therefore the set most exposed to overlap with
the training distribution, and GSM8K and MT-Bench are the cleaner reads.

## Expansion is an upper bound, not the second factor

Tokens per pass decompose as

```
tok/pass = 1 + landing × (target tokens banked per accepted draft unit)
```

where `landing` is the fraction of passes in which at least one drafted token was
accepted. The mean target tokens per *vocabulary entry* — 2.498 and 3.045 above — is an
**upper bound** on that second factor rather than the factor itself, because the entries
that land are not drawn uniformly. `1 + landing × expansion` therefore overshoots the
measured `tok/pass`, and the measured value is the one to quote.

`tok/pass = 1.000` is an arithmetic floor, not a neutral value: every pass emits at
least the target's own bonus token. The benchmark asserts it on every arm.

## Reproduce

```bash
pip install -r requirements.txt
```

`pipeline/e107_clean.py` runs the whole bench in one process: HF `generate()` as the
shared baseline, the harness loop beside it on GSM8K, then the tree arm at each set's
best N. It is written to paste into a single Kaggle or Colab cell.

Point `HEAD_FILES` at one checkpoint — never two in one run, because row *k* of the head
means a different unit in each — and the script reads its vocabulary and its SLEM
expansion table straight out of the file:

```python
HEAD_FILES = [
    ("v8k", ".../e97v8k_qwen3-4b_head_hard.pt"),       # 3.43 M
    # ("post10k", ".../e95_qwen3-4b_head_hard.pt"),    # 11.66 M
]
```

A checkpoint saved without its expansion table is **refused** rather than rebuilt by
re-tokenizing each entry: the table holds the modal expansion the target actually
emitted, and context-free tokenization disagrees with it on about one entry in six. The
cost of getting it wrong is silent — lower landing, no failed assert, no wrong output.

Writes `e107_rows.csv`, `e107_diverge.csv` and `e107_diag.csv`, and asserts they reached
disk before exiting.

## Layout

```
pipeline/e107_clean.py   the benchmark: shared baseline, drafter, tree verifier, tables
results/e89_prompts.json the shared prompt set, sha 5c3a2f181dd5
results/*.csv            the measured rows behind every table above
```

## Caveats

- One card, one target model. `tok/pass` is hardware-independent; the millisecond
  columns are not.
- Batch 1 only. Phrase drafting at batch > 1 is not measured here.
- Maths distillation only. MT-Bench is zero-shot and reported as such.
- Three knobs — `MAX_GRAM`, `H_DIM` and the labelling rule — were never tuned.
- The training scripts for these two heads are not in this repository yet.

## Licence

MIT. See `LICENSE`.
