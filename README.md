# Phrase-level speculative decoding across tokenizers

A **21.9 M-parameter** draft head reads the target model's hidden state and predicts a
whole **1–3 word phrase** as a single entry in its own vocabulary. The phrase is
expanded to target tokens, N candidates are verified in **one** target forward pass
with a 4-D tree-attention mask, and the longest matching prefix is accepted. Output is
identical to greedy decoding up to fp16 tie-breaking.

The point is the ratio: a draft head **10× smaller** than EAGLE-3's, trained on **~3 M
tokens**, measured against EAGLE-3 on the same prompts, same card, same protocol.

## Result

Qwen3-4B, one Tesla T4, fp16, batch 1, temperature 0, EOS-terminated with a 2048-token
cap — DFlash Table 1's protocol. Same 50 GSM8K + 50 MATH-500 + 40 MT-Bench prompts on
both sides, loaded from the same file (`results/e89_prompts.json`, sha `5c3a2f181dd5`).

| set | ours | speedup | tok/pass | ms/tok | EAGLE-3 | speedup | tok/pass | ms/tok | paired Δ ms/tok (95% CI) |
|---|---|---|---|---|---|---|---|---|---|
| **MATH-500** | N16 | 2.037 | 2.620 | 31.3 | tree 16 | 2.061 | 3.035 | 27.9 | **+0.66 [−0.21, +1.58] — tie** |
| **GSM8K** | N32 | 2.211 | 2.742 | 28.8 | tree 60 | 2.451 | 3.811 | 24.3 | +4.44 [+3.77, +5.20] |
| **MT-Bench** | N16 | 1.402 | 1.672 | 45.4 | tree 60 | 2.005 | 3.324 | 29.0 | +14.03 [+11.02, +16.69] |

| | ours | EAGLE-3 (tree 60) |
|---|---|---|
| draft parameters | **21,937,792** (0.545% of target) | 218,245,120 (5.426%) |
| training data | ~3 M distilled maths tokens | ShareGPT + UltraChat |
| drafting vocabulary | its own: 80,000 entries of 1–3 words | 32,000 of the target's own tokens |
| tokens per draft step | **~2.46** (one phrase, expanded by lookup) | 1 |
| draft steps per verification pass | **1** | **7**, autoregressive |
| candidate structure | N chains merged into a prefix tree, ~2.5 deep | depth 7, top-10 beams, 60 nodes |
| draft compute per verification pass | 21.9 M parameter-forwards | ~1,528 M (**70×** ours) |

**The claim: level with EAGLE-3 on MATH-500 and within 18% on GSM8K, with a draft head
10× smaller and three of its knobs never tuned.** Not that it beats EAGLE-3.

### Both methods draft multiple tokens; they get there differently

EAGLE-3 reaches depth by running its draft model **autoregressively, seven times**
per verification pass, building a branching tree of 60 nodes. We reach depth in **one**
head forward, because a vocabulary entry *already is* several tokens — the expansion is
a table lookup, not a forward. So the depth is comparable and the draft cost is not.

Decomposing the measured GSM8K numbers per verification pass:

| | baseline ms/pass | arm ms/pass | drafting overhead | tok/pass |
|---|---|---|---|---|
| ours, N32 | 63.7 | **79.0** | **+24%** | 2.742 |
| EAGLE-3, tree 60 | 59.6 | **92.7** | **+56%** | 3.811 |

**Our pass is 15% cheaper and we still lose, because we bank 28% fewer tokens in it.**
That is the whole result in one line: the verifier and the draft cost are not the
problem — prediction quality is. It is also why the untuned knobs matter so much, since
all three of them act on exactly that.

MT-Bench is the honest bound on generality — the head has only ever seen maths, and
chat is out of distribution. It is reported for that reason.

### Why the paired column is the one to read

A speedup is a ratio to its own baseline, and the two runs use different baselines:
ours is our verifier with drafting switched off, EAGLE's is `naivegenerate` from its
own repository. Ours measures **63.7 ms/token** against EAGLE's **59.6**, consistently
across all three sets — that is our Python, not the model, and it **inflates our
speedup column by ~7%**. Milliseconds per generated token contains no baseline, so the
paired test compares the two methods directly. Pairing is by `(set, prompt index)`, and
baseline token counts agree **50/50 and 40/40 exactly** on GSM8K and MT-Bench
(47/50 on MATH-500, mean difference 8.7 tokens — fp16 tie-breaking between two
different greedy loops).

### Exactness

98 divergences from the baseline across **420 arm runs** (3 arms x 140 prompts) —
**0 of them confident**: worst logit margin 0.0625 against a 0.5 fp16 threshold.

Per token position that is **0.57 divergences per 1,000 tokens** over 170,490 positions.
For scale, **EAGLE-3 through its own official code scores 0.57 per 1,000 on these same
prompts** (64 of 280 runs) — the same rate, so this is what fp16 greedy decoding does
when batch shape changes, not something either method introduces. Ours are classified
as tie-breaks by a margin check; E89 did not classify EAGLE's, so that comparison is of
rates and not of confirmed-benign rates.

## Reproducing it

No weights are shipped. Three stages, in order — the numbering is the dependency.

```bash
pip install -r requirements.txt

python pipeline/1_train_head.py      # distils, builds the vocabulary, trains the head
python pipeline/2_bench_eagle3.py    # EAGLE-3, and writes the shared prompt file
python pipeline/3_bench_ours.py      # our head on those exact prompts
python pipeline/4_leakage_check.py   # optional: 8-gram contamination check, CPU only
```

Each script is the one that produced the numbers above, unchanged. Paths sit in a
config block at the top of each; on anything other than Kaggle you will want to edit
`DISTILL_IN`, `HEADS_IN`, `PROMPTS_IN` and `OUT`.

`2_bench_eagle3.py` pins `transformers==4.53.1`, clones
[SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) at commit `cb7e0841fe0c`, and
applies one patch: Qwen3-4B's attention is wider than its hidden size (32 × 128 = 4096
against 2560) and the official draft code derives head width from hidden size alone, so
**it cannot load the 4B head at all** without honouring `config.head_dim`. EAGLE is
cloned rather than vendored, so no code of theirs lives in this repository.

Budget on one T4: roughly 3 h for stage 1, 2 h for stage 2, 2.5 h for stage 3.

### EAGLE-3 is run, not reimplemented

An earlier version reimplemented the EAGLE-3 draft head from its released checkpoint.
It was architecturally correct — the parameter count matches 218,245,120 exactly — and
it still reached only 1.136 tok/pass where the published acceptance length is ~3.4,
with all four candidate layer triples inside 1.6% of each other. Running the authors'
own code instead **reproduces DFlash's Table 1 to two decimals** (3.811 and 3.497
tok/pass against published 3.77 and 3.52). That agreement is the evidence that the
baseline here is the real one.

## Notebook

`notebook/headtohead.ipynb` runs both arms and renders the comparison. It needs a
trained head from stage 1; everything after that is in the notebook.

## Layout

```
pipeline/   the four stages, as run
results/    per-prompt records, the prompt set, and the aggregate tables
notebook/   both arms plus the comparison
docs/       method.md (how it works) · protocol.md (how it was measured)
```

## Caveats, in the table rather than a footnote

1. **T4 only.** The decode loop is CPU-dispatch-bound — the target forward is 92.6% of
   a pass and essentially all of that is the CPU issuing work — so these numbers do not
   transfer to a faster card until that is fixed. The same pipeline on an A5000 measured
   1.061× with *better* acceptance (tok/pass 2.695), because a faster card shrinks the
   GPU work and exposes the Python a T4 was slow enough to hide. `tok/pass` is
   hardware-independent; the speedups are not.
2. **Undertrained and untuned, so these are floor numbers.** Units cap at 3 words,
   which caps `tok/pass` at **3.46** however good the head gets — EAGLE already exceeds
   that at 3.811. The vocabulary sits at an arbitrary 80,000 cap, not an optimum (the
   parameter budget would allow 150,282). The data ladder has not bent across a 6.5×
   increase. None of the three has been explored.
3. **MATH-500 contamination was checked and is not material.** Max 8-gram overlap with
   the training pool is 0.833 on one prompt against GSM8K's 0.083, with no exact
   duplicates. Dropping every prompt sharing even one 8-gram (8 of 50) moves EAGLE
   2.032 → 2.009. `pipeline/4_leakage_check.py` is that check.

## Licence

Apache-2.0. Qwen3-4B, `AngelSlim/Qwen3-4B_eagle3`, GSM8K, MATH-500 and MT-Bench are
each under their own licences and none of them is redistributed here.
