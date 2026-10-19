# =============================================================================
# E87. THE HEAD WE WILL COMPARE AGAINST EAGLE-3: 3.5 M rows, 80k vocabulary,
#      both T4s, and three blockers fixed. Kaggle, 2 x T4 + 30 GB RAM.
#
# THIS RUN TRAINS AND CACHES. It does not do the head-to-head -- that is E88, and
# splitting them means a bench failure can never cost the training.
#
# WHY THESE NUMBERS
#   E86 reached recall@8 0.694 / top-1 0.364 on 1,763,372 rows and its ladder was
#   STILL unbent (+0.057, +0.058, +0.052 per doubling, last/first 0.91). So rows
#   remain the binding constraint. Two facts set the budget:
#     * 43,550 aligned completions x ~161 word boundaries = 7.0 M CANDIDATE rows
#       already sit in the existing distillation cache. Nothing new is distilled.
#     * 30 GB of CPU RAM holds 3.5 M rows (18.0 GB) with 12 GB left over.
#   So 3.5 M rows, which is 2x E86 and half of what the corpus could give.
#
#   Vocabulary 80,000 -> head 21.9 M = 0.545% of target, still 9.9x smaller than
#   EAGLE-3's verified 218,245,120. NOT larger, because coverage is already 0.998
#   and tok/pass sits at 65% of its 1+expansion ceiling -- so LANDING is the limit,
#   landing is driven by top-1/recall, and recall is driven by ROWS. More entries
#   would buy longer units we cannot land yet; more rows buy landing. 80k keeps
#   39 positions per entry at 3.5 M rows, comfortably above E86's 29.4.
#
# THREE BLOCKERS FIXED, EACH OF WHICH COST A RUN
#   1. MULTI-GPU FEATURES. E86 on 2 x T4 extracted 1,240,415 + 1,240,429 rows,
#      kept the index for BOTH, then assembled FLAT from `SEQ[0]` only -- so the
#      index pointed at 2.48 M rows in an array of 1.24 M (IndexError). The guard
#      only fired on the memmap path and RAM auto-detect chose in-RAM. The fix is
#      not concatenation -- that would peak at 2x18 GB against 30 GB of RAM -- but
#      ONE PREALLOCATED ARRAY with each device writing into its own disjoint view.
#      No concatenate, no doubling, and the row offsets are correct by
#      construction. Using both T4s halves feature extraction: ~92 min, not 184.
#   2. THE FEATURE CACHE WAS KEYED ON THE VOCABULARY. A hidden state depends on
#      (completion, position) and NOTHING ELSE; the vocabulary only decides the
#      LABEL. But the cache key included `nvocab`, so changing 60k -> 80k entries
#      forced a 92-minute re-extraction for what is a pure-CPU relabel. Now the
#      cache stores (flat_row, sid, ti, q, **wi**) -- the WORD INDEX, not the
#      label -- and a `relabel()` step turns wi into y against whatever vocabulary
#      is current. Every future vocabulary experiment is now free, which is what
#      makes item 48 (BPE merges) testable at all.
#   3. `output_hidden_states=True` materialised all 37 hidden states (2.22 GB per
#      batch at FEAT_BATCH=32) to use exactly one. The time cost was negligible --
#      7 ms of write traffic -- but the VRAM was not, and it is what caps
#      FEAT_BATCH. `mm.model(...).last_hidden_state` returns only what we read.
#
# WHAT PHASE 2 ACTUALLY COSTS, so the 92 minutes is not a surprise
#   One prefill per batch over prompt+completion. 43,550 completions x ~334 tokens
#   = 14.5 M tokens through a 4.02 B model = ~128 PFLOP. On a T4 at ~20 TFLOPS
#   effective that is ~105 min on one card. It is genuinely compute-bound; there is
#   no hidden waste beyond ~10% padding (feats does not length-sort, unlike
#   distill) and the FEAT_BATCH that fix 3 unlocks.
#
# THE TEST PROTOCOL, settled here so E88 inherits it
#   EOS-terminated with a 2048 cap, temperature 0 -- which is DFlash's Table 1
#   protocol exactly, so the numbers drop into their table without adjustment.
#   NAT_N is raised to 50 NOW so that E87's bench prompts and E88's are the same
#   50 GSM8K + 50 MATH-500 questions; `rgp.permutation(...)[:NEED]` means changing
#   it later would silently reshuffle the slice. E87 benches 12 of the 50 (cheap,
#   enough to pick the rung and N); E88 runs all 50 against EAGLE-3.
#   FORCED-2048 is kept as a diagnostic only and may never be a headline: EOS falls
#   at ~233 tokens on GSM8K, so ~89% of a forced run is post-EOS filler.
#
# PREDICTION, ON THE RECORD
#   recall@8 0.73-0.76, top-1 0.40-0.43 at 3.5 M rows, with the per-doubling gain
#   finally shrinking below 0.8x of the first step -- the curve bends here without
#   flattening. GSM8K 2.1-2.3x at the best N, MATH-500 1.9-2.1x. If recall@8 lands
#   below 0.72 then 80k entries was too many for 3.5 M rows and the vocabulary,
#   not the rows, was binding.
# =============================================================================

MODEL = "Qwen/Qwen3-4B"

# E82 measured this: same mask, same result, but the score tensor is never written
# to HBM. Asserted below, because a silent fallback to eager would quietly restore
# the depth decay this run is built on top of.
ATTN = "sdpa"
MERGE_LMHEAD = True

# ---- the vocabulary, built from the corpus, capped by a parameter budget
PARAM_BUDGET = 20_000_000      # the CEILING on the head, as asked
MAX_VOCAB    = 80_000          # the cap on vocabulary entries, as asked. This is
                               # what BINDS: head = 1,377,280 + V*(H_DIM+1), so
                               # V=40,000 -> 11.657 M, against 72,461 allowed by
                               # the 20 M ceiling alone. E78's 13.4 M head (the
                               # one that reached 2.259x on MATH-500) was V=46,781,
                               # so raising MAX_VOCAB to ~46,800 reproduces it.
H_DIM        = 256             # E57: H=2048 ties H=256
MIN_WORD_COUNT = 1
# At 10,000 completions, unigram TYPES crowd out phrases under a fixed cap because
# E81's selection gave `uni + numerics` unconditional priority. Phrases are the
# mechanism (findings 8.19), so the unigram slice is bounded and both the cap and
# the resulting coverage are reported -- a coverage regression must be visible.
UNI_SHARE   = 0.45             # at most this fraction of MAX_VOCAB on unigrams
ADD_NUMERIC = 1000             # math is full of integers that each occur once
WORD_RE = r"\s*(?:[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9])"
MAX_GRAM = 3                   # words per unit, AS ASKED (E83 used 5)
MIN_GRAM_COUNT = 5
MAX_UNIT_TOK = 16
N_UNI = 4                      # single-word reserve inside the same N

# ---- data. GSM8K's TEST split is the test set; nothing trains on it.
TEST_SETS = [
    ("gsm8k-test", "openai/gsm8k", "main", "test", "question"),
    ("math500",    "HuggingFaceH4/MATH-500", None, "test", "problem"),
]
# `gsm8k/socratic` is DELIBERATELY ABSENT: it is the same questions as `main` with
# differently styled answers, so training on it would put the test questions into
# the training set. Two extra sources beyond E81's three, because 3 x PER_SOURCE
# barely clears DISTILL_N=10,000 after dedup and E80 lost a run to exactly that
# shortfall.
TRAIN_SOURCES = [
    ("gsm8k-train",  "openai/gsm8k", "main", "train", "question"),
    ("metamath",     "meta-math/MetaMathQA", None, "train", "query"),
    ("orca-math",    "microsoft/orca-math-word-problems-200k", None, "train", "question"),
    ("openmath2",    "nvidia/OpenMathInstruct-2", None, "train", "problem"),
    ("mathinstruct", "TIGER-Lab/MathInstruct", None, "train", "instruction"),
]
PER_SOURCE   = 4000       # per dataset; 5 sources -> ~20,000 candidates for a
                          # 10,000 target, which survives dedup with room spare
# 12,000 saturates a 1.2 M row budget. Raise it and run the script again as many
# times as you like: the cache tops up and saves every DISTILL_SAVE_EVERY
# completions, so a Colab disconnect costs at most that much work.
# The cache holds 49,997 and that is enough: 43,550 of them align and offer 7.0 M
# candidate rows against a 3.5 M budget. Set to what the cache has so PHASE 1 is a
# no-op -- there is nothing to gain from 3 more completions and 175 min to lose.
DISTILL_N    = 49997
DISTILL_NEW  = 256        # 70.8% of the 12,000-completion cache hit the old
                          # 192-token cap EXACTLY, so most training completions
                          # were truncated mid-solution and the head never saw an
                          # answer end. The new truncation rate is measured below.
GEN_BATCH    = 128        # 48 was a T4 number. At 128 the KV cache is
                          # 128 x 366 x 147,456 B = 6.9 GB, so 15 GB of 24 with
                          # the model -- and gen_chunk halves on OOM regardless.
DISTILL_SAVE_EVERY = 2000
# ATT_CELLS caps B x L^2 because EAGER prefill materialises a
# B x heads x L x L score tensor (E72's OOM). Under sdpa nothing is materialised,
# so the old 6 M was throttling a constraint that no longer exists.
ATT_CELLS    = 40_000_000
MAX_PROMPT_TOK = 384
FEAT_BATCH   = 32         # completions per forward in `feats`; was hardcoded at 8
                          # for a 15 GB card

# ---- the bench. Batch 1 only. N=32 only: E78 measured 32 as the optimum for
#      this target and math (64 collapsed 2.26 -> 1.61), and a second N would
#      cost another 2048-token pass per prompt, which is the whole budget.
# EAGLE-3 gets to pick its tree size (16 AND 60 are both published), so we report
# our whole curve. E86 benched only N=8, which turned out to be the best of the
# three by feed efficiency but was never compared at natural length.
N_SCAN      = [8, 16, 32]
# HELD AT E81/E82's VALUES. The test prompts are chosen by
# `rgp.permutation(len(got))[:NEED]` with NEED = max(NAT_N, LONG_N, FORCED_N), and
# the leakage filter removed nothing in either run (8-gram overlap 0.000), so
# keeping NEED at 12 keeps E83's test prompts IDENTICAL to E82's and the comparison
# paired. Changing any of these three silently reshuffles the slice.
# RAISED TO 50 NOW, not in E88. Test prompts come from
# `rgp.permutation(len(got))[:NEED]` with NEED = max(NAT_N, LONG_N, FORCED_N), so
# raising it later would reshuffle the slice and E87's and E88's prompt sets would
# differ. E87 benches LADDER_BENCH_PROMPTS of these 50; E88 runs all 50.
NAT_N       = 50
LONG_N      = 10
FORCED_N    = 8
FORCED_USE  = 6           # GSM8K, EOS ignored, exactly 2048 -- the depth curve
NAT_USE     = 4           # MATH-500, EOS honoured; E82 saw mean 978 tok, cap binding
MAX_NEW_NAT = 2048        # DFlash's cap
MAX_NEW_LONG = 2048
MAX_NEW_FORCED = 2048
BUCKET      = 256         # depth-bucket width for the within-sequence curve
LONG_SUFFIX = (
    "\n\nWork through this completely. State every step, justify each algebraic"
    " manipulation, carry the units, then verify the answer by substituting it"
    " back and finally restate the result.")

MAX_FEED_PER_SEQ = 512
ARGMAX_BUDGET_GB = 0.50   # 0.10 was sized for a T4; this only sets the chunk
                          # width of the lm_head matmul
BOOT     = 4000

MAX_EPOCHS  = 60
PATIENCE    = 6
BATCH       = 1024
LR          = 2e-3
WARMUP_FRAC = 0.03
# Features now live in a disk-backed memmap, so the row budget is DISK, not RAM:
# 4 M rows x 2560 x fp16 = 20.5 GB on disk and ~0 of the 12.7 GB of RAM. E85 was
# capped at 1.2 M purely because FLAT sat in memory.
# 3.5 M rows x 2560 x fp16 = 18.0 GB of Kaggle's 30 GB, leaving 12 GB for the
# python objects, the index and the torch allocator.
MAX_TRAIN_POS = 3_500_000
FEAT_MAX_POS  = 3_500_000
FLAT_ON_GPU   = False     # 5 M rows is 25.6 GB; it would not fit 24 GB of VRAM
                          # beside the 8 GB model, and VRAM is better spent on
                          # distillation batch
# FEAT_MEMMAP is decided BY MEASUREMENT below, not asserted here: in-RAM gives a
# true global shuffle and no disk traffic, the memmap gives unlimited rows. On
# 48 GB RAM the features fit, on 12.7 GB they do not.
FEAT_MEMMAP   = None      # None = auto-detect; True/False to force
# Random access into a 20 GB memmap page-faults per row and is far slower than the
# GPU step, so minibatches come from a CHUNKED SHUFFLED sampler: read a contiguous
# block into RAM, shuffle inside it, train, move on. Chunk order and a random start
# offset are redrawn every epoch so block boundaries do not become fixed.
CHUNK_ROWS    = 262_144   # 1.34 GB per buffer at d_model 2560
# ---- the ladder. Two extra rungs below E85's so that 25% of this run lands near
#      E85's 100% (1.08 M rows) and the two curves share an overlap point.
DATA_FRACS = [0.125, 0.25, 0.5, 1.0]
LADDER_BENCH_PROMPTS = 12  # per head, EOS-terminated -- enough to rank the rungs
DEEP_BENCH_PROMPTS = 3     # forced 2048, best rung only, DIAGNOSTIC not headline

# ---- budget. Colab/Kaggle T4 sessions die; every phase checks the clock and the
#      CSVs are rewritten after each test set, so a timeout leaves whatever
#      finished intact rather than losing the session.
T_BUDGET      = 240      # minutes, total
T_BUDGET_TRAIN = 190     # if training has not finished by here, stop early and
                         # bench with the best checkpoint so far. Both artifacts
                         # are already on disk by then, so a timeout during the
                         # bench costs only the bench.
# The cache-drift control re-prefills for EVERY token, so it is O(L^2): 640 tokens
# cost ~8 min. 384 still clears E79's divergence depth of 251, which is all it has
# to reach.
FLOOR_N   = 3
FLOOR_NEW = 384

# Default to this run's OWN artifacts, so re-running the identical script after a
# timeout picks them up and goes straight to the bench. Both are saved the moment
# they exist, not at the end.
# The distillation cache is reusable across runs and is the expensive artifact.
# HEADS_IN is deliberately None: E85 trains the ladder, that is the point.
# Point this at E85's cache to INHERIT its 12,000 completions -- the script tops
# up from whatever is there rather than starting over. It writes to its own name,
# so E85's artifact is never overwritten.
# Point this at the Kaggle dataset that holds E86's cache. It is read-only there,
# so the run writes its own copy under OUT.
DISTILL_IN = "/kaggle/input/e86-distill/e86_distill_math.pt"
DISTILL_OUT = "e87_distill_math.pt"
HEADS_IN   = None

# ---- references, all measured by us
E78_MEASURED  = {"math500": 2.259, "gsm8k-test": 2.070}   # MAX_NEW=256
E79_GSM8K_512 = 2.145                                     # MAX_NEW=512
E75C_GSM8K    = 2.172
DFLASH_T1 = {"gsm8k-test": dict(e3_16=1.99, e3_60=2.27, dflash=5.15),
             "math500":    dict(e3_16=1.83, e3_60=2.10, dflash=6.09)}
EAGLE3_PARAMS = 218_245_120      # AngelSlim/Qwen3-4B_eagle3, from its manifest

import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import re, sys, time, subprocess, traceback, threading, unicodedata, itertools, math, copy
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "transformers>=4.50", "datasets", "accelerate", "sentencepiece"],
               check=False)
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, DynamicCache
from datasets import load_dataset

OUT = (Path("/kaggle/working/e87") if Path("/kaggle/working").exists()
       else Path("/content/e87") if Path("/content").exists() else Path("e87"))
# d_model is only known once the config loads, but the disk check has to run
# before anything expensive. 2560 is Qwen3-4B's; the guard is approximate on
# purpose and re-reported exactly once the real value is known.
D_MODEL_GUESS = 2560
OUT.mkdir(parents=True, exist_ok=True)
DEVS = ["cuda:%d" % i for i in range(torch.cuda.device_count())] or ["cpu"]
DEV0 = DEVS[0]
DT = torch.float16
NEG = torch.finfo(DT).min
MARGIN_FAIL = 0.5 if DT == torch.float16 else 1.5
torch.manual_seed(0); np.random.seed(0)
T0, LK = time.time(), threading.Lock()
# ROWSX holds the arm rows, PAIRS the per-prompt pairs and MARKS the per-depth
# speedups. E81 and E82 both flushed `ROWS` -- which nothing ever appends to -- so
# the arms table and the HEADLINE depth table were never written to disk at all and
# survived only in stdout. Fixed here.
ROWSX, PAIRS, MARKS, CURVE, DIAG, TRACE, DIVERGE = [], [], [], [], [], [], []
LADDER = []
CURTAG = ["none"]        # the arm whose passes TRACE is currently recording
clk = time.perf_counter


def log(*a):
    with LK:
        print("[%6.1fm]" % ((time.time() - T0) / 60), *a, flush=True)


def mins():
    return (time.time() - T0) / 60


def flush():
    with LK:
        for rows, nm in ((ROWSX, "e87_arms.csv"), (PAIRS, "e87_pairs.csv"),
                         (MARKS, "e87_depth_speedup.csv"),
                         (CURVE, "e87_curve.csv"), (DIAG, "e87_diag.csv"),
                         (TRACE, "e87_depth_tokpass.csv"),
                         (DIVERGE, "e87_diverge.csv"),
                         (LADDER, "e87_ladder.csv")):
            if rows:
                pd.DataFrame(rows).to_csv(OUT / nm, index=False)


# ===================================================== masks (verified E68-E70)
def tree_mask_ref(c0, trunk_len, cand_lens, dev):
    q = trunk_len + sum(cand_lens)
    kv = c0 + q
    m = torch.full((q, kv), NEG, dtype=DT)
    m[:, :c0] = 0.0
    for i in range(trunk_len):
        m[i, c0:c0 + i + 1] = 0.0
    pos = list(range(c0, c0 + trunk_len))
    off = trunk_len
    for L in cand_lens:
        for j in range(L):
            r = off + j
            m[r, c0:c0 + trunk_len] = 0.0
            m[r, c0 + off:c0 + off + j + 1] = 0.0
            pos.append(c0 + trunk_len + j)
        off += L
    return m.view(1, 1, q, kv).to(dev), torch.tensor([pos], device=dev)


def tree_mask_batch(kv_valid, trunk_lens, cand_lens, pos0, dev):
    B, C = kv_valid.shape
    assert len(cand_lens) == B
    assert all(isinstance(x, int) for cl in cand_lens for x in cl), (
        "cand_lens must hold BLOCK LENGTHS, not the blocks")
    Tmax = int(trunk_lens.max().item())
    tot = [sum(cl) for cl in cand_lens]
    Cmax = max(tot) if tot else 0
    q, kv = Tmax + Cmax, C + Tmax + Cmax
    cb = np.full((B, Cmax), -1, dtype=np.int64)
    ci = np.zeros((B, Cmax), dtype=np.int64)
    starts = []
    for b in range(B):
        o, st = 0, []
        for j, L in enumerate(cand_lens[b]):
            st.append(o)
            cb[b, o:o + L] = j
            ci[b, o:o + L] = np.arange(L)
            o += L
        starts.append(st)
    cbt, cit = torch.from_numpy(cb).to(dev), torch.from_numpy(ci).to(dev)
    tl = trunk_lens.to(dev).long()
    tslot = torch.arange(Tmax, device=dev)
    treal = tslot.unsqueeze(0) >= (Tmax - tl).unsqueeze(1)
    tidx = tslot.unsqueeze(0) - (Tmax - tl).unsqueeze(1)
    row_is_tr = torch.cat([torch.ones(Tmax, dtype=torch.bool, device=dev),
                           torch.zeros(Cmax, dtype=torch.bool, device=dev)])
    row_blk = torch.cat([torch.full((B, Tmax), -1, device=dev, dtype=torch.long), cbt], 1)
    row_idx = torch.cat([tidx, cit], 1)
    row_ok = torch.cat([treal, cbt >= 0], 1)
    col_kind = torch.cat([torch.zeros(C, dtype=torch.long, device=dev),
                          torch.ones(Tmax, dtype=torch.long, device=dev),
                          torch.full((Cmax,), 2, device=dev, dtype=torch.long)])
    col_blk = torch.cat([torch.full((B, C + Tmax), -1, device=dev, dtype=torch.long), cbt], 1)
    col_idx = torch.cat([torch.zeros((B, C), device=dev, dtype=torch.long), tidx, cit], 1)
    col_ok = torch.cat([kv_valid.to(dev), treal, cbt >= 0], 1)
    K0 = col_kind.unsqueeze(0).unsqueeze(1)
    RT = row_is_tr.view(1, q, 1)
    vis = (K0 == 0).expand(B, q, kv) & col_ok.unsqueeze(1)
    vis |= ((K0 == 1) & col_ok.unsqueeze(1)
            & (~RT | (col_idx.unsqueeze(1) <= row_idx.unsqueeze(2))))
    vis |= ((K0 == 2) & col_ok.unsqueeze(1)
            & (col_blk.unsqueeze(1) == row_blk.unsqueeze(2))
            & (col_idx.unsqueeze(1) <= row_idx.unsqueeze(2)))
    vis &= row_ok.unsqueeze(2)
    dead = ~vis.any(-1)
    if bool(dead.any()):
        own = (C + torch.arange(q, device=dev)).view(1, q, 1).expand(B, q, 1)
        vis = vis.scatter(2, own, vis.gather(2, own) | dead.unsqueeze(2))
    m = torch.where(vis, torch.zeros((), dtype=DT, device=dev),
                    torch.full((), NEG, dtype=DT, device=dev)).unsqueeze(1)
    pos = torch.cat([pos0.to(dev).unsqueeze(1) + tidx.clamp(min=0),
                     pos0.to(dev).unsqueeze(1) + tl.unsqueeze(1) + cit], 1)
    return m, pos.clamp(min=0), starts, Tmax, Cmax, row_ok


# Decide where the features live BEFORE anything expensive runs. RAM is the fast
# path (true global shuffle, no disk traffic); the memmap is the unlimited one. The
# choice is measured, because the same script has to run on a 12.7 GB Colab box
# and a 48 GB workstation without being edited.
_need_gb = FEAT_MAX_POS * D_MODEL_GUESS * 2 / 1e9
try:
    import psutil
    _ram_gb = psutil.virtual_memory().total / 1e9
    _ram_av = psutil.virtual_memory().available / 1e9
except Exception:
    _ram_gb = _ram_av = float("nan")
if FEAT_MEMMAP is None:
    # leave half the available RAM for everything else: the python objects, the
    # index array, the validation slice and the torch allocator
    FEAT_MEMMAP = not (_ram_av == _ram_av and _ram_av > _need_gb * 2.0)
print("[  0.0m] features need ~%.1f GB | RAM %.0f GB total, %.0f GB available"
      " -> %s" % (_need_gb, _ram_gb, _ram_av,
                  "MEMMAP on disk" if FEAT_MEMMAP else "IN RAM (global shuffle)"),
      flush=True)
if FEAT_MEMMAP:
    import shutil
    _free_gb = shutil.disk_usage(str(OUT)).free / 1e9
    print("[  0.0m] memmap needs ~%.1f GB of disk; %.1f GB free on %s"
          % (_need_gb, _free_gb, OUT), flush=True)
    assert _free_gb > _need_gb * 1.15, (
        "only %.1f GB free but the feature memmap needs ~%.1f GB. Free disk space"
        " or lower FEAT_MAX_POS (and MAX_TRAIN_POS) -- 2e6 rows needs ~%.1f GB."
        " Checked now rather than mid-extraction, because np.memmap allocates"
        " lazily and the failure would arrive an hour in."
        % (_free_gb, _need_gb, 2e6 * D_MODEL_GUESS * 2 / 1e9))
else:
    assert _ram_av > _need_gb * 1.6, (
        "features need ~%.1f GB in RAM but only %.1f GB is available. Set"
        " FEAT_MEMMAP = True to put them on disk, or lower FEAT_MAX_POS."
        % (_need_gb, _ram_av))

log("loading %s on %s" % (MODEL, DEVS))
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
_c = AutoConfig.from_pretrained(MODEL)
_tc = getattr(_c, "text_config", _c)
D_MODEL = getattr(_tc, "hidden_size", None) or _c.hidden_size


def load_on(dev):
    try:
        return AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=DT, device_map={"": dev}, attn_implementation=ATTN).eval()
    except Exception:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(
            MODEL, dtype=DT, device_map={"": dev}, attn_implementation=ATTN).eval()


MODELS = {d: load_on(d) for d in DEVS}
_got = str(getattr(MODELS[DEV0].config, "_attn_implementation", None))
assert _got == ATTN, (
    "ABORT -- asked for attn_implementation=%r but transformers is using %r. E82"
    " measured the difference this makes: under eager the arm's cost per pass grows"
    " 1.21x -> 2.05x with context and the deepest bucket falls to 0.997x, against"
    " 1.22x -> 1.58x and 1.386x under sdpa. Running on the fallback would reproduce"
    " the decay this run is built on top of, and nothing would fail." % (ATTN, _got))
log("attention implementation %r confirmed (E82: sdpa takes the same 4D tree mask"
    " but never writes the score tensor to HBM)" % _got)
EMBS = {d: MODELS[d].get_input_embeddings().weight for d in DEVS}
LMW = {d: MODELS[d].get_output_embeddings().weight for d in DEVS}
TGT = sum(p.numel() for p in MODELS[DEV0].parameters())
PADID = tok.pad_token_id
V_OUT = LMW[DEV0].shape[0]
ARGMAX_CHUNK = max(256, int(ARGMAX_BUDGET_GB * 1e9 // (V_OUT * 2)))
EOS = {tok.eos_token_id}
for s in ("<end_of_turn>", "<eos>", "<|im_end|>"):
    i = tok.convert_tokens_to_ids(s)
    if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
        EOS.add(i)
EOS.discard(None)
log("target %.2f B | d_model %d | vocab %d | EOS ids %s"
    % (TGT / 1e9, D_MODEL, V_OUT, sorted(EOS)))
log("memory at 2048 generated tokens, batch 1: weights %.2f GB + a KV cache that"
    " grows to\n  roughly %.0f MB -- the 2048 regime is not a memory question on"
    " this card, it is a TIME one"
    % (TGT * 2 / 1e9,
       (getattr(_tc, "num_hidden_layers", 0)
        * max(getattr(_tc, "num_key_value_heads", 1), 1)
        * getattr(_tc, "head_dim", 128) * 2 * 2 * 2400) / 1e6))
_HP = {}


def fwd_hidden(mm, dev, ids, mask, pos, cache):
    if _HP.get("m") is None:
        try:
            o = mm.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                         past_key_values=cache, use_cache=True)
            _HP["m"] = "base"
            return o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]
        except Exception:
            _HP["m"] = "full"
    if _HP["m"] == "base":
        o = mm.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                     past_key_values=cache, use_cache=True)
        return o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]
    for kw in ("logits_to_keep", "num_logits_to_keep"):
        try:
            o = mm(input_ids=ids, attention_mask=mask, position_ids=pos,
                   past_key_values=cache, use_cache=True, output_hidden_states=True,
                   **{kw: 1})
            return o.hidden_states[-1]
        except TypeError:
            continue
    return mm(input_ids=ids, attention_mask=mask, position_ids=pos,
              past_key_values=cache, use_cache=True,
              output_hidden_states=True).hidden_states[-1]


def argmax_at(hs, dev, flat_idx):
    if flat_idx.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=hs.device)
    W = LMW[dev]
    f = hs.reshape(-1, hs.shape[-1])
    out = torch.empty(flat_idx.numel(), dtype=torch.long, device=hs.device)
    ch, s0 = ARGMAX_CHUNK, 0
    while s0 < flat_idx.numel():
        idx = flat_idx[s0:s0 + ch]
        try:
            out[s0:s0 + idx.numel()] = (f[idx].to(W.dtype) @ W.t()).argmax(-1)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if ch <= 128:
                raise
            ch = max(128, ch // 2); continue
        s0 += idx.numel()
    return out


def shrink_cache(cache, keep):
    if hasattr(cache, "layers") and cache.layers:
        for l in cache.layers:
            if getattr(l, "keys", None) is not None:
                l.keys = l.keys[keep].contiguous()
                l.values = l.values[keep].contiguous()
        return True
    if hasattr(cache, "key_cache") and len(getattr(cache, "key_cache", [])):
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i][keep].contiguous()
            cache.value_cache[i] = cache.value_cache[i][keep].contiguous()
        return True
    return False


def Enc(s):
    return tok(s, add_special_tokens=False)["input_ids"]


def chat(s):
    m = [{"role": "user", "content": s}]
    try:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)


def cut_eos(g):
    """Truncate a generation at its first EOS, inclusive.

    The baseline emits one token per pass and so stops exactly AT its EOS. An arm
    can accept a whole block whose interior contains EOS and thereby overshoot,
    which is not a lost token but is extra work. Comparing raw lists would then
    report a length divergence on a sequence that agreed everywhere it mattered.
    Both sides are cut here before comparison, and the arm keeps its measured wall
    time including the overshoot -- the conservative direction."""
    for i, t in enumerate(g):
        if t in EOS:
            return g[:i + 1]
    return list(g)


# ============================================================ architecture gate
BAD = ("deltanet", "linearattention", "mamba", "rwkv", "recurrent", "ssm", "lightning")
hits = sorted(n for n in set(type(m).__name__ for m in MODELS[DEV0].modules())
              if any(b in n.lower() for b in BAD))
flags = [k for k in ("linear_attn", "use_mamba")
         if getattr(_tc, k, None) not in (None, False)]
assert not (hits or flags), (
    "ABORT -- non-softmax / stateful attention in %s (%s / %s). The additive tree"
    " mask cannot isolate siblings in such a layer, so verification would stop being"
    " exact WITHOUT failing." % (MODEL, hits, flags))

LT = list(getattr(_tc, "layer_types", None) or [])
SOFTMAX_T = ("full_attention", "sliding_attention", "chunked_attention")
badt = sorted({t for t in LT if t not in SOFTMAX_T})
assert not badt, (
    "ABORT -- %s declares layer types %s, which are not softmax attention over an"
    " additive mask." % (MODEL, badt))

# A WINDOWED softmax layer is still maskable, and what a candidate needs to see is
# its own ancestor chain plus the trunk -- a few dozen positions, never the whole
# context. Truncation of the DISTANT prefix is not an exactness problem, because
# the window applies identically to the baseline pass and the verification pass.
CHAIN_MAX = 64
WIN_NEED = 2 * CHAIN_MAX + 8
SW = getattr(_tc, "sliding_window", None)
nwin = sum(1 for t in LT if t != "full_attention")
if nwin:
    assert SW is None or SW >= WIN_NEED, (
        "ABORT -- %d of %d layers windowed at sliding_window=%s, below the %d"
        " positions a candidate needs to span its own chain plus the trunk."
        % (nwin, len(LT), SW, WIN_NEED))
WIN_BINDS = bool(nwin) and SW is not None and SW < max(MAX_NEW_FORCED, MAX_NEW_LONG)
log("architecture gate: softmax attention throughout (%s)"
    % (("%d layers, %s" % (len(LT), sorted(set(LT)))) if LT else "no layer_types"))
if nwin:
    log("  %d of %d layers windowed at %s; a candidate spans at most %d positions,"
        " so the window never cuts a candidate off from its own prefix"
        % (nwin, len(LT), SW, WIN_NEED))
if WIN_BINDS:
    log("  *** THE WINDOW BINDS AT THIS GENERATION LENGTH (%s < %d). This has never"
        "\n      been exercised: every previous run stopped at or below 512 tokens, so"
        "\n      the window was always slack. The hazard is NOT the window itself -- it"
        "\n      is that we pass a 4D additive mask, and if transformers consumes ours"
        "\n      INSTEAD OF applying its own sliding window, the arm would attend past"
        "\n      the window while the baseline does not. PHASE W tests exactly that"
        "\n      before anything is timed." % (SW, max(MAX_NEW_FORCED, MAX_NEW_LONG)))
DIAG.append(dict(check="arch_gate", value=float(len(LT)),
                 detail="types=%s sliding_window=%s binds=%s"
                 % (sorted(set(LT)), SW, WIN_BINDS)))

log("PHASE A: asserting the vectorised mask")
rg = np.random.default_rng(0)
for t_ in range(30):
    B_ = int(rg.integers(1, 7)); C_ = int(rg.integers(4, 40))
    nv = [int(rg.integers(1, C_ + 1)) for _ in range(B_)]
    kvv = torch.zeros((B_, C_), dtype=torch.bool)
    for b in range(B_):
        kvv[b, C_ - nv[b]:] = True
    tl = torch.tensor([int(rg.integers(1, 5)) for _ in range(B_)])
    cls = [[int(x) for x in rg.integers(1, 6, size=int(rg.integers(1, 7)))]
           for _ in range(B_)]
    m, pos, st_, Tm, Cm, _ok = tree_mask_batch(kvv, tl, cls, torch.tensor(nv), DEV0)
    for b in range(B_):
        T, tt_ = int(tl[b]), sum(cls[b])
        ref, _ = tree_mask_ref(nv[b], T, cls[b], DEV0)
        rws = list(range(Tm - T, Tm)) + list(range(Tm, Tm + tt_))
        cl2 = (list(range(C_ - nv[b], C_)) + list(range(C_ + Tm - T, C_ + Tm))
               + list(range(C_ + Tm, C_ + Tm + tt_)))
        assert torch.equal(m[b, 0][rws][:, cl2], ref[0, 0]), "mask mismatch %d/%d" % (t_, b)
log("  30/30 identical")


# ============================================================ prompts
def pull(srcs, cap, label):
    got = []
    for nm, name, cfg, split, col in srcs:
        if len(got) >= cap:
            break
        try:
            d = load_dataset(name, cfg, split=split, streaming=True) if cfg \
                else load_dataset(name, split=split, streaming=True)
        except Exception as e:
            log("  %-12s unavailable (%s)" % (nm, str(e).split("\n")[0][:60]))
            continue
        n0 = len(got)
        try:
            for row in itertools.islice(iter(d), 60000):
                t = row.get(col)
                if isinstance(t, str) and 40 <= len(t) <= 600:
                    got.append(t.strip())
                if len(got) - n0 >= PER_SOURCE or len(got) >= cap:
                    break
        except Exception:
            pass
        log("  %-12s +%d (%s total %d)" % (nm, len(got) - n0, label, len(got)))
    seen, out = set(), []
    for t in got:
        if t not in seen:
            seen.add(t); out.append(t)
    return out


def norm(t):
    return re.sub(r"\s+", " ", t.lower().strip())


def grams(t, n=8):
    w = norm(t).split()
    return {" ".join(w[i:i + n]) for i in range(max(len(w) - n + 1, 1))}


log("gathering English math prompts (test splits held out, rest trains)")
TRAIN_POOL = pull(TRAIN_SOURCES, DISTILL_N * 3, "train")
assert len(TRAIN_POOL) >= 400, "only %d training prompts" % len(TRAIN_POOL)
TN = {norm(t) for t in TRAIN_POOL}
TG = set()
for t in TRAIN_POOL:
    TG |= grams(t)

rgp = np.random.default_rng(0)
NEED = max(NAT_N, LONG_N, FORCED_N)
TESTS_BY_SET = {}
for nm, name, cfg, split, col in TEST_SETS:
    got = pull([(nm, name, cfg, split, col)], NEED * 6, nm)
    got = [t for t in got if norm(t) not in TN]
    if len(got) < NEED:
        log("  %-12s unusable (%d prompts, need %d) -- skipped" % (nm, len(got), NEED))
        continue
    sel = rgp.permutation(len(got))[:NEED]
    TESTS_BY_SET[nm] = [got[i] for i in sel]
    ov2 = [len(grams(t) & TG) / max(len(grams(t)), 1) for t in TESTS_BY_SET[nm]]
    log("  %-12s %d prompts | 8-gram overlap with training: mean %.3f max %.3f"
        % (nm, len(TESTS_BY_SET[nm]), float(np.mean(ov2)), float(np.max(ov2))))
    DIAG.append(dict(check="leakage_%s" % nm, value=float(np.mean(ov2)),
                     detail="max=%.3f n=%d" % (float(np.max(ov2)), len(got))))
assert "gsm8k-test" in TESTS_BY_SET, (
    "GSM8K test did not load. It is the anchor that ties this run to E78/E79, and"
    " without it the depth curve has no cross-run reference at all.")
_ovg = float(np.mean([len(grams(t) & TG) / max(len(grams(t)), 1)
                      for t in TESTS_BY_SET["gsm8k-test"]]))
assert _ovg < 0.5, (
    "mean 8-gram overlap between the GSM8K test prompts and the training pool is"
    " %.3f -- MetaMathQA is derived from GSM8K, so drop it from TRAIN_SOURCES."
    % _ovg)
rgp.shuffle(TRAIN_POOL)
TRAINP = TRAIN_POOL[:DISTILL_N]
# E80 lost a run to a source shortfall that nothing asserted: it asked for 10,000
# prompts per language, got ~3,000, and carried on. Checked here instead.
assert len(TRAIN_POOL) >= DISTILL_N, (
    "only %d distinct training prompts for DISTILL_N=%d. The head would train on"
    " %.0f%% of the intended data and nothing downstream would fail -- which is"
    " exactly how E80 wasted a session. Raise PER_SOURCE or add a source to"
    " TRAIN_SOURCES." % (len(TRAIN_POOL), DISTILL_N,
                         100.0 * len(TRAIN_POOL) / DISTILL_N))
log("  %d training prompts to distil | test sets %s"
    % (len(TRAINP), {k: len(v) for k, v in TESTS_BY_SET.items()}))
tok.padding_side = "left"


# ============================================================ distillation
def gen_chunk(enc, out, dev):
    L = max(len(e) for e in enc)
    x = torch.full((len(enc), L), PADID, dtype=torch.long)
    am = torch.zeros((len(enc), L), dtype=torch.long)
    for r, e in enumerate(enc):
        x[r, L - len(e):] = torch.tensor(e); am[r, L - len(e):] = 1
    try:
        with torch.no_grad():
            o = MODELS[dev].generate(input_ids=x.to(dev), attention_mask=am.to(dev),
                                     max_new_tokens=DISTILL_NEW, do_sample=False,
                                     pad_token_id=PADID)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False
    for r in range(len(enc)):
        g = [t for t in o[r, L:].tolist() if t != PADID and t not in EOS]
        if len(g) >= 24:
            out.append((enc[r], g))
    del o, x, am
    return True


def distill(dev, ps):
    """The target's own greedy chain-of-thought. Batched to a B*L^2 budget and
    length-sorted: eager attention is O(B * heads * L^2) in PREFILL, and that
    product, not the batch size, is what runs a T4 out of memory (E72)."""
    enc = [Enc(chat(p))[:MAX_PROMPT_TOK] for p in ps]
    order = sorted(range(len(ps)), key=lambda i: len(enc[i]))
    out, i = [], 0
    while i < len(order):
        L = len(enc[order[min(i + GEN_BATCH, len(order)) - 1]])
        B = max(1, min(GEN_BATCH, int(ATT_CELLS // max(L * L, 1))))
        j = min(i + B, len(order))
        sl = order[i:j]
        while not gen_chunk([enc[k] for k in sl], out, dev):
            if len(sl) == 1:
                log("    skipping one %d-token prompt that will not fit" % len(enc[sl[0]]))
                sl = []
                break
            sl = sl[:max(1, len(sl) // 2)]
            log("    OOM at B=%d L=%d, halving" % (j - i, L))
        i += max(len(sl), 1)
        if len(out) % 300 < B:
            log("    %d/%d prompts, %d kept [%5.1fm]" % (i, len(order), len(out), mins()))
    return out


GENS, dl = [], False
for src in [p for p in (DISTILL_IN,) if p]:
    try:
        blob = torch.load(str(src), map_location="cpu", weights_only=False)
        assert blob.get("model") == MODEL, (
            "distillation cache was built on %r but this run targets %r. The labels"
            " ARE the target's own output, so a cache from another model trains the"
            " head to predict the wrong model's words -- and nothing downstream fails."
            % (blob.get("model"), MODEL))
        GENS, dl = blob["gens"], True
        log("distillation LOADED from %s (model tag matches)" % src); break
    except AssertionError:
        raise
    except Exception as e:
        log("  no cache at %s (%s)" % (src, str(e).split("\n")[0][:60]))

HAVE = {tuple(e) for e, _g in GENS}
TODO = [p for p in TRAINP if tuple(Enc(chat(p))[:MAX_PROMPT_TOK]) not in HAVE]
if len(GENS) < DISTILL_N and TODO:
    log("PHASE 1: have %d completions, distilling %d more (%d new tokens each)"
        % (len(GENS), min(len(TODO), DISTILL_N - len(GENS)), DISTILL_NEW))
    TODO = TODO[:DISTILL_N - len(GENS)]
    # Checkpoint in blocks. E83 ran the whole of TODO and saved once at the end,
    # so a disconnect at minute 30 of a 37-minute distillation lost all of it --
    # and at DISTILL_N=50000 that is over two hours of exposure.
    for _b0 in range(0, len(TODO), DISTILL_SAVE_EVERY):
        if mins() > T_BUDGET_TRAIN - 20:
            log("  stopping distillation on the clock with %d completions; re-run"
                " to continue" % len(GENS))
            break
        blk = TODO[_b0:_b0 + DISTILL_SAVE_EVERY]
        RES = {}

        def dw(dev, ps):
            RES[dev] = distill(dev, ps)

        with ThreadPoolExecutor(len(DEVS)) as ex:
            list(ex.map(lambda a: dw(*a),
                        zip(DEVS, [blk[i::len(DEVS)] for i in range(len(DEVS))])))
        for d in DEVS:
            GENS += RES.get(d, [])
        torch.save({"model": MODEL, "gens": GENS},
                   str(OUT / DISTILL_OUT))
        log("  checkpoint: %d completions saved [%5.1fm]" % (len(GENS), mins()))
    torch.save({"model": MODEL, "gens": GENS}, str(OUT / DISTILL_OUT))
    log("  %d usable completions total [%5.1fm] -> SAVED to %s"
        % (len(GENS), mins(), OUT / DISTILL_OUT))
    log("  DOWNLOAD IT. DISTILL_IN already points here, so re-running this exact"
        " script\n  after a timeout skips the single most expensive phase. It also"
        " TOPS UP: a\n  partial cache is extended rather than regenerated.")
else:
    log("PHASE 1: %d cached completions already meet DISTILL_N=%d" % (len(GENS),
                                                                      DISTILL_N))
assert len(GENS) >= 300, "only %d completions; not enough to train on" % len(GENS)
# E85 discovered after the fact that 70.8% of its completions hit the token cap
# exactly, so most were cut off mid-solution. Measured here rather than inferred.
_ls = [len(g) for _p, g in GENS]
_cap_hit = sum(1 for x in _ls if x >= DISTILL_NEW)
log("completion lengths: mean %.1f, median %d, max %d of a %d cap"
    % (float(np.mean(_ls)), int(np.median(_ls)), max(_ls), DISTILL_NEW))
log("  %d of %d (%.1f%%) hit the cap exactly, i.e. were TRUNCATED mid-solution."
    % (_cap_hit, len(_ls), 100 * _cap_hit / len(_ls)))
if _cap_hit > 0.4 * len(_ls):
    log("  -> still mostly truncated. The head will be weakest exactly where it"
        " must draft\n     the end of an answer. Raise DISTILL_NEW next run; the"
        " cache keeps old\n     completions, so lengths may be mixed freely.")
DIAG.append(dict(check="truncation", value=_cap_hit / max(len(_ls), 1),
                 detail="mean=%.1f median=%d cap=%d n=%d"
                 % (float(np.mean(_ls)), int(np.median(_ls)), DISTILL_NEW,
                    len(_ls))))


# ===================== the vocabulary: WORDS *AND PHRASES* (E75c, unchanged)
def split_words(txt):
    w = re.findall(WORD_RE, txt)
    return w if "".join(w) == txt else None


def align(ids):
    """(words, per-word token tuples, per-word first-token index) or None."""
    txt = tok.decode(ids)
    ws = split_words(txt)
    if ws is None:
        return None
    enc = tok(txt, add_special_tokens=False, return_offsets_mapping=True)
    if enc["input_ids"] != list(ids):
        return None
    offs = enc["offset_mapping"]
    pos, spans = 0, []
    for w in ws:
        spans.append((pos, pos + len(w))); pos += len(w)
    wt, wstart, ti = [], [], 0
    for a, b in spans:
        while ti < len(ids) and offs[ti][1] <= a:
            ti += 1
        wstart.append(ti)
        t = []
        while ti < len(ids) and offs[ti][0] < b:
            t.append(ids[ti]); ti += 1
        wt.append(tuple(t))
    return ws, wt, wstart


log("PHASE 1b: building a word+phrase vocabulary from %d completions" % len(GENS))
ALIGNED, nbad = [], 0
for p, g in GENS:
    a = align(g)
    if a is None:
        nbad += 1
        continue
    ALIGNED.append((p, list(g)) + a)
log("  %d completions aligned, %d rejected by the reconstruction check"
    % (len(ALIGNED), nbad))
assert nbad < 0.25 * len(GENS), (
    "%d of %d completions failed exact reconstruction -- WORD_RE does not cover this"
    " text, so SLEM could not reassemble a draft. Fix WORD_RE." % (nbad, len(GENS)))

CNT = [None] * (MAX_GRAM + 1)
EXPN = [None] * (MAX_GRAM + 1)
CNT[1], EXPN[1] = Counter(), defaultdict(Counter)
for _p, _ids, ws, wt, _st in ALIGNED:
    for i, w in enumerate(ws):
        if wt[i]:
            CNT[1][w] += 1
            EXPN[1][w][wt[i]] += 1
log("  n=1: %d distinct" % len(CNT[1]))
alive = {w for w, c in CNT[1].items() if c >= MIN_GRAM_COUNT}
for n in range(2, MAX_GRAM + 1):
    CNT[n], EXPN[n] = Counter(), defaultdict(Counter)
    for _p, _ids, ws, wt, _st in ALIGNED:
        for i in range(len(ws) - n + 1):
            if "".join(ws[i:i + n - 1]) not in alive:
                continue
            toks = tuple(t for j in range(i, i + n) for t in wt[j])
            if not toks or len(toks) > MAX_UNIT_TOK:
                continue
            k = "".join(ws[i:i + n])
            CNT[n][k] += 1
            EXPN[n][k][toks] += 1
    alive = {k for k, c in CNT[n].items() if c >= MIN_GRAM_COUNT}
    log("  n=%d: %d distinct, %d above count %d" % (n, len(CNT[n]), len(alive),
                                                    MIN_GRAM_COUNT))
    if not alive:
        break

FIXED_PAR = 2 * D_MODEL * H_DIM + H_DIM * H_DIM + 4 * H_DIM
BUDGET_WORDS = max(1000, int((PARAM_BUDGET - FIXED_PAR) // (H_DIM + 1)))
MAX_WORDS = min(MAX_VOCAB, BUDGET_WORDS)
log("  head arithmetic: %d fixed + V x %d. The %.0f M ceiling allows V=%d; the"
    " explicit\n  cap is %d, so %s binds -> a %.3f M head at full capacity"
    % (FIXED_PAR, H_DIM + 1, PARAM_BUDGET / 1e6, BUDGET_WORDS, MAX_VOCAB,
       "MAX_VOCAB" if MAX_VOCAB < BUDGET_WORDS else "PARAM_BUDGET",
       (FIXED_PAR + MAX_WORDS * (H_DIM + 1)) / 1e6))

uni_all = [w for w, c in CNT[1].most_common() if c >= MIN_WORD_COUNT]
nums = [f(i) for i in range(ADD_NUMERIC) for f in (str, lambda i: " %d" % i)]
extra = [w for w in nums if w not in CNT[1]]
# BOUND THE UNIGRAM SLICE. E81 took `uni + numerics` first and gave phrases only
# the room left over; at 1,200 completions that was 7,761 unigrams leaving 15,312
# for phrases, but unigram TYPES grow with the corpus and at 10,000 completions
# they would crowd the phrases out of a fixed 40,000-entry cap. Phrases are the
# mechanism (findings 8.19), so unigrams are capped by FREQUENCY at UNI_SHARE of
# the budget. Coverage is reported either way, so a regression shows up instead of
# hiding: unigrams past the frequent head of the distribution are count-1 types
# that contribute little coverage and one vocabulary row each.
UNI_CAP = max(1000, int(UNI_SHARE * MAX_WORDS))
uni = uni_all[:UNI_CAP]
if len(uni_all) > len(uni):
    log("  unigram cap BINDS: %d distinct unigrams, keeping the %d most frequent"
        " (%.0f%% of a %d cap).\n  The %d dropped are the count-%d-and-below tail;"
        " greedy longest match falls back\n  to a shorter unit where one was"
        " dropped, so the cost is graceful."
        % (len(uni_all), len(uni), 100 * UNI_SHARE, MAX_WORDS,
           len(uni_all) - len(uni), CNT[1][uni[-1]] if uni else 0))
base = uni + extra
ph = []
for n in range(2, MAX_GRAM + 1):
    if CNT[n] is None:
        break
    for k, c in CNT[n].items():
        if c < MIN_GRAM_COUNT:
            continue
        L = len(max(EXPN[n][k].items(), key=lambda z: z[1])[0])
        ph.append((c * (L - 1), k, n))
ph.sort(reverse=True)
room = max(0, MAX_WORDS - len(base))
VOCAB = base + [k for _s, k, _n in ph[:room]]
WROW = {w: i for i, w in enumerate(VOCAB)}
NG = {}
for n in range(1, MAX_GRAM + 1):
    if EXPN[n] is None:
        break
    for k, c in EXPN[n].items():
        if k in WROW and k not in NG:
            NG[k] = max(c.items(), key=lambda z: z[1])[0]
PRE, nretok = [], 0
for w in VOCAB:
    if w in NG:
        PRE.append(NG[w])
    else:
        PRE.append(tuple(tok(w, add_special_tokens=False)["input_ids"]))
        nretok += 1
NWORDS = {w: len(split_words(w) or [w]) for w in VOCAB}
MASK = {d: torch.zeros(len(VOCAB), dtype=torch.float32).to(d) for d in DEVS}
log("  vocabulary %d of a %d cap: %d unigrams + %d numerics + %d phrases"
    % (len(VOCAB), MAX_WORDS, len(uni), len(extra), len(VOCAB) - len(base)))
log("  -> head will be %.3f M = %.4f%% of target (ceiling %.0f M, vocabulary cap %d)"
    % ((FIXED_PAR + len(VOCAB) * (H_DIM + 1)) / 1e6,
       100 * (FIXED_PAR + len(VOCAB) * (H_DIM + 1)) / TGT, PARAM_BUDGET / 1e6,
       MAX_VOCAB))
log("  %d entries re-tokenized context-free (never observed in the corpus)" % nretok)
by_n = Counter(NWORDS[w] for w in VOCAB)
log("  entries by word count: %s" % sorted(by_n.items()))


def segment(ws):
    """Greedy longest match at EVERY word boundary -- sliding, not tiling, because
    at inference the loop re-drafts at whatever boundary it just accepted up to."""
    out = []
    for i in range(len(ws)):
        best = None
        for n in range(min(MAX_GRAM, len(ws) - i), 0, -1):
            k = "".join(ws[i:i + n])
            if k in WROW:
                best = (k, n)
                break
        out.append(best)
    return out


SEGS, tot_tok, tot_pos, cov_hit, cov_all = [], 0, 0, 0, 0
for _p, _ids, ws, wt, st in ALIGNED[:600]:
    sg = segment(ws)
    for i in range(len(ws)):
        cov_all += 1
        b = sg[i]
        if b is None:
            continue
        cov_hit += 1
        tot_tok += len(PRE[WROW[b[0]]])
        tot_pos += 1
        SEGS.append(NWORDS[b[0]])
EXP = tot_tok / max(tot_pos, 1)
COV_W = cov_hit / max(cov_all, 1)
_na = _nd = 0
for n in range(1, MAX_GRAM + 1):
    if EXPN[n] is None:
        break
    for k, c in EXPN[n].items():
        if k not in WROW:
            continue
        _nd += sum(c.values())
        _na += c.get(PRE[WROW[k]], 0)
agree = _na / max(_nd, 1)
log("  OCCURRENCE-WEIGHTED expansion %.3f target tokens per unit" % EXP)
log("  mean words per drafted unit %.2f" % float(np.mean(SEGS) if SEGS else 1.0))
log("  coverage %.3f of word boundaries have a vocabulary entry"
    " (E81 at 1,200 completions: 0.973)" % COV_W)
assert COV_W > 0.90, (
    "coverage fell to %.3f. The unigram cap (UNI_SHARE=%.2f -> %d entries) has cut"
    " into the frequent head of the distribution rather than its count-1 tail, and"
    " coverage is a hard multiplier on tok/pass (E75b). Raise UNI_SHARE or"
    " MAX_VOCAB." % (COV_W, UNI_SHARE, UNI_CAP))
log("  SLEM exposure gap: the modal expansion is what the target emitted %.1f%% of"
    " the time" % (100 * agree))
log("  tok/pass ceiling at landing 1.0: %.2f" % (1 + EXP))
DIAG.append(dict(check="expansion", value=EXP,
                 detail="vocab=%d cov=%.3f agree=%.3f words/unit=%.2f"
                 % (len(VOCAB), COV_W, agree,
                    float(np.mean(SEGS) if SEGS else 1.0))))


# ============================================================ features
_FRG = np.random.default_rng(19)


def feats(dev, chunk, sq, row0, cap):
    """`chunk` is [(global_index, (p, ids, ws, wt, st)), ...]. The global index is
    carried through to `sid_src` so `relabel()` can find each sid's word list --
    completions get SKIPPED here, so positional recovery afterwards is wrong."""
    """One row per word boundary: (sid, ti, q, y).

    ALIGNMENT. The decode loop supplies (h_t, x_{t+1}) and must predict the unit
    starting at x_{t+2}. E75 v1 paired (h_{z-1}, x_z) with the unit at x_z, handing
    the head the unit's own first token -- loss 0.02, top-1 0.961, landing 0.016.
    The pass is over PROMPT + COMPLETION, not the completion alone (E66, +0.115x)."""
    # ONE HIDDEN VECTOR PER TRAINING ROW, PREALLOCATED.
    #
    # E81 kept the hidden state of EVERY token of every completion. At 1,200
    # completions that is 1.54 GB; at DISTILL_N=10,000 it is 12.80 GB against
    # Colab's 12.7 GB limit, and the `np.concatenate` below then allocated a
    # SECOND copy before the first was freed. That is exactly how E80 died, and
    # this is E80's fix: write only the positions a training row actually needs,
    # straight into a preallocated buffer, with no concatenate and a hard cap.
    # Column layout of `ix`: (flat_row, sid, ti, q, y).
    # `sq` is a VIEW into one array shared by every device, starting at `row0`
    # and `cap` rows long. E86 allocated a fresh array per device and then
    # assembled FLAT from the FIRST shard only, so on 2 x T4 the index pointed at
    # 2.48 M rows inside 1.24 M (IndexError). Concatenating afterwards is not the
    # answer either: at 3.5 M rows that peaks at 2 x 18 GB against 30 GB of RAM.
    # One preallocated array with disjoint views has no concatenate, no doubling,
    # and row offsets that are correct by construction.
    nrow = 0
    ix, sid_ids, sid_src = [], [], []
    with torch.no_grad():
        for i in range(0, len(chunk), FEAT_BATCH):
            grp = []
            for _gi, (p, ids, ws, wt, st) in chunk[i:i + FEAT_BATCH]:
                if len(ids) < 12:
                    continue
                sg = segment(ws)
                ent = []
                for wi, b in enumerate(sg):
                    if b is None:
                        continue
                    z = st[wi]
                    if z < 2:
                        continue
                    # wi, the WORD INDEX, not WROW[b[0]]. A hidden state depends
                    # on (completion, position) and nothing else; only the LABEL
                    # depends on the vocabulary. Storing wi makes the whole feature
                    # cache vocabulary-independent, so switching vocabularies is a
                    # pure-CPU relabel instead of a 92-minute re-extraction.
                    ent.append((z - 2, ids[z - 1], wi))
                if ROWS_PER_SEQ and len(ent) > ROWS_PER_SEQ:
                    # keep a random scatter across the whole completion rather
                    # than its first ROWS_PER_SEQ positions: the end of an answer
                    # is where the head is weakest and a prefix would exclude it
                    pick = _FRG.choice(len(ent), ROWS_PER_SEQ, replace=False)
                    pick.sort()
                    ent = [ent[k] for k in pick]
                if ent:
                    grp.append((Enc(chat(p))[:MAX_PROMPT_TOK], ids, ent, _gi))
            if not grp:
                continue
            L = max(len(p) + len(g) for p, g, _e, _g in grp)
            x = torch.full((len(grp), L), PADID, dtype=torch.long)
            am = torch.zeros((len(grp), L), dtype=torch.long)
            for r, (p, g, _e, _g) in enumerate(grp):
                s_ = list(p) + list(g)
                x[r, :len(s_)] = torch.tensor(s_); am[r, :len(s_)] = 1
            # `output_hidden_states=True` builds a tuple of all 37 hidden states
            # -- 2.22 GB per batch at FEAT_BATCH=32 -- to use exactly one. The
            # write traffic is only ~7 ms, but the VRAM is what caps FEAT_BATCH,
            # and `mm.model(...)` returns just the last layer. fwd_hidden already
            # does this in the decode loop; feats never caught up.
            _o = MODELS[dev].model(input_ids=x.to(dev), attention_mask=am.to(dev))
            hs = _o.last_hidden_state if hasattr(_o, "last_hidden_state") else _o[0]
            del _o
            for r, (p, g, ent, _gi) in enumerate(grp):
                sid = len(sid_ids)
                sid_ids.append(list(g))
                sid_src.append(_gi)
                if nrow + len(ent) > cap:
                    ent = ent[:max(cap - nrow, 0)]
                if not ent:
                    continue
                want = torch.tensor([len(p) + e[0] for e in ent], device=hs.device)
                sq[nrow:nrow + len(ent)] = hs[r].index_select(0, want).half(
                    ).cpu().numpy()
                for j, (ti, q, wi) in enumerate(ent):
                    # flat_row is ABSOLUTE in the shared array, so the caller does
                    # not have to offset anything
                    ix.append((row0 + nrow + j, sid, ti, q, wi))
                nrow += len(ent)
            del hs
            if nrow >= cap:
                log("  %s feature cap reached at %d of %d rows" % (dev, nrow, cap))
                break
    return nrow, ix, sid_ids, sid_src


class Head(nn.Module):
    """E69's winner, unchanged. Nothing new is being tried here -- only the length."""

    def __init__(self, d_model, v, H=256):
        super().__init__()
        self.i = nn.Linear(d_model * 2, H)
        self.n = nn.LayerNorm(H)
        self.f = nn.Linear(H, H)
        self.n2 = nn.LayerNorm(H)
        self.o = nn.Linear(H, v)

    def forward(self, hv, qe):
        z = self.n(self.i(torch.cat([hv, qe], -1)))
        return self.o(self.n2(z + F.silu(self.f(z))))


HD, hl = None, False
for src in [p for p in (HEADS_IN,) if p]:
    try:
        blob = torch.load(str(src), map_location="cpu", weights_only=False)
        assert blob.get("model") == MODEL and blob.get("vocab") == VOCAB, \
            "head checkpoint does not match this run's target or vocabulary"
        hd = Head(D_MODEL, len(VOCAB), H_DIM).to(DEV0)
        hd.load_state_dict(blob["sd"])
        HD = dict(head=hd.eval(), **{k: v for k, v in blob.items()
                                     if k not in ("sd", "vocab", "PRE",
                                                  "NWORDS")})
        hl = True
        log("head LOADED from %s" % src); break
    except AssertionError:
        raise
    except Exception as e:
        log("  no head at %s (%s)" % (src, str(e).split("\n")[0][:60]))

# ---- Feature extraction is ~an hour at 4 M rows, and the memmap plus its index
#      is all that training needs, so it is cached. The cache is keyed on the
#      vocabulary, the model and d_model: a different vocabulary means every stored
#      LABEL refers to a different unit, which would train silently on nonsense.
FEAT_IDX = OUT / "e87_feats_index.npz"


def FEAT_DAT_FOR(dev):
    return str(OUT / ("e87_feats.dat"))


# How many positions to take per completion, so every completion is represented
# and the total lands near the cap. Computed from the real completion count.
#
# E86's feats walked completions in ORDER and stopped at the cap, which at ~161
# rows each would have left ~31,000 of 43,550 UNREAD while logging "feature cap
# reached" and looking fine. Rows inside one chain-of-thought are correlated, so
# the SEQUENCE count is what diversity means: 43,550 x 80 beats 21,700 x 161 at
# equal rows.
ROWS_PER_SEQ = 0
if len(ALIGNED) > 0:
    _per = max(1, FEAT_MAX_POS // len(ALIGNED))
    _avg_rows = float(np.mean([max(len(a[2]) - 2, 1) for a in ALIGNED[:200]]))
    if _avg_rows * len(ALIGNED) > FEAT_MAX_POS:
        ROWS_PER_SEQ = int(_per)
        log("ROWS PER COMPLETION capped at %d: %d completions x ~%.0f rows ="
            " ~%.1f M candidate\n  rows against a %.1f M cap. Sampling %d"
            " SCATTERED positions from each keeps all\n  %d sequences in the"
            " training set, which is the diversity this run is buying."
            % (ROWS_PER_SEQ, len(ALIGNED), _avg_rows,
               _avg_rows * len(ALIGNED) / 1e6, FEAT_MAX_POS / 1e6,
               ROWS_PER_SEQ, len(ALIGNED)))
        DIAG.append(dict(check="rows_per_seq", value=float(ROWS_PER_SEQ),
                         detail="completions=%d avg_rows=%.1f cap=%d"
                         % (len(ALIGNED), _avg_rows, FEAT_MAX_POS)))
    else:
        log("the row cap does not bind: %d completions x ~%.0f rows = %.1f M,"
            " under the %.1f M cap"
            % (len(ALIGNED), _avg_rows, _avg_rows * len(ALIGNED) / 1e6,
               FEAT_MAX_POS / 1e6))


def feature_cache_key():
    """NO VOCABULARY IN THE KEY. A hidden state depends on (completion, position)
    and nothing else -- only the LABEL depends on the vocabulary, and labels are
    recomputed by `relabel()` below in seconds. E86 keyed on `nvocab`, so changing
    60k -> 80k entries forced a 92-minute re-extraction for a pure-CPU relabel.
    The corpus IS in the key, because a different distillation cache means
    different positions."""
    return dict(model=MODEL, d_model=D_MODEL, ncomp=len(ALIGNED),
                rows_per_seq=ROWS_PER_SEQ)


FEAT_CACHED = None
if FEAT_IDX.exists() and os.path.exists(FEAT_DAT_FOR(DEV0)):
    try:
        _z = np.load(str(FEAT_IDX), allow_pickle=True)
        _k = feature_cache_key()
        if (str(_z["model"]) == _k["model"] and int(_z["d_model"]) == _k["d_model"]
                and int(_z["ncomp"]) == _k["ncomp"]
                and int(_z["rows_per_seq"]) == _k["rows_per_seq"]):
            FEAT_CACHED = _z
            log("FEATURES LOADED from %s -- PHASE 2 SKIPPED (%d rows). The cache is"
                " vocabulary-\n  INDEPENDENT, so this works even though the"
                " vocabulary is now %d entries."
                % (os.path.basename(FEAT_DAT_FOR(DEV0)),
                   int(_z["nrow_total"]), len(VOCAB)))
        else:
            log("feature cache exists but the corpus or geometry differs"
                " (model/d_model/ncomp/rows_per_seq); re-extracting")
    except Exception as e:
        log("feature cache unreadable (%s); re-extracting" % str(e)[:70])


def relabel(idx_wi, sid_src):
    """(flat_row, sid, ti, q, WI) -> (flat_row, sid, ti, q, Y) for the CURRENT
    vocabulary. Pure CPU: group the rows by completion, run greedy longest match
    once per completion, and map each word index to its vocabulary row. Rows whose
    word has no entry are dropped -- at coverage 0.998 that is a rounding error."""
    t0 = clk()
    by_sid = defaultdict(list)
    for k, r in enumerate(idx_wi):
        by_sid[r[1]].append(k)
    out, dropped = [], 0
    for sid, ks in by_sid.items():
        gi = int(sid_src[sid])
        if gi >= len(ALIGNED):
            dropped += len(ks)
            continue
        ws = ALIGNED[gi][2]
        sg = segment(ws)
        for k in ks:
            r = idx_wi[k]
            wi = int(r[4])
            b = sg[wi] if 0 <= wi < len(sg) else None
            if b is None:
                dropped += 1
                continue
            out.append((int(r[0]), int(r[1]), int(r[2]), int(r[3]), WROW[b[0]]))
    log("  relabelled %d rows against the %d-entry vocabulary in %.1f s"
        " (%d dropped, no entry)" % (len(out), len(VOCAB), clk() - t0, dropped))
    DIAG.append(dict(check="relabel", value=float(len(out)),
                     detail="dropped=%d nvocab=%d" % (dropped, len(VOCAB))))
    return out


if not hl:
    # ---- ONE array, device-disjoint views. See the header: a per-device array
    #      plus a concatenate would peak at 2 x 18 GB against 30 GB of RAM, and
    #      E86's bug was assembling FLAT from the first shard only.
    NDEV = max(len(DEVS), 1)
    CAP_PER_DEV = FEAT_MAX_POS // NDEV
    CAP_TOTAL = CAP_PER_DEV * NDEV
    if FEAT_CACHED is None:
        log("PHASE 2: features over %d completions on %d device(s), into one"
            " %.1f GB array\n  (%d rows per device, disjoint views -- no"
            " concatenate, no second copy)"
            % (len(ALIGNED), NDEV, CAP_TOTAL * D_MODEL * 2 / 1e9, CAP_PER_DEV))
        if FEAT_MEMMAP:
            SQ_ALL = np.memmap(FEAT_DAT_FOR(DEV0), dtype=np.float16, mode="w+",
                               shape=(CAP_TOTAL, D_MODEL))
        else:
            SQ_ALL = np.empty((CAP_TOTAL, D_MODEL), dtype=np.float16)
        RES2 = {}

        def fw(di, dev, ch):
            try:
                r0 = di * CAP_PER_DEV
                view = SQ_ALL[r0:r0 + CAP_PER_DEV]
                RES2[dev] = feats(dev, ch, view, r0, CAP_PER_DEV)
                log("  %s %d rows at offset %d, %d completions kept [%5.1fm]"
                    % (dev, RES2[dev][0], r0, len(RES2[dev][2]), mins()))
            except Exception as e:
                log("  [FAIL features] %s: %s" % (dev, e))
                traceback.print_exc()

        # carry each completion's GLOBAL index so relabel can find its word list
        TAGGED = list(enumerate(ALIGNED))
        with ThreadPoolExecutor(NDEV) as ex:
            list(ex.map(lambda a: fw(*a),
                        [(di, d, TAGGED[di::NDEV]) for di, d in enumerate(DEVS)]))
        IDX_WI, SIDIDS, SID_SRC = [], [], []
        for di, d in enumerate(DEVS):
            if d not in RES2:
                continue
            nrow_d, ix, sids, src = RES2[d]
            assert len(src) == len(sids), (
                "feats returned %d sids but %d source indices" % (len(sids),
                                                                  len(src)))
            bsid = len(SIDIDS)
            SIDIDS += sids
            SID_SRC += src
            IDX_WI += [tuple(r[:1]) + (bsid + r[1],) + tuple(r[2:]) for r in ix]
        del RES2
        assert len(SID_SRC) == len(SIDIDS)
        assert IDX_WI, "no training positions were produced"
        if FEAT_MEMMAP:
            SQ_ALL.flush()
        NROW_TOTAL = CAP_TOTAL
        np.savez(str(FEAT_IDX), idx=np.array(IDX_WI, dtype=np.int64),
                 sidids=np.array(SIDIDS, dtype=object),
                 sid_src=np.array(SID_SRC, dtype=np.int64),
                 nrow_total=NROW_TOTAL, cap=CAP_TOTAL,
                 naligned=len(ALIGNED), **feature_cache_key())
        log("  feature index SAVED to %s (vocabulary-independent) -- download it"
            " with\n  %s and PHASE 2 never runs again for this corpus"
            % (FEAT_IDX.name, os.path.basename(FEAT_DAT_FOR(DEV0))))
        if not FEAT_MEMMAP:
            try:
                log("  writing %.1f GB of features to disk for reuse..."
                    % (CAP_TOTAL * D_MODEL * 2 / 1e9))
                SQ_ALL.tofile(FEAT_DAT_FOR(DEV0))
            except Exception as e:
                log("  could not write the feature cache (%s); this run is fine"
                    " but a re-run will re-extract" % str(e)[:60])
        FLAT_NP = SQ_ALL
    else:
        IDX_WI = [tuple(r) for r in FEAT_CACHED["idx"]]
        SIDIDS = list(FEAT_CACHED["sidids"])
        SID_SRC = list(FEAT_CACHED["sid_src"])
        CAP_TOTAL = int(FEAT_CACHED["cap"])
        _mm = np.memmap(FEAT_DAT_FOR(DEV0), dtype=np.float16, mode="r",
                        shape=(CAP_TOTAL, D_MODEL))
        if FEAT_MEMMAP:
            FLAT_NP = _mm
        else:
            log("  loading %.1f GB of cached features into RAM..."
                % (CAP_TOTAL * D_MODEL * 2 / 1e9))
            FLAT_NP = np.array(_mm)
            del _mm

    # ---- the relabel. This is the step that makes the cache reusable.
    IDX = relabel(IDX_WI, SID_SRC)
    del IDX_WI
    N_ALIGNED, N_GENS = len(ALIGNED), len(GENS)
    ALIGNED.clear()
    del GENS[:]
    import gc as _gc
    _gc.collect()

    # ---- alignment check, now that labels exist
    _seqs = {i: g for i, g in enumerate(SIDIDS)}
    _rg = np.random.default_rng(5)
    _chk = _rg.choice(len(IDX), min(2000, len(IDX)), replace=False)
    _ok = _q = 0
    for _i in _chk:
        _fr, sid, ti, q, y = IDX[_i]
        g = _seqs.get(sid)
        if g is None or ti + 2 + len(PRE[y]) > len(g):
            continue
        _ok += int(tuple(g[ti + 2:ti + 2 + len(PRE[y])]) == tuple(PRE[y]))
        _q += int(g[ti + 1] == q)
    log("  alignment check on %d rows: unit begins at ti+2 for %.1f%%,"
        " q == x_(ti+1) for %.1f%%"
        % (len(_chk), 100 * _ok / max(len(_chk), 1), 100 * _q / max(len(_chk), 1)))
    assert _q > 0.98 * len(_chk), "q is not the token at ti+1; the pairing is wrong"
    assert _ok > 0.5 * len(_chk), (
        "the drafted unit's expansion begins at ti+2 on only %.1f%% of rows -- the"
        " head is being trained on a different question than it is asked at"
        " inference. If this fired right after a RELABEL, the word-index mapping"
        " is wrong, not the alignment." % (100 * _ok / max(len(_chk), 1)))

    if len(IDX) > MAX_TRAIN_POS:
        s_ = np.random.default_rng(3).choice(len(IDX), MAX_TRAIN_POS, replace=False)
        IDX = [IDX[i] for i in s_]
        log("  row cap binds: subsampled to %d" % len(IDX))

    if FLAT_ON_GPU:
        FLAT = torch.from_numpy(np.asarray(FLAT_NP)).to(DEV0, torch.float16)
    elif FEAT_MEMMAP:
        FLAT = FLAT_NP
    else:
        FLAT = torch.from_numpy(FLAT_NP)
    log("  features ready: %d of %d allocated rows used, %.2f GB %s"
        % (len(IDX), CAP_TOTAL, CAP_TOTAL * D_MODEL * 2 / 1e9,
           "on disk (memmap)" if FEAT_MEMMAP and not FLAT_ON_GPU
           else ("on " + DEV0 if FLAT_ON_GPU else "in RAM")))

    A_ = np.array(IDX, dtype=np.int64)
    nseq = int(A_[:, 1].max()) + 1
    sp = np.random.default_rng(7).permutation(nseq)
    vs = set(sp[:max(1, nseq // 10)].tolist())
    isv = np.array([int(r) in vs for r in A_[:, 1]])
    TR_ALL = A_[~isv]
    VA_ALL = A_[isv][:30000]
    # sort the validation rows by memmap position: evaluation gathers them every
    # epoch, and sorted order turns 30,000 scattered page faults into a near
    # sequential scan.
    VA_ALL = VA_ALL[np.argsort(VA_ALL[:, 0], kind="stable")]
    log("  split by SEQUENCE: %d train / %d val rows from %d / %d sequences"
        % (len(TR_ALL), len(VA_ALL), nseq - len(vs), len(vs)))
    log("  positions per vocabulary entry: %.1f (below ~10 the head memorises --"
        " E67; E85 had 27)" % (len(TR_ALL) / max(len(VOCAB), 1)))
    log("  rows drawn from %d distinct completions, %.0f rows each on average --"
        " and since\n  rows inside one completion are correlated, the SEQUENCE"
        " count is what diversity\n  means here (E85: 10,379 sequences)"
        % (nseq, len(A_) / max(nseq, 1)))
    if len(TR_ALL) / max(len(VOCAB), 1) > 60:
        log("  -> at %.0f positions per entry the ROWS are no longer the binding"
            " constraint:\n     the 40,000-entry vocabulary is. That is one"
            " constant (MAX_VOCAB; 20 M params\n     allows 72,461) and it is the"
            " next experiment, not this one."
            % (len(TR_ALL) / max(len(VOCAB), 1)))
    DIAG.append(dict(check="rows", value=float(len(TR_ALL)),
                     detail="val=%d seqs=%d per_entry=%.2f completions=%d"
                     % (len(VA_ALL), nseq, len(TR_ALL) / max(len(VOCAB), 1),
                        N_ALIGNED)))

    emb = EMBS[DEV0]
    VAL_CHUNK = max(256, int(0.5e9 // (len(VOCAB) * 4)))
    PRE_LEN = torch.tensor([len(x) for x in PRE], device=DEV0, dtype=torch.long)

    # A gather out of pinned memory returns an UNPINNED tensor, so a following
    # .to(non_blocking=True) would be synchronous after all. Gather straight into
    # a preallocated pinned staging buffer instead: that one is pinned, so the
    # transfer really does overlap with the previous step's compute.
    _STAGE = None
    if not FLAT_ON_GPU and not FEAT_MEMMAP:
        try:
            _STAGE = torch.empty((max(BATCH, VAL_CHUNK), D_MODEL),
                                 dtype=torch.float16).pin_memory()
        except Exception as e:
            log("  no pinned staging buffer (%s); transfers will be synchronous"
                % str(e).split("\n")[0][:50])

    def to_gpu(rows_np, hv_np=None):
        """Ship one minibatch to the GPU.

        `hv_np` is the already-gathered hidden vectors when the caller is the
        chunked sampler (which gathers out of its RAM buffer, not out of the
        memmap). Without it, gather here -- fine for a RAM-resident FLAT, and the
        slow path for a memmap."""
        T = torch.from_numpy(np.ascontiguousarray(rows_np)).to(DEV0)
        if hv_np is not None:
            src = torch.from_numpy(hv_np)
            if _STAGE is not None and len(src) <= _STAGE.shape[0]:
                out = _STAGE[:len(src)]
                out.copy_(src)
                return out.to(DEV0, non_blocking=True).float(), T
            return src.to(DEV0).float(), T
        idx = torch.from_numpy(np.ascontiguousarray(rows_np[:, 0]))
        if FLAT_ON_GPU:
            hv = FLAT.index_select(0, idx.to(DEV0))
        elif FEAT_MEMMAP:
            hv = torch.from_numpy(np.asarray(FLAT[rows_np[:, 0]])).to(DEV0)
        elif _STAGE is not None and len(idx) <= _STAGE.shape[0]:
            out = _STAGE[:len(idx)]
            torch.index_select(FLAT, 0, idx, out=out)
            hv = out.to(DEV0, non_blocking=True)
        else:
            hv = FLAT.index_select(0, idx).to(DEV0)
        return hv.float(), T

    @torch.no_grad()
    def evaluate(hd, VA):
        """top-1, recall@8, recall@64 and EBANK -- the expected tokens a top-8
        draft would actually bank. E80 showed ebank is NOT a usable selector (it
        inverted against measured speed), so it is reported as a diagnostic and
        never used to choose. Selection stays on recall@8 (E67)."""
        t1 = rk = r64 = tt = 0
        eb = 0.0
        for s0 in range(0, len(VA), VAL_CHUNK):
            blk = VA[s0:s0 + VAL_CHUNK]
            hv, T = to_gpu(blk, None)
            lo = hd(hv, emb[T[:, 3]].float())
            y = T[:, 4]
            t1 += int((lo.argmax(-1) == y).sum())
            top8 = lo.topk(min(8, lo.shape[-1]), -1).indices
            rk += int((top8 == y[:, None]).any(-1).sum())
            r64 += int((lo.topk(min(64, lo.shape[-1]), -1).indices
                        == y[:, None]).any(-1).sum())
            # tokens banked if the true unit is in the top-8, else 0
            hit = (top8 == y[:, None]).any(-1)
            eb += float((hit.float() * PRE_LEN[y].float()).sum())
            tt += len(blk)
        return t1 / tt, rk / tt, r64 / tt, eb / tt

    def chunk_batches(rows_np, seed):
        """Yield (batch_rows, batch_hidden) with disk reads kept SEQUENTIAL.

        Random access into a 20 GB memmap page-faults per row: a 1024-row batch
        would touch 1024 scattered pages, which on Colab's network-backed disk is
        far slower than the ~25 ms GPU step it feeds. So the walk is organised by
        MEMMAP POSITION, not by row index:
          * sort this rung's rows by their position in the memmap,
          * step through the memmap in CHUNK_ROWS-wide windows, reading each
            window once into a RAM buffer (1.34 GB at CHUNK_ROWS=262,144),
          * emit shuffled batches of whichever rung rows fall inside that window.

        WHY BY POSITION AND NOT BY ROW COUNT. A first version took CHUNK_ROWS
        *rows of the rung* and read from the first to the last of their memmap
        positions. For the 12.5% rung those 262,144 rows are spread over ~2 M
        memmap positions, so the read would have been 10 GB -- a RAM blowup that
        grows worse the SMALLER the rung, which would have hit the cheapest rung
        first and looked like a mysterious crash. Windowing by position makes the
        buffer exactly CHUNK_ROWS wide for every rung.

        A random offset shifts the window boundaries each epoch so the partition
        is not fixed, and the windows are non-overlapping and exhaustive, so no
        row is emitted twice. Partial batches are CARRIED between windows rather
        than dropped: dropping them would discard ~3% of the smallest rung
        against ~0.4% of the largest, which is a confound in a ladder whose whole
        job is comparing rungs.

        The honest caveat: this is weaker than a global shuffle, because rows in
        one window come from nearby completions and are correlated. That is why
        the buffer is deliberately large -- 262,144 rows spans roughly 1,500
        completions -- and why the window order is redrawn every epoch.
        """
        rg = np.random.default_rng(seed)
        if not FEAT_MEMMAP:
            # Features are in RAM: gathering is O(1) per row, so take the real
            # global permutation. The windowing below buys nothing here and costs
            # correlation, because neighbouring rows come from one completion.
            order = rg.permutation(len(rows_np))
            for s0 in range(0, len(rows_np) - BATCH + 1, BATCH):
                yield rows_np[order[s0:s0 + BATCH]], None
            return
        srt = rows_np[np.argsort(rows_np[:, 0], kind="stable")]
        pos = srt[:, 0]
        nmax = int(pos[-1]) + 1
        off = int(rg.integers(0, max(CHUNK_ROWS, 1)))
        edges = [0] + ([off] if 0 < off < nmax else [])
        while edges[-1] + CHUNK_ROWS < nmax:
            edges.append(edges[-1] + CHUNK_ROWS)
        edges.append(nmax)
        wins = [(edges[k], edges[k + 1]) for k in range(len(edges) - 1)
                if edges[k + 1] > edges[k]]
        rg.shuffle(wins)
        car_r, car_h = [], []
        for lo, hi in wins:
            s_, e_ = np.searchsorted(pos, lo, "left"), np.searchsorted(pos, hi,
                                                                       "left")
            blk = srt[s_:e_]
            if len(blk) == 0:
                continue
            buf = np.asarray(FLAT[lo:hi]) if FEAT_MEMMAP else None
            inner = rg.permutation(len(blk))
            for s0 in range(0, len(blk), BATCH):
                sel = blk[inner[s0:s0 + BATCH]]
                hv = buf[sel[:, 0] - lo] if buf is not None else None
                if len(sel) == BATCH:
                    yield sel, hv
                    continue
                car_r.append(sel)
                if hv is not None:
                    car_h.append(hv)
                if sum(len(x) for x in car_r) >= BATCH:
                    R = np.concatenate(car_r)
                    H = np.concatenate(car_h) if car_h else None
                    yield R[:BATCH], (H[:BATCH] if H is not None else None)
                    if len(R) > BATCH:
                        car_r = [R[BATCH:]]
                        car_h = [H[BATCH:]] if H is not None else []
                    else:
                        car_r, car_h = [], []

    def train_head(rows_np, tag):
        """One rung of the ladder. Fresh head, fresh optimiser, same features."""
        hd = Head(D_MODEL, len(VOCAB), H_DIM).to(DEV0)
        npar = sum(p.numel() for p in hd.parameters())
        assert npar < PARAM_BUDGET, (
            "head is %.2f M, over the %.0f M budget" % (npar / 1e6,
                                                        PARAM_BUDGET / 1e6))
        n = len(rows_np)
        opt = torch.optim.AdamW(hd.parameters(), lr=LR, weight_decay=0.01)
        spe = max(n // BATCH, 1)
        warm = max(int(spe * MAX_EPOCHS * WARMUP_FRAC), 1)
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda z: (z + 1) / warm if z < warm else 0.5 * (1 + np.cos(
                np.pi * min((z - warm) / max(spe * MAX_EPOCHS - warm, 1), 1.0))))
        best, bsd, bad = -1.0, None, 0
        a1 = ak = a64 = ebk = ml = 0.0
        ep = 0
        rg_ = np.random.default_rng(11)
        for ep in range(1, MAX_EPOCHS + 1):
            hd.train()
            acc, nb = 0.0, 0
            for blk, hvb in chunk_batches(rows_np, 1000 * ep + 7):
                hv, T = to_gpu(blk, hvb)
                lo = hd(hv, emb[T[:, 3]].float())
                loss = F.cross_entropy(lo, T[:, 4])
                opt.zero_grad(); loss.backward(); opt.step(); sch.step()
                acc += float(loss.detach()); nb += 1
            ml = acc / max(nb, 1)
            hd.eval()
            a1, ak, a64, ebk = evaluate(hd, VA_ALL)
            if ak > best + 1e-4:
                best, bad = ak, 0
                bsd = {k: v.detach().clone() for k, v in hd.state_dict().items()}
            else:
                bad += 1
            CURVE.append(dict(tag=tag, rows=n, epoch=ep, loss=ml, top1=a1,
                              recall8=ak, recall64=a64, ebank8=ebk))
            if ep <= 2 or ep % 4 == 0 or bad >= PATIENCE:
                log("    %-6s ep %2d loss %.4f | top-1 %.3f | recall@8 %.3f |"
                    " recall@64 %.3f | ebank %.3f%s"
                    % (tag, ep, ml, a1, ak, a64, ebk,
                       "  <- best" if bad == 0 else ""))
            if bad >= PATIENCE:
                log("    %-6s early stop ep %d, best recall@8 %.3f"
                    % (tag, ep, best))
                break
            if mins() > T_BUDGET_TRAIN:
                log("    %-6s STOPPING on the clock at ep %d, best recall@8 %.3f"
                    % (tag, ep, best))
                break
        hd.load_state_dict(bsd)
        a1b, akb, a64b, ebkb = evaluate(hd.eval(), VA_ALL)
        return dict(head=hd.eval(), npar=npar, rows=n, tag=tag, epochs=ep,
                    loss=ml, top1=a1b, recall8=akb, recall64=a64b, ebank8=ebkb)

    # ---- the ladder: NESTED subsets, so rung k's rows are a superset of rung k-1's
    log("PHASE 2: training the ladder %s of %d rows | head %.3f M over %d entries"
        % (DATA_FRACS, len(TR_ALL),
           (2 * D_MODEL * H_DIM + H_DIM * H_DIM + 4 * H_DIM
            + len(VOCAB) * (H_DIM + 1)) / 1e6, len(VOCAB)))
    log("  for scale: AngelSlim/Qwen3-4B_eagle3 is %d params = %.3f%% of this"
        " target" % (EAGLE3_PARAMS, 100 * EAGLE3_PARAMS / TGT))
    _perm = np.random.default_rng(23).permutation(len(TR_ALL))
    HEADS = []
    for _f in DATA_FRACS:
        _k = max(BATCH * 2, int(_f * len(TR_ALL)))
        if _k > len(TR_ALL):
            _k = len(TR_ALL)
        _tag = "%d%%" % int(round(100 * _f))
        log("  rung %-5s %d rows (%.0f%% of %d)" % (_tag, _k, 100 * _k
                                                    / len(TR_ALL), len(TR_ALL)))
        HEADS.append(train_head(TR_ALL[_perm[:_k]], _tag))
        # PRE and NWORDS go IN the checkpoint. They are derived from the
        # corpus, not from the head, so without them a bench has to reload the
        # 2.7 GB distillation cache and spend ~4 min rebuilding the vocabulary
        # just to know what each output row expands to -- and the 17.9 GB feature
        # file is too large to move between sessions at all. With them, this
        # ~88 MB file is everything a bench needs.
        torch.save({"model": MODEL, "vocab": VOCAB, "PRE": PRE,
                    "NWORDS": NWORDS,
                    "sd": HEADS[-1]["head"].state_dict(),
                    **{k: v for k, v in HEADS[-1].items() if k != "head"}},
                   str(OUT / ("e87_head_%s.pt" % _tag.replace("%", "pct"))))
        flush()
        if mins() > T_BUDGET_TRAIN:
            log("  stopping the ladder on the clock after rung %s" % _tag)
            break
    HD = HEADS[-1]
    log("head ladder done: %s"
        % ", ".join("%s top-1 %.3f recall@8 %.3f" % (h["tag"], h["top1"],
                                                     h["recall8"])
                    for h in HEADS))
    del FLAT, TR_ALL, VA_ALL
    _gc.collect(); torch.cuda.empty_cache()
    log("  features cached at %s (+ e87_feats_index.npz); keep BOTH and the next"
        "\n  run skips PHASE 2 entirely" % os.path.basename(FEAT_DAT_FOR(DEV0)))

HEAD = {d: (HD["head"] if d == DEV0 else copy.deepcopy(HD["head"]).to(d)).eval()
        for d in DEVS}
flush()
log("head ready: %.3f M params (%.4f%% of target) | top-1 %.3f | recall@8 %.3f"
    % (HD["npar"] / 1e6, 100 * HD["npar"] / TGT, HD["top1"], HD["recall8"]))


# ============================================================ the decode loop
@torch.no_grad()
def run_batch(prompts, drafter, dev, max_new, allow_eos=True, marks=None,
              trace=False):
    """One verifier for every arm.

    Two things are new in E81, both for the length question:

    `allow_eos=False` keeps generating past the target's own EOS so the length axis
    is clean. EOS is NOT masked out of the logits -- the token is still emitted, we
    simply do not stop -- so both sides compute the same function and exactness is
    unaffected. Past EOS the text degenerates, which is EASY to draft, so that arm
    is reported separately and never merged into a headline.

    `marks` collects (depth, cumulative seconds) at every BUCKET boundary, with one
    cuda synchronize per boundary. Eight syncs against ~900 passes is noise, and it
    is what makes the per-depth speedup a MEASUREMENT rather than an extrapolation
    from tok/pass. Only filled for a single sequence, since a per-depth wall clock
    is meaningless once sequences retire at different times.

    `trace=True` records one row per pass for the by-depth tok/pass table. It is
    passed ONLY for arm runs: a baseline emits exactly one token per pass by
    construction, so tracing it would mix a column of 1.000 into the arm's mean."""
    mm = MODELS[dev]
    B0 = len(prompts)
    P = max(len(p) for p in prompts)
    ids = [list(p) for p in prompts]
    x = torch.full((B0, P), PADID, dtype=torch.long)
    kvv = torch.zeros((B0, P), dtype=torch.bool)
    ppos = torch.zeros((B0, P), dtype=torch.long)
    for b, p in enumerate(prompts):
        x[b, P - len(p):] = torch.tensor(p)
        kvv[b, P - len(p):] = True
        ppos[b, P - len(p):] = torch.arange(len(p))
    cache = DynamicCache()
    t0 = clk()
    am = kvv.to(dev).unsqueeze(1).unsqueeze(1).expand(B0, 1, P, P).clone()
    am &= torch.tril(torch.ones((P, P), dtype=torch.bool, device=dev)).view(1, 1, P, P)
    am = torch.where(am, torch.zeros((), dtype=DT, device=dev),
                     torch.full((), NEG, dtype=DT, device=dev))
    hsp = fwd_hidden(mm, dev, x.to(dev), am, ppos.to(dev), cache)
    h = hsp[:, -1].float()
    nxt = argmax_at(hsp[:, -1:], dev, torch.arange(B0, device=dev)).tolist()
    for b in range(B0):
        ids[b].append(int(nxt[b]))
    del hsp, am
    fed = [len(p) for p in prompts]
    ntrue = [len(p) for p in prompts]
    kvalid = kvv.to(dev)
    act = list(range(B0))
    passes, nacc, nwords, nfeed = 1, [], [], 0
    eos_at = [None] * B0
    nextmark = BUCKET
    while act:
        A = len(act)
        trunks = [ids[b][fed[b]:] for b in act]
        tl = torch.tensor([len(t_) for t_ in trunks])
        cands = [[] for _ in range(A)]
        bnds = [[] for _ in range(A)]
        if drafter is not None:
            cands, bnds = drafter(h, torch.tensor([ids[b][-1] for b in act], device=dev), A)
        clen = [[len(blk) for blk in cands[i]] for i in range(A)]
        m, pos, starts, Tmax, Cmax, rok = tree_mask_batch(
            kvalid, tl, clen, torch.tensor([ntrue[b] for b in act]), dev)
        q = Tmax + Cmax
        fnp = np.full((A, q), PADID, dtype=np.int64)
        for i in range(A):
            T = len(trunks[i])
            fnp[i, Tmax - T:Tmax] = trunks[i]
            fl = [tk for blk in cands[i] for tk in blk]
            if fl:
                fnp[i, Tmax:Tmax + len(fl)] = fl
        feed = torch.from_numpy(fnp).to(dev, non_blocking=True)
        nfeed += A * q
        hs = fwd_hidden(mm, dev, feed, m, pos, cache)
        passes += 1
        del m
        base = Tmax - 1
        # ONE lm_head call per pass, not two. E81 called argmax_at for the base
        # slot, used it to decide which blocks were live, then called it AGAIN for
        # those blocks -- each call streaming the whole 151,936 x 2,560 output
        # matrix (778 MB in fp16). Computing all q positions once is a strict
        # SUPERSET of the two-call result (verified on 3,000 random trees in E82).
        # E82 measured this as roughly NEUTRAL rather than the ~9% I predicted: it
        # saves a 778 MB read but costs ~4x the FLOPs, and at batch 1 on a T4 those
        # are the same order. Kept because it is free and marginally better deep.
        if MERGE_LMHEAD:
            allidx = (torch.arange(A, device=dev).unsqueeze(1) * q
                      + torch.arange(q, device=dev).unsqueeze(0)).reshape(-1)
            amap = dict(zip(allidx.tolist(), argmax_at(hs, dev, allidx).tolist()))
            g0 = [amap[i * q + base] for i in range(A)]
            live = [[j for j, blk in enumerate(cands[i]) if blk[0] == g0[i]]
                    for i in range(A)]
        else:
            g0 = argmax_at(hs, dev, torch.arange(A, device=dev) * q + base).tolist()
            sel, live = [], []
            for i in range(A):
                js = [j for j, blk in enumerate(cands[i]) if blk[0] == g0[i]]
                live.append(js)
                for j in js:
                    st = Tmax + starts[i][j]
                    sel.extend(i * q + st + k for k in range(len(cands[i][j])))
            amap = dict(zip(sel, argmax_at(hs, dev,
                                           torch.tensor(sel, device=dev)).tolist())) \
                if sel else {}
        pick = []
        for i, b in enumerate(act):
            bj, bi = 0, -1
            for j in live[i]:
                blk = cands[i][j]; st = Tmax + starts[i][j]; k = 1
                while k < len(blk) and amap.get(i * q + st + k - 1) == blk[k]:
                    k += 1
                if k > bj:
                    bj, bi = k, j
            if bj:
                st = Tmax + starts[i][bi]
                at = cands[i][bi][:bj]
                bonus = amap[i * q + st + bj - 1]
                pick.append(st + bj - 1)
                nwords.append(sum(1 for c in bnds[i][bi] if c <= bj) if bnds[i] else 0)
            else:
                at, bonus = [], g0[i]
                pick.append(base)
                nwords.append(0)
            nacc.append(bj)
            d0 = len(ids[b]) - len(prompts[b])
            fed[b] = len(ids[b])
            ids[b] = ids[b] + at + [bonus]
            ntrue[b] += len(trunks[i])
            if eos_at[b] is None:
                for _k, _t in enumerate(at + [bonus]):
                    if _t in EOS:
                        eos_at[b] = d0 + _k
                        break
            if trace:
                with LK:
                    TRACE.append(dict(tag=CURTAG[0], depth=d0,
                                      bucket=(d0 // BUCKET) * BUCKET,
                                      banked=bj, emitted=bj + 1, fed=q,
                                      ctx=ntrue[b]))
        h = hs[torch.arange(A, device=dev), torch.tensor(pick, device=dev)].float()
        if Cmax:
            cache.crop(-Cmax)
        treal = (torch.arange(Tmax, device=dev).unsqueeze(0)
                 >= (Tmax - tl.to(dev)).unsqueeze(1))
        kvalid = torch.cat([kvalid, treal], 1)
        if marks is not None and B0 == 1:
            dnow = len(ids[0]) - len(prompts[0])
            while dnow >= nextmark:
                torch.cuda.synchronize(dev)
                marks.append((nextmark, clk() - t0))
                nextmark += BUCKET
        # EOS ANYWHERE in what this pass emitted, not only at its end. A drafted
        # block can carry EOS in its interior, and the sequence then never stops:
        # it runs to max_new, cut_eos discards the overshoot, and every pass that
        # produced it stays in the tok/pass denominator. That is how E88's EAGLE
        # arm reported 0.929 tok/pass -- below the arithmetic floor of 1.000.
        # OUR arm is immune today, because a drafted unit is a word phrase and so
        # EOS can only arrive as the bonus token, which the old test did catch.
        # It stops being immune the moment a vocabulary entry can end at EOS.
        # ids[b][fed[b]:] is exactly this pass's output.
        keep = [i for i, b in enumerate(act)
                if len(ids[b]) - len(prompts[b]) < max_new
                and (not allow_eos or not EOS.intersection(ids[b][fed[b]:]))]
        if len(keep) < A:
            if keep:
                kt = torch.tensor(keep, device=dev)
                shrink_cache(cache, kt); kvalid = kvalid[kt]; h = h[kt]
            act = [act[i] for i in keep]
    torch.cuda.synchronize(dev)
    dt = clk() - t0
    gen = [ids[b][len(prompts[b]):len(prompts[b]) + max_new] for b in range(B0)]
    return gen, passes, nacc, nwords, dt, nfeed, eos_at


IS_UNI = torch.tensor([NWORDS[w] == 1 for w in VOCAB])
UNI_BIAS = {d: torch.where(IS_UNI, torch.zeros(len(VOCAB)),
                           torch.full((len(VOCAB),), NEG, dtype=torch.float32)).to(d)
            for d in DEVS}


def mk_width(dev, N, head=None):
    head = head if head is not None else HEAD[dev]
    emb, msk = EMBS[dev], MASK[dev]
    ub = UNI_BIAS[dev]
    nu = min(N_UNI, max(N - 1, 0))

    def drafter(h, last, A):
        lo = head(h.float(), emb[last].float()) + msk
        top = lo.topk(min(N - nu, lo.shape[-1]), -1).indices
        if nu:
            tu = (lo + ub).topk(min(nu, int(IS_UNI.sum())), -1).indices
            top = torch.cat([top, tu], -1)
        out, bd = [], []
        for i in range(A):
            blks, bb, used, seen = [], [], 0, set()
            for r in top[i].tolist():
                ex = PRE[r]
                if not ex or ex in seen:
                    continue
                if used + len(ex) <= MAX_FEED_PER_SEQ - 4:
                    seen.add(ex)
                    blks.append(list(ex)); bb.append([len(ex)]); used += len(ex)
            z = sorted(range(len(blks)), key=lambda j: -len(blks[j]))
            out.append([blks[j] for j in z]); bd.append([bb[j] for j in z])
        return out, bd
    return drafter


def boot_ci(d):
    """Bootstrap of the MEAN of per-prompt values. Correct for the mean of
    per-prompt speedups; NOT the interval for an aggregate ratio of sums -- use
    boot_ratio_ci for that."""
    if len(d) < 2:
        return float("nan"), float("nan")
    r = np.random.default_rng(11)
    bs = np.array([r.choice(d, len(d), replace=True).mean() for _ in range(BOOT)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def boot_ratio_ci(b, a):
    """Bootstrap of sum(b)/sum(a), resampling PROMPTS -- the interval that belongs
    to the aggregate speedup. Resampling the per-prompt ratios instead and taking
    their mean estimates a different quantity, and printing that interval beside
    the aggregate point estimate produced a CI that excluded it (E89 MATH-500:
    2.032 against [2.114, 2.280]). Report both, each with its own bootstrap."""
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)
    assert b.shape == a.shape, "paired arrays must align, got %s and %s" % (
        b.shape, a.shape)
    if b.size < 2:
        return float("nan"), float("nan")
    r = np.random.default_rng(11)
    ix = r.integers(0, b.size, (BOOT, b.size))
    s = b[ix].sum(1) / np.maximum(a[ix].sum(1), 1e-9)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def enc_prompt(t):
    """Truncate the prompt TEXT, then apply the chat template, so the generation
    prompt's trailing turn marker is never cut off."""
    e = Enc(chat(t))
    if len(e) <= MAX_PROMPT_TOK:
        return e
    lo, hi = 0, len(t)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(Enc(chat(t[:mid]))) <= MAX_PROMPT_TOK:
            lo = mid
        else:
            hi = mid - 1
    return Enc(chat(t[:lo]))


POOLS = {k: [enc_prompt(p) for p in v] for k, v in TESTS_BY_SET.items()}
POOLS_LONG = {k: [enc_prompt(p + LONG_SUFFIX) for p in v]
              for k, v in TESTS_BY_SET.items()}
log("prompt budget %d tokens | timing pools %s"
    % (MAX_PROMPT_TOK, {k: len(v) for k, v in POOLS.items()}))



# ================================================= PHASE W: does the window bind?
# Only runs when the target windows its attention BELOW the generation length.
# Qwen3-4B does not window at all, so this is a no-op there and the real purpose
# is to make the Gemma-3-1B switch safe: at 2048 tokens Gemma's 512-token window
# binds on 22 of 26 layers, and we hand transformers a 4D additive mask. If that
# mask REPLACES the library's own sliding-window mask rather than composing with
# it, our verification pass would attend past the window while the plain baseline
# forward does not -- and the result would be silently non-exact.
#
# The test does not reason about it, it measures it: take one prompt, decode past
# the window with the cached baseline, then recompute the final logits with a
# single plain uncached forward (which definitely applies the library's own
# windowing) and compare the argmax.
if WIN_BINDS:
    log("PHASE W: the window binds -- testing whether our 4D mask SUPPRESSES it")
    _wp = POOLS["gsm8k-test"][0]
    _need = int(SW) + 2 * BUCKET
    _wgo, _, _, _, _, _, _ = run_batch([_wp], None, DEV0, _need, allow_eos=False)
    _wg = _wgo[0]
    assert len(_wg) >= _need, (
        "the window test only reached %d of the %d tokens it needs to get past the"
        " %s-token window" % (len(_wg), _need, SW))
    _full = _wp + _wg[:_need - 1]
    _lg = MODELS[DEV0](input_ids=torch.tensor([_full]).to(DEV0)).logits[0, -1].float()
    _ref = int(_lg.argmax())
    _tv = _lg.topk(2).values
    _mg = float(_tv[0] - _tv[1])
    _agree = _ref == _wg[_need - 1]
    log("  at depth %d (window %s): cached-4D-mask token %d, plain forward token %d,"
        " margin %.4f -> %s" % (_need - 1, SW, _wg[_need - 1], _ref, _mg,
                                "AGREE" if _agree else "DISAGREE"))
    DIAG.append(dict(check="window_binds", value=float(_agree),
                     detail="depth=%d window=%s margin=%.4f" % (_need - 1, SW, _mg)))
    assert _agree or _mg < MARGIN_FAIL, (
        "ABORT -- past the %s-token sliding window our 4D-masked decode and a plain"
        " forward pass disagree at a CONFIDENT position (margin %.4f). That means the"
        " 4D mask we pass is replacing the library's sliding-window mask instead of"
        " composing with it, so the verification pass sees context the baseline does"
        " not and NOTHING past token %s is greedy-exact. Compose the window into"
        " tree_mask_batch before running any timing on this target."
        % (SW, _mg, SW))
    log("  the 4D mask composes with the window correctly; long generation is safe"
        " on this target")


# ============================================= PHASE 0b: the decode-control floor
# E79's floor control was broken -- it reported divergence at token 0 with margins
# of 2.0-4.6, which is not an fp16 story, so it was measuring something else and
# could not be used to attribute E79's one confident arm-side divergence. It also
# BUNDLED three different things into one comparison, which is the bug family this
# log keeps recording. So it is decomposed here into three separately reported
# quantities, each with exactly one possible cause:
#
#   (1) ARGMAX PATH. Our two-stage argmax (hidden state -> LMW matmul, chunked)
#       against the model's own lm_head on the SAME hidden state. Isolates our
#       matmul and nothing else. If this fails, EVERY number in this programme is
#       suspect and the run should stop.
#   (2) CACHE DRIFT. Our cached baseline against a fully uncached re-prefill decode.
#       Isolates the KV cache. Whatever this reports is the FLOOR: the arm cannot be
#       expected to agree with the baseline more closely than the baseline agrees
#       with an uncached reference.
#   (3) TREE PATH, later, in the bench: arm against baseline. Only interpretable
#       once (1) and (2) are known.
log("PHASE 0b: decode controls, decomposed (E79's version bundled all three)")
# E81 took ONE prompt per test set, so FLOOR_N=4 actually sampled 2 and the floor
# was underpowered. Several per set now.
_fp = [q for k in POOLS for q in POOLS[k][:3]][:FLOOR_N]


@torch.no_grad()
def uncached_greedy(prompt, n, dev):
    """Fully uncached reference: re-prefill the whole sequence for every token.
    O(L^2) and far too slow to use as a baseline, but it is the only decode with no
    cache to drift, so it is the right thing to measure cached decode against."""
    ids = list(prompt)
    for _ in range(n):
        lg = MODELS[dev](input_ids=torch.tensor([ids]).to(dev)).logits[0, -1]
        t = int(lg.argmax())
        ids.append(t)
        if t in EOS:
            break
    return ids[len(prompt):]


# ---- (1) the argmax path, on identical hidden states
_amax_bad, _amax_n, _amax_marg = 0, 0, []
for _p in _fp:
    _o = MODELS[DEV0](input_ids=torch.tensor([_p]).to(DEV0), output_hidden_states=True)
    _hs = _o.hidden_states[-1]
    _mine = int(argmax_at(_hs[:, -1:], DEV0, torch.tensor([0], device=DEV0))[0])
    _lg = _o.logits[0, -1].float()
    _theirs = int(_lg.argmax())
    _tv = _lg.topk(2).values
    _amax_n += 1
    if _mine != _theirs:
        _amax_bad += 1
        _amax_marg.append(float(_tv[0] - _tv[1]))
    del _o, _hs
log("  (1) argmax path: %d of %d prompts disagree with the model's own lm_head%s"
    % (_amax_bad, _amax_n,
       (" (margins %s)" % ["%.4f" % m for m in _amax_marg]) if _amax_marg else ""))
DIAG.append(dict(check="argmax_path", value=_amax_bad / max(_amax_n, 1),
                 detail="margins=%s" % ["%.4f" % m for m in _amax_marg]))
assert _amax_bad == 0 or max(_amax_marg) < MARGIN_FAIL, (
    "ABORT -- our two-stage argmax disagrees with the model's own lm_head at a"
    " CONFIDENT position (margin %.4f) on the same hidden state. That is not an fp16"
    " story and it is not specific to speculation: it would corrupt the BASELINE as"
    " well as every arm, in this run and in every run that used argmax_at. Fix this"
    " before anything else is interpreted." % (max(_amax_marg) if _amax_marg else 0))

# ---- (2) cache drift: our cached baseline vs a fully uncached decode
FLOOR = []
for _i, _p in enumerate(_fp):
    _g, _, _, _, _, _, _ = run_batch([_p], None, DEV0, FLOOR_NEW)
    _u = uncached_greedy(_p, FLOOR_NEW, DEV0)
    _gc, _uc = cut_eos(_g[0]), cut_eos(_u)
    _d = next((k for k in range(min(len(_gc), len(_uc))) if _gc[k] != _uc[k]), None)
    if _d is None:
        log("  (2) prompt %d: cached == uncached over %d tokens"
            % (_i, min(len(_gc), len(_uc))))
        continue
    _lf = MODELS[DEV0](input_ids=torch.tensor([_p + _uc[:_d]]).to(DEV0)
                       ).logits[0, -1].float()
    _tv = _lf.topk(2).values
    _mg = float(_tv[0] - _tv[1])
    FLOOR.append(dict(prompt=_i, depth=_d, margin=_mg))
    log("  (2) prompt %d: cached decode diverges from uncached at token %d of %d,"
        " margin %.4f" % (_i, _d, min(len(_gc), len(_uc)), _mg))
FLOOR_CONF = sum(1 for f in FLOOR if f["margin"] >= MARGIN_FAIL)
log("  (2) cache-drift floor: %d of %d prompts diverge by %d tokens, %d of them at a"
    " CONFIDENT position" % (len(FLOOR), len(_fp), FLOOR_NEW, FLOOR_CONF))
log("      This is the FLOOR. The arm cannot be held to a stricter standard than the"
    " baseline\n      itself meets against an uncached reference, and any arm"
    " divergence at a comparable\n      depth and margin is this, not the verifier.")
DIAG.append(dict(check="cache_floor", value=float(len(FLOOR)) / max(len(_fp), 1),
                 detail="confident=%d depths=%s margins=%s"
                 % (FLOOR_CONF, [f["depth"] for f in FLOOR],
                    ["%.3f" % f["margin"] for f in FLOOR])))
flush()


@torch.no_grad()
def exact_check(g, a, p, name, ts, idx, dev, cut=None):
    """Classify a divergence instead of attributing it.

    Both sides come from OUR loop, so a mismatch does not by itself say which is
    wrong. One fresh UNCACHED forward over prompt + common prefix settles it:
    whichever side equals that pass's argmax is correct.
      side="arm"   baseline matches the fresh pass -> the verifier is wrong
      side="base"  the arm matches it -> our cached baseline drifted
      side="both-wrong"  neither matches -> cached/uncached disagree outright
    Nothing is asserted; divergences are judged together against the PHASE 0b
    floor, because one sample cannot distinguish a systematic verifier bug from a
    handful of low-margin fp16 coincidences."""
    cut = cut or cut_eos
    g, a = cut(g), cut(a)
    if g == a:
        return 0
    d = next((k for k in range(min(len(g), len(a))) if g[k] != a[k]), None)
    if d is None:
        with LK:
            DIVERGE.append(dict(arm=name, test=ts, seq=idx, kind="length", at=-1,
                                margin=float("nan"), side="length",
                                depth=min(len(g), len(a))))
        return 1
    lf = MODELS[dev](input_ids=torch.tensor([p + g[:d]]).to(dev)).logits[0, -1].float()
    tv, ti2 = lf.topk(2)
    mg = float(tv[0] - tv[1])
    ref = int(ti2[0])
    side = ("arm" if ref == g[d] else "base" if ref == a[d] else "both-wrong")
    with LK:
        DIVERGE.append(dict(arm=name, test=ts, seq=idx, kind="token", at=d,
                            margin=mg, side=side, depth=d,
                            ref=ref, base_tok=g[d], arm_tok=a[d]))
    return 1


# ====================================================== PHASE 3: the length bench
def bench(ts, pool, nmax, tag, allow_eos, nprompt):
    """Batch 1. The baseline runs once per prompt and the arm immediately after, so
    both share the same back-to-back conditions (E67). Depth marks are collected for
    both sides, which is what gives a per-depth speedup rather than an aggregate."""
    drs = [("N%d" % n, mk_width(DEV0, n)) for n in N_SCAN]
    ag = {nm: dict(sec=0.0, tok=0, ps=0, acc=[], feed=0, ndiv=0, eos=[])
          for nm, _ in drs}
    bs, bt, blen = 0.0, 0, []
    pool = pool[:nprompt]
    # In forced mode neither side stops at EOS, so both run the full length and the
    # raw lists are the right things to compare and to count. Cutting there would
    # divide the arm's FULL pass count by a TRUNCATED token count.
    cut = cut_eos if allow_eos else (lambda g: list(g))
    CURTAG[0] = tag
    for k, p in enumerate(pool):
        bmk = []
        go, bps, _, _, bdt, _, beos = run_batch([p], None, DEV0, nmax,
                                                allow_eos=allow_eos, marks=bmk)
        gcut = cut(go[0])
        bs += bdt; bt += len(gcut); blen.append(len(gcut))
        for nm, dr in drs:
            amk = []
            ao, pq, na, nw, adt, nf, aeos = run_batch([p], dr, DEV0, nmax,
                                                      allow_eos=allow_eos, marks=amk,
                                                      trace=True)
            a = ag[nm]
            a["ndiv"] += exact_check(go[0], ao[0], p, nm, ts, k, DEV0, cut)
            a["sec"] += adt; a["tok"] += len(cut(ao[0])); a["ps"] += pq
            a["acc"] += na; a["feed"] += nf
            a["eos"].append(aeos[0] if aeos[0] is not None else -1)
            with LK:
                PAIRS.append(dict(test=ts, arm=nm, tag=tag, seq=k,
                                  base_secs=bdt, arm_secs=adt, speedup=bdt / adt,
                                  gen_len=len(gcut),
                                  eos_at=beos[0] if beos[0] is not None else -1))
                # per-depth speedup from the paired marks: same prompt, same head,
                # same session -- the only thing that differs is how deep we are.
                bd_, ad_ = dict(bmk), dict(amk)
                for dep in sorted(set(bd_) & set(ad_)):
                    pb = bd_[dep] - bd_.get(dep - BUCKET, 0.0)
                    pa = ad_[dep] - ad_.get(dep - BUCKET, 0.0)
                    if pa > 0 and pb > 0:
                        MARKS.append(dict(test=ts, arm=nm, tag=tag, seq=k,
                                          bucket=dep - BUCKET, base_secs=pb,
                                          arm_secs=pa, speedup=pb / pa))
        if k % 4 == 0 or k == len(pool) - 1:
            log("    %-12s %-7s %d/%d | gen %d tok [%5.1fm]"
                % (ts, tag, k + 1, len(pool), len(gcut), mins()))
        flush()
        if mins() > T_BUDGET:
            log("    %-12s %-7s BUDGET REACHED after %d prompts" % (ts, tag, k + 1))
            pool = pool[:k + 1]
            break
    for nm, _ in drs:
        a = ag[nm]
        ar = np.array(a["acc"]) if a["acc"] else np.array([0])
        with LK:
            ROWSX.append(dict(test=ts, arm=nm, tag=tag, N=int(nm[1:]), max_new=nmax,
                              allow_eos=allow_eos, n_prompts=len(pool),
                              base_secs=bs, arm_secs=a["sec"], tokens=bt,
                              speedup=bs / max(a["sec"], 1e-9),
                              base_tok_s=bt / max(bs, 1e-9),
                              tok_s=a["tok"] / max(a["sec"], 1e-9),
                              tok_per_pass=bt / max(a["ps"], 1),
                              feed_per_seq=a["feed"] / max(a["ps"], 1),
                              landing=float((ar > 0).mean()),
                              gen_len_mean=float(np.mean(blen)) if blen else 0.0,
                              gen_len_max=int(np.max(blen)) if blen else 0,
                              eos_depth_mean=float(np.mean([e for e in a["eos"]
                                                            if e >= 0]))
                              if any(e >= 0 for e in a["eos"]) else -1.0,
                              ndiverge=a["ndiv"]))
    flush()


# ===================================================== PHASE 2b: tree-mask gate
log("PHASE 2b: tree-mask gate under %r" % ATTN)
_n0 = len(DIVERGE)
_gq = POOLS["gsm8k-test"][0]
_gb, _, _, _, _, _, _ = run_batch([_gq], None, DEV0, 256)
_ga, _, _, _, _, _, _ = run_batch([_gq], mk_width(DEV0, N_SCAN[0]), DEV0, 256)
exact_check(_gb[0], _ga[0], _gq, "gate", "gsm8k-test", -1, DEV0)
_bad = [d for d in DIVERGE[_n0:] if d.get("side") == "arm"
        and d.get("margin", 0) >= MARGIN_FAIL]
log("  %d divergences on one 256-token prompt, %d CONFIDENT arm-side"
    % (len(DIVERGE) - _n0, len(_bad)))
DIAG.append(dict(check="tree_gate", value=float(not _bad),
                 detail="impl=%s confident_arm=%d" % (ATTN, len(_bad))))
assert not _bad, (
    "ABORT -- %r is not honouring the tree mask (margin %.4f at depth %d)."
    % (ATTN, _bad[0]["margin"] if _bad else 0, _bad[0]["depth"] if _bad else 0))
flush()


# ============================ PHASE 3: bench every rung on the SAME baseline
# One baseline per prompt, shared by every rung, so the rungs differ only in the
# head. Four proxies have already inverted against measured speed (findings
# 8.23), so a rung's offline numbers are never allowed to stand in for this.
log("PHASE 3: benching %d rungs on GSM8K at natural length, N=%d"
    % (len(HEADS), N_SCAN[0]))
_pool = POOLS["gsm8k-test"][:LADDER_BENCH_PROMPTS]
_bas = {}
for _k, _pm in enumerate(_pool):
    _go, _, _, _, _bdt, _, _ = run_batch([_pm], None, DEV0, MAX_NEW_NAT)
    _bas[_k] = (cut_eos(_go[0]), _bdt)
log("  %d baselines, %.1f s total [%5.1fm]"
    % (len(_bas), sum(v[1] for v in _bas.values()), mins()))

for _h in HEADS:
    agg = dict(sec=0.0, bsec=0.0, tok=0, ps=0, acc=[], feed=0, ndiv=0)
    dr = mk_width(DEV0, N_SCAN[0], head=_h["head"])
    CURTAG[0] = _h["tag"]
    for _k, _pm in enumerate(_pool):
        go, bdt = _bas[_k]
        ao, pq, na, nw, adt, nf, _ = run_batch([_pm], dr, DEV0, MAX_NEW_NAT,
                                               trace=True)
        agg["ndiv"] += exact_check(go, ao[0], _pm, _h["tag"], "gsm8k-test", _k,
                                   DEV0)
        agg["sec"] += adt; agg["bsec"] += bdt
        agg["tok"] += len(cut_eos(ao[0])); agg["ps"] += pq
        agg["acc"] += na; agg["feed"] += nf
        PAIRS.append(dict(test="gsm8k-test", arm=_h["tag"], tag="nat", seq=_k,
                          base_secs=bdt, arm_secs=adt, speedup=bdt / adt,
                          gen_len=len(go), eos_at=-1))
    ar = np.array(agg["acc"]) if agg["acc"] else np.array([0])
    tpp = sum(len(v[0]) for v in _bas.values()) / max(agg["ps"], 1)
    fed = agg["feed"] / max(agg["ps"], 1)
    LADDER.append(dict(tag=_h["tag"], rows=_h["rows"], npar=_h["npar"],
                       epochs=_h["epochs"], loss=_h["loss"], top1=_h["top1"],
                       recall8=_h["recall8"], recall64=_h["recall64"],
                       ebank8=_h["ebank8"],
                       speedup=agg["bsec"] / max(agg["sec"], 1e-9),
                       tok_per_pass=tpp, feed_per_pass=fed,
                       banked_per_fed=(tpp - 1) / max(fed, 1e-9),
                       landing=float((ar > 0).mean()), ndiverge=agg["ndiv"]))
    log("  %-6s rows %-9d top-1 %.3f recall@8 %.3f | speedup %.3fx tok/pass %.3f"
        " fed %.1f banked/fed %.4f [%5.1fm]"
        % (_h["tag"], _h["rows"], _h["top1"], _h["recall8"],
           LADDER[-1]["speedup"], tpp, fed, LADDER[-1]["banked_per_fed"], mins()))
    flush()

# ============ PHASE 3c: the N sweep on BOTH sets, for the best rung only
# The protocol is EOS-TERMINATED WITH A 2048 CAP at temperature 0, which is
# DFlash's Table 1 protocol exactly, so E88's numbers drop into their table with
# no adjustment. Forced-2048 stays a diagnostic (PHASE 3b) and may never be a
# headline: EOS falls at ~233 tokens on GSM8K, so ~89% of a forced run is
# post-EOS filler.
LD = pd.DataFrame(LADDER)
if not LD.empty:
    _best_tag = LD.loc[LD["speedup"].idxmax(), "tag"]
    _BH = next(h for h in HEADS if h["tag"] == _best_tag)
    log("PHASE 3c: best rung is %r; sweeping N=%s on GSM8K and MATH-500,"
        " EOS-terminated" % (_best_tag, N_SCAN))
    for _ts in [t for t in ("gsm8k-test", "math500") if t in POOLS]:
        if mins() > T_BUDGET - 25:
            log("  skipping %s: budget" % _ts)
            continue
        _pl = POOLS[_ts][:LADDER_BENCH_PROMPTS]
        _bb = {}
        for _k, _pm in enumerate(_pl):
            _go, _, _, _, _bdt, _, _ = run_batch([_pm], None, DEV0, MAX_NEW_NAT)
            _bb[_k] = (cut_eos(_go[0]), _bdt)
        _blen = float(np.mean([len(v[0]) for v in _bb.values()]))
        log("  %-11s %d baselines, %.1f s, mean achieved length %.0f tokens"
            " [%5.1fm]" % (_ts, len(_bb), sum(v[1] for v in _bb.values()),
                           _blen, mins()))
        for _n in N_SCAN:
            dr = mk_width(DEV0, _n, head=_BH["head"])
            CURTAG[0] = "N%d" % _n
            ag = dict(sec=0.0, bsec=0.0, tok=0, ps=0, acc=[], feed=0, ndiv=0)
            for _k, _pm in enumerate(_pl):
                go, bdt = _bb[_k]
                ao, pq, na, nw, adt, nf, _ = run_batch([_pm], dr, DEV0,
                                                       MAX_NEW_NAT, trace=True)
                ag["ndiv"] += exact_check(go, ao[0], _pm, "N%d" % _n, _ts, _k,
                                          DEV0)
                ag["sec"] += adt; ag["bsec"] += bdt
                ag["tok"] += len(cut_eos(ao[0])); ag["ps"] += pq
                ag["acc"] += na; ag["feed"] += nf
                PAIRS.append(dict(test=_ts, arm="N%d" % _n, tag="nsweep", seq=_k,
                                  base_secs=bdt, arm_secs=adt,
                                  speedup=bdt / adt, gen_len=len(go), eos_at=-1))
            ar = np.array(ag["acc"]) if ag["acc"] else np.array([0])
            tpp = sum(len(v[0]) for v in _bb.values()) / max(ag["ps"], 1)
            fed = ag["feed"] / max(ag["ps"], 1)
            ROWSX.append(dict(test=_ts, arm="N%d" % _n, tag="nsweep", N=_n,
                              max_new=MAX_NEW_NAT, allow_eos=True,
                              n_prompts=len(_pl), base_secs=ag["bsec"],
                              arm_secs=ag["sec"], tokens=ag["tok"],
                              speedup=ag["bsec"] / max(ag["sec"], 1e-9),
                              base_tok_s=0.0, tok_s=0.0, tok_per_pass=tpp,
                              feed_per_seq=fed,
                              landing=float((ar > 0).mean()),
                              gen_len_mean=_blen, gen_len_max=int(
                                  max(len(v[0]) for v in _bb.values())),
                              eos_depth_mean=-1.0, ndiverge=ag["ndiv"]))
            log("    N=%-3d %-11s speedup %.3fx | tok/pass %.3f | fed %.1f |"
                " banked/fed %.4f | diverge %d [%5.1fm]"
                % (_n, _ts, ROWSX[-1]["speedup"], tpp, fed,
                   (tpp - 1) / max(fed, 1e-9), ag["ndiv"], mins()))
            flush()
BEST = None
if not LD.empty:
    BEST = LD.loc[LD["speedup"].idxmax(), "tag"]
    HD = next(h for h in HEADS if h["tag"] == BEST)
    log("PHASE 3b: best rung by MEASURED speedup is %r; running the forced-2048"
        " depth curve on it" % BEST)
    if mins() < T_BUDGET - 20:
        dr = mk_width(DEV0, N_SCAN[0], head=HD["head"])
        CURTAG[0] = "deep"
        for _k, _pm in enumerate(POOLS["gsm8k-test"][:DEEP_BENCH_PROMPTS]):
            bmk = []
            go, _, _, _, bdt, _, _ = run_batch([_pm], None, DEV0, MAX_NEW_FORCED,
                                               allow_eos=False, marks=bmk)
            amk = []
            ao, pq, na, nw, adt, nf, _ = run_batch([_pm], dr, DEV0,
                                                   MAX_NEW_FORCED,
                                                   allow_eos=False, marks=amk,
                                                   trace=True)
            exact_check(go[0], ao[0], _pm, "deep", "gsm8k-test", _k, DEV0,
                        lambda g: list(g))
            bd_, ad_ = dict(bmk), dict(amk)
            for dep in sorted(set(bd_) & set(ad_)):
                pb = bd_[dep] - bd_.get(dep - BUCKET, 0.0)
                pa = ad_[dep] - ad_.get(dep - BUCKET, 0.0)
                if pa > 0 and pb > 0:
                    MARKS.append(dict(test="gsm8k-test", arm="deep", tag="forced",
                                      seq=_k, bucket=dep - BUCKET, base_secs=pb,
                                      arm_secs=pa, speedup=pb / pa))
            ROWSX.append(dict(test="gsm8k-test", arm="deep", tag="forced",
                              N=N_SCAN[0], max_new=MAX_NEW_FORCED,
                              allow_eos=False, n_prompts=1, base_secs=bdt,
                              arm_secs=adt, tokens=len(go[0]),
                              speedup=bdt / adt, base_tok_s=0.0, tok_s=0.0,
                              tok_per_pass=len(go[0]) / max(pq, 1),
                              feed_per_seq=nf / max(pq, 1),
                              landing=float((np.array(na) > 0).mean()),
                              gen_len_mean=len(go[0]), gen_len_max=len(go[0]),
                              eos_depth_mean=-1.0, ndiverge=0))
            log("    deep %d/%d %.3fx [%5.1fm]"
                % (_k + 1, DEEP_BENCH_PROMPTS, bdt / adt, mins()))
            flush()
            if mins() > T_BUDGET - 4:
                break
    else:
        log("  SKIPPED on the clock")


# ============================================================ report
R = pd.DataFrame(ROWSX)
M = pd.DataFrame(MARKS)
D = pd.DataFrame(DIVERGE)
CV = pd.DataFrame(CURVE)
LD = pd.DataFrame(LADDER)
flush()
# E85, measured: the curve this run extends. 10,379 completions, 1,080,346 rows,
# MAX_GRAM=3, 40,000-entry vocabulary, the same head shape.
E85_LADDER = [(270086, 0.284, 0.573, 2.184, 2.380),
              (540173, 0.318, 0.631, 2.201, 2.432),
              (1080346, 0.350, 0.684, 2.301, 2.530)]
E85_FORCED = {0: 2.010, 256: 1.799, 512: 1.649, 768: 1.635, 1024: 1.740,
              1280: 1.611, 1536: 1.470, 1792: 1.536}
RIVALS = [("DFlash", 16, 6.53), ("EAGLE-3 (16)", 16, 3.30),
          ("EAGLE-3 (60)", 60, 3.77)]
print("\n" + "=" * 78, flush=True)
print("E86 -- SCALE: %d completions, features out of core" % N_GENS, flush=True)
print("=" * 78, flush=True)
print("  %s | attention %r | batch 1 | N=%d | MAX_GRAM=%d"
      % (MODEL, ATTN, N_SCAN[0], MAX_GRAM), flush=True)
try:
    _gpu = torch.cuda.get_device_name(0)
    _vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    _cc = torch.cuda.get_device_capability(0)
except Exception:
    _gpu, _vram, _cc = "unknown", float("nan"), (0, 0)
print("  %s, %.0f GB VRAM, sm%d%d | features %s | distil batch %d"
      % (_gpu, _vram, _cc[0], _cc[1],
         "memmap" if FEAT_MEMMAP else "in RAM", GEN_BATCH), flush=True)
print("  %d completions (E85: 10,379) | vocabulary %d of a %d cap | head %.3f M"
      % (N_GENS, len(VOCAB), MAX_WORDS, HD["npar"] / 1e6), flush=True)
print("  expansion %.3f | coverage %.3f | SLEM gap %.1f%%"
      % (EXP, COV_W, 100 * agree), flush=True)
_rw = [d for d in DIAG if d.get("check") == "rows"]
if _rw:
    print("  %s" % _rw[0]["detail"], flush=True)
_tr = [d for d in DIAG if d.get("check") == "truncation"]
if _tr:
    print("  truncation: %.1f%% of completions hit the %d-token cap (E85 at 192:"
          " 70.8%%)" % (100 * float(_tr[0]["value"]), DISTILL_NEW), flush=True)

if not LD.empty:
    print("\n=== THE LADDER, E85's three rungs plus this run's ===", flush=True)
    print("  Nested subsets of ONE feature set; rungs differ only in rows seen.",
          flush=True)
    print("  !! E85 ran on a T4 and this run did not. top-1, recall@8 and tok/pass"
          " are\n  acceptance counts and compare directly; SPEEDUP DOES NOT -- the"
          " affordable-feed\n  constant I = FLOPS/bandwidth is ~203 on a T4 and"
          " ~152 here (findings 8.16), so\n  this card affords fewer fed tokens"
          " per sequence and the whole speedup column\n  shifts. Compare speedups"
          " only WITHIN this run.", flush=True)
    print("  %-7s %-11s %-8s %-10s %-9s %-10s %s"
          % ("run", "rows", "top-1", "recall@8", "speedup", "tok/pass", "rung"),
          flush=True)
    for rows, t1, r8, su, tp in E85_LADDER:
        print("  %-7s %-11d %-8.3f %-10.3f %-9.3f %-10.3f %s"
              % ("E85", rows, t1, r8, su, tp, ""), flush=True)
    for _, r in LD.iterrows():
        print("  %-7s %-11d %-8.3f %-10.3f %-9.3f %-10.3f %s"
              % ("E86", r["rows"], r["top1"], r["recall8"], r["speedup"],
                 r["tok_per_pass"], r["tag"]), flush=True)

    print("\n  marginal gain in recall@8 per doubling of rows:", flush=True)
    pts = [(rows, r8) for rows, _t, r8, _s, _p in E85_LADDER] + \
          [(int(r["rows"]), r["recall8"]) for _, r in LD.iterrows()]
    pts.sort()
    prev = None
    for k in range(1, len(pts)):
        if pts[k - 1][0] <= 0:
            continue
        ratio = pts[k][0] / pts[k - 1][0]
        if ratio < 1.5:
            continue
        d = pts[k][1] - pts[k - 1][1]
        per = d / np.log2(ratio)
        tag = ""
        if prev is not None:
            tag = "  (%.2fx the previous step)" % (per / prev) if abs(prev) > 1e-9 \
                else ""
        print("    %9d -> %-9d  %+.4f recall@8  = %+.4f per doubling%s"
              % (pts[k - 1][0], pts[k][0], d, per, tag), flush=True)
        prev = per
    if prev is not None:
        first = None
        for k in range(1, len(pts)):
            r_ = pts[k][0] / max(pts[k - 1][0], 1)
            if r_ >= 1.5:
                first = (pts[k][1] - pts[k - 1][1]) / np.log2(r_)
                break
        if first:
            print("\n  last step is %.2fx the first step's gain per doubling."
                  % (prev / first), flush=True)
            if prev > 0.5 * first:
                print("  -> STILL NOT BENT. Data remains the binding constraint;"
                      " raise DISTILL_N again\n     and/or lift the 40,000-entry"
                      " vocabulary cap (positions per entry is now the number"
                      "\n     to watch -- well above E85's 27 means the VOCABULARY"
                      " binds, not the rows).", flush=True)
            elif prev > 0.15 * first:
                print("  -> BENDING. Diminishing but still worth one more"
                      " doubling. The vocabulary cap\n     is now the more likely"
                      " binding constraint: check positions per entry.",
                      flush=True)
            else:
                print("  -> FLATTENED. More completions will not help this"
                      " vocabulary. The next levers are\n     a bigger vocabulary"
                      " (40,000 was a chosen cap, 20 M params allows 72,461) and"
                      "\n     the drafting policy (fewer, better candidates --"
                      " E84).", flush=True)

    print("\n=== FEED EFFICIENCY ===", flush=True)
    print("  %-16s %-10s %-14s %-14s" % ("method", "feed", "accepted/pass",
                                         "banked per fed tok"), flush=True)
    for _, r in LD.iterrows():
        print("  %-16s %-10.1f %-14.3f %-14.4f"
              % ("ours " + r["tag"], r["feed_per_pass"], r["tok_per_pass"],
                 r["banked_per_fed"]), flush=True)
    print("  %-16s %-10.1f %-14.3f %-14.4f" % ("ours E85 100%", 66.8, 2.530,
                                               0.0229), flush=True)
    for nm, tree, tau in RIVALS:
        print("  %-16s %-10d %-14.2f %-14.4f  (published)"
              % (nm, tree, tau, (tau - 1) / tree), flush=True)
    print("  N is pinned at %d in every rung, so this measures 'more tok/pass at"
          " the SAME feed'.\n  Whether a better head lets us feed LESS is E84's"
          " surface, not this run." % N_SCAN[0], flush=True)

if not CV.empty and "tag" in CV.columns:
    print("\n--- training curves: is it still overfitting? ---", flush=True)
    for tg in CV["tag"].unique():
        g = CV[CV["tag"] == tg]
        pk = int(g.loc[g["recall8"].idxmax(), "epoch"])
        print("  %-6s %2d epochs, best at %-3d | loss %.3f -> %.3f | recall@8"
              " %.3f -> %.3f%s"
              % (tg, len(g), pk, g["loss"].iloc[0], g["loss"].iloc[-1],
                 g["recall8"].iloc[0], g["recall8"].max(),
                 "   <- peaked early, still data-limited" if pk < len(g) - 2
                 else "   <- ran to the end"), flush=True)
    print("  Training loss RANKS THE RUNGS BACKWARDS by design: it measures fit to"
          "\n  the training set, and more data makes memorisation harder. recall@8"
          " on held-out\n  SEQUENCES is the progress measure (E67, findings 8.23)."
          , flush=True)

if not M.empty:
    print("\n--- depth curve, best rung, forced 2048 ---", flush=True)
    print("  %-14s %-20s %-9s %s" % ("tokens", "E86", "E85", "delta"), flush=True)
    for bkt in sorted(M["bucket"].unique()):
        d = M[M["bucket"] == bkt]["speedup"].values
        lo, hi = boot_ci(d)
        cell = ("%.3f [%.2f,%.2f]" % (float(np.mean(d)), lo, hi) if len(d) >= 2
                else "%.3f (n=1)" % float(d[0]))
        e = E85_FORCED.get(bkt, float("nan"))
        print("  %-14s %-20s %-9.3f %+.3f"
              % ("%d-%d" % (bkt, bkt + BUCKET), cell, e,
                 float(np.mean(d)) - e), flush=True)
    bs = sorted(M["bucket"].unique())
    if len(bs) >= 2:
        x = M[M["bucket"] == bs[0]].set_index("seq")["speedup"]
        y = M[M["bucket"] == bs[-1]].set_index("seq")["speedup"]
        com = sorted(set(x.index) & set(y.index))
        if len(com) >= 2:
            dd = (y.loc[com] - x.loc[com]).values
            r2 = np.random.default_rng(13)
            bb = np.array([r2.choice(dd, len(dd), replace=True).mean()
                           for _ in range(BOOT)])
            print("  slope last-minus-first: %+.3f [%+.3f, %+.3f]  (E85: -0.474,"
                  " E83: -0.363)"
                  % (dd.mean(), np.percentile(bb, 2.5), np.percentile(bb, 97.5)),
                  flush=True)
            print("  A better head RAISES this curve; it flattens only if the feed"
                  " falls, and the\n  feed is fixed here. Expect the slope to"
                  " stay put -- that is E84's problem.", flush=True)

print("\n--- EXACTNESS ---", flush=True)
for c in ("argmax_path", "cache_floor", "tree_gate"):
    rr = [d for d in DIAG if d.get("check") == c]
    if rr:
        print("  control %-14s value %.3f | %s"
              % (c, float(rr[0]["value"]), rr[0]["detail"]), flush=True)
if D.empty:
    print("  0 divergences -- greedy-exact in every rung", flush=True)
else:
    dd = D[D["arm"] != "gate"]
    cf = dd[(dd["side"] == "arm") & (dd["margin"] >= MARGIN_FAIL)]
    print("  %d divergences, %d arm-side, %d CONFIDENT (worst arm margin %.4f)"
          % (len(dd), int((dd["side"] == "arm").sum()), len(cf),
             float(dd[dd["side"] == "arm"]["margin"].max())
             if (dd["side"] == "arm").any() else 0.0), flush=True)
    if len(cf):
        print("  -> CONFIDENT arm-side divergence is a verifier bug. Do not quote"
              " any speedup\n     from this run until it is explained.", flush=True)

print("\n--- RESUMING ---", flush=True)
print("  Re-running this identical script continues from wherever it stopped:",
      flush=True)
print("    %-28s %s" % (DISTILL_OUT,
                        "%d of %d completions" % (N_GENS, DISTILL_N)), flush=True)
print("    %-28s %s" % ("e87_feats_index.npz + .dat",
                        "present -- PHASE 2 will be skipped"
                        if FEAT_IDX.exists() else "not written"), flush=True)
print("  Keep both artifacts between sessions and nothing above is recomputed."
      " Distillation\n  to %d took %.0f min of this run; feature extraction is"
      " the other big one." % (DISTILL_N, mins()), flush=True)
print("\nWRITTEN to %s: %s" % (OUT, sorted(q.name for q in OUT.glob("*"))),
      flush=True)
print("[%6.1fm] done" % mins(), flush=True)
