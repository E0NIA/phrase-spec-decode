# =============================================================================
# E91. OUR 21.9 M HEAD on E89's EXACT prompts: 50 GSM8K + 50 MATH-500 + 40
#      MT-Bench, EOS-terminated with a 2048 cap, temperature 0. Kaggle T4.
#
# WHY THIS RUN EXISTS
#   E87's headline -- 2.341x on GSM8K, 2.620x on MATH-500 at N=16 -- was measured
#   on seqs 0-11 only. TWELVE prompts of each 50-prompt pool, because E88 never
#   completed and E87's N sweep was sized for a budget that assumed E88 would do
#   the full bench. E89 then measured EAGLE-3 on all 50. So the 27% MATH-500 gap
#   that looked like a result was 12 prompts against 50, and the two were never
#   the same sample. This run puts our head on the same 50+50+40 EAGLE saw.
#
# THE PROMPTS ARE LOADED, NOT DRAWN
#   E89 wrote e89_prompts.json. E91 reads it and asserts its sha. Re-deriving the
#   pool is precisely what went wrong before: PER_SOURCE was 20000 in E88 and
#   4000 in E87, so the training pool differed, the `norm(t) not in TN` dedup
#   removed different candidates, and the rng(0) permutation landed on a different
#   50 -- while an assert built from a hand-typed constant table certified that
#   the sets matched. Loading the file removes the whole class of error: there is
#   no selection code here to disagree with anything.
#
# WHAT IS COMPARED, AND ON WHAT
#   Primary: MS PER GENERATED TOKEN, paired per prompt against E89's per-prompt
#   records. A speedup is a ratio to its own baseline and the two runs used
#   different baselines -- ours is this verifier with drafting off, E89's was
#   EAGLE's `naivegenerate`. Milliseconds per token has no baseline in it, so
#   pairing on it compares the two methods directly and the fact that the two
#   baselines agree to 0.3% (59.4 against 59.6/59.8/59.6 ms/tok) becomes a check
#   rather than a load-bearing assumption. Both speedups are still reported.
#   The pairing is guarded: both sides are greedy at temperature 0 on the same
#   question, so the baseline token counts must agree PER PROMPT. If they do not,
#   the prompt sets are not identical and that set is skipped rather than reported.
#
# NO EAGLE IN THIS SCRIPT
#   E88's reimplementation of the EAGLE-3 draft head reached 1.136 tok/pass on
#   MT-Bench where the published acceptance length is ~3.4, with all four aux-layer
#   triples inside 1.6% of each other -- the aux path was inert. E89 replaced it
#   with SafeAILab's own code at a pinned commit and reproduced DFlash's published
#   numbers to two decimals (3.811/3.497 against 3.77/3.52). EAGLE's numbers come
#   from there; nothing here drafts for EAGLE. `run_tree` keeps its `want_aux`
#   branches because it is verified code and is NOT edited here; the function they
#   would call is a stub that raises.
#
# E87's HEAD, and why the distillation cache is still needed
#   The checkpoint stores its VOCABULARY but not `PRE` -- each entry's token
#   expansion, derived from the corpus. So the vocabulary is rebuilt from the
#   distillation cache (~4 min of CPU) and asserted EQUAL to the one the head was
#   trained against. If that assert fires, the head's output rows mean something
#   different from what we would be drafting and every number would be silently
#   wrong. This run RE-SAVES the head WITH `PRE`, so later benches need the ~88 MB
#   .pt alone -- no 17.9 GB feature file, no distillation cache.
#
# WHAT COULD STILL INVALIDATE THIS, each checked rather than hoped
#   1. bf16 -> fp16. The target ships bf16 and a T4 is sm75, so there is no bf16.
#      E89 ran both sides in fp16 on the same card, so the cast is common to both.
#   2. The rebuilt vocabulary must equal the head's, entry for entry. Asserted.
#   3. tok/pass cannot fall below 1.000 -- every pass emits at least the target's
#      own bonus token. Asserted on every arm. This is what caught E88's EOS bug,
#      where a drafted block carrying EOS in its INTERIOR never stopped the
#      sequence, so it ran to 2048 and cut_eos discarded the overshoot while the
#      passes that produced it stayed in the denominator.
#   4. The aggregate speedup must lie inside its own bootstrap interval. The old
#      report bootstrapped the mean of per-prompt ratios and printed it beside a
#      ratio of sums; when speedup correlates with answer length -- which depth
#      decay guarantees -- that interval misses the point estimate essentially
#      always (400/400 in simulation; E89 printed 2.032 with [2.114, 2.280]).
#
# PREDICTION, ON THE RECORD
#   Our 12-prompt numbers came from seqs 0-11, and MATH-500's mean answer length
#   over all 50 is 562 tokens against the 361 I had estimated -- so the full 50
#   are LONGER on average than the 12. Depth decay therefore predicts our MATH-500
#   figure will come in BELOW 2.620x. If it lands near 2.0x we are level with
#   EAGLE-3(60) at a tenth of the parameters; if it holds above 2.3x the win is
#   real and larger than the parameter count alone would suggest. Either way the
#   12-prompt figure should stop being quoted.
# =============================================================================
# =============================================================================

MODEL   = "Qwen/Qwen3-4B"
PARAMS_EAGLE3 = 218_245_120        # verified from the safetensors manifest

# E89's EAGLE-3 numbers, measured through SafeAILab's OWN implementation on the
# prompts this run loads. Printed beside ours for reference only: each side is
# divided by its own baseline, and E89's naive baselines (59.6 / 59.8 / 59.6
# ms/tok) are what make the two speedups comparable at all.
E89_EAGLE = {
    "gsm8k-test": dict(e60=2.451, e16=2.418, tau60=3.811, tau16=3.294,
                       ms60=24.3, ms16=24.5, naive_ms=59.6, gen_len=276),
    "math500":    dict(e60=2.032, e16=2.061, tau60=3.497, tau16=3.035,
                       ms60=28.9, ms16=27.9, naive_ms=59.8, gen_len=562),
    "mtbench":    dict(e60=2.005, e16=1.898, tau60=3.324, tau16=2.871,
                       ms60=29.0, ms16=29.7, naive_ms=59.6, gen_len=362),
}

# ---- our drafter. E87 measured N=16 as the optimum on BOTH sets (GSM8K 2.341,
#      MATH-500 2.620); 8 and 32 are carried so the curve is reported, since
#      EAGLE publishes two tree sizes and we should show ours.
N_SCAN = [8, 16, 32]

# ---- the vocabulary, rebuilt from E87's corpus. EVERY ONE of these must match
#      E87 exactly or the rebuilt vocabulary will differ from the head's and the
#      assert below will fire.
PARAM_BUDGET = 20_000_000
MAX_VOCAB    = 80_000
H_DIM        = 256
MIN_WORD_COUNT = 1
UNI_SHARE   = 0.45
ADD_NUMERIC = 1000
WORD_RE = r"\s*(?:[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9])"
MAX_GRAM = 3
MIN_GRAM_COUNT = 5
MAX_UNIT_TOK = 16
N_UNI = 4

# ---- the protocol. The test prompts are LOADED from E89's dump, not drawn, so
#      no selection code here can diverge from what E89 benched.
PROMPTS_IN    = "/kaggle/input/e89-prompts/e89_prompts.json"
E89_PAIRS_IN  = "/kaggle/input/e89-prompts/e89_pairs.csv"
PROMPTS_SHA = "5c3a2f181dd5"       # E89's; asserted on load
SET_ORDER = ("mtbench", "gsm8k-test", "math500")   # MT-Bench FIRST: it is the
# diagnostic, and if the time budget runs out it is the one result we must have.

PER_SOURCE  = 4000
NAT_N       = 50          # MUST equal E87's, or the 50 prompts differ
LONG_N      = 10
FORCED_N    = 8
BENCH_N     = 50          # prompts per set actually benched
MAX_NEW     = 2048        # a CAP; EOS terminates. DFlash's Table 1 protocol.
MAX_PROMPT_TOK = 384
MAX_FEED_PER_SEQ = 512    # per-sequence fed-token cap inside a drafter
MT_BENCH_N  = 40          # MT-Bench prompts benched, exactly as E89 did
ATT_CELLS   = 40_000_000
ARGMAX_BUDGET_GB = 0.50
BOOT        = 4000
T_BUDGET    = 420         # minutes; Kaggle allows 12 h

DISTILL_IN = "/kaggle/input/e86-distill/e86_distill_math.pt"
HEADS_IN   = "/kaggle/input/e87-head/e87_head_100pct.pt"

# DFlash Table 1, Qwen3-4B, temperature 0, up to 2048 generated tokens.
DFLASH_T1 = {"gsm8k-test": dict(e3_16=1.99, e3_60=2.27, dflash=5.15,
                                tau_e3_16=3.30, tau_e3_60=3.77, tau_df=6.53),
             "math500":    dict(e3_16=1.83, e3_60=2.10, dflash=6.09,
                                tau_e3_16=3.08, tau_e3_60=3.52, tau_df=7.84)}
# E87, same head, same 50 prompts, our own verifier (not the node mask)
E87_OURS = {"gsm8k-test": {8: 2.224, 16: 2.341, 32: 2.130},
            "math500":    {8: 2.431, 16: 2.620, 32: 2.352}}

import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import io, re, json, sys, time, subprocess, traceback, threading, unicodedata, itertools, math, copy
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "transformers>=4.50", "datasets", "accelerate", "sentencepiece"],
               check=False)
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, DynamicCache
from datasets import load_dataset

OUT = (Path("/kaggle/working/e88") if Path("/kaggle/working").exists()
       else Path("/content/e88") if Path("/content").exists()
       else Path("e88"))
OUT.mkdir(parents=True, exist_ok=True)
DEVS = ["cuda:%d" % i for i in range(torch.cuda.device_count())] or ["cpu"]
DEV0 = DEVS[0]
DT = torch.float16
NEG = torch.finfo(DT).min
MARGIN_FAIL = 0.5 if DT == torch.float16 else 1.5
torch.manual_seed(0); np.random.seed(0)
T0, LK = time.time(), threading.Lock()
ROWS, SENT, CURVE, DIAG = [], [], [], []
clk = time.perf_counter


def log(*a):
    with LK:
        print("[%6.1fm]" % ((time.time() - T0) / 60), *a, flush=True)


def mins():
    return (time.time() - T0) / 60


TAG = "e91"          # prefixes every CSV this run writes


def flush():
    """Write every table that is actually appended to, named by TAG.

    This wrote `ROWS` for three runs. Nothing appends to ROWS -- the headline table
    is ROWSX and the per-prompt records are PAIRS -- so E81, E82 and E91 all
    finished and saved neither. PAIRS is what the leakage filter and any CI
    recomputation read, so losing it costs the whole run, not a column.

    Resolved through globals() because flush() is defined before ROWSX/PAIRS exist
    and is called from inside the bench, after they do."""
    g = globals()
    with LK:
        for key, nm in (("ROWSX", "rows"), ("PAIRS", "pairs"),
                        ("DIVERGE", "diverge"), ("CURVE", "curve"),
                        ("SENT", "sent"), ("DIAG", "diag")):
            rows = g.get(key)
            if rows:
                pd.DataFrame(rows).to_csv(OUT / ("%s_%s.csv" % (TAG, nm)),
                                          index=False)


def flush_check():
    """Post-condition on the DISK, not on the write list above.

    The bug this guards is "the numbers exist only in memory", and no amount of
    care reading the tuple above detects that -- only looking at the filesystem
    does. Called once at the end, after the final flush()."""
    g = globals()
    bad = []
    for key, nm in (("ROWSX", "rows"), ("PAIRS", "pairs")):
        rows = g.get(key) or []
        f = OUT / ("%s_%s.csv" % (TAG, nm))
        if not rows:
            bad.append("%s is EMPTY -- the bench produced no %s" % (key, nm))
        elif not f.exists():
            bad.append("%s has %d entries but %s was never written"
                       % (key, len(rows), f.name))
        else:
            n = sum(1 for _ in io.open(str(f), encoding="utf-8")) - 1
            if n != len(rows):
                bad.append("%s holds %d rows but %s has %d"
                           % (key, len(rows), f.name, n))
    assert not bad, (
        "the results did not reach disk: %s. A finished run whose tables exist only"
        " in memory is a lost run -- E81, E82 and E91 each lost one this way."
        % "; ".join(bad))
    print("\nSAVED to %s:" % OUT, flush=True)
    for q in sorted(OUT.glob("%s_*.csv" % TAG)):
        n = sum(1 for _ in io.open(str(q), encoding="utf-8")) - 1
        print("  %-22s %d rows" % (q.name, n), flush=True)


def script_rate(t, want):
    n = ok = 0
    for ch in t:
        if ch.isalpha():
            n += 1
            try:
                if unicodedata.name(ch).split()[0] == want:
                    ok += 1
            except ValueError:
                pass
    return ok / max(n, 1)

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



# ============================================ general tree mask (nodes + parents)
# Our block mask gave each candidate an isolated chain. EAGLE's tree shares long
# prefixes, so expressing it as independent root-to-leaf paths would duplicate those
# prefixes and inflate its fed-token count by the mean root-to-leaf path length of
# the shared structure: measured at 1.0x for a tree of disjoint chains and ~1.7x
# for a bushy one, so it is shape-dependent and the run reports nodes/seq beside
# feed/seq rather than asserting a factor. This mask takes nodes with parent pointers instead, so a
# shared prefix is fed ONCE. Our own drafter goes through it too (one chain per
# word), so neither side gets a structural advantage.
def anc_matrix(parents):
    """anc[i][j] = True iff j is i itself or an ancestor of i. parents[i] < i, or -1
    for a node attached directly to the end of the trunk."""
    C = len(parents)
    A = np.zeros((C, C), dtype=bool)
    for i, p in enumerate(parents):
        A[i, i] = True
        if p >= 0:
            assert p < i, "parents must be topologically ordered (parent index < child)"
            A[i] |= A[p]
    return A


def node_depth(parents):
    d = [0] * len(parents)
    for i, p in enumerate(parents):
        d[i] = 0 if p < 0 else d[p] + 1
    return d


def tree_mask_ref_nodes(c0, trunk_len, parents, dev):
    """Brute force, one sequence, for the assert. Deliberately written the slow
    obvious way so a bug in the vectorised version cannot hide in shared code."""
    C = len(parents)
    q, kv = trunk_len + C, c0 + trunk_len + C
    m = torch.full((q, kv), NEG, dtype=DT)
    anc, dep = anc_matrix(parents), node_depth(parents)
    pos = []
    for i in range(trunk_len):                       # trunk: causal over the trunk
        m[i, :c0] = 0.0
        m[i, c0:c0 + i + 1] = 0.0
        pos.append(c0 + i)
    for i in range(C):
        r = trunk_len + i
        m[r, :c0] = 0.0                              # whole cache
        m[r, c0:c0 + trunk_len] = 0.0                # whole trunk
        for j in range(C):                           # ancestors and self only
            if anc[i, j]:
                m[r, c0 + trunk_len + j] = 0.0
        pos.append(c0 + trunk_len + dep[i])
    return m.view(1, 1, q, kv).to(dev), torch.tensor([pos], device=dev)


def tree_mask_nodes(kv_valid, trunk_lens, parents_b, pos0, dev):
    B, C = kv_valid.shape
    assert len(parents_b) == B
    Tmax = int(trunk_lens.max().item())
    Cs = [len(p) for p in parents_b]
    Cmax = max(Cs) if Cs else 0
    q, kv = Tmax + Cmax, C + Tmax + Cmax
    ancn = np.zeros((B, Cmax, Cmax), dtype=bool)
    depn = np.zeros((B, Cmax), dtype=np.int64)
    okn = np.zeros((B, Cmax), dtype=bool)
    for b, par in enumerate(parents_b):
        if not par:
            continue
        a = anc_matrix(par)
        ancn[b, :len(par), :len(par)] = a
        depn[b, :len(par)] = node_depth(par)
        okn[b, :len(par)] = True
    anc_t = torch.from_numpy(ancn).to(dev)
    dep_t = torch.from_numpy(depn).to(dev)
    ok_t = torch.from_numpy(okn).to(dev)
    tl = trunk_lens.to(dev).long()
    tslot = torch.arange(Tmax, device=dev)
    treal = tslot.unsqueeze(0) >= (Tmax - tl).unsqueeze(1)          # [B, Tmax]
    tidx = tslot.unsqueeze(0) - (Tmax - tl).unsqueeze(1)
    row_ok = torch.cat([treal, ok_t], 1)                            # [B, q]

    vis = torch.zeros((B, q, kv), dtype=torch.bool, device=dev)
    # --- cache columns
    vis[:, :, :C] = kv_valid.to(dev).unsqueeze(1)
    # --- trunk columns: trunk rows are causal in trunk index, node rows see it all
    tcol = treal.unsqueeze(1).expand(B, q, Tmax).clone()
    caus = (tidx.unsqueeze(1) <= tidx.unsqueeze(2))                 # [B, Tmax, Tmax]
    tcol[:, :Tmax, :] &= caus
    vis[:, :, C:C + Tmax] = tcol
    # --- node columns: only ancestors-or-self, and only for node rows
    if Cmax:
        ncol = torch.zeros((B, q, Cmax), dtype=torch.bool, device=dev)
        ncol[:, Tmax:, :] = anc_t
        vis[:, :, C + Tmax:] = ncol
    vis &= row_ok.unsqueeze(2)
    dead = ~vis.any(-1)
    if bool(dead.any()):     # padded rows attend to themselves so softmax is defined
        own = (C + torch.arange(q, device=dev)).view(1, q, 1).expand(B, q, 1)
        vis = vis.scatter(2, own, vis.gather(2, own) | dead.unsqueeze(2))
    m = torch.where(vis, torch.zeros((), dtype=DT, device=dev),
                    torch.full((), NEG, dtype=DT, device=dev)).unsqueeze(1)
    pos = torch.cat([pos0.to(dev).unsqueeze(1) + tidx.clamp(min=0),
                     pos0.to(dev).unsqueeze(1) + tl.unsqueeze(1) + dep_t], 1)
    return m, pos.clamp(min=0), Tmax, Cmax, row_ok

# ============================================================ the head
class Head(nn.Module):
    """E69's winner, unchanged. Reads the target's hidden state at the current
    position plus the embedding of the token the verifier just handed us, and
    predicts the next whole Indic WORD as one draft-vocabulary row.

    No `g` (E71's hidden-state predictor): depth is dead, so carrying a 0.3 M map
    we never call would inflate the published parameter count for nothing.
    """

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
            MODEL, dtype=DT, device_map={"": dev}, attn_implementation="eager").eval()
    except Exception:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(
            MODEL, dtype=DT, device_map={"": dev}, attn_implementation="eager").eval()


MODELS = {d: load_on(d) for d in DEVS}
EMBS = {d: MODELS[d].get_input_embeddings().weight for d in DEVS}
LMW = {d: MODELS[d].get_output_embeddings().weight for d in DEVS}
TGT = sum(p.numel() for p in MODELS[DEV0].parameters())
PADID = tok.pad_token_id
V_OUT = LMW[DEV0].shape[0]
ARGMAX_CHUNK = max(256, int(ARGMAX_BUDGET_GB * 1e9 // (V_OUT * 2)))
EOS = {tok.eos_token_id}
for s in ("<end_of_turn>", "<eos>"):
    i = tok.convert_tokens_to_ids(s)
    if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
        EOS.add(i)
EOS.discard(None)
log("target %.2f B | d_model %d | vocab %d" % (TGT / 1e9, D_MODEL, V_OUT))
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


# ======================================= E87's data pipeline, lifted verbatim
# No distillation, no feature extraction, no training. The distillation cache is
# read ONLY to rebuild the vocabulary and PRE (each entry's token expansion),
# because the head checkpoint stores its vocabulary but not the expansions.
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


# ===================================================== the prompts, LOADED
# E89 dumped the exact prompts it benched. Loading them is the only way to be sure
# both runs saw the same text. Re-deriving the pool is what went wrong before:
# PER_SOURCE was 20000 here and 4000 in E87, so the training pool differed, the
# `norm(t) not in TN` dedup removed different candidates, and the rng(0) draw
# landed on a different 50 -- while an assert claimed the sets matched.
_pp = [p for p in (PROMPTS_IN,
                   "results/e89_prompts.json",      # the repo layout
                   "../results/e89_prompts.json",
                   "e89_prompts.json",
                   "/kaggle/working/e89_prompts.json",
                   "/kaggle/input/e89-prompts/e89_prompts.json")
       if os.path.exists(p)]
assert _pp, (
    "e89_prompts.json not found. It holds the 50+50+40 prompts E89 benched, and"
    " without it this run would measure a DIFFERENT set, making the comparison"
    " against E89's EAGLE-3 numbers meaningless. Looked in: %s"
    % str((PROMPTS_IN, "e89_prompts.json", "/kaggle/working/")))
_pc = json.loads(io.open(_pp[0], encoding="utf-8").read())
assert str(_pc.get("sha", "")).startswith(PROMPTS_SHA), (
    "prompt file sha is %s but E89 ran %s -- a different prompt set"
    % (str(_pc.get("sha"))[:12], PROMPTS_SHA))
TESTS_BY_SET = {k: list(v) for k, v in _pc["sets"].items()}
for _k, _n in (("gsm8k-test", BENCH_N), ("math500", BENCH_N),
               ("mtbench", MT_BENCH_N)):
    assert _k in TESTS_BY_SET and len(TESTS_BY_SET[_k]) >= _n, (
        "%s holds %d prompts, need %d"
        % (_k, len(TESTS_BY_SET.get(_k, [])), _n))
log("prompts LOADED from %s (sha %s): %s"
    % (_pp[0], _pc["sha"][:12], {k: len(v) for k, v in TESTS_BY_SET.items()}))
DIAG.append(dict(check="prompt_sha", value=1.0, detail=_pc["sha"][:12]))

tok.padding_side = "left"


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

assert dl, (
    "DISTILL_IN did not load. E88 does not distil: it reads the cache ONLY to"
    " rebuild the vocabulary and PRE, because the head checkpoint stores its"
    " vocabulary but not each entry's token expansion. Point DISTILL_IN at the"
    " same cache E87 used -- a different corpus gives a different vocabulary and"
    " the equality assert below will fire.")
assert len(GENS) >= 300, "only %d completions; not enough to train on" % len(GENS)
# E85 discovered after the fact that 70.8% of its completions hit the token cap
# exactly, so most were cut off mid-solution. Measured here rather than inferred.
_ls = [len(g) for _p, g in GENS]
log("corpus for the vocabulary rebuild: %d completions, mean %.1f tokens"
    % (len(_ls), float(np.mean(_ls))))
DIAG.append(dict(check="corpus", value=float(len(_ls)),
                 detail="mean=%.1f median=%d"
                 % (float(np.mean(_ls)), int(np.median(_ls)))))


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



# ============================================================ the head
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



# ---- load E87's head and PROVE the rebuilt vocabulary is the one it was trained
#      against. If these differ, the head's output rows index different units than
#      we would draft, and every number below would be silently wrong.
assert HEADS_IN, "E88 benches E87's head; point HEADS_IN at e87_head_100pct.pt"
_blob = torch.load(str(HEADS_IN), map_location="cpu", weights_only=False)
assert _blob.get("model") == MODEL, (
    "head was trained against %r, not %r" % (_blob.get("model"), MODEL))
_hv = list(_blob["vocab"])
assert len(_hv) == len(VOCAB), (
    "the head was trained on a %d-entry vocabulary and this run rebuilt %d. Every"
    " vocabulary constant must match E87 exactly (PARAM_BUDGET, MAX_VOCAB, H_DIM,"
    " MAX_GRAM, MIN_GRAM_COUNT, MIN_WORD_COUNT, UNI_SHARE, ADD_NUMERIC, WORD_RE)"
    " AND the distillation cache must be the same one."
    % (len(_hv), len(VOCAB)))
_nd = sum(1 for a_, b_ in zip(_hv, VOCAB) if a_ != b_)
assert _nd == 0, (
    "%d of %d vocabulary entries differ between the head checkpoint and the"
    " rebuild, so row k of the head's output does not mean unit k here. First"
    " mismatch: %r vs %r"
    % (_nd, len(VOCAB), next(a_ for a_, b_ in zip(_hv, VOCAB) if a_ != b_),
       next(b_ for a_, b_ in zip(_hv, VOCAB) if a_ != b_)))
_hd = Head(D_MODEL, len(VOCAB), H_DIM).to(DEV0)
_hd.load_state_dict(_blob["sd"])
HD = dict(head=_hd.eval(), **{k: v for k, v in _blob.items()
                              if k not in ("sd", "vocab", "PRE", "NWORDS")})
HEAD = {d: (HD["head"] if d == DEV0 else copy.deepcopy(HD["head"]).to(d)).eval()
        for d in DEVS}
log("head LOADED from %s: %.3f M params (%.4f%% of target) | top-1 %.3f |"
    " recall@8 %.3f" % (os.path.basename(str(HEADS_IN)), HD["npar"] / 1e6,
                        100 * HD["npar"] / TGT, HD.get("top1", float("nan")),
                        HD.get("recall8", float("nan"))))
log("  vocabulary rebuild MATCHES the checkpoint on all %d entries" % len(VOCAB))

# ---- re-save WITH PRE so future benches need only this ~88 MB file
try:
    torch.save({"model": MODEL, "vocab": VOCAB, "sd": _hd.state_dict(),
                "PRE": PRE, "NWORDS": NWORDS,
                **{k: v for k, v in HD.items() if k != "head"}},
               str(OUT / "e88_head_with_pre.pt"))
    log("  re-saved as %s WITH PRE and NWORDS -- a future bench needs this file"
        "\n  alone: no 17.9 GB feature cache, no 2.7 GB distillation cache."
        % (OUT / "e88_head_with_pre.pt").name)
except Exception as e:
    log("  could not re-save the head with PRE (%s); harmless for this run"
        % str(e)[:60])

MASK = {d: torch.zeros(len(VOCAB), dtype=torch.float32).to(d) for d in DEVS}
IS_UNI = torch.tensor([NWORDS[w] == 1 for w in VOCAB])
UNI_BIAS = {d: torch.where(IS_UNI, torch.zeros(len(VOCAB)),
                           torch.full((len(VOCAB),), NEG, dtype=torch.float32)).to(d)
            for d in DEVS}
PRE_LIST = [list(x) for x in PRE]


# ===================================================== no EAGLE in this run
# run_tree carries `want_aux` branches for EAGLE's fused hidden state. E91 never
# passes want_aux=True, and run_tree is verified code that is not being edited
# here, so the branches stay and the function they call becomes a stub. Reaching
# it means a caller asked for aux features that nothing in this script produces.
AUX = {}


def aux_cat(dev):
    raise AssertionError(
        "aux_cat reached in E91, which has no EAGLE and no layer hooks. Some"
        " caller passed want_aux=True; nothing here can supply aux features.")


# ============================================================ the two drafters
def close_ancestors(toks, pars, scores, total):
    """Keep the `total` highest-scoring nodes, but a node is only usable if its
    parent is kept too -- that is EAGLE-2's selection rule, and dropping a parent
    while keeping a child would silently corrupt the tree."""
    if len(toks) <= total:
        return toks, pars
    keep = set()
    for i in sorted(range(len(toks)), key=lambda z: -scores[z]):
        if len(keep) >= total:
            break
        chain, j = [], i
        while j >= 0 and j not in keep:
            chain.append(j); j = pars[j]
        if len(keep) + len(chain) <= total:
            keep.update(chain)
    old = sorted(keep)
    rn = {o: n for n, o in enumerate(old)}
    return ([toks[o] for o in old],
            [(-1 if pars[o] < 0 else rn[pars[o]]) for o in old])


def mk_ours(dev, N):
    """Our drafter through the SAME tree mask EAGLE uses: the drafted units are
    merged into a PREFIX TREE rooted at the trunk, so a phrase and an EAGLE branch
    are verified by identical machinery and neither side gains from the mask.

    The N_UNI single-word reserve is E87's: greedy-longest-match labelling never
    makes a short prefix the target, so the head can under-rank it, and reserving
    slots for the best single-word units keeps word-level drafting inside the same
    N. Without it this is not the same drafter E87 measured."""
    head, emb, msk = HEAD[dev], EMBS[dev], MASK[dev]
    ub = UNI_BIAS[dev]
    nu = min(N_UNI, max(N - 1, 0))

    def drafter(ctx, A):
        lo = head(ctx["hid"].float(), emb[ctx["last"]].float()) + msk
        top = lo.topk(min(N - nu, lo.shape[-1]), -1).indices
        if nu:
            tu = (lo + ub).topk(min(nu, int(IS_UNI.sum())), -1).indices
            # reserve LAST, as E87 had it. With a trie the order cannot change
            # WHAT is accepted, but it decides which candidates survive the feed
            # cap, and a unigram evicting a phrase is a bad trade.
            top = torch.cat([top, tu], -1)
        out = []
        for a in range(A):
            # dedup by EXPANSION, not by row: two vocabulary entries can expand to
            # the same token sequence, and `r in seen` never fired because topk
            # rows are distinct already.
            rows, sx = [], set()
            for r in top[a].tolist():
                ex = PRE_LIST[r]
                if not ex:
                    continue
                k = tuple(ex)
                if k in sx:
                    continue
                sx.add(k); rows.append(ex)
            rows.sort(key=lambda e: -len(e))      # longest first, for the cap
            # ---- build a PREFIX TREE over the expansions.
            # Independent chains let two candidates share a first token, and
            # run_tree's walk takes the FIRST matching child -- so a 1-token
            # unigram fed ahead of a 3-token phrase with the same first token
            # truncates the phrase. 91.1% of our multi-word entries have their
            # first word as a standalone unigram entry, so this is the common
            # case: it cost 2.589 -> 2.018 tok/pass on GSM8K between E87 and E91.
            # In a trie no two siblings share a token, the walk reaches the
            # deepest match by construction, the result is independent of
            # candidate order, and a shared prefix is fed once.
            t, p, kids = [], [], {}
            for ex in rows:
                par, i = -1, 0
                while i < len(ex):                # how far the trie already goes
                    nd = kids.get((par, int(ex[i])))
                    if nd is None:
                        break
                    par = nd; i += 1
                if len(t) + (len(ex) - i) > MAX_FEED_PER_SEQ - 4:
                    continue
                for j in range(i, len(ex)):
                    tk = int(ex[j])
                    t.append(tk); p.append(par)
                    par = len(t) - 1
                    kids[(p[-1], tk)] = par
            if not t:
                for ex in rows:
                    par = -1
                    for tk in ex:
                        t.append(int(tk)); p.append(par); par = len(t) - 1
                    break
            out.append((t, p))
        return out
    return drafter


# ============================================================ one verifier, both arms
@torch.no_grad()
def run_tree(prompts, drafter, dev, want_aux=False):
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
    hid = hsp[:, -1].float()
    aux = aux_cat(dev)[:, -1] if want_aux else None
    nxt = argmax_at(hsp[:, -1:], dev, torch.arange(B0, device=dev)).tolist()
    for b in range(B0):
        ids[b].append(int(nxt[b]))
    del hsp, am
    fed = [len(p) for p in prompts]
    ntrue = [len(p) for p in prompts]
    kvalid = kvv.to(dev)
    # If PREFILL already produced EOS the answer is complete and the loop must
    # not run: one wasted pass against one kept token reads as 0.5 tok/pass and
    # would trip the floor assert on a sequence that was in fact handled right.
    # Rows are dropped only inside the loop, via shrink_cache, so doing it here
    # for a subset would desynchronise hid/kvalid from `act` -- hence the B0==1
    # restriction, which every call site in this script satisfies.
    assert B0 == 1 or all(i not in EOS for i in nxt), (
        "a batched prefill produced EOS for some rows but not others; dropping"
        " them here needs the same cache shrink the main loop does")
    act = [] if int(nxt[0]) in EOS else list(range(B0))
    passes, nacc, nfeed, nnodes = 1, [], 0, 0
    while act:
        A = len(act)
        trunks = [ids[b][fed[b]:] for b in act]
        tl = torch.tensor([len(t_) for t_ in trunks])
        ctx = dict(hid=hid, aux=aux,
                   last=torch.tensor([ids[b][-1] for b in act], device=dev))
        trees = drafter(ctx, A) if drafter is not None else [([], [])] * A
        pars_b = [t[1] for t in trees]
        m, pos, Tmax, Cmax, rok = tree_mask_nodes(
            kvalid, tl, pars_b, torch.tensor([ntrue[b] for b in act]), dev)
        q = Tmax + Cmax
        fnp = np.full((A, q), PADID, dtype=np.int64)
        for i in range(A):
            T = len(trunks[i])
            fnp[i, Tmax - T:Tmax] = trunks[i]
            tk = trees[i][0]
            if tk:
                fnp[i, Tmax:Tmax + len(tk)] = tk
        feed = torch.from_numpy(fnp).to(dev, non_blocking=True)
        nfeed += A * q
        nnodes += sum(len(t[0]) for t in trees)
        hs = fwd_hidden(mm, dev, feed, m, pos, cache)
        naux = aux_cat(dev) if want_aux else None
        passes += 1
        del m
        base = Tmax - 1
        # walk the tree: argmax only where a path is still alive. Depth+1 rounds of
        # A rows each, instead of an argmax over every node -- at vocab 151936 the
        # difference is hundreds of MB per pass.
        kids = [{} for _ in range(A)]
        for i in range(A):
            for n_, p_ in enumerate(pars_b[i]):
                kids[i].setdefault(p_, []).append(n_)
        cur = [base] * A                      # slot index within the row
        node = [-1] * A                       # current accepted node (-1 = trunk end)
        alive = list(range(A))
        acc = [[] for _ in range(A)]
        bonus = [None] * A
        while alive:
            idx = torch.tensor([i * q + cur[i] for i in alive], device=dev)
            g = argmax_at(hs, dev, idx).tolist()
            nxt_alive = []
            for s, i in enumerate(alive):
                a_ = int(g[s])
                hit = next((c for c in kids[i].get(node[i], [])
                            if trees[i][0][c] == a_), None)
                if hit is None:
                    bonus[i] = a_
                else:
                    acc[i].append(hit)
                    node[i] = hit
                    cur[i] = Tmax + hit
                    nxt_alive.append(i)
            alive = nxt_alive
        pick = []
        for i, b in enumerate(act):
            at = [trees[i][0][c] for c in acc[i]]
            pick.append(Tmax + acc[i][-1] if acc[i] else base)
            nacc.append(len(at))
            fed[b] = len(ids[b])
            ids[b] = ids[b] + at + [bonus[i]]
            ntrue[b] += len(trunks[i])
        pk = torch.tensor(pick, device=dev)
        ar = torch.arange(A, device=dev)
        hid = hs[ar, pk].float()
        aux = naux[ar, pk] if want_aux else None
        if Cmax:
            cache.crop(-Cmax)
        treal = (torch.arange(Tmax, device=dev).unsqueeze(0)
                 >= (Tmax - tl.to(dev)).unsqueeze(1))
        kvalid = torch.cat([kvalid, treal], 1)
        # EOS ANYWHERE in what this pass emitted, not just at the end of it.
        # A drafted block can carry EOS in its INTERIOR; the old test looked only
        # at ids[b][-1] -- the bonus token -- so such a sequence never stopped and
        # ran on to the full MAX_NEW. cut_eos then threw the overshoot away while
        # every pass that produced it stayed in the denominator, which is how an
        # arm reported 0.929 tok/pass: below 1.000, a figure no correct verifier
        # can reach, since each pass emits at least the bonus token.
        # The baseline drafts nothing, so for it EOS is ALWAYS the bonus token and
        # the old test was sufficient -- the bug hit only the arms, and hit hardest
        # the arms that accept the longest blocks. It is therefore not a wash: it
        # penalised exactly the configurations that draft best.
        # ids[b][fed[b]:] is precisely this pass's output; next pass reads the same
        # slice as its trunk.
        keep = [i for i, b in enumerate(act)
                if len(ids[b]) - len(prompts[b]) < MAX_NEW
                and not EOS.intersection(ids[b][fed[b]:])]
        if len(keep) < A:
            if keep:
                kt = torch.tensor(keep, device=dev)
                shrink_cache(cache, kt)
                kvalid = kvalid[kt]; hid = hid[kt]
                if want_aux:
                    aux = aux[kt]
            act = [act[i] for i in keep]
    torch.cuda.synchronize(dev)
    dt = clk() - t0
    gen = [ids[b][len(prompts[b]):len(prompts[b]) + MAX_NEW] for b in range(B0)]
    return gen, passes, nacc, dt, nfeed, nnodes



# ================================= helpers the bench needs (E74 + E87)
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


# ============================================================ PHASE A2: the new mask
log("PHASE A2: asserting the node/parent tree mask against brute force")
_rg = np.random.default_rng(1)
for _t in range(24):
    B_ = int(_rg.integers(1, 6)); C_ = int(_rg.integers(4, 30))
    nv = [int(_rg.integers(1, C_ + 1)) for _ in range(B_)]
    kvv = torch.zeros((B_, C_), dtype=torch.bool)
    for b in range(B_):
        kvv[b, C_ - nv[b]:] = True
    tls = torch.tensor([int(_rg.integers(1, 5)) for _ in range(B_)])
    pars = []
    for b in range(B_):
        n = int(_rg.integers(0, 14))
        pars.append([-1 if (i == 0 or _rg.random() < 0.35) else int(_rg.integers(0, i))
                     for i in range(n)])
    m, pos, Tm, Cm, _r = tree_mask_nodes(kvv, tls, pars, torch.tensor(nv), DEV0)
    for b in range(B_):
        T, Cb = int(tls[b]), len(pars[b])
        ref, rp = tree_mask_ref_nodes(nv[b], T, pars[b], DEV0)
        rows = list(range(Tm - T, Tm)) + list(range(Tm, Tm + Cb))
        cols = (list(range(C_ - nv[b], C_)) + list(range(C_ + Tm - T, C_ + Tm))
                + list(range(C_ + Tm, C_ + Tm + Cb)))
        assert torch.equal(m[b, 0][rows][:, cols], ref[0, 0]), \
            "tree mask mismatch %d/%d parents=%s" % (_t, b, pars[b])
        assert torch.equal(pos[b][rows], rp[0]), "tree positions mismatch %d/%d" % (_t, b)
log("  24/24 identical, mask and positions")

POOLS = {k: [enc_prompt(t) for t in v] for k, v in TESTS_BY_SET.items()}
log("timing pools: %s" % {k: len(v) for k, v in POOLS.items()})
PAIRS, ROWSX, DIVERGE = [], [], []

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



# ============================================ PHASE H: the head-to-head
# One shared verifier, one shared baseline per prompt, interleaved so no arm sees
# a different machine state. EOS-terminated with a 2048 cap, temperature 0.
def bench_set(ts, nprompt):
    pool = POOLS[ts][:nprompt]
    base = {}
    for k, pm in enumerate(pool):
        go, pq, na, dt_, nf, nn = run_tree([pm], None, DEV0, want_aux=False)
        base[k] = (cut_eos(go[0]), dt_)
    blen = float(np.mean([len(v[0]) for v in base.values()]))
    bsec = sum(v[1] for v in base.values())
    btok = sum(len(v[0]) for v in base.values())
    log("  %-11s %d baselines, %.1f s, %.0f tok/s, mean achieved length %.0f"
        " [%5.1fm]" % (ts, len(pool), bsec, btok / max(bsec, 1e-9), blen, mins()))

    arms = [("ours@N%d" % n, mk_ours(DEV0, n), False) for n in N_SCAN]
    for nm, dr, waux in arms:
        if mins() > T_BUDGET - 10:
            log("    budget reached, skipping %s" % nm)
            continue
        ag = dict(sec=0.0, tok=0, ps=0, acc=[], feed=0, nodes=0, ndiv=0)
        for k, pm in enumerate(pool):
            go, bdt = base[k]
            ao, pq, na, adt, nf, nn = run_tree([pm], dr, DEV0, want_aux=waux)
            ag["ndiv"] += exact_check(cut_eos(ao[0]), go, pm, nm, ts, k, DEV0)
            ag["sec"] += adt; ag["tok"] += len(cut_eos(ao[0])); ag["ps"] += pq
            ag["acc"] += na; ag["feed"] += nf; ag["nodes"] += nn
            PAIRS.append(dict(test=ts, arm=nm, seq=k, base_secs=bdt,
                              arm_secs=adt, speedup=bdt / adt,
                              gen_len=len(go), base_tok=len(go),
                              arm_tok=len(cut_eos(ao[0]))))
        ar = np.array(ag["acc"]) if ag["acc"] else np.array([0])
        tpp = btok / max(ag["ps"], 1)
        assert tpp >= 1.0, (
            "%s on %s reports %.3f tok/pass, below the arithmetic floor of 1.000."
            " Every pass emits at least the bonus token, so this is a verifier bug"
            " -- most likely a sequence that ran past EOS and had the overshoot cut"
            " from the numerator while its passes stayed in the denominator."
            % (nm, ts, tpp))
        fed = ag["feed"] / max(ag["ps"], 1)
        nod = ag["nodes"] / max(ag["ps"] - 1, 1)
        ROWSX.append(dict(test=ts, arm=nm, n_prompts=len(pool), base_secs=bsec,
                          arm_secs=ag["sec"], tokens=btok,
                          speedup=bsec / max(ag["sec"], 1e-9),
                          base_tok_s=btok / max(bsec, 1e-9),
                          tok_s=ag["tok"] / max(ag["sec"], 1e-9),
                          tok_per_pass=tpp, feed_per_pass=fed,
                          nodes_per_pass=nod,
                          banked_per_fed=(tpp - 1) / max(fed, 1e-9),
                          landing=float((ar > 0).mean()),
                          gen_len_mean=blen, ndiverge=ag["ndiv"]))
        log("    %-12s %-11s speedup %.3fx | tok/pass %.3f | nodes %.1f | fed %.1f"
            " | banked/fed %.4f | diverge %d [%5.1fm]"
            % (nm, ts, ROWSX[-1]["speedup"], tpp, nod, fed,
               ROWSX[-1]["banked_per_fed"], ag["ndiv"], mins()))
        flush()


log("PHASE H: our head on E89's prompts, EOS-terminated, cap %d" % MAX_NEW)
for _ts in [t for t in SET_ORDER if t in POOLS]:
    bench_set(_ts, MT_BENCH_N if _ts == "mtbench" else BENCH_N)


# ============================================================ report
R = pd.DataFrame(ROWSX)
P = pd.DataFrame(PAIRS)
D = pd.DataFrame(DIVERGE)
flush()
print("\n" + "=" * 78, flush=True)
print("E88 -- OURS vs EAGLE-3, ONE VERIFIER, %d MT-BENCH + %d + %d MATHS PROMPTS,"
      " EOS-TERMINATED" % (len(POOLS.get("mtbench", [])), BENCH_N, BENCH_N),
      flush=True)
print("=" * 78, flush=True)
try:
    _g = torch.cuda.get_device_name(0)
    _cc = torch.cuda.get_device_capability(0)
except Exception:
    _g, _cc = "unknown", (0, 0)
print("  %s | %s sm%d%d | %s | batch 1 | temperature 0, cap %d, EOS honoured"
      % (MODEL, _g, _cc[0], _cc[1], DT, MAX_NEW), flush=True)
print("  ours   %.3f M = %.4f%% of target | vocabulary %d entries | prompts %s"
      % (HD["npar"] / 1e6, 100 * HD["npar"] / TGT, len(VOCAB),
         _pc["sha"][:12]), flush=True)
print("  EAGLE-3 for reference: %d params = %.3f%% of target, i.e. %.1fx our head"
      % (PARAMS_EAGLE3, 100 * PARAMS_EAGLE3 / TGT,
         PARAMS_EAGLE3 / HD["npar"]), flush=True)

if not R.empty:
    for ts in [t for t in SET_ORDER if (R["test"] == t).any()]:
        g = R[R["test"] == ts].copy()
        z = DFLASH_T1.get(ts, {})
        print("\n--- %s (%d prompts, mean achieved length %.0f tokens) ---"
              % (ts, int(g["n_prompts"].iloc[0]), g["gen_len_mean"].iloc[0]),
              flush=True)
        print("  %-13s %-9s %-20s %-7s %-9s %-8s %-8s %-10s %s"
              % ("arm", "aggregate", "95% CI (aggregate)", "per-seq",
                 "tok/pass", "nodes", "fed", "banked/fed", "diverge"), flush=True)
        for _, r in g.sort_values("speedup", ascending=False).iterrows():
            pr = [q for q in PAIRS
                  if q["test"] == ts and q["arm"] == r["arm"]]
            d = np.array([q["speedup"] for q in pr]) if pr else np.array([np.nan])
            # the interval for the AGGREGATE, which is what r["speedup"] is; the
            # mean-of-ratios bootstrap belongs to a different number entirely
            lo, hi = (boot_ratio_ci([q["base_secs"] for q in pr],
                                    [q["arm_secs"] for q in pr])
                      if pr else (float("nan"), float("nan")))
            assert not pr or lo <= r["speedup"] <= hi, (
                "%s/%s: aggregate %.3f outside its own interval [%.3f, %.3f]"
                % (ts, r["arm"], r["speedup"], lo, hi))
            print("  %-13s %-9.3f [%.3f, %.3f] %-7.3f %-9.3f %-8.1f %-8.1f %-10.4f %d"
                  % (r["arm"], r["speedup"], lo, hi, float(np.nanmean(d)),
                     r["tok_per_pass"], r["nodes_per_pass"], r["feed_per_pass"],
                     r["banked_per_fed"], int(r["ndiverge"])), flush=True)
        if z:
            print("  --- published, DFlash Table 1, same target and protocol ---",
                  flush=True)
            for nm, su, tau in (("EAGLE-3 (16)", z["e3_16"], z["tau_e3_16"]),
                                ("EAGLE-3 (60)", z["e3_60"], z["tau_e3_60"]),
                                ("DFlash (16)", z["dflash"], z["tau_df"])):
                print("  %-13s %-9.2f %-20s %-9.2f" % (nm, su, "(published)", tau),
                      flush=True)
        if ts in E87_OURS:
            print("  --- E87, same head and prompts, our OWN verifier ---",
                  flush=True)
            print("  %s" % "  ".join("N%d %.3f" % (k, v)
                                     for k, v in sorted(E87_OURS[ts].items())),
                  flush=True)

    # ---- paired against E89, on MS PER TOKEN rather than on speedup.
    # A speedup is a ratio to its own baseline, and the two runs used different
    # baselines (our verifier with drafting off, versus EAGLE's naivegenerate).
    # Milliseconds per generated token has no baseline in it at all, so pairing
    # on that quantity compares the two methods directly and the matched 59.4 /
    # 59.8 ms/tok baselines become a check rather than a load-bearing assumption.
    _e89 = [q for q in (E89_PAIRS_IN, "e89_pairs.csv", "/kaggle/working/e89_pairs.csv",
                        "/kaggle/input/e89-prompts/e89_pairs.csv")
            if os.path.exists(q)]
    if not _e89:
        print("\n=== vs E89's EAGLE-3 (aggregate only; e89_pairs.csv not found) ===",
              flush=True)
        for ts in [t for t in SET_ORDER if (R["test"] == t).any()]:
            z = E89_EAGLE.get(ts)
            g = R[(R["test"] == ts) & (R["arm"].str.startswith("ours"))]
            if z is None or g.empty:
                continue
            b = g.loc[g["speedup"].idxmax()]
            print("  %-11s ours %s %.3fx  |  EAGLE-3 (60) %.3fx  (16) %.3fx"
                  % (ts, b["arm"], b["speedup"], z["e60"], z["e16"]), flush=True)
    else:
        EP = pd.read_csv(_e89[0])
        print("\n=== PAIRED vs E89's EAGLE-3, same prompts, on MS/TOKEN ===",
              flush=True)
        print("  Pairing is by (test, seq) and both runs loaded the SAME prompt"
              "\n  file, so seq k is the same question on both sides. ms/token"
              "\n  carries no baseline, so this does not depend on the two"
              "\n  baselines matching -- though they do, to 0.3%.", flush=True)
        print("  %-11s %-16s %-16s %-11s %-22s %s"
              % ("test", "ours (best)", "EAGLE-3 (best)", "delta ms/tok",
                 "95% CI on the delta", "n"), flush=True)
        for ts in [t for t in SET_ORDER if (R["test"] == t).any()]:
            ours = {}
            for q in PAIRS:
                if q["test"] == ts and str(q["arm"]).startswith("ours"):
                    ours.setdefault(q["arm"], {})[q["seq"]] = q
            if not ours:
                continue
            # the arm with the lowest mean ms/token is "best"; same rule for EAGLE
            def _mspt(d):
                return float(np.mean([1000.0 * r["arm_secs"] / max(r["arm_tok"], 1)
                                      for r in d.values()]))
            bo = min(ours, key=lambda k: _mspt(ours[k]))
            eg = EP[EP["test"] == ts]
            if eg.empty:
                continue
            ea = {}
            for arm in eg["arm"].unique():
                d = eg[eg["arm"] == arm]
                ea[arm] = {int(r.seq): r for r in d.itertuples()}
            be = min(ea, key=lambda k: float(np.mean(
                [1000.0 * r.arm_secs / max(r.arm_tok, 1) for r in ea[k].values()])))
            common = sorted(set(ours[bo]) & set(ea[be]))
            if len(common) < 2:
                continue
            # Prompt-identity check, as a TOLERANCE not an equality. Both runs
            # are greedy at temperature 0 on the same weights, but their baselines
            # are different loops, and in fp16 two greedy implementations
            # tie-break differently at near-equal logits -- the same effect our
            # runs report as MARGIN_FAIL. Measured E87 against E89: 9/12 and 11/12
            # exact, mean |diff| 6.1 and 0.4 tokens. DIFFERENT prompts would not
            # look like that; lengths would disagree wildly and at random.
            _d = np.array([abs(ours[bo][k]["base_tok"] - int(ea[be][k].base_tok))
                           for k in common], dtype=float)
            _mn = float(np.mean([ours[bo][k]["base_tok"] for k in common]))
            _exact = int((_d == 0).sum())
            print("  %-11s prompt identity: %d/%d baselines exact, mean |diff|"
                  " %.1f tok (%.2f%% of %.0f)"
                  % (ts, _exact, len(common), _d.mean(),
                     100 * _d.mean() / max(_mn, 1), _mn), flush=True)
            DIAG.append(dict(check="prompt_identity_%s" % ts,
                             value=_exact / max(len(common), 1),
                             detail="mean_abs_diff=%.1f of %.0f" % (_d.mean(), _mn)))
            if _exact < 0.6 * len(common) or _d.mean() > 0.08 * _mn:
                print("    SKIPPED: that is too far apart to be fp16 tie-breaking."
                      "\n    The two runs did not see the same prompts.", flush=True)
                continue
            u = np.array([1000.0 * ours[bo][k]["arm_secs"]
                          / max(ours[bo][k]["arm_tok"], 1) for k in common])
            v = np.array([1000.0 * ea[be][k].arm_secs
                          / max(int(ea[be][k].arm_tok), 1) for k in common])
            dd = u - v
            r2 = np.random.default_rng(7)
            bb = np.array([r2.choice(dd, len(dd), replace=True).mean()
                           for _ in range(BOOT)])
            lo, hi = float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))
            print("  %-11s %-16s %-16s %+-11.2f [%+.2f, %+.2f]%s   %d"
                  % (ts, "%s %.1f" % (bo, u.mean()), "%s %.1f" % (be, v.mean()),
                     dd.mean(), lo, hi,
                     "  OURS faster " if hi < 0 else
                     ("  EAGLE faster" if lo > 0 else "  tie         "),
                     len(dd)), flush=True)
            ROWSX.append(dict(test=ts, arm="paired_vs_%s" % be, n_prompts=len(dd),
                              speedup=float(v.mean() / max(u.mean(), 1e-9)),
                              tok_per_pass=float("nan"),
                              ms_ours=float(u.mean()), ms_eagle=float(v.mean()),
                              delta_ms=float(dd.mean()), ci_lo=lo, ci_hi=hi))

print("\n--- EXACTNESS, both arms through the same verifier ---", flush=True)
for c in ("argmax_path", "cache_floor"):
    rr = [d for d in DIAG if d.get("check") == c]
    if rr:
        print("  control %-14s %.3f | %s"
              % (c, float(rr[0]["value"]), rr[0]["detail"]), flush=True)
if D.empty:
    print("  0 divergences in any arm", flush=True)
else:
    dd = D[D["arm"] != "gate"] if "arm" in D.columns else D
    cf = dd[dd["margin"] >= MARGIN_FAIL] if "margin" in dd.columns else dd[:0]
    print("  %d divergences, %d CONFIDENT (worst margin %.4f)"
          % (len(dd), len(cf),
             float(dd["margin"].max()) if "margin" in dd.columns and len(dd)
             else 0.0), flush=True)
    if "arm" in dd.columns:
        print("  by arm: %s" % dict(dd["arm"].value_counts()), flush=True)
    if len(cf):
        print("  -> CONFIDENT divergence means the verifier is losing tokens."
              " Do not quote\n     any speedup from this run until it is"
              " explained.", flush=True)

print("\n--- CAVEATS THAT GO IN THE TABLE, NOT A FOOTNOTE ---", flush=True)
print("  1. T4 only. Our decode loop is CPU-dispatch-bound (E84: the target"
      " forward is\n     92.6%% of the pass and 100%% of it is the CPU issuing"
      " work), so no number here\n     transfers to a modern card until that is"
      " fixed.", flush=True)
print("  2. This head is UNDERTRAINED and its shape is UNTUNED. ~3 M tokens of"
      " distilled\n     maths, 21.9 M parameters, units capped at MAX_GRAM=3"
      " words, vocabulary pinned at\n     the 80,000 cap rather than at an"
      " optimum -- MAX_VOCAB binds, not the parameter\n     ceiling, so we never"
      " found where efficiency peaks. Everything below is a FLOOR\n     on the"
      " method, not its ceiling.", flush=True)
print("  3. The three knobs that decide that ceiling are all unexplored:"
      "\n       - unit length: MAX_GRAM 3 -> 5 or 6 words raises the tok/pass"
      " ceiling\n         (1 + expansion), which is 3.46 here at landing 1.0."
      "\n       - vocabulary size: the cap binds at 80,000, so the"
      " positions-per-entry\n         floor from E67 was never approached from"
      " above.\n       - training tokens: the data ladder was still unbent at"
      " 3.1 M rows (E86).", flush=True)
# Leakage was MEASURED, not assumed, so it is no longer a caveat: E90 dropped every
# prompt sharing even one 8-gram with the training pool and EAGLE moved 2.032 ->
# 2.009 while our 12-prompt figure moved 2.620 -> 2.632. Both inside their CIs.
print("  4. MATH-500 leakage was checked and is NOT material: dropping every"
      " prompt sharing\n     even one 8-gram with the training pool (8 of 50)"
      " moves EAGLE 2.032 -> 2.009.", flush=True)
flush()
flush_check()
print("[%6.1fm] done" % mins(), flush=True)
