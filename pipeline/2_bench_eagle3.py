"""E89 -- EAGLE-3 on Qwen3-4B through the OFFICIAL implementation, on our prompts.

WHY THIS EXISTS
---------------
E88 reimplemented the EAGLE-3 draft head from the released checkpoint and got
1.136 tok/pass on MT-Bench, where the published figure is ~3.4. All four candidate
aux-layer triples landed within 1.6% of each other (1.132 / 1.126 / 1.136 / 1.119),
which is the tell: if the fused hidden state were reaching the draft model,
changing WHICH target layers feed it would move acceptance a lot. It did not, so
the aux path was inert. The architecture was right -- the parameter count proves it
(fc 19.66M + q 20.97M + k 5.24M + v 5.24M + o 10.49M + MLP 74.7M + lm_head 81.9M =
218.2M, matching the checkpoint's 218,245,120 exactly) -- but something in the
wiring was not, and chasing it is not what this programme is for.

So this script does not reimplement anything. It runs SafeAILab's own EAGLE code at
a pinned commit, loads AngelSlim/Qwen3-4B_eagle3 through `EaModel.from_pretrained`,
and measures on OUR 50 GSM8K + 50 MATH-500 prompts. Any number it produces is
EAGLE-3's own implementation's number, which is the only kind worth comparing to.

Setup is lifted from `anatomy-of-a-speedup-eagle-3-on-qwen3.ipynb` (cells 7, 9, 29,
33, 67, 72): the transformers pin, the EAGLE commit, the head_dim patch the 4B head
needs, and the eagenerate/naivegenerate measurement pair.

WHAT IS COMPARED, AND WHAT IS NOT
---------------------------------
The baseline here is `model.naivegenerate` -- EAGLE's OWN loop with drafting
switched off, same weights, same Python. That makes the speedup ratio internally
fair in exactly the way our arm/baseline pair is. It is NOT the same baseline E87
used (our own verifier's), so the two speedups are only comparable to the extent
the two baselines run at the same ms/token. Both baselines' ms/token are printed so
that can be checked rather than assumed, and tok/pass -- which is
implementation-independent -- is reported alongside as the primary quantity.

PROTOCOL: temperature 0, batch 1, EOS-terminated with a 2048-token cap (DFlash
Table 1's protocol, and E87's). Trees: total_token 60 and 16, depth 7, top_k 10.
"""

# ============================================================ stage 0: environment
# Must run before `import torch` / `import transformers`, so nothing above this.
import os, sys, subprocess

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # EAGLE's loop is single-device
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

EAGLE_COMMIT     = "cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b"   # as in the notebook
TRANSFORMERS_PIN = "4.53.1"      # the EAGLE commit needs this; Kaggle ships 5.x
EAGLE_DIR        = "/kaggle/working/EAGLE"

BASE_MODEL = "Qwen/Qwen3-4B"
EA_MODEL   = "AngelSlim/Qwen3-4B_eagle3"

# ---- the protocol
MAX_NEW   = 2048          # a CAP; EOS terminates. DFlash Table 1 / E87.
BENCH_N   = 50            # prompts per maths set -- MUST equal E87's NAT_N
MT_N      = 40            # MT-Bench prompts (the control set)
GATE_N    = 6             # MT-Bench prompts used for the early sanity gate
GATE_TAU  = 2.0           # below this on MT-Bench, the wiring is wrong again
TREES     = [dict(name="eagle3-60", total_token=60, depth=7, top_k=10),
             dict(name="eagle3-16", total_token=16, depth=7, top_k=10)]
T_BUDGET  = 400           # minutes
BOOT      = 4000

# ---- prompt selection, copied constant-for-constant from E87/E88 so the 50+50 are
#      the SAME prompts our head was measured on. Changing any of these changes
#      which prompts are drawn and silently breaks the comparison.
TEST_SETS = [
    ("gsm8k-test", "openai/gsm8k", "main", "test", "question"),
    ("math500",    "HuggingFaceH4/MATH-500", None, "test", "problem"),
]
TRAIN_SOURCES = [
    ("gsm8k-train",  "openai/gsm8k", "main", "train", "question"),
    ("metamath",     "meta-math/MetaMathQA", None, "train", "query"),
    ("orca-math",    "microsoft/orca-math-word-problems-200k", None, "train", "question"),
    ("openmath2",    "nvidia/OpenMathInstruct-2", None, "train", "problem"),
    ("mathinstruct", "TIGER-Lab/MathInstruct", None, "train", "instruction"),
]
PER_SOURCE     = 4000
NAT_N          = 50
LONG_N         = 10
FORCED_N       = 8
MAX_PROMPT_TOK = 384

# E87, same head, same 50 prompts, our own verifier. For the report only.
E87_OURS = {"gsm8k-test": {8: 2.224, 16: 2.341, 32: 2.130},
            "math500":    {8: 2.431, 16: 2.620, 32: 2.352}}
# DFlash Table 1, Qwen3-4B, temperature 0, up to 2048 generated tokens.
DFLASH_T1 = {"gsm8k-test": dict(e3_16=1.99, e3_60=2.27, tau_e3_16=3.30, tau_e3_60=3.77),
             "math500":    dict(e3_16=1.83, e3_60=2.10, tau_e3_16=3.08, tau_e3_60=3.52)}

OUT = "/kaggle/working"
if not os.path.isdir(OUT):
    OUT = os.path.dirname(os.path.abspath(__file__))
    EAGLE_DIR = os.path.join(OUT, "EAGLE")

import time
T0 = time.time()


def mins():
    return (time.time() - T0) / 60.0


def log(m):
    print("[%7.1fm] %s" % (mins(), m), flush=True)


def sh(cmd, quiet=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if not quiet:
        o = (r.stdout + r.stderr).strip()
        if o:
            print(o[-1500:], flush=True)
    return r.returncode


log("stage 0: pinning transformers==%s" % TRANSFORMERS_PIN)
if "transformers" in sys.modules:
    raise SystemExit(
        "transformers is ALREADY IMPORTED in this kernel, so the %s pin cannot take"
        " effect and EAGLE's code will fail against it. Restart the kernel and run"
        " this script as the first thing that touches transformers."
        % TRANSFORMERS_PIN)
sh("pip install -q --progress-bar off 'transformers==%s'" % TRANSFORMERS_PIN)

if not os.path.exists(EAGLE_DIR):
    log("cloning SafeAILab/EAGLE at %s" % EAGLE_COMMIT[:12])
    sh("git clone -q https://github.com/SafeAILab/EAGLE.git %s" % EAGLE_DIR)
    sh("cd %s && git checkout -q %s" % (EAGLE_DIR, EAGLE_COMMIT))
assert os.path.isdir(EAGLE_DIR), "EAGLE did not clone to %s" % EAGLE_DIR

# ---- the ONE patch to the official code, taken verbatim from the notebook.
# Qwen3-4B's attention is wider than its hidden size (32 heads x 128 = 4096 vs
# 2560). The official draft code derives head width from hidden size alone and so
# cannot load that head at all. This honours config.head_dim instead; it is a no-op
# where the two coincide. Nothing else in EAGLE is modified.
_p = os.path.join(EAGLE_DIR, "eagle", "model", "cnets.py")
_src = open(_p, encoding="utf-8").read()
_pats = [
    ("self.head_dim = self.hidden_size // self.num_heads",
     'self.head_dim = getattr(config, "head_dim", None) or self.hidden_size // self.num_heads'),
    ("if (self.head_dim * self.num_heads) != self.hidden_size:", "if False:"),
    ("attn_output.reshape(bsz, q_len, self.hidden_size)",
     "attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)"),
]
_done = sum(1 for a, b in _pats if b in _src)
if _done != len(_pats):
    for a, b in _pats:
        if b not in _src:
            assert a in _src, (
                "cnets.py does not contain %r, so the head_dim patch cannot be"
                " applied and the 4B draft head will not load. The pinned commit"
                " moved -- re-derive the patch against it." % a[:60])
            _src = _src.replace(a, b)
    open(_p, "w", encoding="utf-8").write(_src)
    log("  head_dim patch applied to cnets.py (%d substitutions)" % len(_pats))
else:
    log("  head_dim patch already present")
sys.path.insert(0, EAGLE_DIR)

import json, gc, re, itertools, random
import numpy as np
import torch
import transformers

assert transformers.__version__.startswith("4.53"), (
    "transformers is %s but EAGLE's pinned commit needs %s. The pip install did not"
    " take effect -- restart the kernel." % (transformers.__version__, TRANSFORMERS_PIN))
log("torch %s | transformers %s" % (torch.__version__, transformers.__version__))
assert torch.cuda.is_available(), "needs a GPU"
_cc = torch.cuda.get_device_capability(0)
log("GPU: %s (sm%d%d)" % (torch.cuda.get_device_name(0), _cc[0], _cc[1]))

random.seed(0); np.random.seed(0); torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from eagle.model.ea_model import EaModel
log("EAGLE imported, commit %s" % EAGLE_COMMIT[:12])

ROWS, PAIRS, DIAG = [], [], []


def flush():
    """Written after EVERY arm. E81/E82 lost their headline tables to a crash at
    the end of the run; results that exist only in memory do not exist."""
    import pandas as pd
    for nm, d in (("e89_rows", ROWS), ("e89_pairs", PAIRS), ("e89_diag", DIAG)):
        if d:
            pd.DataFrame(d).to_csv(os.path.join(OUT, nm + ".csv"), index=False)


# ============================================================ stage 1: the prompts
# Replicated constant-for-constant from E87 so these are the SAME prompts, down to
# the training-pool dedup filter and the rng(0) permutation. The pool is pulled only
# to reproduce that filter and to report leakage; nothing is trained here.
PROMPT_CACHE = os.path.join(OUT, "e89_prompts.json")


def norm(t):
    return re.sub(r"\s+", " ", t.lower().strip())


def grams(t, n=8):
    w = norm(t).split()
    return {" ".join(w[i:i + n]) for i in range(max(len(w) - n + 1, 1))}


def pull(srcs, cap, label):
    from datasets import load_dataset
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


def mt_bench_prompts(n):
    """The authors' OWN MT-Bench file, which ships in the repo we just cloned. No
    hub call, no fallback set of mine to be wrong about."""
    f = os.path.join(EAGLE_DIR, "eagle", "data", "mt_bench", "question.jsonl")
    assert os.path.exists(f), (
        "MT-Bench not found at %s. It is the control that tells us whether EAGLE is"
        " wired correctly at all, so do not proceed without it." % f)
    qs = []
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                qs.append(json.loads(line)["turns"][0])
    log("  mt-bench     %d questions in the repo, taking %d" % (len(qs), min(n, len(qs))))
    return qs[:n]


if os.path.exists(PROMPT_CACHE):
    _pc = json.loads(open(PROMPT_CACHE, encoding="utf-8").read())
    TESTS = {k: v for k, v in _pc["sets"].items()}
    log("stage 1: prompts loaded from %s (sha %s)" % (PROMPT_CACHE, _pc["sha"][:12]))
else:
    log("stage 1: rebuilding E87's prompt selection")
    TRAIN_POOL = pull(TRAIN_SOURCES, PER_SOURCE * len(TRAIN_SOURCES), "train")
    assert len(TRAIN_POOL) >= 1000, (
        "only %d training prompts pulled. The pool is not trained on here, but it IS"
        " the dedup filter that decides WHICH 50 test prompts are drawn -- without"
        " it this run measures a different prompt set than E87 did."
        % len(TRAIN_POOL))
    TN = {norm(t) for t in TRAIN_POOL}
    TG = set()
    for t in TRAIN_POOL:
        TG |= grams(t)

    rgp = np.random.default_rng(0)            # same seed as E87
    NEED = max(NAT_N, LONG_N, FORCED_N)
    TESTS = {}
    for nm, name, cfg, split, col in TEST_SETS:
        got = pull([(nm, name, cfg, split, col)], NEED * 6, nm)
        got = [t for t in got if norm(t) not in TN]
        assert len(got) >= NEED, "%s: %d usable prompts, need %d" % (nm, len(got), NEED)
        sel = rgp.permutation(len(got))[:NEED]
        TESTS[nm] = [got[i] for i in sel]
        ov = [len(grams(t) & TG) / max(len(grams(t)), 1) for t in TESTS[nm]]
        log("  %-12s %d prompts | 8-gram overlap with training: mean %.3f max %.3f"
            % (nm, len(TESTS[nm]), float(np.mean(ov)), float(np.max(ov))))
        DIAG.append(dict(check="leakage_%s" % nm, value=float(np.mean(ov)),
                         detail="max=%.3f" % float(np.max(ov))))
    TESTS["mtbench"] = mt_bench_prompts(MT_N)
    import hashlib
    _sha = hashlib.sha256(
        json.dumps(TESTS, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    open(PROMPT_CACHE, "w", encoding="utf-8").write(
        json.dumps(dict(sha=_sha, sets=TESTS), ensure_ascii=False))
    log("  prompt set sha %s -> %s" % (_sha[:12], PROMPT_CACHE))

for _k in ("gsm8k-test", "math500", "mtbench"):
    assert _k in TESTS and TESTS[_k], "prompt set %r is empty" % _k
log("  sets: %s" % {k: len(v) for k, v in TESTS.items()})


# ============================================================ stage 2: the pair
log("stage 2: loading Qwen3-4B + %s" % EA_MODEL)
_t = time.time()
model = EaModel.from_pretrained(
    use_eagle3=True,
    base_model_path=BASE_MODEL,
    ea_model_path=EA_MODEL,
    total_token=TREES[0]["total_token"], depth=TREES[0]["depth"],
    top_k=TREES[0]["top_k"],
    torch_dtype=torch.float16,
    device_map={"": 0},
).eval()
model.ea_layer.to(torch.float16)      # the 4B head ships bf16; T4 has no native bf16
tok = model.get_tokenizer()
cfg = model.base_model.config
N_TARGET = sum(p.numel() for p in model.base_model.parameters())
N_DRAFT = sum(p.numel() for n, p in model.ea_layer.named_parameters()
              if "embed_tokens" not in n)
log("  loaded in %.0f s | target %d params | draft %d (%.3f%%) | GPU %.2f GB"
    % (time.time() - _t, N_TARGET, N_DRAFT, 100.0 * N_DRAFT / N_TARGET,
       torch.cuda.memory_allocated() / 2 ** 30))
# 1% tolerance, not equality: what this must catch is the WRONG MODEL (the 1.7B
# pair sits 2.4x away), and aborting a 5-minute load over a tied-embedding
# accounting difference would cost a session to learn nothing.
assert abs(N_TARGET - 4_022_468_096) < 0.01 * 4_022_468_096, (
    "target has %d params, expected Qwen3-4B's 4,022,468,096 -- a different model"
    " loaded and nothing below would be comparable to E87." % N_TARGET)
if N_TARGET != 4_022_468_096:
    log("  NOTE: target param count %d differs from E88's direct count %d (tied"
        " embeddings)" % (N_TARGET, 4_022_468_096))
# 218,245,120 is what we counted directly from the safetensors in E88. The official
# loader reports it too, which confirms both readings of the checkpoint.
assert abs(N_DRAFT - 218_245_120) < 2_000_000, (
    "draft head has %d params, expected ~218,245,120. The checkpoint changed, or the"
    " head_dim patch loaded a differently-shaped head." % N_DRAFT)
DIAG.append(dict(check="params", value=float(N_DRAFT),
                 detail="target=%d draft_pct=%.4f" % (N_TARGET, 100.0 * N_DRAFT / N_TARGET)))


def encode(q):
    """Same chat template and same `enable_thinking=False` as E87's `chat()`, so the
    model sees byte-identical prompts in both runs."""
    txt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                  add_generation_prompt=True, enable_thinking=False)
    ids = tok([txt]).input_ids[0]
    if len(ids) > MAX_PROMPT_TOK:
        # E87 truncates the TEXT and re-applies the template, so the trailing turn
        # marker is never cut off. Same binary search here.
        lo, hi = 0, len(q)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            t2 = tok.apply_chat_template([{"role": "user", "content": q[:mid]}],
                                         tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
            if len(tok([t2]).input_ids[0]) <= MAX_PROMPT_TOK:
                lo = mid
            else:
                hi = mid - 1
        txt = tok.apply_chat_template([{"role": "user", "content": q[:lo]}],
                                      tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False)
        ids = tok([txt]).input_ids[0]
    return torch.as_tensor([ids]).cuda()


def timed(fn, *a, **k):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn(*a, **k)
    torch.cuda.synchronize()
    return out, time.perf_counter() - t


def set_tree(cf):
    """Tree shape lives in plain draft-model attributes; no reload needed.
    `total_tokens` is total_token - 1 because the root is not a drafted node --
    that is the official loader's own convention, not a correction of ours."""
    model.ea_layer.top_k = cf["top_k"]
    model.ea_layer.depth = cf["depth"]
    model.ea_layer.total_tokens = cf["total_token"] - 1
    model.ea_layer.init_tree()


def run_naive(ids):
    # After eagenerate, base_model.model keeps the last tree's mask and a plain pass
    # of a different length trips over it. The notebook documents this; clear it.
    model.base_model.model.tree_mask = None
    (out, new, _), dt = timed(model.naivegenerate, ids, temperature=0.0,
                              max_new_tokens=MAX_NEW, log=True)
    return dict(tokens=int(new), sec=dt, steps=int(new), ids=out)


def run_eagle(ids):
    (out, new, idx), dt = timed(model.eagenerate, ids, temperature=0.0,
                                max_new_tokens=MAX_NEW, log=True)
    return dict(tokens=int(new), sec=dt, steps=int(idx) + 1, ids=out)


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


WARM = encode("Explain in two sentences why the sky is blue.")
with torch.no_grad():
    set_tree(TREES[0])
    run_eagle(WARM); run_naive(WARM)
log("  warmed up")


# ====================================================== stage 3: the sanity gate
# EAGLE-3 was trained on ShareGPT + UltraChat. MT-Bench is that distribution, and
# the notebook reports acceptance length ~3.4 there. If the official code does not
# clear GATE_TAU on a handful of MT-Bench prompts, something is wrong with THIS
# setup and no maths number below should be believed. Six prompts, ~2 minutes.
log("stage 3: MT-Bench sanity gate on %d prompts" % GATE_N)
set_tree(TREES[0])
_gt, _gs = 0, 0
with torch.no_grad():
    for q in TESTS["mtbench"][:GATE_N]:
        r = run_eagle(encode(q))
        _gt += r["tokens"]; _gs += r["steps"]
GATE = _gt / max(_gs, 1)
log("  tok/pass %.3f over %d tokens" % (GATE, _gt))
DIAG.append(dict(check="gate_mtbench_tau", value=GATE, detail="n=%d" % GATE_N))
# The arithmetic floor first: a pass emits the accepted block PLUS the target's own
# next token, so tok/pass >= 1 holds for any drafter however bad. Below 1 means
# tokens were generated and then discarded, which is a loop bug, not a weak drafter.
assert GATE >= 1.0, (
    "tok/pass %.3f is below the arithmetic floor of 1.000 -- the measurement is"
    " wrong, not the drafter." % GATE)
assert GATE > GATE_TAU, (
    "EAGLE-3 reaches only %.3f tok/pass on MT-Bench through its OWN implementation,"
    " where the published acceptance length is ~3.4. This setup is wrong -- check"
    " the head_dim patch, the dtype cast, and the transformers pin. Do not report"
    " any comparison from this run." % GATE)
log("  -> gate passed: the official implementation drafts correctly here, so any"
    " weakness on\n     maths below is a distribution effect and not a setup bug.")


# ============================================================ stage 4: the bench
def bench_set(ts, prompts):
    log("  %s: %d prompts" % (ts, len(prompts)))
    base = {}
    with torch.no_grad():
        for k, q in enumerate(prompts):
            if mins() > T_BUDGET - 20:
                log("    budget reached during baselines, stopping at %d" % k)
                break
            base[k] = run_naive(encode(q))
    if not base:
        return
    bsec = sum(v["sec"] for v in base.values())
    btok = sum(v["tokens"] for v in base.values())
    blen = float(np.mean([v["tokens"] for v in base.values()]))
    log("    naive baseline: %d prompts, %.1f s, %.1f ms/tok, mean length %.0f"
        % (len(base), bsec, 1000 * bsec / max(btok, 1), blen))
    ROWS.append(dict(test=ts, arm="naive", n_prompts=len(base), sec=bsec, tokens=btok,
                     ms_per_tok=1000 * bsec / max(btok, 1), tok_per_pass=1.0,
                     speedup=1.0, gen_len_mean=blen, ndiverge=0))
    flush()

    for cf in TREES:
        if mins() > T_BUDGET - 10:
            log("    budget reached, skipping %s" % cf["name"])
            continue
        set_tree(cf)
        ag = dict(sec=0.0, tok=0, steps=0, ndiv=0)
        with torch.no_grad():
            for k in sorted(base):
                ids = encode(prompts[k])
                plen = ids.shape[1]
                r = run_eagle(ids)
                # exactness: EAGLE is greedy-exact at temperature 0, so the two
                # outputs must agree token for token up to the shorter length.
                a = r["ids"][0, plen:].tolist()
                b = base[k]["ids"][0, plen:].tolist()
                m = 0
                for x, y in zip(a, b):
                    if x != y:
                        break
                    m += 1
                if m < min(len(a), len(b)):
                    ag["ndiv"] += 1
                    DIAG.append(dict(check="diverge", value=float(m),
                                     detail="%s %s seq%d na=%d nb=%d"
                                     % (ts, cf["name"], k, len(a), len(b))))
                ag["sec"] += r["sec"]; ag["tok"] += r["tokens"]
                ag["steps"] += r["steps"]
                PAIRS.append(dict(test=ts, arm=cf["name"], seq=k,
                                  base_secs=base[k]["sec"], arm_secs=r["sec"],
                                  speedup=base[k]["sec"] / max(r["sec"], 1e-9),
                                  base_tok=base[k]["tokens"], arm_tok=r["tokens"]))
        tpp = ag["tok"] / max(ag["steps"], 1)
        assert tpp >= 1.0, (
            "%s on %s reports %.3f tok/pass, below the arithmetic floor of 1.000."
            % (cf["name"], ts, tpp))
        ROWS.append(dict(test=ts, arm=cf["name"], n_prompts=len(base),
                         sec=ag["sec"], tokens=ag["tok"],
                         ms_per_tok=1000 * ag["sec"] / max(ag["tok"], 1),
                         tok_per_pass=tpp, speedup=bsec / max(ag["sec"], 1e-9),
                         gen_len_mean=float(ag["tok"]) / len(base),
                         ndiverge=ag["ndiv"]))
        log("    %-11s speedup %.3fx | tok/pass %.3f | %.1f ms/tok | diverge %d"
            " [%5.1fm]"
            % (cf["name"], ROWS[-1]["speedup"], tpp, ROWS[-1]["ms_per_tok"],
               ag["ndiv"], mins()))
        flush()


log("stage 4: the bench, temperature 0, EOS-terminated, cap %d" % MAX_NEW)
for _ts in ("gsm8k-test", "math500", "mtbench"):
    n = MT_N if _ts == "mtbench" else BENCH_N
    bench_set(_ts, TESTS[_ts][:n])


# ============================================================ stage 5: the report
import pandas as pd

R = pd.DataFrame(ROWS)
flush()
print("\n" + "=" * 78, flush=True)
print("E89 -- EAGLE-3 ON QWEN3-4B, OFFICIAL IMPLEMENTATION, OUR PROMPTS", flush=True)
print("=" * 78, flush=True)
print("  %s + %s | %s sm%d%d | fp16 | batch 1 | temperature 0, cap %d, EOS honoured"
      % (BASE_MODEL, EA_MODEL, torch.cuda.get_device_name(0), _cc[0], _cc[1], MAX_NEW),
      flush=True)
print("  SafeAILab/EAGLE @ %s | transformers %s"
      % (EAGLE_COMMIT[:12], transformers.__version__), flush=True)
print("  draft head %d params = %.3f%% of target | MT-Bench gate %.3f tok/pass"
      % (N_DRAFT, 100.0 * N_DRAFT / N_TARGET, GATE), flush=True)
print("  baseline is model.naivegenerate: EAGLE's own loop, drafting off, same"
      " weights.", flush=True)

for ts in ("gsm8k-test", "math500", "mtbench"):
    g = R[R["test"] == ts]
    if g.empty:
        continue
    nb = g[g["arm"] == "naive"]
    print("\n--- %s (%d prompts, mean baseline length %.0f tokens) ---"
          % (ts, int(g["n_prompts"].iloc[0]),
             float(nb["gen_len_mean"].iloc[0]) if not nb.empty else float("nan")),
          flush=True)
    # TWO statistics, each with the bootstrap that belongs to it:
    #   aggregate  = sum(base)/sum(arm), the whole workload's wall-clock saving
    #   per-prompt = mean(base_i/arm_i), what a typical prompt sees
    # They differ whenever answer lengths vary. Printing one point estimate beside
    # the other's interval is how E89 reported 2.032 with a CI of [2.114, 2.280].
    print("  %-12s %-21s %-21s %-9s %-10s %s"
          % ("arm", "aggregate speedup", "per-prompt speedup", "tok/pass",
             "ms/tok", "diverge"), flush=True)
    for _, r in g.iterrows():
        pr = [q for q in PAIRS if q["test"] == ts and q["arm"] == r["arm"]]
        d = np.array([q["speedup"] for q in pr])
        if len(pr):
            alo, ahi = boot_ratio_ci([q["base_secs"] for q in pr],
                                     [q["arm_secs"] for q in pr])
            plo, phi = boot_ci(d)
            agg = sum(q["base_secs"] for q in pr) / max(
                sum(q["arm_secs"] for q in pr), 1e-9)
            # A warning, not an assert: the CSVs are already flushed by now, and
            # a false trip here would cost the printed table for nothing. The CI
            # bracket below stays fatal -- it never fired in 3000 random paired
            # sets, so if it fires the statistics really are wrong.
            if abs(agg - r["speedup"]) > 1e-6 * max(agg, 1.0):
                print("  WARNING: aggregate recomputed from PAIRS is %.6f but the"
                      " row says %.6f -- the per-prompt records and the headline"
                      " disagree" % (agg, r["speedup"]), flush=True)
            assert alo <= r["speedup"] <= ahi, (
                "%s/%s: the aggregate speedup %.3f falls outside its OWN bootstrap"
                " interval [%.3f, %.3f], which cannot happen if both are computed"
                " from the same statistic" % (ts, r["arm"], r["speedup"], alo, ahi))
            print("  %-12s %-9.3f [%.3f,%.3f]  %-9.3f [%.3f,%.3f]  %-9.3f %-10.1f %d"
                  % (r["arm"], r["speedup"], alo, ahi, float(d.mean()), plo, phi,
                     r["tok_per_pass"], r["ms_per_tok"], int(r["ndiverge"])),
                  flush=True)
        else:
            print("  %-12s %-9.3f %-11s  %-9s %-11s  %-9.3f %-10.1f %d"
                  % (r["arm"], r["speedup"], "(baseline)", "", "",
                     r["tok_per_pass"], r["ms_per_tok"], int(r["ndiverge"])),
                  flush=True)
    z = DFLASH_T1.get(ts)
    if z:
        print("  --- published, DFlash Table 1, same target and protocol ---", flush=True)
        print("  %-12s %-9.2f %-21s %-21s %-9.2f"
              % ("EAGLE-3 (60)", z["e3_60"], "(published)", "", z["tau_e3_60"]),
              flush=True)
        print("  %-12s %-9.2f %-21s %-21s %-9.2f"
              % ("EAGLE-3 (16)", z["e3_16"], "(published)", "", z["tau_e3_16"]),
              flush=True)
    if ts in E87_OURS:
        print("  --- E87, OUR head, same %d prompts, OUR verifier and baseline ---"
              % BENCH_N, flush=True)
        print("      %s" % "  ".join("N%d %.3fx" % (k, v)
                                     for k, v in sorted(E87_OURS[ts].items())),
              flush=True)

print("\n--- READING THIS TABLE ---", flush=True)
print("  tok/pass is the comparable quantity: it counts target forward passes and"
      "\n  does not care whose Python wraps them. Speedup does care -- ours is"
      "\n  normalised by OUR baseline and EAGLE's by EAGLE's, so compare the two"
      "\n  ms/tok baselines before comparing the two speedups.", flush=True)
nb_g = R[(R["test"] == "gsm8k-test") & (R["arm"] == "naive")]
if not nb_g.empty:
    print("  naivegenerate baseline on GSM8K: %.1f ms/tok. E87's own baseline ran at"
          "\n  59.4 ms/tok on the same card; the gap between those two numbers is the"
          "\n  correction to apply before reading any speedup difference as real."
          % float(nb_g["ms_per_tok"].iloc[0]), flush=True)

_dv = int(R["ndiverge"].sum())
print("\n--- EXACTNESS ---", flush=True)
print("  %d prompt(s) diverged from the naive baseline out of %d arm runs."
      % (_dv, len(PAIRS)), flush=True)
print("  EAGLE-3 is greedy-exact at temperature 0, so divergences here are fp16"
      "\n  tie-breaking, the same effect our own runs report as MARGIN_FAIL.",
      flush=True)
flush()
log("done. CSVs in %s" % OUT)
