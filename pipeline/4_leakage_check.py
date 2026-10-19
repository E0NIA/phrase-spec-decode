"""E90 -- the MATH-500 leakage filter, and the headline recomputed without it.

CPU only. No GPU, no torch, no model. Nothing is re-benched: E89 and E87 both
recorded PER-PROMPT base and arm wall times, so dropping a contaminated prompt and
recomputing the headline is arithmetic.

WHY THIS MATTERS HERE SPECIFICALLY
----------------------------------
E89 measured our 27% win over EAGLE-3 on MATH-500, and MATH-500 is also where the
8-gram overlap with our training pool runs mean 0.052 / max 0.833, against GSM8K's
mean 0.003 / max 0.083. The asymmetry is not incidental: our head was trained on
completions that Qwen3-4B itself generated from MetaMathQA, OpenMathInstruct-2 and
friends, several of which are derived from the same problem sources as MATH-500.
EAGLE-3 was trained on ShareGPT and UltraChat. So a leaked prompt is one where our
head has effectively seen this model's own solution to this problem and EAGLE's has
not -- it moves the comparison in our favour and in no one else's. The win is not
reportable until the leaked prompts are out.

WHAT "LEAKED" MEANS, AND WHY A SWEEP RATHER THAN A THRESHOLD
------------------------------------------------------------
Overlap is |8-grams(test) INTERSECT 8-grams(train)| / |8-grams(test)|. Any single
cut-off is arbitrary: 8-gram coincidences do happen innocently in maths phrasing
("what is the value of x such that"), while 0.833 is memorisation. So this reports
the whole distribution and recomputes the headline at several cut-offs. If the
conclusion survives the strictest one it does not depend on where the line is
drawn, and that is the only version worth putting in a paper.
"""
import ast, io, itertools, json, os, re, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KAGGLE = "/kaggle/working"
# --dir <path> searched first, so the CSVs can be downloaded anywhere
_cli = []
if "--dir" in sys.argv:
    _cli = [sys.argv[sys.argv.index("--dir") + 1]]
DIRS = [d for d in _cli + [KAGGLE, HERE] if os.path.isdir(d)]
assert DIRS, "no readable directory among %s" % (_cli + [KAGGLE, HERE])

PROMPTS_JSON = "e89_prompts.json"
POOL_CACHE = "e90_train_grams.json"       # so the pull happens at most once
PAIR_FILES = [("E89 EAGLE-3", "e89_pairs.csv"), ("E87 ours", "e87_pairs.csv")]
THRESHOLDS = [1.01, 0.50, 0.25, 0.10, 1e-9]   # 1.01 = keep everything
BOOT = 4000
N_GRAM = 8


def find(name):
    for d in DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def say(m):
    print(m, flush=True)


# ---- the prompt-selection constants are READ from E87, never retyped. A
# hand-copied table is how PER_SOURCE sat at 20000 in two scripts while E87 used
# 4000, which silently changed which 50 prompts were drawn.
def e87_consts():
    p = find("e87_train_head.py") or os.path.join(HERE, "e87_train_head.py")
    if not os.path.exists(p):
        # Fallback so this runs on a Kaggle session that has the CSVs but not the
        # trainer. Stated loudly, because a copied constant table is exactly how
        # PER_SOURCE sat at 20000 in two scripts while E87 used 4000.
        say("  WARNING: e87_train_head.py not found, using INLINED constants."
            "\n  Put the trainer beside this script to have them verified"
            " against the source.")
        return dict(PER_SOURCE=4000, TRAIN_SOURCES=[
            ("gsm8k-train", "openai/gsm8k", "main", "train", "question"),
            ("metamath", "meta-math/MetaMathQA", None, "train", "query"),
            ("orca-math", "microsoft/orca-math-word-problems-200k", None,
             "train", "question"),
            ("openmath2", "nvidia/OpenMathInstruct-2", None, "train", "problem"),
            ("mathinstruct", "TIGER-Lab/MathInstruct", None, "train",
             "instruction")])
    t = ast.parse(io.open(p, encoding="utf-8").read())
    c = {}
    for n in t.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            try:
                c[n.targets[0].id] = ast.literal_eval(n.value)
            except Exception:
                pass
    for k in ("TRAIN_SOURCES", "PER_SOURCE"):
        assert k in c, "E87 has no module-level %s" % k
    return c


def norm(t):
    return re.sub(r"\s+", " ", t.lower().strip())


def grams(t, n=N_GRAM):
    w = norm(t).split()
    return {" ".join(w[i:i + n]) for i in range(max(len(w) - n + 1, 1))}


# ============================================================ 1. the prompts
pj = find(PROMPTS_JSON)
assert pj, (
    "%s not found in %s. It is written by E89 stage 1 and holds the exact prompts"
    " that were benched; without it this filter would be guessing at a different"
    " prompt set." % (PROMPTS_JSON, DIRS))
_pc = json.loads(io.open(pj, encoding="utf-8").read())
TESTS = _pc["sets"]
say("prompts: %s (sha %s)" % ({k: len(v) for k, v in TESTS.items()}, _pc["sha"][:12]))


# ============================================================ 2. the training pool
cache = find(POOL_CACHE)
if cache:
    _cc = json.loads(io.open(cache, encoding="utf-8").read())
    TG = set(_cc["grams"])
    TN = set(_cc["norms"])
    say("training pool: %d 8-grams, %d normalised prompts (from %s)"
        % (len(TG), len(TN), POOL_CACHE))
else:
    C87 = e87_consts()
    say("training pool: pulling %d sources x %d (E87's constants)"
        % (len(C87["TRAIN_SOURCES"]), C87["PER_SOURCE"]))
    try:
        from datasets import load_dataset
    except ImportError:
        say("\n`datasets` is not installed here. Run this script where it is"
            "\n(the Kaggle session that produced the CSVs), or install it:"
            "\n    pip install datasets"
            "\nIt writes %s, after which this script runs anywhere." % POOL_CACHE)
        sys.exit(2)
    got, PER = [], C87["PER_SOURCE"]
    for nm, name, cfg, split, col in C87["TRAIN_SOURCES"]:
        n0 = len(got)
        try:
            d = load_dataset(name, cfg, split=split, streaming=True) if cfg \
                else load_dataset(name, split=split, streaming=True)
            for row in itertools.islice(iter(d), 60000):
                t = row.get(col)
                if isinstance(t, str) and 40 <= len(t) <= 600:
                    got.append(t.strip())
                if len(got) - n0 >= PER:
                    break
        except Exception as e:
            say("  %-12s unavailable (%s)" % (nm, str(e).split("\n")[0][:60]))
            continue
        say("  %-12s +%d (total %d)" % (nm, len(got) - n0, len(got)))
    seen, pool = set(), []
    for t in got:
        if t not in seen:
            seen.add(t); pool.append(t)
    assert len(pool) >= 1000, (
        "only %d training prompts pulled; the overlap figures would be meaningless"
        % len(pool))
    TN = {norm(t) for t in pool}
    TG = set()
    for t in pool:
        TG |= grams(t)
    io.open(os.path.join(DIRS[0], POOL_CACHE), "w", encoding="utf-8").write(
        json.dumps(dict(grams=sorted(TG), norms=sorted(TN))))
    say("  %d prompts -> %d 8-grams, cached to %s"
        % (len(pool), len(TG), os.path.join(DIRS[0], POOL_CACHE)))


# ============================================================ 3. per-prompt overlap
OV = {}
for ts, prompts in TESTS.items():
    o = []
    for t in prompts:
        g = grams(t)
        o.append(len(g & TG) / max(len(g), 1))
    OV[ts] = np.array(o)

say("\n--- 8-gram overlap with the training pool, per test set ---")
say("  %-12s %-6s %-7s %-7s %-7s %-7s %-7s %s"
    % ("set", "n", "mean", "median", "p90", "max", "dup", ">0.5"))
for ts in TESTS:
    o = OV[ts]
    dup = sum(1 for t in TESTS[ts] if norm(t) in TN)
    say("  %-12s %-6d %-7.3f %-7.3f %-7.3f %-7.3f %-7d %d"
        % (ts, len(o), o.mean(), float(np.median(o)),
           float(np.percentile(o, 90)), o.max(), dup, int((o > 0.5).sum())))

say("\n--- the worst offenders ---")
for ts in TESTS:
    o = OV[ts]
    idx = np.argsort(-o)[:4]
    say("  %s:" % ts)
    for i in idx:
        if o[i] <= 0:
            continue
        say("    seq %-3d overlap %.3f | %s" % (i, o[i], TESTS[ts][i][:84]))

# An exact normalised duplicate is not a judgement call, so it is asserted rather
# than swept: if a test prompt IS a training prompt, no threshold choice saves it.
for ts in TESTS:
    d = [i for i, t in enumerate(TESTS[ts]) if norm(t) in TN]
    if d:
        say("\n  NOTE: %s has %d EXACT normalised duplicate(s) of training prompts"
            " at seq %s.\n  These are dropped at every threshold below."
            % (ts, len(d), d))


# ============================================================ 4. the recomputation
def boot_ci(d):
    d = np.asarray(d, float)
    if d.size < 2:
        return float("nan"), float("nan")
    r = np.random.default_rng(11)
    bs = np.array([r.choice(d, d.size, replace=True).mean() for _ in range(BOOT)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_ratio_ci(b, a):
    """Interval for sum(b)/sum(a) -- the statistic the headline IS. The
    mean-of-ratios bootstrap belongs to a different number and, when per-prompt
    speedup correlates with answer length (which depth decay guarantees), it
    misses the aggregate essentially always: 400/400 in simulation."""
    b = np.asarray(b, float); a = np.asarray(a, float)
    assert b.shape == a.shape
    if b.size < 2:
        return float("nan"), float("nan")
    r = np.random.default_rng(11)
    ix = r.integers(0, b.size, (BOOT, b.size))
    s = b[ix].sum(1) / np.maximum(a[ix].sum(1), 1e-9)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def headline(df, keep):
    d = df[df["seq"].isin(keep)]
    if len(d) < 2:
        return None
    agg = d["base_secs"].sum() / max(d["arm_secs"].sum(), 1e-9)
    lo, hi = boot_ratio_ci(d["base_secs"].values, d["arm_secs"].values)
    assert lo <= agg <= hi, (
        "aggregate %.3f outside its own interval [%.3f, %.3f]" % (agg, lo, hi))
    per = float((d["base_secs"] / d["arm_secs"]).mean())
    plo, phi = boot_ci((d["base_secs"] / d["arm_secs"]).values)
    return dict(n=len(d), agg=agg, lo=lo, hi=hi, per=per, plo=plo, phi=phi)


FOUND = [(lab, find(f)) for lab, f in PAIR_FILES]
if not any(p for _, p in FOUND):
    say("\nNo per-prompt CSV found (%s). Nothing to recompute -- the overlap table"
        " above still stands." % ", ".join(f for _, f in PAIR_FILES))
    sys.exit(0)

for lab, path in FOUND:
    if not path:
        say("\n%s: %s not found, skipped. Copy it next to this script to include it."
            % (lab, dict(PAIR_FILES)[lab] if lab in dict(PAIR_FILES) else lab))
        continue
    P = pd.read_csv(path)
    need = {"test", "arm", "seq", "base_secs", "arm_secs"}
    assert need <= set(P.columns), (
        "%s is missing %s" % (path, need - set(P.columns)))
    if "tag" in P.columns:
        # E87 records two EOS-terminated families: "nat" is the data-ladder rungs
        # (arms 100%/50%/25%/12%) and "nsweep" is the N sweep (N8/N16/N32). The
        # quoted headline numbers are the N sweep, so keeping only "nat" dropped
        # exactly the rows that matter. Keep both and carry the tag in the label.
        P = P.copy()
        P["arm"] = P["tag"].astype(str) + "/" + P["arm"].astype(str)
        say("  tags present: %s" % sorted(P["tag"].unique()))
    say("\n" + "=" * 78)
    say("%s -- %s (%d per-prompt records)" % (lab, os.path.basename(path), len(P)))
    say("=" * 78)
    _cov = P.groupby(["test", "arm"])["seq"].nunique()
    _u = sorted(_cov.unique())
    say("  prompts per arm: %s" % (_u[0] if len(_u) == 1 else dict(_cov)))
    if len(_u) == 1 and _u[0] < 50:
        say("  NOTE: %d prompts per arm, not 50. A headline from this file is NOT"
            " on the\n  same sample as a 50-prompt run and must not be compared"
            " to one directly." % _u[0])
    for ts in [t for t in TESTS if (P["test"] == t).any()]:
        o = OV[ts]
        say("\n--- %s ---" % ts)
        say("  %-13s %-5s %-5s | %-10s %-22s | %s"
            % ("threshold", "kept", "drop", "arm", "aggregate speedup",
               "per-prompt mean"))
        for th in THRESHOLDS:
            keep = {i for i in range(len(o))
                    if o[i] < th and norm(TESTS[ts][i]) not in TN}
            lbl = ("all" if th > 1.0 else
                   ("overlap>0" if th < 1e-6 else "overlap>=%.2f" % th))
            for arm in sorted(P[P["test"] == ts]["arm"].unique()):
                d = P[(P["test"] == ts) & (P["arm"] == arm)]
                h = headline(d, keep)
                if h is None:
                    continue
                say("  %-13s %-5d %-5d | %-10s %.3f [%.3f,%.3f]      | %.3f"
                    " [%.3f,%.3f]"
                    % (lbl, h["n"], len(o) - h["n"], arm, h["agg"], h["lo"],
                       h["hi"], h["per"], h["plo"], h["phi"]))
            say("")

say("\n--- HOW TO READ THIS ---")
say("  Each block repeats the headline on a smaller, cleaner prompt set. If the")
say("  number moves materially as the threshold tightens, the result depended on")
say("  contaminated prompts and the strictest row is the one to report. If it")
say("  barely moves, leakage was not driving it and the full-set number stands.")
say("  `overlap>0` is the conservative bound: every prompt sharing even one")
say("  8-gram with training is gone, which also removes innocent coincidences,")
say("  so it UNDERSTATES us if anything.")
say("\n  Note the two sets are not symmetric. GSM8K's test split is held out from")
say("  its own train split, so its overlap is near zero and filtering changes")
say("  nothing. MATH-500 is the one at risk, and it is also where we win.")

# ---- the surviving prompt set, for any re-run
CLEAN = {ts: [t for i, t in enumerate(TESTS[ts])
              if OV[ts][i] <= 0 and norm(t) not in TN] for ts in TESTS}
op = os.path.join(DIRS[0], "e90_clean_prompts.json")
io.open(op, "w", encoding="utf-8").write(
    json.dumps(dict(sha=_pc["sha"], threshold="overlap>0", sets=CLEAN),
               ensure_ascii=False))
say("\nstrictest surviving set %s -> %s"
    % ({k: len(v) for k, v in CLEAN.items()}, op))
