"""Diagnostic: pooled vs within-suite AUROC of the released recipe (S1, layer 28 PCA-32), grouped 5-fold OOF.

Same data, folds (seeds 13, 0, 1, 2) and probe recipe as eval_two_stage.py. The suite-base-rate
row uses every label to order the four suites, so it is a diagnostic, not a held-out detector.
Usage: within_suite_auroc.py <chat dir> <vllm-hook repo> <out.json>
"""
import json, re, random, sys
from pathlib import Path
import numpy as np

chat, out = Path(sys.argv[1]), Path(sys.argv[3])
gen = [json.loads(l) for l in open(chat / "suite_full.jsonl")]
lab = {json.loads(l)["case_id"]: bool(json.loads(l)["unsafe"]) for l in open(chat / "suite_full.labels.jsonl")}
ids = [g["case_id"] for g in gen]
att = np.array([g["attack_present"] for g in gen]); uns = np.array([lab.get(c, False) and a for c, a in zip(ids, att)])
suite = np.array([c.split("_user_task")[0] for c in ids])
X = np.load(chat / "suite_full.npz")["layer_28"].astype(np.float64)
groups = np.array([(lambda m: f"{m.group(1) or 'workspace'}/{m.group(2)}")(re.match(r"(?:(\w+?)_)?(user_task_\d+)", c)) for c in ids])

def logreg_fit(X, y, l2=1.0, iters=500, lr=0.1):  # verbatim from commitment_probes.py
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd
    w = np.zeros(Xs.shape[1]); b = 0.0; n = len(y)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
        g = p - y
        w -= lr * (Xs.T @ g / n + l2 * w / n)
        b -= lr * g.mean()
    return mu, sd, w, b

def fit_score(tr, te, k=32):
    mu = X[tr].mean(0); Xc = X[tr] - mu
    lam, U = np.linalg.eigh(Xc @ Xc.T); top = np.argsort(lam)[::-1][:k]
    P = Xc.T @ (U[:, top] / np.sqrt(lam[top]))
    mz, sz, w, b = logreg_fit(Xc @ P, uns[tr].astype(float))
    return 1 / (1 + np.exp(-((((X[te] - mu) @ P) - mz) / sz @ w + b)))

def auroc(s, y):
    s, y = np.asarray(s), np.asarray(y, bool)
    r = np.argsort(np.argsort(s, kind="mergesort"), kind="mergesort") + 1.0
    for v in np.unique(s):  # average ranks on ties
        m = s == v
        r[m] = r[m].mean()
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))

res = {}
for seed in (13, 0, 1, 2):
    keys = sorted(set(groups)); random.Random(seed).shuffle(keys)
    s = np.full(len(ids), np.nan)
    for i in range(5):
        te = np.flatnonzero(np.isin(groups, keys[i::5])); tr = np.setdiff1d(np.flatnonzero(att), te)
        s[te] = fit_score(tr, te)
    a = att
    r = {"pooled": auroc(s[a], uns[a])}
    for su in ("workspace", "travel", "banking", "slack"):
        m = a & (suite == su); r[su] = auroc(s[m], uns[m])
    # suite identity alone: score = training base rate of the step's suite (no hidden state at all)
    base = {su: uns[a & (suite == su)].mean() for su in ("workspace", "travel", "banking", "slack")}
    r["suite_base_rate_only"] = auroc([base[x] for x in suite[a]], uns[a])
    res[seed] = r
summary = {k: [round(float(np.mean([res[s][k] for s in res])), 3), round(float(np.std([res[s][k] for s in res])), 3)] for k in res[13]}
print(json.dumps(summary)); out.write_text(json.dumps({"per_seed": res, "mean_sd": summary}, indent=1))
