"""Held-out evaluation of two-stage FPR reduction for the tool-call risk probe (CPU).

Data (from chat/gen_capture.py + chat/label_steps.py): suite_full.{npz,jsonl,labels.jsonl}.
1826 AgentDojo v1 steps of Qwen2.5-7B-Instruct (bf16, vLLM-Hook capture, chat template):
1729 attacked (358 unsafe) and 97 clean. Features are last-prompt-token hidden states at
layers 7/14/21/28. This is the same data the released probe (#41) was fit on.

Protocol. No test label reaches any fit or threshold.
  outer   5 folds grouped by suite-qualified user task, seeds 13, 0, 1, 2
  inner   inside each outer-train split, 5 grouped folds give out-of-fold (OOF) scores for
          stage 1 and every stage-2 candidate. Thresholds come from those OOF scores
          (calibrate_cascade for cascades, threshold_for_recall for single stages).
          Final models are refit on the whole outer-train split.
  test    unseen user tasks: recall on unsafe steps, FPR on attacked-but-safe steps, FPR on
          clean steps (never trained on), and the fraction of steps that reach stage 2.

Detectors, fixed before any test result was seen:
  S1     released recipe, layer 28, PCA-32 + logistic (single stage)
  L21    layer 21, PCA-64 + logistic (single stage)
  AVG    mean of S1 and L21 probabilities (single-stage ensemble on the same features as C-A)
  C-A    cascade, S1 gate, stage 2 = L21 probe trained on every attacked step
  C-HN   cascade, S1 gate, stage 2 trained only on the steps S1 passes at 97% OOF recall
         (its hard negatives), features layers 14+21+28 concatenated, PCA-64 + logistic

Usage: eval_two_stage.py <chat dir> <vllm-hook repo> <out dir> [--seeds 13,0,1,2]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("chat", type=Path)
ap.add_argument("repo", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--seeds", default="13,0,1,2")
ap.add_argument("--targets", default="0.8,0.9,0.95")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--stage2-features", type=Path, default=None,
                help="npz from capture_decision.py; adds DEC (single stage) and C-DEC (S1 gate, DEC stage 2)")
ap.add_argument("--stage2-key", default="dec4_layer_28")
args = ap.parse_args()

# Load the shipped cascade module by path: importing the mia package would pull in vLLM.
spec = importlib.util.spec_from_file_location("cascade", args.repo / "mia/utils/two_stage/cascade.py")
cascade = importlib.util.module_from_spec(spec)
sys.modules["cascade"] = cascade   # dataclasses look their module up here
spec.loader.exec_module(cascade)

FOLDS, HN_GATE_RECALL = 5, 0.97
SEEDS = [int(s) for s in args.seeds.split(",")]
TARGETS = [float(t) for t in args.targets.split(",")]

gen = [json.loads(l) for l in open(args.chat / "suite_full.jsonl")]
relab = {json.loads(l)["case_id"]: bool(json.loads(l)["unsafe"]) for l in open(args.chat / "suite_full.labels.jsonl")}
ids = [g["case_id"] for g in gen]
attacked = np.array([g["attack_present"] for g in gen])
unsafe = np.array([relab.get(c, False) if a else False for c, a in zip(ids, attacked)])
assert attacked.sum() == 1729 and unsafe.sum() == 358, (attacked.sum(), unsafe.sum())
Z = np.load(args.chat / "suite_full.npz")
L = {k: Z[f"layer_{k}"].astype(np.float64) for k in (14, 21, 28)}
DEC = None
if args.stage2_features:
    E = np.load(args.stage2_features)
    assert list(E["case_id"]) == ids, "decision features must follow suite_full.jsonl order"
    DEC = E[args.stage2_key].astype(np.float64)
    keep = E["same_action"] & ~np.isnan(DEC).any(1)
    print(f"decision features: keeping {keep.sum()} of {len(keep)} steps (same action, full length)")
    ids = [c for c, k in zip(ids, keep) if k]
    attacked, unsafe, DEC = attacked[keep], unsafe[keep], DEC[keep]
    L = {k: v[keep] for k, v in L.items()}
CAT = np.hstack([L[14], L[21], L[28]])


def group_key(cid):
    m = re.match(r"(?:(\w+?)_)?(user_task_\d+)", cid)
    return f"{m.group(1) or 'workspace'}/{m.group(2)}"


groups = np.array([group_key(c) for c in ids])


def grouped_folds(idx, seed):
    keys = sorted(set(groups[idx]))
    random.Random(seed).shuffle(keys)
    return [idx[np.isin(groups[idx], keys[i::FOLDS])] for i in range(FOLDS)]


def logreg_fit(X, y, l2=1.0, iters=500, lr=0.1):  # verbatim from commitment_probes.py (released probe)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd
    w = np.zeros(Xs.shape[1]); b = 0.0; n = len(y)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
        g = p - y
        w -= lr * (Xs.T @ g / n + l2 * w / n)
        b -= lr * g.mean()
    return mu, sd, w, b


class Probe:
    """Centered PCA-k + logistic, as fit_chat.Probe. PCA via the Gram matrix (same subspace as SVD)."""
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.mu = X.mean(0)
        Xc = X - self.mu
        lam, U = np.linalg.eigh(Xc @ Xc.T)
        top = np.argsort(lam)[::-1][:self.k]
        self.P = Xc.T @ (U[:, top] / np.sqrt(np.maximum(lam[top], 1e-12)))
        self.m = logreg_fit(Xc @ self.P, y.astype(float))
        return self

    def score(self, X):
        mz, sz, w, b = self.m
        z = (((X - self.mu) @ self.P - mz) / sz) @ w + b
        return 1.0 / (1.0 + np.exp(-z))

    def export(self):
        """Fold into x -> sigmoid(((x - mean) / scale) @ w + b), the vLLM-Hook probe format."""
        mz, sz, w, b = self.m
        return self.mu, np.ones_like(self.mu), self.P @ (w / sz), float(b - (mz / sz) @ w)


def fit_models(tr, gate_scores=None):
    """Fit every component on attacked rows of ``tr``. ``gate_scores`` (row -> S1 OOF) selects C-HN's region."""
    a = tr[attacked[tr]]
    m = {"S1": Probe(32).fit(L[28][a], unsafe[a]), "L21": Probe(64).fit(L[21][a], unsafe[a])}
    if DEC is not None:
        m["DEC"] = Probe(64).fit(DEC[a], unsafe[a])
    if gate_scores is not None:
        reg = a[gate_scores[a] >= gate_scores["__gate__"]]
        m["HN"] = Probe(64).fit(CAT[reg], unsafe[reg])
    return m


def score_models(m, rows):
    s = {"S1": m["S1"].score(L[28][rows]), "L21": m["L21"].score(L[21][rows])}
    if "HN" in m:
        s["HN"] = m["HN"].score(CAT[rows])
    if "DEC" in m:
        s["DEC"] = m["DEC"].score(DEC[rows])
    return s


class GateScores(dict):
    """Row-indexed S1 OOF scores plus the gate threshold under key '__gate__'."""
    def __getitem__(self, k):
        if isinstance(k, np.ndarray):
            return np.array([dict.__getitem__(self, int(i)) for i in k])
        return dict.__getitem__(self, k)


def inner_oof(tr, seed):
    """OOF scores on ``tr`` for all components. Two passes: S1/L21 first, then C-HN on S1's region."""
    folds = grouped_folds(tr, seed + 1000)
    oof = {k: np.full(len(L[28]), np.nan) for k in ("S1", "L21", "HN") + (("DEC",) if DEC is not None else ())}
    for f in folds:
        s = score_models(fit_models(np.setdiff1d(tr, f)), f)
        for k in oof:
            if k != "HN":
                oof[k][f] = s[k]
    a = tr[attacked[tr]]
    gs = GateScores({int(i): oof["S1"][i] for i in tr})
    gs["__gate__"] = cascade.threshold_for_recall(oof["S1"][a[unsafe[a]]], HN_GATE_RECALL)
    for f in folds:
        oof["HN"][f] = score_models(fit_models(np.setdiff1d(tr, f), gs), f)["HN"]
    return oof, gs


def detectors(s):
    """Single-stage scores and cascade score pairs from one score dict."""
    d = {"S1": s["S1"], "L21": s["L21"], "AVG": (s["S1"] + s["L21"]) / 2,
         "C-A": (s["S1"], s["L21"]), "C-HN": (s["S1"], s["HN"])}
    if "DEC" in s:
        d.update({"DEC": s["DEC"], "C-DEC": (s["S1"], s["DEC"])})
    return d


def run_seed(seed):
    allrows = np.arange(len(ids))
    pred = {}   # (detector, target) -> flags over all rows (each row tested once per seed)
    reach = {}
    for fi, te in enumerate(grouped_folds(allrows, seed)):
        t0 = time.time()
        tr = np.setdiff1d(allrows, te)
        oof, gs = inner_oof(tr, seed)
        a = tr[attacked[tr]]
        cal = {k: oof[k][a] for k in oof}
        final = fit_models(tr, gs)
        test = detectors(score_models(final, te))
        caldet = detectors(cal)
        for name, sc in test.items():
            for t in TARGETS:
                key = (name, t)
                pred.setdefault(key, np.zeros(len(ids), bool))
                reach.setdefault(key, np.zeros(len(ids), bool))
                if isinstance(sc, tuple):
                    op = cascade.calibrate_cascade(caldet[name][0], caldet[name][1], unsafe[a], t)
                    pred[key][te] = cascade.cascade_flags(sc[0], sc[1], op.stage1_threshold, op.stage2_threshold)
                    reach[key][te] = sc[0] >= op.stage1_threshold
                else:
                    thr = cascade.threshold_for_recall(caldet[name][unsafe[a]], t)
                    pred[key][te] = sc >= thr
                    reach[key][te] = True
        print(f"seed {seed} fold {fi} done in {time.time() - t0:.0f}s", flush=True)
    return pred, reach


def metrics(flags, reached, rows=None):
    rows = np.arange(len(ids)) if rows is None else rows
    f, r, att, uns = flags[rows], reached[rows], attacked[rows], unsafe[rows]
    return {"recall": float(f[uns].mean()), "fpr_attacked_safe": float(f[att & ~uns].mean()),
            "fpr_clean": float(f[~att].mean()), "stage2_rate": float(r.mean())}


args.out.mkdir(parents=True, exist_ok=True)
per_seed = {}
for seed in SEEDS:
    pred, reach = run_seed(seed)
    per_seed[seed] = (pred, reach)
    np.savez_compressed(args.out / f"pred_seed{seed}.npz",
                        **{f"{n}@{t}": pred[(n, t)] for n, t in pred},
                        **{f"reach:{n}@{t}": reach[(n, t)] for n, t in reach})

names = ["S1", "L21", "AVG", "C-A", "C-HN"] + (["DEC", "C-DEC"] if DEC is not None else [])
summary = {}
for t in TARGETS:
    for n in names:
        ms = [metrics(*[x[(n, t)] for x in per_seed[s]]) for s in SEEDS]
        summary[f"{n}@{t}"] = {k: [round(float(np.mean([m[k] for m in ms])), 3),
                                   round(float(np.std([m[k] for m in ms])), 3)] for k in ms[0]}

# Paired bootstrap over user-task groups (each seed's pooled test predictions cover every row once).
rng = np.random.default_rng(0)
ukeys = np.array(sorted(set(groups)))
members = {k: np.flatnonzero(groups == k) for k in ukeys}
paired = {}
for t in TARGETS:
    for n in names[1:]:
        draws = []
        for _ in range(args.boot):
            rows = np.concatenate([members[k] for k in rng.choice(ukeys, len(ukeys))])
            d = []
            for s in SEEDS:
                mb = metrics(*[x[(n, t)] for x in per_seed[s]], rows)
                m0 = metrics(*[x[("S1", t)] for x in per_seed[s]], rows)
                d.append([mb["recall"] - m0["recall"], mb["fpr_attacked_safe"] - m0["fpr_attacked_safe"],
                          mb["fpr_clean"] - m0["fpr_clean"]])
            draws.append(np.mean(d, 0))
        draws = np.array(draws)
        paired[f"{n}-S1@{t}"] = {k: [round(float(np.percentile(draws[:, i], 2.5)), 3),
                                     round(float(np.percentile(draws[:, i], 97.5)), 3)]
                                 for i, k in enumerate(("d_recall", "d_fpr_attacked_safe", "d_fpr_clean"))}

report = {"protocol": __doc__, "seeds": SEEDS, "targets": TARGETS, "hn_gate_recall": HN_GATE_RECALL,
          "n": {"attacked": int(attacked.sum()), "unsafe": int(unsafe.sum()), "clean": int((~attacked).sum()),
                "user_task_groups": int(len(ukeys))},
          "summary_mean_sd_over_seeds": summary, "paired_bootstrap_95ci_vs_S1": paired}
(args.out / "report.json").write_text(json.dumps(report, indent=2))
for k, v in summary.items():
    print(k, json.dumps(v))
for k, v in paired.items():
    print(k, json.dumps(v))
