"""Shared data loading, folds and probe recipe for the r1 scripts (diagnose.py, eval_r1.py).

Same data, fold rule and probe as eval_two_stage.py (r0), which keeps its own copy so its
published numbers stay reproducible byte for byte.
"""
from __future__ import annotations

import importlib.util
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

FOLDS = 5
SUITES = ("workspace", "travel", "banking", "slack")


def load_cascade(path: Path):
    """Load mia/utils/two_stage/cascade.py by path (importing mia would pull in vLLM)."""
    spec = importlib.util.spec_from_file_location("cascade", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cascade"] = mod
    spec.loader.exec_module(mod)
    return mod


class Data:
    def __init__(self, chat: Path, layers=(7, 14, 21, 28)):
        gen = [json.loads(l) for l in open(chat / "suite_full.jsonl")]
        self.labels = {json.loads(l)["case_id"]: json.loads(l) for l in open(chat / "suite_full.labels.jsonl")}
        self.gen = gen
        self.ids = [g["case_id"] for g in gen]
        self.attacked = np.array([g["attack_present"] for g in gen])
        self.unsafe = np.array([bool(self.labels.get(c, {}).get("unsafe", False)) and a
                                for c, a in zip(self.ids, self.attacked)])
        assert self.attacked.sum() == 1729 and self.unsafe.sum() == 358
        self.suite = np.array([c.split("_user_task")[0] for c in self.ids])
        self.groups = np.array([self.group_key(c) for c in self.ids])
        z = np.load(chat / "suite_full.npz")
        self.X = {k: z[f"layer_{k}"].astype(np.float64) for k in layers}
        self.n = len(self.ids)

    @staticmethod
    def group_key(cid):
        m = re.match(r"(?:(\w+?)_)?(user_task_\d+)", cid)
        return f"{m.group(1) or 'workspace'}/{m.group(2)}"

    def folds(self, rows, seed):
        keys = sorted(set(self.groups[rows]))
        random.Random(seed).shuffle(keys)
        return [rows[np.isin(self.groups[rows], keys[i::FOLDS])] for i in range(FOLDS)]


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
    """Centered PCA-k + logistic (fit_chat.Probe recipe; PCA via the Gram matrix)."""
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.mu = X.mean(0)
        Xc = X - self.mu
        lam, U = np.linalg.eigh(Xc @ Xc.T)
        top = np.argsort(lam)[::-1][:self.k]
        self.P = Xc.T @ (U[:, top] / np.sqrt(np.maximum(lam[top], 1e-12)))
        self.m = logreg_fit(Xc @ self.P, np.asarray(y, float))
        return self

    def score(self, X):
        mz, sz, w, b = self.m
        z = (((X - self.mu) @ self.P - mz) / sz) @ w + b
        return 1.0 / (1.0 + np.exp(-z))

    def export(self):
        """Fold into x -> sigmoid(((x - mean) / scale) @ w + b), the vLLM-Hook probe format."""
        mz, sz, w, b = self.m
        return self.mu, np.ones_like(self.mu), self.P @ (w / sz), float(b - (mz / sz) @ w)


def auroc(s, y):
    """Mann-Whitney AUROC with average ranks for ties."""
    s, y = np.asarray(s, float), np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.cumsum(counts) - (counts - 1) / 2.0)[inv]   # 1-based average rank of each value
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def group_bootstrap(groups, stat, n_boot=2000, seed=0):
    """95% percentile interval of stat(rows) over user-task-group resamples."""
    rng = np.random.default_rng(seed)
    keys = np.array(sorted(set(groups)))
    members = {k: np.flatnonzero(groups == k) for k in keys}
    vals = []
    for _ in range(n_boot):
        rows = np.concatenate([members[k] for k in rng.choice(keys, len(keys))])
        vals.append(stat(rows))
    vals = np.array(vals, float)
    return [round(float(np.nanpercentile(vals, 2.5)), 3), round(float(np.nanpercentile(vals, 97.5)), 3)]
