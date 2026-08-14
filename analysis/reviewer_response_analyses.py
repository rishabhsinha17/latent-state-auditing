"""Analyses answering the external-review items, on the primary expanded run.

1. Per-step commitment-trained probes: step-1-only (t-2) and step-2-only (t-1).
2. Stronger text baseline: TF-IDF (uni+bigram) logistic regression over the
   visible text, commitment contrast: stratified CI, grouped means, transfer.
3. Grouped-fold uncertainty: per-seed values and paired differences
   (activation minus text) across the same grouped splits.
4. Cluster bootstrap (resampling user tasks) for the headline commitment cells.

Usage: python3 reviewer_response_analyses.py --run-dir <exp_ws_full> \
         --templates-run <run> --travel-run <run> --output <json>
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(REPO))

from latent_agent_auditing.detectors.activation_probe import CentroidActivationProbe  # noqa: E402
from latent_agent_auditing.detectors.text_classifier import BagOfWordsNaiveBayes  # noqa: E402
from latent_agent_auditing.eval.metrics import binary_metrics, bootstrap_metric_ci  # noqa: E402
from latent_agent_auditing.experiments import summarize_paper_evidence as spe  # noqa: E402
from latent_agent_auditing.logging.jsonl import read_jsonl  # noqa: E402

SEED = 13
FOLDS = 5
RESAMPLES = 1000
ACTION_STEP = 3
TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_'-]*")


def auroc_metric(scores, labels):
    return binary_metrics(scores, labels).auroc


def ci_dict(scores, labels):
    ci = bootstrap_metric_ci(scores, labels, auroc_metric, resamples=RESAMPLES, seed=SEED)
    return {"auroc": round(ci.mean, 3), "lower": round(ci.lower, 3), "upper": round(ci.upper, 3), "n": len(scores)}


def logreg_fit(X, y, l2=1.0, iters=500, lr=0.1):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd
    w = np.zeros(Xs.shape[1]); b = 0.0; n = len(y)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
        g = p - y
        w -= lr * (Xs.T @ g / n + l2 * w / n)
        b -= lr * g.mean()
    return mu, sd, w, b


def logreg_score(m, X):
    mu, sd, w, b = m
    return 1.0 / (1.0 + np.exp(-(((X - mu) / sd) @ w + b)))


def featurize(text):
    toks = TOKEN_RE.findall(text.lower())
    return toks + [f"{a}__{b}" for a, b in zip(toks, toks[1:])]


class TfidfLogreg:
    def __init__(self, max_vocab=5000):
        self.max_vocab = max_vocab

    def fit(self, texts, labels):
        df = Counter()
        docs = [Counter(featurize(t)) for t in texts]
        for d in docs:
            df.update(d.keys())
        vocab = [t for t, _ in df.most_common(self.max_vocab)]
        self.index = {t: i for i, t in enumerate(vocab)}
        self.idf = np.array([math.log(len(docs) / (1 + df[t])) + 1.0 for t in vocab])
        X = self._matrix(docs)
        self.model = logreg_fit(X, np.array(labels, dtype=float))

    def _matrix(self, docs):
        X = np.zeros((len(docs), len(self.index)))
        for i, d in enumerate(docs):
            for t, c in d.items():
                j = self.index.get(t)
                if j is not None:
                    X[i, j] = c
        X = X * self.idf
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        return X / norms

    def score_many(self, texts):
        return list(logreg_score(self.model, self._matrix([Counter(featurize(t)) for t in texts])))


def load(run):
    case_rows = read_jsonl(run / "case_scores.jsonl")
    cases = {str(c["id"]): c for c in read_jsonl(run / "cases.jsonl")}
    audits = read_jsonl(run / "audits.jsonl")
    labels = {str(r["case_id"]): bool(r["attack_success"]) for r in case_rows}
    attacked = {cid: bool(c.get("attack_present")) for cid, c in cases.items()}
    acts_all = spe._activations_by_case(audits)
    att_ids = sorted(c for c in labels if attacked[c])
    def vis(cid):
        c = cases[cid]
        return f"{c.get('user_instruction', '')} {c.get('external_content', '')}"
    return att_ids, {c: labels[c] for c in att_ids}, acts_all, vis


def user_task(cid):
    m = re.search(r"(user_task_\d+)", cid)
    return m.group(1) if m else cid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--templates-run", type=Path, required=True)
    ap.add_argument("--travel-run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    att_ids, att_labels, acts_all, vis = load(args.run_dir)
    y_all = [att_labels[c] for c in att_ids]
    out = {"n_attacked": len(att_ids), "n_unsafe": sum(y_all)}

    def acts_step(cid, steps):
        return [r for r in acts_all.get(cid, []) if getattr(r, "step", 99) in steps]

    # ---- 1. per-step commitment probes (stratified folds, seed 13) ----
    folds = spe._stratified_case_folds(att_ids, att_labels, folds=FOLDS, seed=SEED)
    per_step = {}
    for tag, steps in (("t_minus_2_step1", {1}), ("t_minus_1_step2", {2}), ("pre_action_both", {1, 2})):
        oof_c, oof_l = {}, {}
        for test_ids in folds:
            train_ids = [c for c in att_ids if c not in set(test_ids)]
            ta, tl = [], []
            for c in train_ids:
                for r in acts_step(c, steps):
                    ta.append(r); tl.append(att_labels[c])
            apb = CentroidActivationProbe(); apb.fit(ta, tl)
            lg = logreg_fit(np.array([r.vector for r in ta]), np.array(tl, dtype=float))
            for c in test_ids:
                rs = acts_step(c, steps)
                oof_c[c] = max([apb.score(r) for r in rs] or [0.5])
                oof_l[c] = float(np.max(logreg_score(lg, np.array([r.vector for r in rs])))) if rs else 0.5
        per_step[tag] = {
            "centroid": ci_dict([oof_c[c] for c in att_ids], y_all),
            "logreg": ci_dict([oof_l[c] for c in att_ids], y_all),
        }
    out["per_step_commitment"] = per_step

    # ---- 2. TF-IDF logistic text baseline (commitment contrast) ----
    oof_t = {}
    for test_ids in folds:
        train_ids = [c for c in att_ids if c not in set(test_ids)]
        tf = TfidfLogreg()
        tf.fit([vis(c) for c in train_ids], [att_labels[c] for c in train_ids])
        for c, s in zip(test_ids, tf.score_many([vis(c) for c in test_ids])):
            oof_t[c] = s
    out["tfidf_stratified"] = ci_dict([oof_t[c] for c in att_ids], y_all)

    grouped = {}
    for seed in (13, 0, 1, 2):
        tasks = sorted({user_task(c) for c in att_ids})
        rng = random.Random(seed); rng.shuffle(tasks)
        gf = [[c for c in att_ids if user_task(c) in set(tasks[i::FOLDS])] for i in range(FOLDS)]
        oof = {}
        for test_ids in gf:
            train_ids = [c for c in att_ids if c not in set(test_ids)]
            if len(set(att_labels[c] for c in train_ids)) < 2:
                continue
            tf = TfidfLogreg()
            tf.fit([vis(c) for c in train_ids], [att_labels[c] for c in train_ids])
            for c, s in zip(test_ids, tf.score_many([vis(c) for c in test_ids])):
                oof[c] = s
        grouped[str(seed)] = round(auroc_metric([oof[c] for c in att_ids], y_all), 3)
    out["tfidf_grouped_per_seed"] = grouped
    out["tfidf_grouped_mean"] = round(float(np.mean(list(grouped.values()))), 3)

    tf_full = TfidfLogreg()
    tf_full.fit([vis(c) for c in att_ids], [att_labels[c] for c in att_ids])
    xfer = {}
    for name, run in (("templates", args.templates_run), ("travel", args.travel_run)):
        te_ids, te_lab, _, te_vis = load(run)
        xfer[name] = ci_dict(tf_full.score_many([te_vis(c) for c in te_ids]), [te_lab[c] for c in te_ids])
    out["tfidf_transfer"] = xfer

    # ---- 3. grouped-fold paired differences (from EXP_commitment artifact) ----
    commit = json.load(open(Path(__file__).parent / "EXP_commitment.json"))
    ps = commit["grouped_per_seed"]
    diffs_l = [ps[s]["logreg_max"] - ps[s]["transcript"] for s in ps]
    diffs_c = [ps[s]["centroid_max"] - ps[s]["transcript"] for s in ps]
    out["grouped_seed_stats"] = {
        "logreg_range": [min(ps[s]["logreg_max"] for s in ps), max(ps[s]["logreg_max"] for s in ps)],
        "centroid_range": [min(ps[s]["centroid_max"] for s in ps), max(ps[s]["centroid_max"] for s in ps)],
        "transcript_range": [min(ps[s]["transcript"] for s in ps), max(ps[s]["transcript"] for s in ps)],
        "paired_diff_logreg_minus_text": [round(min(diffs_l), 3), round(max(diffs_l), 3)],
        "paired_diff_centroid_minus_text": [round(min(diffs_c), 3), round(max(diffs_c), 3)],
    }

    # ---- 4. cluster bootstrap (user tasks) for headline commitment cells ----
    oof_scores = {"centroid": {}, "logreg": {}, "transcript_nb": {}}
    for test_ids in folds:
        train_ids = [c for c in att_ids if c not in set(test_ids)]
        ta, tl = [], []
        for c in train_ids:
            for r in acts_step(c, {1, 2}):
                ta.append(r); tl.append(att_labels[c])
        apb = CentroidActivationProbe(); apb.fit(ta, tl)
        lg = logreg_fit(np.array([r.vector for r in ta]), np.array(tl, dtype=float))
        nb = BagOfWordsNaiveBayes()
        nb.fit([vis(c) for c in train_ids], [att_labels[c] for c in train_ids])
        for c in test_ids:
            rs = acts_step(c, {1, 2})
            oof_scores["centroid"][c] = max([apb.score(r) for r in rs] or [0.5])
            oof_scores["logreg"][c] = float(np.max(logreg_score(lg, np.array([r.vector for r in rs])))) if rs else 0.5
            oof_scores["transcript_nb"][c] = nb.score(vis(c))
    tasks = sorted({user_task(c) for c in att_ids})
    by_task = defaultdict(list)
    for c in att_ids:
        by_task[user_task(c)].append(c)
    rng = random.Random(SEED)
    cluster = {}
    for mon, sc in oof_scores.items():
        vals = []
        for _ in range(RESAMPLES):
            sample_tasks = [rng.choice(tasks) for _ in tasks]
            ids = [c for t in sample_tasks for c in by_task[t]]
            ys = [att_labels[c] for c in ids]
            if len(set(ys)) < 2:
                continue
            vals.append(auroc_metric([sc[c] for c in ids], ys))
        vals.sort()
        cluster[mon] = {
            "auroc": round(auroc_metric([sc[c] for c in att_ids], y_all), 3),
            "cluster_lower": round(vals[int(0.025 * len(vals))], 3),
            "cluster_upper": round(vals[int(0.975 * len(vals))], 3),
        }
    out["cluster_bootstrap"] = cluster

    args.output.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, indent=1)[:2400])


if __name__ == "__main__":
    main()
