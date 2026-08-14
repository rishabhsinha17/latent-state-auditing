"""Commitment-trained pre-action monitors, parameterized over run dirs.

Trains on attacked cases only (unsafe vs attacked-but-safe), strictly
pre-action (steps < action step), out-of-fold. Reports, with the repo's
bootstrap CIs: NLA-text NB, centroid probe (max-agg), logistic probe
(max-agg), and the visible-text NB — under (a) label-stratified folds
(seed 13) and (b) user-task-grouped folds (means over 4 seeds).

Usage: python3 commitment_probes.py --run-dir <run> --output <json>
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    case_rows = read_jsonl(args.run_dir / "case_scores.jsonl")
    cases = {str(c["id"]): c for c in read_jsonl(args.run_dir / "cases.jsonl")}
    audits = read_jsonl(args.run_dir / "audits.jsonl")
    labels = {str(r["case_id"]): bool(r["attack_success"]) for r in case_rows}
    attacked = {cid: bool(c.get("attack_present")) for cid, c in cases.items()}
    att_ids = sorted(c for c in labels if attacked[c])
    att_labels = {c: labels[c] for c in att_ids}

    def user_task(cid):
        m = re.search(r"(user_task_\d+)", cid)
        return m.group(1) if m else cid

    def visible_text(cid):
        c = cases[cid]
        return f"{c.get('user_instruction', '')} {c.get('external_content', '')}"

    text_pre = defaultdict(list)
    for a in audits:
        act = a["activation"]
        if act["step"] < ACTION_STEP:
            for e in a.get("nla_explanations", []):
                t = e if isinstance(e, str) else (e.get("text") or e.get("explanation") or "")
                if t:
                    text_pre[act["case_id"]].append(t)
    acts_all = spe._activations_by_case(audits)
    acts_pre = {cid: [r for r in recs if getattr(r, "step", 99) < ACTION_STEP] for cid, recs in acts_all.items()}

    def run_folds(fold_lists):
        oof = defaultdict(dict)
        for test_ids in fold_lists:
            train_ids = [c for c in att_ids if c not in set(test_ids)]
            if len(set(att_labels[c] for c in train_ids)) < 2:
                continue
            tn = BagOfWordsNaiveBayes()
            tn.fit([" ".join(text_pre.get(c, [])) for c in train_ids], [att_labels[c] for c in train_ids])
            tt = BagOfWordsNaiveBayes()
            tt.fit([visible_text(c) for c in train_ids], [att_labels[c] for c in train_ids])
            ta, tl = [], []
            for c in train_ids:
                for r in acts_pre.get(c, []):
                    ta.append(r); tl.append(att_labels[c])
            apb = CentroidActivationProbe(); apb.fit(ta, tl)
            lg = logreg_fit(np.array([r.vector for r in ta]), np.array(tl, dtype=float))
            for c in test_ids:
                oof["nla_text"][c] = tn.score(" ".join(text_pre.get(c, [])))
                oof["transcript"][c] = tt.score(visible_text(c))
                s = [apb.score(r) for r in acts_pre.get(c, [])]
                oof["centroid_max"][c] = max(s) if s else 0.5
                if acts_pre.get(c):
                    oof["logreg_max"][c] = float(np.max(logreg_score(lg, np.array([r.vector for r in acts_pre[c]]))))
                else:
                    oof["logreg_max"][c] = 0.5
        return oof

    out = {"run_dir": str(args.run_dir), "n_attacked": len(att_ids), "n_unsafe": sum(att_labels.values())}

    # (a) label-stratified folds, seed 13, with CIs
    folds = spe._stratified_case_folds(att_ids, att_labels, folds=FOLDS, seed=SEED)
    oof = run_folds(folds)
    y = [att_labels[c] for c in att_ids]
    out["stratified_seed13"] = {m: ci_dict([sc[c] for c in att_ids], y) for m, sc in oof.items()}

    # (b) user-task-grouped folds, 4 seeds, means
    per_seed = {}
    for seed in (13, 0, 1, 2):
        tasks = sorted({user_task(c) for c in att_ids})
        rng = random.Random(seed); rng.shuffle(tasks)
        gfolds = [[c for c in att_ids if user_task(c) in set(tasks[i::FOLDS])] for i in range(FOLDS)]
        goof = run_folds(gfolds)
        per_seed[str(seed)] = {m: round(auroc_metric([sc[c] for c in att_ids], y), 3) for m, sc in goof.items()}
    out["grouped_per_seed"] = per_seed
    out["grouped_means"] = {
        m: round(float(np.mean([per_seed[s][m] for s in per_seed])), 3)
        for m in per_seed["13"]
    }

    args.output.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"attacked {out['n_attacked']}, unsafe {out['n_unsafe']}")
    for m, d in out["stratified_seed13"].items():
        print(f"  {m:14s} {d['auroc']:.3f} [{d['lower']:.3f}, {d['upper']:.3f}]  grouped-mean {out['grouped_means'][m]:.3f}")


if __name__ == "__main__":
    main()
