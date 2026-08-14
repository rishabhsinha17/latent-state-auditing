"""Cross-suite / cross-template transfer of commitment-trained pre-action monitors.

Fits on ALL attacked cases of the train run (unsafe vs attacked-but-safe,
strictly pre-action), then scores the attacked cases of a held-out test run
(different attack templates or a different suite). No case overlap by design.

Usage: python3 cross_transfer.py --train-run <run> --test-run <run> --output <json>
"""
from __future__ import annotations

import argparse
import json
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


def load(run):
    case_rows = read_jsonl(run / "case_scores.jsonl")
    cases = {str(c["id"]): c for c in read_jsonl(run / "cases.jsonl")}
    audits = read_jsonl(run / "audits.jsonl")
    labels = {str(r["case_id"]): bool(r["attack_success"]) for r in case_rows}
    attacked = {cid: bool(c.get("attack_present")) for cid, c in cases.items()}
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
    att_ids = sorted(c for c in labels if attacked[c])
    def vis(cid):
        c = cases[cid]
        return f"{c.get('user_instruction', '')} {c.get('external_content', '')}"
    return att_ids, {c: labels[c] for c in att_ids}, text_pre, acts_pre, vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-run", type=Path, required=True)
    ap.add_argument("--test-run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    tr_ids, tr_lab, tr_text, tr_acts, tr_vis = load(args.train_run)
    te_ids, te_lab, te_text, te_acts, te_vis = load(args.test_run)

    tn = BagOfWordsNaiveBayes()
    tn.fit([" ".join(tr_text.get(c, [])) for c in tr_ids], [tr_lab[c] for c in tr_ids])
    tt = BagOfWordsNaiveBayes()
    tt.fit([tr_vis(c) for c in tr_ids], [tr_lab[c] for c in tr_ids])
    ta, tl = [], []
    for c in tr_ids:
        for r in tr_acts.get(c, []):
            ta.append(r); tl.append(tr_lab[c])
    apb = CentroidActivationProbe(); apb.fit(ta, tl)
    lg = logreg_fit(np.array([r.vector for r in ta]), np.array(tl, dtype=float))

    y = [te_lab[c] for c in te_ids]
    scores = {
        "nla_text": [tn.score(" ".join(te_text.get(c, []))) for c in te_ids],
        "transcript": [tt.score(te_vis(c)) for c in te_ids],
        "centroid_max": [max([apb.score(r) for r in te_acts.get(c, [])] or [0.5]) for c in te_ids],
        "logreg_max": [
            float(np.max(logreg_score(lg, np.array([r.vector for r in te_acts[c]]))))
            if te_acts.get(c) else 0.5 for c in te_ids
        ],
    }
    out = {
        "train_run": str(args.train_run), "test_run": str(args.test_run),
        "test_n": len(te_ids), "test_unsafe": sum(y),
        "monitors": {m: ci_dict(s, y) for m, s in scores.items()},
    }
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"test {out['test_n']} attacked, {out['test_unsafe']} unsafe")
    for m, d in out["monitors"].items():
        print(f"  {m:14s} {d['auroc']:.3f} [{d['lower']:.3f}, {d['upper']:.3f}]")


if __name__ == "__main__":
    main()
