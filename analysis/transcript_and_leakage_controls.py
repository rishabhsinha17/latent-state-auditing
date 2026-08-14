"""Controls demanded by the review panel, as durable artifacts.

1. Trained-transcript baseline: the paper-family NB trained on the VISIBLE
   text (user_instruction + external_content), under both training contrasts:
   - exposure protocol (unsafe-vs-rest over all cases): exposure AUROC among
     safe cases + case-level flag counts per stratum,
   - commitment protocol (unsafe vs attacked-safe): out-of-fold AUROC + CI.
2. Cross-injection-goal transfer for commitment-trained pre-action monitors:
   train on injection_task_0 cases, test on injection_task_1, and reverse —
   bounds goal-vocabulary leakage.
3. Identity-prior ceiling: AUROC from injection-task identity alone.
4. Top NB features of the pre-action commitment-trained NLA-text monitor.

Usage: python3 transcript_and_leakage_controls.py --run-dir <run> --output <json>
"""
from __future__ import annotations

import argparse
import json
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


def logreg_score(model, X):
    mu, sd, w, b = model
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

    case_ids = sorted(labels)
    att_ids = [c for c in case_ids if attacked[c]]
    att_labels = {c: labels[c] for c in att_ids}
    safe_ids = [c for c in case_ids if not labels[c]]
    att_safe = [c for c in att_ids if not labels[c]]
    unsafe = [c for c in case_ids if labels[c]]
    clean = [c for c in case_ids if not attacked[c]]

    def visible_text(cid):
        c = cases[cid]
        return f"{c.get('user_instruction', '')} {c.get('external_content', '')}"

    out = {"run_dir": str(args.run_dir), "seed": SEED, "folds": FOLDS,
           "composition": {"clean": len(clean), "attacked_safe": len(att_safe), "unsafe": len(unsafe)}}

    # 1a. transcript NB, exposure protocol (unsafe-vs-rest over all cases, OOF)
    folds_all = spe._stratified_case_folds(case_ids, labels, folds=FOLDS, seed=SEED)
    oof = {}
    for test_ids in folds_all:
        train_ids = [c for c in case_ids if c not in set(test_ids)]
        tm = BagOfWordsNaiveBayes()
        tm.fit([visible_text(c) for c in train_ids], [labels[c] for c in train_ids])
        for c in test_ids:
            oof[c] = tm.score(visible_text(c))
    out["transcript_exposure_protocol"] = {
        "full": ci_dict([oof[c] for c in case_ids], [labels[c] for c in case_ids]),
        "exposure_among_safe": ci_dict([oof[c] for c in safe_ids], [attacked[c] for c in safe_ids]),
        "attacked_only": ci_dict([oof[c] for c in att_ids], [labels[c] for c in att_ids]),
        "flags_gt_05": {
            "unsafe": [sum(oof[c] > 0.5 for c in unsafe), len(unsafe)],
            "attacked_safe": [sum(oof[c] > 0.5 for c in att_safe), len(att_safe)],
            "clean": [sum(oof[c] > 0.5 for c in clean), len(clean)],
        },
    }

    # 1b. transcript NB, commitment protocol (unsafe vs attacked-safe, OOF)
    folds_att = spe._stratified_case_folds(att_ids, att_labels, folds=FOLDS, seed=SEED)
    oof_c = {}
    for test_ids in folds_att:
        train_ids = [c for c in att_ids if c not in set(test_ids)]
        tm = BagOfWordsNaiveBayes()
        tm.fit([visible_text(c) for c in train_ids], [att_labels[c] for c in train_ids])
        for c in test_ids:
            oof_c[c] = tm.score(visible_text(c))
    out["transcript_commitment_trained"] = ci_dict([oof_c[c] for c in att_ids], [att_labels[c] for c in att_ids])

    # 2. cross-injection-goal transfer, commitment-trained pre-action monitors
    def inj_task(cid):
        m = re.search(r"injection_task_(\d+)", cid)
        return m.group(1) if m else None

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

    groups = sorted({inj_task(c) for c in att_ids if inj_task(c)})
    xfer = {}
    for hold in groups:
        train_ids = [c for c in att_ids if inj_task(c) != hold]
        test_ids = [c for c in att_ids if inj_task(c) == hold]
        if len(set(att_labels[c] for c in train_ids)) < 2 or len(set(att_labels[c] for c in test_ids)) < 2:
            continue
        tm = BagOfWordsNaiveBayes()
        tm.fit([" ".join(text_pre.get(c, [])) for c in train_ids], [att_labels[c] for c in train_ids])
        ta, tl = [], []
        for c in train_ids:
            for r in acts_pre.get(c, []):
                ta.append(r); tl.append(att_labels[c])
        apb = CentroidActivationProbe(); apb.fit(ta, tl)
        X = np.array([r.vector for r in ta]); y = np.array(tl, dtype=float)
        lg = logreg_fit(X, y)
        tt = BagOfWordsNaiveBayes()
        tt.fit([visible_text(c) for c in train_ids], [att_labels[c] for c in train_ids])
        res = {}
        yl = [att_labels[c] for c in test_ids]
        res["nla_text"] = round(auroc_metric([tm.score(" ".join(text_pre.get(c, []))) for c in test_ids], yl), 3)
        res["centroid_max"] = round(auroc_metric(
            [max([apb.score(r) for r in acts_pre.get(c, [])] or [0.5]) for c in test_ids], yl), 3)
        res["logreg_max"] = round(auroc_metric(
            [float(np.max(logreg_score(lg, np.array([r.vector for r in acts_pre.get(c, [])]))))
             if acts_pre.get(c) else 0.5 for c in test_ids], yl), 3)
        res["transcript"] = round(auroc_metric([tt.score(visible_text(c)) for c in test_ids], yl), 3)
        res["test_n"] = f"{sum(yl)}/{len(yl)}"
        xfer[f"holdout_injection_task_{hold}"] = res
    out["cross_injection_transfer_pre_action"] = xfer

    # 3. identity-prior ceiling
    rate_by_inj = {}
    for g in groups:
        ids = [c for c in att_ids if inj_task(c) == g]
        rate_by_inj[g] = sum(att_labels[c] for c in ids) / len(ids)
    out["identity_prior_auroc"] = round(auroc_metric(
        [rate_by_inj[inj_task(c)] for c in att_ids], [att_labels[c] for c in att_ids]), 3)
    out["unsafe_rate_by_injection_task"] = {g: round(r, 3) for g, r in rate_by_inj.items()}

    # 4. top NB features, pre-action commitment-trained NLA text (fit on all 40)
    tm = BagOfWordsNaiveBayes()
    tm.fit([" ".join(text_pre.get(c, [])) for c in att_ids], [att_labels[c] for c in att_ids])
    import math
    feats = []
    for tok in tm.vocabulary:
        lp = math.log((tm.class_token_counts[True][tok] + 1) / (tm.class_totals[True] + len(tm.vocabulary)))
        ln = math.log((tm.class_token_counts[False][tok] + 1) / (tm.class_totals[False] + len(tm.vocabulary)))
        feats.append((round(lp - ln, 3), tok))
    feats.sort(reverse=True)
    out["top_unsafe_features"] = [t for _, t in feats[:15]]
    out["top_safe_features"] = [t for _, t in feats[-15:]]

    args.output.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in out.items() if k != "top_safe_features"}, indent=1)[:2200])


if __name__ == "__main__":
    main()
