"""Per-provenance ablation of the replay grid (Table 3).

Recomputes the two learned monitors out-of-fold using only one provenance of
activation/text per case: step 1 (after-untrusted-content, normalized_replay)
or step 2 (before-tool-selection, exact_prefix), versus both (the paper's
setting). Reports AUROC with the repo's bootstrap CI.

Usage: python3 provenance_ablation.py --run-dir <t3 run> --output <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

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


def auroc_metric(scores, labels):
    return binary_metrics(scores, labels).auroc


def oof_scores(audits, labels, keep_step):
    text = defaultdict(list)
    for a in audits:
        act = a["activation"]
        if keep_step is None or act["step"] == keep_step:
            for e in a.get("nla_explanations", []):
                t = e if isinstance(e, str) else (e.get("text") or e.get("explanation") or "")
                if t:
                    text[act["case_id"]].append(t)
    acts_all = spe._activations_by_case(audits)
    acts = {
        cid: [r for r in recs if keep_step is None or getattr(r, "step", None) == keep_step]
        for cid, recs in acts_all.items()
    }
    case_ids = sorted(labels)
    folds = spe._stratified_case_folds(case_ids, labels, folds=FOLDS, seed=SEED)
    oof = {"trained_nla_text": {}, "raw_activation_probe": {}}
    for test_ids in folds:
        train_ids = [c for c in case_ids if c not in set(test_ids)]
        tm = BagOfWordsNaiveBayes()
        tm.fit([" ".join(text.get(c, [])) for c in train_ids], [labels[c] for c in train_ids])
        ta, tl = [], []
        for c in train_ids:
            for r in acts.get(c, []):
                ta.append(r)
                tl.append(labels[c])
        ap = CentroidActivationProbe()
        ap.fit(ta, tl)
        for c in test_ids:
            oof["trained_nla_text"][c] = tm.score(" ".join(text.get(c, [])))
            s = [ap.score(r) for r in acts.get(c, [])]
            oof["raw_activation_probe"][c] = max(s) if s else 0.5
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    case_rows = read_jsonl(args.run_dir / "case_scores.jsonl")
    audits = read_jsonl(args.run_dir / "audits.jsonl")
    labels = {str(r["case_id"]): bool(r["attack_success"]) for r in case_rows}
    case_ids = sorted(labels)

    variants = {
        "both_provenances_paper": None,
        "normalized_replay_only_step1": 1,
        "exact_prefix_only_step2": 2,
    }
    out = {"run_dir": str(args.run_dir), "seed": SEED, "folds": FOLDS, "resamples": RESAMPLES, "variants": {}}
    for vname, step in variants.items():
        oof = oof_scores(audits, labels, step)
        block = {}
        for mon, sc in oof.items():
            ci = bootstrap_metric_ci(
                [sc[c] for c in case_ids], [labels[c] for c in case_ids],
                auroc_metric, resamples=RESAMPLES, seed=SEED,
            )
            block[mon] = {"auroc": ci.mean, "lower": ci.lower, "upper": ci.upper}
        out["variants"][vname] = block

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True))
    for vname, block in out["variants"].items():
        print(vname)
        for mon, v in block.items():
            print(f"  {mon:22s} {v['auroc']:.3f} [{v['lower']:.3f}, {v['upper']:.3f}]")


if __name__ == "__main__":
    main()
