"""Bootstrap CIs for the transfer splits (Table 4) from saved predictions.

Reads a transfer_experiments.jsonl produced by evaluate_benchmark_transfer and
attaches the repo's bootstrap CI to each split x monitor AUROC.

Usage: python3 transfer_cis.py --experiments <transfer_experiments.jsonl> --output <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(REPO))

from latent_agent_auditing.eval.metrics import binary_metrics, bootstrap_metric_ci  # noqa: E402
from latent_agent_auditing.logging.jsonl import read_jsonl  # noqa: E402

RESAMPLES = 1000
SEED = 13
MONITORS = ("trained_nla_text", "raw_activation_probe", "hybrid")


def auroc_metric(scores, labels):
    return binary_metrics(scores, labels).auroc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    out = {"source": str(args.experiments), "resamples": RESAMPLES, "seed": SEED, "splits": {}}
    for exp in read_jsonl(args.experiments):
        preds = exp["predictions"]
        labels = [bool(p["label"]) for p in preds]
        block = {}
        for mon in MONITORS:
            scores = [float(p[f"{mon}_score"]) for p in preds]
            ci = bootstrap_metric_ci(scores, labels, auroc_metric, resamples=RESAMPLES, seed=SEED)
            block[mon] = {"auroc": ci.mean, "lower": ci.lower, "upper": ci.upper}
        out["splits"][exp["name"]] = block

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True))
    for name, block in out["splits"].items():
        print(name)
        for mon, v in block.items():
            print(f"  {mon:22s} {v['auroc']:.3f} [{v['lower']:.3f}, {v['upper']:.3f}]")


if __name__ == "__main__":
    main()
