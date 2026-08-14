"""AUROCs for the zero-shot LLM-judge baseline, on the paper's stratifications.

Inputs: judge JSONL files (id, attack_present, attack_success,
exposure_score, commitment_score). Zero-shot, so no folds: scores are used
as-is. Slices match Table 4/Appendix D:
  exposure_among_safe : attacked-but-safe vs clean, exposure_score
  commitment_attacked : unsafe vs attacked-but-safe, commitment_score

Usage: python3 judge_baseline.py --primary <jsonl> --templates <jsonl> --travel <jsonl> --output <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(REPO))

from latent_agent_auditing.eval.metrics import binary_metrics, bootstrap_metric_ci  # noqa: E402

RESAMPLES = 1000
SEED = 13


def auroc_metric(scores, labels):
    return binary_metrics(scores, labels).auroc


def ci_dict(scores, labels):
    ci = bootstrap_metric_ci(scores, labels, auroc_metric, resamples=RESAMPLES, seed=SEED)
    return {"auroc": round(ci.mean, 3), "lower": round(ci.lower, 3), "upper": round(ci.upper, 3), "n": len(scores)}


def load(path):
    rows = [json.loads(l) for l in open(path)]
    for r in rows:
        r["exposure_score"] = 50 if r["exposure_score"] is None else r["exposure_score"]
        r["commitment_score"] = 50 if r["commitment_score"] is None else r["commitment_score"]
    return rows


def slices(rows):
    safe = [r for r in rows if not r["attack_success"]]
    att = [r for r in rows if r["attack_present"]]
    out = {}
    if any(r["attack_present"] for r in safe) and any(not r["attack_present"] for r in safe):
        out["exposure_among_safe"] = ci_dict(
            [r["exposure_score"] for r in safe], [r["attack_present"] for r in safe])
    if any(r["attack_success"] for r in att) and any(not r["attack_success"] for r in att):
        out["commitment_attacked"] = ci_dict(
            [r["commitment_score"] for r in att], [r["attack_success"] for r in att])
        out["commitment_attacked_via_exposure_score"] = ci_dict(
            [r["exposure_score"] for r in att], [r["attack_success"] for r in att])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", type=Path, required=True)
    ap.add_argument("--templates", type=Path, required=True)
    ap.add_argument("--travel", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    out = {}
    for name, path in (("primary_ws_full", args.primary), ("templates", args.templates), ("travel", args.travel)):
        rows = load(path)
        out[name] = {"n": len(rows), **slices(rows)}
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True))
    for name, blk in out.items():
        print(name)
        for k, v in blk.items():
            if isinstance(v, dict):
                print(f"  {k:38s} {v['auroc']:.3f} [{v['lower']:.3f}, {v['upper']:.3f}] n={v['n']}")


if __name__ == "__main__":
    main()
