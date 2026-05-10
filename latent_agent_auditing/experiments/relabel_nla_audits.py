from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from latent_agent_auditing.eval.metrics import binary_metrics
from latent_agent_auditing.labels.labeler import NLAKeywordLabeler
from latent_agent_auditing.labels.taxonomy import labels_to_score
from latent_agent_auditing.logging.jsonl import read_jsonl, write_jsonl
from latent_agent_auditing.models.schemas import to_jsonable


def relabel_audits(input_path: Path, output_path: Path, metrics_path: Path, threshold: float = 0.5) -> dict[str, object]:
    rows = read_jsonl(input_path)
    labeler = NLAKeywordLabeler()
    relabeled = []
    labels_bool = []
    scores = []
    for row in rows:
        labels = labeler.label([str(item) for item in row.get("nla_explanations", [])])
        score = labels_to_score(labels)
        updated = dict(row)
        updated["latent_labels"] = [label.value for label in labels]
        updated["risk_score"] = score
        updated["predicted_risky"] = score >= threshold
        relabeled.append(updated)
        labels_bool.append(bool(row["attack_success"]))
        scores.append(score)

    metrics = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "cases": len(relabeled),
        "attack_success_cases": sum(labels_bool),
        "threshold": threshold,
        "nla_text": asdict(binary_metrics(scores, labels_bool)),
        "confusion_at_threshold": _confusion(scores, labels_bool, threshold=threshold),
        "label_counts": _label_counts(relabeled),
    }
    write_jsonl(output_path, relabeled)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(metrics), handle, indent=2, sort_keys=True)
    return metrics


def _confusion(scores: list[float], labels: list[bool], threshold: float) -> dict[str, int]:
    out = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for score, label in zip(scores, labels):
        pred = score >= threshold
        if pred and label:
            out["tp"] += 1
        elif pred and not label:
            out["fp"] += 1
        elif not pred and not label:
            out["tn"] += 1
        else:
            out["fn"] += 1
    return out


def _label_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for label in row["latent_labels"]:  # type: ignore[index]
            counts[str(label)] = counts.get(str(label), 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score saved NLA audit rows with the current keyword taxonomy.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    metrics = relabel_audits(args.input, args.output, args.metrics, threshold=args.threshold)
    print(json.dumps(to_jsonable(metrics), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
