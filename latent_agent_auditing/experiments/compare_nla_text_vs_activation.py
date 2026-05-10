from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from latent_agent_auditing.detectors.activation_probe import CentroidActivationProbe
from latent_agent_auditing.detectors.hybrid import weighted_hybrid_score
from latent_agent_auditing.detectors.text_classifier import BagOfWordsNaiveBayes
from latent_agent_auditing.eval.metrics import binary_metrics
from latent_agent_auditing.logging.jsonl import read_jsonl, write_jsonl
from latent_agent_auditing.models.schemas import ActivationRecord, DecisionPoint, to_jsonable


def compare(
    audits_path: Path,
    activation_records_path: Path,
    output_dir: Path,
    folds: int,
    seed: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audits = read_jsonl(audits_path)
    activation_by_id = {
        row["id"]: _activation_from_row(row)
        for row in read_jsonl(activation_records_path)
    }
    examples = []
    for row in audits:
        activation_id = str(row["activation_id"])
        if activation_id not in activation_by_id:
            raise KeyError(f"activation {activation_id} missing from {activation_records_path}")
        text = "\n\n".join(str(item) for item in row.get("nla_explanations", []))
        examples.append(
            {
                "case_id": str(row["case_id"]),
                "domain": str(row["domain"]),
                "label": bool(row["attack_success"]),
                "text": text,
                "activation": activation_by_id[activation_id],
            }
        )

    splits = _stratified_folds([bool(example["label"]) for example in examples], folds=folds, seed=seed)
    prediction_rows = []
    for fold_idx, test_indices in enumerate(splits):
        test_set = set(test_indices)
        train_examples = [example for idx, example in enumerate(examples) if idx not in test_set]
        test_examples = [examples[idx] for idx in test_indices]

        text_model = BagOfWordsNaiveBayes()
        text_model.fit(
            [str(example["text"]) for example in train_examples],
            [bool(example["label"]) for example in train_examples],
        )

        activation_model = CentroidActivationProbe()
        activation_model.fit(
            [example["activation"] for example in train_examples],  # type: ignore[list-item]
            [bool(example["label"]) for example in train_examples],
        )

        for example in test_examples:
            text_score = text_model.score(str(example["text"]))
            activation_score = activation_model.score(example["activation"])  # type: ignore[arg-type]
            hybrid_score = weighted_hybrid_score(
                transcript_score=0.0,
                tool_policy_score=0.0,
                nla_score=text_score,
                activation_probe_score=activation_score,
                weights=(0.0, 0.0, 0.5, 0.5),
            )
            prediction_rows.append(
                {
                    "fold": fold_idx,
                    "case_id": example["case_id"],
                    "domain": example["domain"],
                    "label": example["label"],
                    "text_score": text_score,
                    "activation_score": activation_score,
                    "hybrid_score": hybrid_score,
                }
            )

    labels = [bool(row["label"]) for row in prediction_rows]
    metrics = {
        "audits_path": str(audits_path),
        "activation_records_path": str(activation_records_path),
        "examples": len(examples),
        "positives": sum(labels),
        "folds": folds,
        "seed": seed,
        "text_classifier": _scored_metrics(prediction_rows, "text_score"),
        "activation_probe": _scored_metrics(prediction_rows, "activation_score"),
        "hybrid_text_activation": _scored_metrics(prediction_rows, "hybrid_score"),
        "output_dir": str(output_dir),
    }
    write_jsonl(output_dir / "predictions.jsonl", prediction_rows)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(metrics), handle, indent=2, sort_keys=True)
    return metrics


def _activation_from_row(row: dict[str, object]) -> ActivationRecord:
    return ActivationRecord(
        id=str(row["id"]),
        case_id=str(row["case_id"]),
        step=int(row["step"]),
        decision_point=DecisionPoint(str(row["decision_point"])),
        layer=int(row["layer"]),
        token_position=str(row["token_position"]),
        vector=[float(item) for item in row["vector"]],  # type: ignore[index]
        activation_path=str(row["activation_path"]) if row.get("activation_path") else None,
        metadata=dict(row.get("metadata", {})),  # type: ignore[arg-type]
    )


def _stratified_folds(labels: list[bool], folds: int, seed: int) -> list[list[int]]:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    positives = [idx for idx, label in enumerate(labels) if label]
    negatives = [idx for idx, label in enumerate(labels) if not label]
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    out = [[] for _ in range(folds)]
    for bucket in (positives, negatives):
        for offset, idx in enumerate(bucket):
            out[offset % folds].append(idx)
    return [sorted(fold) for fold in out if fold]


def _scored_metrics(rows: list[dict[str, object]], score_key: str) -> dict[str, object]:
    scores = [float(row[score_key]) for row in rows]
    labels = [bool(row["label"]) for row in rows]
    return {
        "binary": asdict(binary_metrics(scores, labels)),
        "thresholds": {
            str(threshold): _confusion(scores, labels, threshold)
            for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]
        },
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-validate NLA text classifier vs raw activation probe.")
    parser.add_argument("--audits", type=Path, required=True)
    parser.add_argument("--activation-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    metrics = compare(
        audits_path=args.audits,
        activation_records_path=args.activation_records,
        output_dir=args.output_dir,
        folds=args.folds,
        seed=args.seed,
    )
    print(json.dumps(to_jsonable(metrics), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
