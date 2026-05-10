from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from latent_agent_auditing.adapters.injecagent import ATTACK_TEMPLATE_NAMES
from latent_agent_auditing.detectors.activation_probe import CentroidActivationProbe
from latent_agent_auditing.detectors.hybrid import weighted_hybrid_score
from latent_agent_auditing.detectors.text_classifier import BagOfWordsNaiveBayes
from latent_agent_auditing.eval.metrics import binary_metrics
from latent_agent_auditing.logging.jsonl import read_jsonl, write_jsonl
from latent_agent_auditing.models.schemas import ActivationRecord, DecisionPoint, to_jsonable


def evaluate_generalization(
    audits_path: Path,
    activation_records_path: Path,
    cases_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = _load_examples(audits_path, activation_records_path, cases_path)
    experiments: list[dict[str, object]] = []

    for train_domain, test_domain in [("email", "file"), ("file", "email")]:
        train = [idx for idx, example in enumerate(examples) if example["domain"] == train_domain]
        test = [idx for idx, example in enumerate(examples) if example["domain"] == test_domain]
        experiments.append(_run_split(f"domain:{train_domain}_to_{test_domain}", examples, train, test))

    benign_indices = [idx for idx, example in enumerate(examples) if not example["label"]]
    for template_idx, template in enumerate(ATTACK_TEMPLATE_NAMES):
        test_attacks = [
            idx
            for idx, example in enumerate(examples)
            if example["label"] and example["attack_template"] == template
        ]
        test_benign = [idx for offset, idx in enumerate(benign_indices) if offset % len(ATTACK_TEMPLATE_NAMES) == template_idx]
        test = sorted(test_attacks + test_benign)
        train = [
            idx
            for idx, example in enumerate(examples)
            if idx not in set(test) and (not example["label"] or example["attack_template"] != template)
        ]
        if test_attacks and test_benign:
            experiments.append(_run_split(f"template_holdout:{template}", examples, train, test))

    metrics = {
        "audits_path": str(audits_path),
        "activation_records_path": str(activation_records_path),
        "cases_path": str(cases_path),
        "examples": len(examples),
        "positives": sum(bool(example["label"]) for example in examples),
        "experiments": experiments,
        "summary": _summarize(experiments),
        "output_dir": str(output_dir),
    }
    write_jsonl(output_dir / "experiments.jsonl", experiments)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(metrics), handle, indent=2, sort_keys=True)
    return metrics


def _load_examples(audits_path: Path, activation_records_path: Path, cases_path: Path) -> list[dict[str, object]]:
    cases = {str(row["id"]): row for row in read_jsonl(cases_path)}
    activation_by_id = {
        str(row["id"]): _activation_from_row(row)
        for row in read_jsonl(activation_records_path)
    }
    examples: list[dict[str, object]] = []
    for row in read_jsonl(audits_path):
        case = cases[str(row["case_id"])]
        activation = activation_by_id[str(row["activation_id"])]
        metadata = case.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        examples.append(
            {
                "case_id": str(row["case_id"]),
                "domain": str(row["domain"]),
                "attack_template": str(metadata.get("attack_template", "none")),
                "label": bool(row["attack_success"]),
                "text": "\n\n".join(str(item) for item in row.get("nla_explanations", [])),
                "activation": activation,
            }
        )
    return examples


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


def _run_split(name: str, examples: list[dict[str, object]], train_indices: list[int], test_indices: list[int]) -> dict[str, object]:
    train = [examples[idx] for idx in train_indices]
    test = [examples[idx] for idx in test_indices]
    _validate_split(name, train, test)

    text_model = BagOfWordsNaiveBayes()
    text_model.fit([str(example["text"]) for example in train], [bool(example["label"]) for example in train])

    activation_model = CentroidActivationProbe()
    activation_model.fit(
        [example["activation"] for example in train],  # type: ignore[list-item]
        [bool(example["label"]) for example in train],
    )

    prediction_rows = []
    for example in test:
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
                "case_id": example["case_id"],
                "domain": example["domain"],
                "attack_template": example["attack_template"],
                "label": example["label"],
                "text_score": text_score,
                "activation_score": activation_score,
                "hybrid_score": hybrid_score,
            }
        )

    return {
        "name": name,
        "train_cases": len(train),
        "test_cases": len(test),
        "train_positives": sum(bool(example["label"]) for example in train),
        "test_positives": sum(bool(example["label"]) for example in test),
        "text_classifier": _scored_metrics(prediction_rows, "text_score"),
        "activation_probe": _scored_metrics(prediction_rows, "activation_score"),
        "hybrid_text_activation": _scored_metrics(prediction_rows, "hybrid_score"),
        "predictions": prediction_rows,
    }


def _validate_split(name: str, train: list[dict[str, object]], test: list[dict[str, object]]) -> None:
    for split_name, split in [("train", train), ("test", test)]:
        positives = sum(bool(example["label"]) for example in split)
        negatives = len(split) - positives
        if positives == 0 or negatives == 0:
            raise ValueError(f"{name} {split_name} split needs positives and negatives, got {positives=} {negatives=}")


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


def _summarize(experiments: list[dict[str, object]]) -> dict[str, object]:
    keys = ["text_classifier", "activation_probe", "hybrid_text_activation"]
    out: dict[str, object] = {}
    for key in keys:
        aurocs = [float(exp[key]["binary"]["auroc"]) for exp in experiments]  # type: ignore[index]
        auprcs = [float(exp[key]["binary"]["auprc"]) for exp in experiments]  # type: ignore[index]
        out[key] = {
            "mean_auroc": sum(aurocs) / len(aurocs) if aurocs else None,
            "mean_auprc": sum(auprcs) / len(auprcs) if auprcs else None,
            "min_auroc": min(aurocs) if aurocs else None,
            "max_auroc": max(aurocs) if aurocs else None,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cross-domain and cross-template generalization.")
    parser.add_argument("--audits", type=Path, required=True)
    parser.add_argument("--activation-records", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metrics = evaluate_generalization(
        audits_path=args.audits,
        activation_records_path=args.activation_records,
        cases_path=args.cases,
        output_dir=args.output_dir,
    )
    print(json.dumps(to_jsonable(metrics), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
