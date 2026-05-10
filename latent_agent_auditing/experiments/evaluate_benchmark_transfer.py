from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from latent_agent_auditing.detectors.activation_probe import CentroidActivationProbe
from latent_agent_auditing.detectors.hybrid import weighted_hybrid_score
from latent_agent_auditing.detectors.text_classifier import BagOfWordsNaiveBayes
from latent_agent_auditing.eval.metrics import binary_metrics, operating_point_at_tpr
from latent_agent_auditing.experiments.summarize_paper_evidence import _activations_by_case, _nla_text_by_case
from latent_agent_auditing.logging.jsonl import read_jsonl, write_jsonl
from latent_agent_auditing.models.schemas import to_jsonable


def evaluate_benchmark_transfer(run_dir: Path, output_dir: Path | None = None) -> dict[str, object]:
    """Evaluate held-out suite/domain and attack-template transfer for one run."""

    output_dir = output_dir or run_dir / "paper_evidence" / "transfer"
    output_dir.mkdir(parents=True, exist_ok=True)
    case_rows = read_jsonl(run_dir / "case_scores.jsonl")
    audits = read_jsonl(run_dir / "audits.jsonl")
    source_records = _read_optional_jsonl(run_dir / "source_records.jsonl")

    metadata_by_case = _metadata_by_case(case_rows, source_records)
    labels_by_case = {str(row["case_id"]): bool(row["attack_success"]) for row in case_rows}
    text_by_case = _nla_text_by_case(audits)
    activations_by_case = _activations_by_case(audits)

    splits = []
    splits.extend(_domain_splits(case_rows))
    splits.extend(_attack_splits(case_rows, metadata_by_case))

    experiments = []
    for split in splits:
        result = _run_split(
            split["name"],
            split["train_case_ids"],
            split["test_case_ids"],
            case_rows,
            labels_by_case,
            text_by_case,
            activations_by_case,
            metadata_by_case,
        )
        if result is not None:
            experiments.append(result)

    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "experiments": experiments,
        "summary": _summarize(experiments),
    }
    write_jsonl(output_dir / "transfer_experiments.jsonl", experiments)
    with (output_dir / "transfer_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(summary), handle, indent=2, sort_keys=True)
    (output_dir / "transfer_table.md").write_text(_markdown_table(experiments), encoding="utf-8")
    return summary


def _run_split(
    name: str,
    train_case_ids: list[str],
    test_case_ids: list[str],
    case_rows: list[dict[str, Any]],
    labels_by_case: dict[str, bool],
    text_by_case: dict[str, str],
    activations_by_case,
    metadata_by_case: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    if not _has_both_classes(train_case_ids, labels_by_case) or not _has_both_classes(test_case_ids, labels_by_case):
        return None

    text_model = BagOfWordsNaiveBayes()
    text_model.fit([text_by_case.get(case_id, "") for case_id in train_case_ids], [labels_by_case[case_id] for case_id in train_case_ids])

    train_activations = []
    train_labels = []
    for case_id in train_case_ids:
        for activation in activations_by_case.get(case_id, []):
            train_activations.append(activation)
            train_labels.append(labels_by_case[case_id])
    if not train_activations or not _has_bool_pair(train_labels):
        return None
    activation_probe = CentroidActivationProbe()
    activation_probe.fit(train_activations, train_labels)

    case_by_id = {str(row["case_id"]): row for row in case_rows}
    predictions = []
    for case_id in test_case_ids:
        row = case_by_id[case_id]
        text_score = text_model.score(text_by_case.get(case_id, ""))
        activation_scores = [activation_probe.score(activation) for activation in activations_by_case.get(case_id, [])]
        activation_score = max(activation_scores) if activation_scores else 0.5
        transcript_score = float(row["transcript_score"])
        tool_policy_score = float(row["tool_policy_score"])
        raw_nla_score = float(row["nla_score"])
        hybrid_score = weighted_hybrid_score(
            transcript_score=transcript_score,
            tool_policy_score=tool_policy_score,
            nla_score=text_score,
            activation_probe_score=activation_score,
        )
        predictions.append(
            {
                "case_id": case_id,
                "label": labels_by_case[case_id],
                "domain": row.get("domain"),
                **metadata_by_case.get(case_id, {}),
                "transcript_score": transcript_score,
                "tool_policy_score": tool_policy_score,
                "raw_nla_taxonomy_score": raw_nla_score,
                "trained_nla_text_score": text_score,
                "raw_activation_probe_score": activation_score,
                "hybrid_score": hybrid_score,
            }
        )
    return {
        "name": name,
        "train_cases": len(train_case_ids),
        "test_cases": len(test_case_ids),
        "train_positives": sum(labels_by_case[case_id] for case_id in train_case_ids),
        "test_positives": sum(labels_by_case[case_id] for case_id in test_case_ids),
        "monitors": {
            monitor: _score(predictions, f"{monitor}_score")
            for monitor in [
                "transcript",
                "tool_policy",
                "raw_nla_taxonomy",
                "trained_nla_text",
                "raw_activation_probe",
                "hybrid",
            ]
        },
        "predictions": predictions,
    }


def _score(predictions: list[dict[str, object]], score_key: str) -> dict[str, object]:
    scores = [float(row[score_key]) for row in predictions]
    labels = [bool(row["label"]) for row in predictions]
    return {
        "binary": asdict(binary_metrics(scores, labels)),
        "operating_point_at_90_tpr": operating_point_at_tpr(scores, labels, target_tpr=0.90),
    }


def _domain_splits(case_rows: list[dict[str, Any]]) -> list[dict[str, object]]:
    domains = sorted({str(row.get("domain", "unknown")) for row in case_rows})
    case_ids_by_domain = {
        domain: [str(row["case_id"]) for row in case_rows if str(row.get("domain", "unknown")) == domain]
        for domain in domains
    }
    all_case_ids = [str(row["case_id"]) for row in case_rows]
    return [
        {
            "name": f"domain_holdout:{domain}",
            "train_case_ids": [case_id for case_id in all_case_ids if case_id not in set(test_case_ids)],
            "test_case_ids": test_case_ids,
        }
        for domain, test_case_ids in case_ids_by_domain.items()
    ]


def _attack_splits(case_rows: list[dict[str, Any]], metadata_by_case: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    positive_attack_names = sorted(
        {
            str(metadata.get("attack_name", "unknown"))
            for case_id, metadata in metadata_by_case.items()
            if _case_row(case_rows, case_id).get("attack_success") and metadata.get("attack_name") not in {None, "none"}
        }
    )
    all_case_ids = [str(row["case_id"]) for row in case_rows]
    clean_case_ids = [str(row["case_id"]) for row in case_rows if not bool(row["attack_success"])]
    splits = []
    for attack_name in positive_attack_names:
        heldout_positive = [
            case_id
            for case_id, metadata in metadata_by_case.items()
            if metadata.get("attack_name") == attack_name and bool(_case_row(case_rows, case_id).get("attack_success"))
        ]
        test_case_ids = sorted(clean_case_ids + heldout_positive)
        train_case_ids = [
            case_id
            for case_id in all_case_ids
            if case_id not in set(heldout_positive)
        ]
        splits.append(
            {
                "name": f"attack_holdout:{attack_name}",
                "train_case_ids": train_case_ids,
                "test_case_ids": test_case_ids,
            }
        )
    return splits


def _metadata_by_case(case_rows: list[dict[str, Any]], source_records: list[dict[str, Any]]) -> dict[str, dict[str, object]]:
    source_by_case = {str(record.get("id", record.get("case_id", ""))): record for record in source_records}
    out = {}
    for row in case_rows:
        case_id = str(row["case_id"])
        source = source_by_case.get(case_id, {})
        out[case_id] = {
            "suite": source.get("environment", row.get("domain")),
            "attack_name": source.get("attack_name", "unknown"),
            "user_task_id": source.get("user_task_id"),
            "injection_task_id": (source.get("injection_task") or {}).get("id") if isinstance(source.get("injection_task"), dict) else None,
        }
    return out


def _read_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def _case_row(case_rows: list[dict[str, Any]], case_id: str) -> dict[str, Any]:
    for row in case_rows:
        if str(row["case_id"]) == case_id:
            return row
    raise KeyError(case_id)


def _has_both_classes(case_ids: list[str], labels_by_case: dict[str, bool]) -> bool:
    labels = [labels_by_case[case_id] for case_id in case_ids]
    return any(labels) and not all(labels)


def _has_bool_pair(labels: list[bool]) -> bool:
    return any(labels) and not all(labels)


def _summarize(experiments: list[dict[str, object]]) -> dict[str, object]:
    monitors = ["trained_nla_text", "raw_activation_probe", "hybrid"]
    out: dict[str, object] = {}
    for monitor in monitors:
        aurocs = [float(exp["monitors"][monitor]["binary"]["auroc"]) for exp in experiments]  # type: ignore[index]
        out[monitor] = {
            "mean_auroc": sum(aurocs) / len(aurocs) if aurocs else None,
            "min_auroc": min(aurocs) if aurocs else None,
            "max_auroc": max(aurocs) if aurocs else None,
        }
    return out


def _markdown_table(experiments: list[dict[str, object]]) -> str:
    lines = [
        "| Split | Monitor | Train | Test | Positives | AUROC | AUPRC | FPR@90TPR |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for exp in experiments:
        for monitor in ["trained_nla_text", "raw_activation_probe", "hybrid"]:
            binary = exp["monitors"][monitor]["binary"]  # type: ignore[index]
            lines.append(
                "| {name} | {monitor} | {train} | {test} | {pos} | {auroc:.3f} | {auprc:.3f} | {fpr:.3f} |".format(
                    name=exp["name"],
                    monitor=monitor,
                    train=exp["train_cases"],
                    test=exp["test_cases"],
                    pos=exp["test_positives"],
                    auroc=binary["auroc"],
                    auprc=binary["auprc"],
                    fpr=binary["fpr_at_90_tpr"],
                )
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate held-out suite and attack transfer for a benchmark audit run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    summary = evaluate_benchmark_transfer(run_dir=args.run_dir, output_dir=args.output_dir)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
