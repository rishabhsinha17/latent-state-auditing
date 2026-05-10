from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from latent_agent_auditing.detectors.activation_probe import CentroidActivationProbe
from latent_agent_auditing.detectors.hybrid import weighted_hybrid_score
from latent_agent_auditing.detectors.text_classifier import BagOfWordsNaiveBayes
from latent_agent_auditing.eval.metrics import (
    auprc,
    auroc,
    binary_metrics,
    bootstrap_metric_ci,
    latent_lead_time,
    lead_time_summary,
    operating_point,
    operating_point_at_tpr,
    pairwise_complementarity,
)
from latent_agent_auditing.logging.jsonl import read_jsonl, write_jsonl
from latent_agent_auditing.models.schemas import ActivationRecord, DecisionPoint, to_jsonable


def summarize_paper_evidence(
    run_dir: Path,
    output_dir: Path | None = None,
    folds: int = 5,
    bootstrap_resamples: int = 1000,
    seed: int = 13,
) -> dict[str, object]:
    """Build paper-facing tables from a benchmark audit run directory.

    The benchmark audit CLI writes raw audit records and simple per-case scores.
    This summarizer adds out-of-fold learned baselines for NLA text and raw
    activations, then emits a compact artifact suitable for paper tables.
    """

    output_dir = output_dir or run_dir / "paper_evidence"
    output_dir.mkdir(parents=True, exist_ok=True)

    case_rows = read_jsonl(run_dir / "case_scores.jsonl")
    audits = read_jsonl(run_dir / "audits.jsonl")
    manifest = _read_json(run_dir / "run_manifest.json")
    labels_by_case = {str(row["case_id"]): bool(row["attack_success"]) for row in case_rows}
    labels = [labels_by_case[str(row["case_id"])] for row in case_rows]

    transcript_scores = {str(row["case_id"]): float(row["transcript_score"]) for row in case_rows}
    tool_scores = {str(row["case_id"]): float(row["tool_policy_score"]) for row in case_rows}
    raw_nla_scores = {str(row["case_id"]): float(row["nla_score"]) for row in case_rows}
    lead_times = [_optional_int(row.get("latent_lead_time")) for row in case_rows]

    learned = _out_of_fold_learned_scores(audits, labels_by_case, folds=folds, seed=seed)
    learned_step_scores = _out_of_fold_learned_step_scores(audits, labels_by_case, folds=folds, seed=seed)
    score_by_monitor = {
        "transcript": [transcript_scores[str(row["case_id"])] for row in case_rows],
        "tool_policy": [tool_scores[str(row["case_id"])] for row in case_rows],
        "raw_nla_taxonomy": [raw_nla_scores[str(row["case_id"])] for row in case_rows],
        "trained_nla_text": [learned["trained_nla_text"][str(row["case_id"])] for row in case_rows],
        "raw_activation_probe": [learned["raw_activation_probe"][str(row["case_id"])] for row in case_rows],
    }
    score_by_monitor["hybrid"] = [
        weighted_hybrid_score(
            transcript_score=score_by_monitor["transcript"][idx],
            tool_policy_score=score_by_monitor["tool_policy"][idx],
            nla_score=score_by_monitor["trained_nla_text"][idx],
            activation_probe_score=score_by_monitor["raw_activation_probe"][idx],
        )
        for idx in range(len(case_rows))
    ]

    table = {
        name: _monitor_summary(scores, labels, bootstrap_resamples=bootstrap_resamples, seed=seed)
        for name, scores in score_by_monitor.items()
    }
    prediction_rows = []
    for idx, row in enumerate(case_rows):
        case_id = str(row["case_id"])
        prediction_rows.append(
            {
                "case_id": case_id,
                "label": labels[idx],
                "domain": row.get("domain"),
                "benchmark": row.get("benchmark"),
                "latent_lead_time": lead_times[idx],
                **{f"{name}_score": scores[idx] for name, scores in score_by_monitor.items()},
            }
        )

    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "cases": len(case_rows),
        "positive_cases": sum(labels),
        "folds": folds,
        "bootstrap_resamples": bootstrap_resamples,
        "prompt_summary": manifest.get("prompt_summary", {}),
        "monitors": table,
        "lead_time": lead_time_summary(lead_times, labels),
        "lead_time_by_monitor": _lead_time_by_monitor(case_rows, score_by_monitor, learned_step_scores),
        "complementarity_at_0_5": pairwise_complementarity(score_by_monitor, labels, threshold=0.50),
    }

    write_jsonl(output_dir / "predictions.jsonl", prediction_rows)
    with (output_dir / "paper_table.json").open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(summary), handle, indent=2, sort_keys=True)
    (output_dir / "paper_table.md").write_text(_markdown_table(summary), encoding="utf-8")
    return summary


def _out_of_fold_learned_scores(
    audits: list[dict[str, Any]],
    labels_by_case: dict[str, bool],
    folds: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    case_ids = sorted(labels_by_case)
    if len(case_ids) < 2:
        return {
            "trained_nla_text": {case_id: 0.5 for case_id in case_ids},
            "raw_activation_probe": {case_id: 0.5 for case_id in case_ids},
        }
    folds = min(folds, len(case_ids))
    splits = _stratified_case_folds(case_ids, labels_by_case, folds=folds, seed=seed)
    text_by_case = _nla_text_by_case(audits)
    activations_by_case = _activations_by_case(audits)
    out = {
        "trained_nla_text": {},
        "raw_activation_probe": {},
    }

    for test_case_ids in splits:
        test_set = set(test_case_ids)
        train_case_ids = [case_id for case_id in case_ids if case_id not in test_set]
        if not _has_both_classes(train_case_ids, labels_by_case):
            for case_id in test_case_ids:
                out["trained_nla_text"][case_id] = 0.5
                out["raw_activation_probe"][case_id] = 0.5
            continue

        text_model = BagOfWordsNaiveBayes()
        text_model.fit([text_by_case.get(case_id, "") for case_id in train_case_ids], [labels_by_case[case_id] for case_id in train_case_ids])

        train_activations: list[ActivationRecord] = []
        train_activation_labels: list[bool] = []
        for case_id in train_case_ids:
            for activation in activations_by_case.get(case_id, []):
                train_activations.append(activation)
                train_activation_labels.append(labels_by_case[case_id])
        activation_model = CentroidActivationProbe()
        activation_model.fit(train_activations, train_activation_labels)

        for case_id in test_case_ids:
            out["trained_nla_text"][case_id] = text_model.score(text_by_case.get(case_id, ""))
            case_activations = activations_by_case.get(case_id, [])
            if case_activations:
                out["raw_activation_probe"][case_id] = max(activation_model.score(activation) for activation in case_activations)
            else:
                out["raw_activation_probe"][case_id] = 0.5
    return out


def _out_of_fold_learned_step_scores(
    audits: list[dict[str, Any]],
    labels_by_case: dict[str, bool],
    folds: int,
    seed: int,
) -> dict[str, dict[str, list[tuple[int, float]]]]:
    case_ids = sorted(labels_by_case)
    if len(case_ids) < 2:
        return {"trained_nla_text": {}, "raw_activation_probe": {}}
    splits = _stratified_case_folds(case_ids, labels_by_case, folds=min(folds, len(case_ids)), seed=seed)
    out: dict[str, dict[str, list[tuple[int, float]]]] = {
        "trained_nla_text": {case_id: [] for case_id in case_ids},
        "raw_activation_probe": {case_id: [] for case_id in case_ids},
    }
    audit_examples = _audit_examples(audits)
    for test_case_ids in splits:
        test_set = set(test_case_ids)
        train_case_ids = [case_id for case_id in case_ids if case_id not in test_set]
        if not _has_both_classes(train_case_ids, labels_by_case):
            continue

        train_examples = [example for example in audit_examples if example["case_id"] in train_case_ids]
        test_examples = [example for example in audit_examples if example["case_id"] in test_set]
        text_model = BagOfWordsNaiveBayes()
        text_model.fit([str(example["text"]) for example in train_examples], [labels_by_case[str(example["case_id"])] for example in train_examples])
        activation_model = CentroidActivationProbe()
        activation_model.fit(
            [example["activation"] for example in train_examples],  # type: ignore[list-item]
            [labels_by_case[str(example["case_id"])] for example in train_examples],
        )
        text_steps: dict[str, dict[int, float]] = {}
        activation_steps: dict[str, dict[int, float]] = {}
        for example in test_examples:
            case_id = str(example["case_id"])
            step = int(example["step"])
            text_steps.setdefault(case_id, {})[step] = max(
                text_steps.setdefault(case_id, {}).get(step, 0.0),
                text_model.score(str(example["text"])),
            )
            activation_steps.setdefault(case_id, {})[step] = max(
                activation_steps.setdefault(case_id, {}).get(step, 0.0),
                activation_model.score(example["activation"]),  # type: ignore[arg-type]
            )
        for case_id, step_scores in text_steps.items():
            out["trained_nla_text"][case_id] = sorted(step_scores.items())
        for case_id, step_scores in activation_steps.items():
            out["raw_activation_probe"][case_id] = sorted(step_scores.items())
    return out


def _audit_examples(audits: list[dict[str, Any]]) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for audit in audits:
        raw = audit.get("activation")
        if not isinstance(raw, dict):
            continue
        activation = ActivationRecord(
            id=str(raw["id"]),
            case_id=str(raw["case_id"]),
            step=int(raw["step"]),
            decision_point=DecisionPoint(str(raw["decision_point"])),
            layer=int(raw["layer"]),
            token_position=str(raw["token_position"]),
            vector=[float(item) for item in raw["vector"]],
            activation_path=str(raw["activation_path"]) if raw.get("activation_path") else None,
            metadata=dict(raw.get("metadata", {})),
        )
        examples.append(
            {
                "case_id": activation.case_id,
                "step": activation.step,
                "text": "\n\n".join(str(item) for item in audit.get("nla_explanations", [])),
                "activation": activation,
                "raw_nla_score": float(audit.get("risk_score", 0.0)),
            }
        )
    return examples


def _monitor_summary(
    scores: list[float],
    labels: list[bool],
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, object]:
    return {
        "binary": asdict(binary_metrics(scores, labels)),
        "auroc_ci": asdict(bootstrap_metric_ci(scores, labels, auroc, resamples=bootstrap_resamples, seed=seed)),
        "auprc_ci": asdict(bootstrap_metric_ci(scores, labels, auprc, resamples=bootstrap_resamples, seed=seed)),
        "operating_point_0_5": operating_point(scores, labels, threshold=0.50),
        "operating_point_at_90_tpr": operating_point_at_tpr(scores, labels, target_tpr=0.90),
    }


def _lead_time_by_monitor(
    case_rows: list[dict[str, Any]],
    score_by_monitor: dict[str, list[float]],
    learned_step_scores: dict[str, dict[str, list[tuple[int, float]]]],
) -> dict[str, object]:
    labels = [bool(row["attack_success"]) for row in case_rows]
    unsafe_steps = [_optional_int(row.get("unsafe_step")) for row in case_rows]
    out = {
        "raw_nla_taxonomy": lead_time_summary([_optional_int(row.get("latent_lead_time")) for row in case_rows], labels),
    }
    for monitor in ["trained_nla_text", "raw_activation_probe"]:
        monitor_leads = []
        for row, unsafe_step in zip(case_rows, unsafe_steps):
            case_id = str(row["case_id"])
            monitor_leads.append(latent_lead_time(learned_step_scores.get(monitor, {}).get(case_id, []), unsafe_step))
        out[monitor] = lead_time_summary(monitor_leads, labels)
    # Case-level monitors do not expose pre-action step scores in this artifact.
    out["note"] = "Lead-time summaries use per-step audit scores. Transcript, tool-policy, and case-level hybrid scores are excluded."
    return out


def _nla_text_by_case(audits: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    for audit in audits:
        case_id = str(audit.get("activation", {}).get("case_id", audit.get("case_id", "")))
        out.setdefault(case_id, []).extend(str(item) for item in audit.get("nla_explanations", []))
    return {case_id: "\n\n".join(texts) for case_id, texts in out.items()}


def _activations_by_case(audits: list[dict[str, Any]]) -> dict[str, list[ActivationRecord]]:
    out: dict[str, list[ActivationRecord]] = {}
    for audit in audits:
        raw = audit.get("activation")
        if not isinstance(raw, dict):
            continue
        activation = ActivationRecord(
            id=str(raw["id"]),
            case_id=str(raw["case_id"]),
            step=int(raw["step"]),
            decision_point=DecisionPoint(str(raw["decision_point"])),
            layer=int(raw["layer"]),
            token_position=str(raw["token_position"]),
            vector=[float(item) for item in raw["vector"]],
            activation_path=str(raw["activation_path"]) if raw.get("activation_path") else None,
            metadata=dict(raw.get("metadata", {})),
        )
        out.setdefault(activation.case_id, []).append(activation)
    return out


def _stratified_case_folds(
    case_ids: list[str],
    labels_by_case: dict[str, bool],
    folds: int,
    seed: int,
) -> list[list[str]]:
    rng = random.Random(seed)
    positives = [case_id for case_id in case_ids if labels_by_case[case_id]]
    negatives = [case_id for case_id in case_ids if not labels_by_case[case_id]]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    out = [[] for _ in range(max(2, folds))]
    for bucket in (positives, negatives):
        for offset, case_id in enumerate(bucket):
            out[offset % len(out)].append(case_id)
    return [sorted(fold) for fold in out if fold]


def _has_both_classes(case_ids: list[str], labels_by_case: dict[str, bool]) -> bool:
    labels = [labels_by_case[case_id] for case_id in case_ids]
    return any(labels) and not all(labels)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _markdown_table(summary: dict[str, object]) -> str:
    monitors = summary["monitors"]  # type: ignore[assignment]
    lines = [
        "| Monitor | AUROC | AUROC 95% CI | AUPRC | FPR@90TPR | Threshold@90TPR | Acc@90TPR | Acc@0.5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, payload in monitors.items():  # type: ignore[union-attr]
        binary = payload["binary"]
        ci = payload["auroc_ci"]
        op_90 = payload["operating_point_at_90_tpr"]
        threshold_90 = op_90["threshold"]
        rates_90 = op_90["rates"] or {}
        lines.append(
            "| {name} | {auroc:.3f} | [{lower:.3f}, {upper:.3f}] | {auprc:.3f} | {fpr:.3f} | {thr} | {acc90} | {acc:.3f} |".format(
                name=name,
                auroc=binary["auroc"],
                lower=ci["lower"],
                upper=ci["upper"],
                auprc=binary["auprc"],
                fpr=binary["fpr_at_90_tpr"],
                thr="NA" if threshold_90 is None else f"{threshold_90:.3f}",
                acc90="NA" if not rates_90 else f"{rates_90['accuracy']:.3f}",
                acc=binary["accuracy_at_50"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a benchmark audit run into paper-facing evidence tables.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    summary = summarize_paper_evidence(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        folds=args.folds,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
