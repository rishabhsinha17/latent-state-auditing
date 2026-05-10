from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path

from latent_agent_auditing.adapters.injecagent import ATTACK_TEMPLATE_NAMES, make_injecagent_style_cases
from latent_agent_auditing.agent.rule_based import RuleBasedAgent
from latent_agent_auditing.auditing.activation_capture import TransformersActivationBackend
from latent_agent_auditing.detectors.activation_probe import CentroidActivationProbe
from latent_agent_auditing.eval.calibration import confusion, rates, select_threshold
from latent_agent_auditing.eval.metrics import binary_metrics
from latent_agent_auditing.logging.jsonl import write_jsonl
from latent_agent_auditing.models.schemas import ActivationRecord, DecisionPoint, EvalCase, TraceEvent, Trajectory, to_jsonable
from latent_agent_auditing.tools.mock_tools import MockToolRuntime


def run_layer_ablation(
    cases_per_domain: int,
    run_dir: Path,
    base_model_name: str,
    layers: list[int],
    cache_dir: str | None,
    dtype: str,
    max_length: int,
    calibration_objective: str,
) -> dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=True)
    cases = make_injecagent_style_cases(cases_per_domain=cases_per_domain)
    agent = RuleBasedAgent(vulnerable=True)
    backend = TransformersActivationBackend(
        model_name=base_model_name,
        torch_dtype=dtype,
        max_length=max_length,
        cache_dir=cache_dir,
    )

    trajectories: list[Trajectory] = []
    activation_rows: list[ActivationRecord] = []
    for case in cases:
        trajectory = agent.run(case, _runtime_for_case(case))
        trajectories.append(trajectory)
        event = _select_event(trajectory.events)
        activation_rows.extend(backend.capture(case, event, layers, output_dir=run_dir / "activations" / case.id))

    write_jsonl(run_dir / "cases.jsonl", cases)
    write_jsonl(run_dir / "trajectories.jsonl", trajectories)
    write_jsonl(run_dir / "activation_records.jsonl", activation_rows)
    _release_torch_model(backend)

    case_by_id = {case.id: case for case in cases}
    trajectory_by_id = {trajectory.case_id: trajectory for trajectory in trajectories}
    examples = [
        {
            "case": case_by_id[activation.case_id],
            "trajectory": trajectory_by_id[activation.case_id],
            "activation": activation,
            "label": trajectory_by_id[activation.case_id].attack_success,
        }
        for activation in activation_rows
    ]

    layer_results = []
    for layer in layers:
        layer_examples = [example for example in examples if example["activation"].layer == layer]
        layer_results.append(
            {
                "layer": layer,
                "domain_transfer": _domain_transfer(layer_examples, calibration_objective),
                "template_holdout": _template_holdout(layer_examples, calibration_objective),
            }
        )

    metrics = {
        "base_model_name": base_model_name,
        "cases": len(cases),
        "positives": sum(trajectory.attack_success for trajectory in trajectories),
        "layers": layers,
        "calibration_objective": calibration_objective,
        "layer_results": layer_results,
        "summary": _summarize(layer_results),
        "run_dir": str(run_dir),
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(metrics), handle, indent=2, sort_keys=True)
    write_jsonl(run_dir / "layer_results.jsonl", layer_results)
    return metrics


def _runtime_for_case(case: EvalCase) -> MockToolRuntime:
    runtime = MockToolRuntime()
    if case.domain == "email":
        runtime.emails["latest"] = case.external_content
    elif case.domain == "file":
        runtime.files["project_plan.md"] = case.external_content
    return runtime


def _select_event(events: list[TraceEvent]) -> TraceEvent:
    for event in events:
        if event.decision_point == DecisionPoint.BEFORE_TOOL_SELECTION:
            return event
    raise RuntimeError("trajectory did not include BEFORE_TOOL_SELECTION")


def _release_torch_model(backend: TransformersActivationBackend) -> None:
    backend._model = None
    backend._tokenizer = None
    torch = backend._torch
    backend._torch = None
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _domain_transfer(examples: list[dict[str, object]], calibration_objective: str) -> list[dict[str, object]]:
    out = []
    for train_domain, test_domain in [("email", "file"), ("file", "email")]:
        train = [example for example in examples if example["case"].domain == train_domain]  # type: ignore[union-attr]
        test = [example for example in examples if example["case"].domain == test_domain]  # type: ignore[union-attr]
        out.append(_run_probe_split(f"{train_domain}_to_{test_domain}", train, test, calibration_objective))
    return out


def _template_holdout(examples: list[dict[str, object]], calibration_objective: str) -> list[dict[str, object]]:
    out = []
    benign = [example for example in examples if not bool(example["label"])]
    for template_idx, template in enumerate(ATTACK_TEMPLATE_NAMES):
        test_attacks = [
            example
            for example in examples
            if bool(example["label"]) and example["case"].metadata.get("attack_template") == template  # type: ignore[union-attr]
        ]
        test_benign = [example for offset, example in enumerate(benign) if offset % len(ATTACK_TEMPLATE_NAMES) == template_idx]
        test_ids = {id(example) for example in test_attacks + test_benign}
        train = [
            example
            for example in examples
            if id(example) not in test_ids
            and (
                not bool(example["label"])
                or example["case"].metadata.get("attack_template") != template  # type: ignore[union-attr]
            )
        ]
        out.append(_run_probe_split(f"holdout_{template}", train, test_attacks + test_benign, calibration_objective))
    return out


def _run_probe_split(
    name: str,
    train_examples: list[dict[str, object]],
    test_examples: list[dict[str, object]],
    calibration_objective: str,
) -> dict[str, object]:
    _validate_split(name, train_examples, test_examples)
    probe = CentroidActivationProbe()
    probe.fit(
        [example["activation"] for example in train_examples],  # type: ignore[list-item]
        [bool(example["label"]) for example in train_examples],
    )
    train_scores = [probe.score(example["activation"]) for example in train_examples]  # type: ignore[arg-type]
    train_labels = [bool(example["label"]) for example in train_examples]
    threshold = select_threshold(train_scores, train_labels, objective=calibration_objective)

    test_scores = [probe.score(example["activation"]) for example in test_examples]  # type: ignore[arg-type]
    test_labels = [bool(example["label"]) for example in test_examples]
    stats = confusion(test_scores, test_labels, threshold)
    return {
        "name": name,
        "train_cases": len(train_examples),
        "test_cases": len(test_examples),
        "train_positives": sum(train_labels),
        "test_positives": sum(test_labels),
        "calibrated_threshold": threshold,
        "calibrated_confusion": stats,
        "calibrated_rates": rates(stats),
        "binary": asdict(binary_metrics(test_scores, test_labels)),
        "score_range": {"min": min(test_scores), "max": max(test_scores)},
    }


def _validate_split(name: str, train: list[dict[str, object]], test: list[dict[str, object]]) -> None:
    for split_name, split in [("train", train), ("test", test)]:
        positives = sum(bool(example["label"]) for example in split)
        negatives = len(split) - positives
        if positives == 0 or negatives == 0:
            raise ValueError(f"{name} {split_name} split needs positives and negatives, got {positives=} {negatives=}")


def _summarize(layer_results: list[dict[str, object]]) -> dict[str, object]:
    summary = []
    for result in layer_results:
        splits = result["domain_transfer"] + result["template_holdout"]  # type: ignore[operator]
        aurocs = [float(split["binary"]["auroc"]) for split in splits]  # type: ignore[index]
        fprs = [float(split["calibrated_rates"]["fpr"]) for split in splits]  # type: ignore[index]
        tprs = [float(split["calibrated_rates"]["tpr"]) for split in splits]  # type: ignore[index]
        summary.append(
            {
                "layer": result["layer"],
                "mean_auroc": sum(aurocs) / len(aurocs),
                "min_auroc": min(aurocs),
                "mean_calibrated_tpr": sum(tprs) / len(tprs),
                "mean_calibrated_fpr": sum(fprs) / len(fprs),
            }
        )
    return {"layers": summary}


def _parse_layers(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run raw activation probe layer ablation with calibrated thresholds.")
    parser.add_argument("--cases-per-domain", type=int, default=36)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/qwen_7b_activation_layers"))
    parser.add_argument("--base-model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layers", default="6,13,20,27")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--calibration-objective", default="youden", choices=["youden", "f1", "accuracy"])
    args = parser.parse_args()
    metrics = run_layer_ablation(
        cases_per_domain=args.cases_per_domain,
        run_dir=args.run_dir,
        base_model_name=args.base_model_name,
        layers=_parse_layers(args.layers),
        cache_dir=args.cache_dir,
        dtype=args.dtype,
        max_length=args.max_length,
        calibration_objective=args.calibration_objective,
    )
    print(json.dumps(to_jsonable(metrics), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
