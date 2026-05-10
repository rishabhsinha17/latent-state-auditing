from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_agent_auditing.auditing.nla_runner import LocalTransformersAVRunner
from latent_agent_auditing.labels.labeler import NLAKeywordLabeler
from latent_agent_auditing.labels.taxonomy import labels_to_score
from latent_agent_auditing.logging.jsonl import read_jsonl, write_jsonl
from latent_agent_auditing.models.schemas import ActivationRecord, DecisionPoint, to_jsonable


def verbalize_saved_activation(
    activation_records_path: Path,
    output_path: Path,
    av_model_name: str,
    layer: int,
    n_samples: int,
    cache_dir: str | None,
    dtype: str,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, object]:
    rows = read_jsonl(activation_records_path)
    records = [_activation_from_row(row) for row in rows]
    selected = [record for record in records if record.layer == layer]
    if not selected:
        raise ValueError(f"no activation record found for layer {layer} in {activation_records_path}")
    activation = selected[0]

    runner = LocalTransformersAVRunner(
        av_model_name=av_model_name,
        cache_dir=cache_dir,
        torch_dtype=dtype,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    explanations = runner.explain(activation, n_samples=n_samples)
    labeler = NLAKeywordLabeler()
    labels = labeler.label(explanations)
    result = {
        "activation_id": activation.id,
        "case_id": activation.case_id,
        "decision_point": activation.decision_point.value,
        "layer": activation.layer,
        "token_position": activation.token_position,
        "av_model_name": av_model_name,
        "n_samples": n_samples,
        "nla_explanations": explanations,
        "latent_labels": [label.value for label in labels],
        "risk_score": labels_to_score(labels),
        "activation_metadata": activation.metadata,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, [result])
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(result), handle, indent=2, sort_keys=True)
    return result


def _activation_from_row(row: dict[str, object]) -> ActivationRecord:
    decision_point = DecisionPoint(str(row["decision_point"]))
    return ActivationRecord(
        id=str(row["id"]),
        case_id=str(row["case_id"]),
        step=int(row["step"]),
        decision_point=decision_point,
        layer=int(row["layer"]),
        token_position=str(row["token_position"]),
        vector=[float(item) for item in row["vector"]],  # type: ignore[index]
        activation_path=str(row["activation_path"]) if row.get("activation_path") else None,
        metadata=dict(row.get("metadata", {})),  # type: ignore[arg-type]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a released NLA AV checkpoint on a saved activation record.")
    parser.add_argument("--activation-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--av-model-name", default="kitft/nla-qwen2.5-7b-L20-av")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    result = verbalize_saved_activation(
        activation_records_path=args.activation_records,
        output_path=args.output,
        av_model_name=args.av_model_name,
        layer=args.layer,
        n_samples=args.n_samples,
        cache_dir=args.cache_dir,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(json.dumps(to_jsonable(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
