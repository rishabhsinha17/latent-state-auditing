from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_agent_auditing.adapters.injecagent import make_injecagent_style_cases
from latent_agent_auditing.agent.rule_based import RuleBasedAgent
from latent_agent_auditing.auditing.activation_capture import TransformersActivationBackend
from latent_agent_auditing.logging.jsonl import write_jsonl
from latent_agent_auditing.models.schemas import DecisionPoint, TraceEvent, to_jsonable
from latent_agent_auditing.tools.mock_tools import MockToolRuntime


def capture_smoke(
    model_name: str,
    run_dir: Path,
    case_index: int,
    layers: list[int] | None,
    dtype: str,
    max_length: int,
    cache_dir: str | None,
) -> dict[str, object]:
    cases = make_injecagent_style_cases(cases_per_domain=max(2, case_index + 1))
    case = cases[case_index]
    runtime = MockToolRuntime()
    if case.domain == "email":
        runtime.emails["latest"] = case.external_content
    elif case.domain == "file":
        runtime.files["project_plan.md"] = case.external_content

    trajectory = RuleBasedAgent(vulnerable=True).run(case, runtime)
    event = _select_event(trajectory.events)
    backend = TransformersActivationBackend(
        model_name=model_name,
        torch_dtype=dtype,
        max_length=max_length,
        cache_dir=cache_dir,
    )
    selected_layers = layers or backend.default_layers()
    records = backend.capture(case, event, selected_layers, output_dir=run_dir / "activations")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(run_dir / "activation_records.jsonl", records)
    write_jsonl(run_dir / "trajectory.jsonl", [trajectory])
    write_jsonl(run_dir / "case.jsonl", [case])

    summary = {
        "model_name": model_name,
        "case_id": case.id,
        "attack_present": case.attack_present,
        "decision_point": event.decision_point.value,
        "proposed_tool": event.tool_call.name if event.tool_call else None,
        "layers": selected_layers,
        "records": len(records),
        "hidden_size": records[0].metadata["hidden_size"] if records else None,
        "n_tokens": records[0].metadata["n_tokens"] if records else None,
        "activation_paths": [record.activation_path for record in records],
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(summary), handle, indent=2, sort_keys=True)
    return summary


def _select_event(events: list[TraceEvent]) -> TraceEvent:
    for event in events:
        if event.decision_point == DecisionPoint.BEFORE_TOOL_SELECTION:
            return event
    raise RuntimeError("trajectory did not include BEFORE_TOOL_SELECTION")


def _parse_layers(raw: str | None) -> list[int] | None:
    if raw is None or not raw.strip():
        return None
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture real Hugging Face hidden-state activations for one audit point.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/transformer_smoke"))
    parser.add_argument("--case-index", type=int, default=1)
    parser.add_argument("--layers", default=None, help="Comma-separated transformer block layers. Defaults to quartiles.")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    summary = capture_smoke(
        model_name=args.model_name,
        run_dir=args.run_dir,
        case_index=args.case_index,
        layers=_parse_layers(args.layers),
        dtype=args.dtype,
        max_length=args.max_length,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
