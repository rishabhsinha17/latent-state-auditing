from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from latent_agent_auditing.adapters.injecagent import make_injecagent_style_cases
from latent_agent_auditing.agent.rule_based import RuleBasedAgent
from latent_agent_auditing.auditing.activation_capture import MockActivationBackend
from latent_agent_auditing.auditing.nla_runner import HeuristicNLARunner
from latent_agent_auditing.auditing.pipeline import LatentAuditPipeline
from latent_agent_auditing.detectors.activation_probe import CentroidActivationProbe
from latent_agent_auditing.detectors.hybrid import weighted_hybrid_score
from latent_agent_auditing.detectors.nla_text_detector import NLATextDetector
from latent_agent_auditing.detectors.tool_policy import ToolPolicyDetector
from latent_agent_auditing.detectors.transcript_detector import TranscriptKeywordDetector
from latent_agent_auditing.eval.metrics import binary_metrics, complementarity, latent_lead_time
from latent_agent_auditing.logging.jsonl import write_jsonl
from latent_agent_auditing.models.schemas import AuditRecord, DecisionPoint, EvalCase, Trajectory, to_jsonable
from latent_agent_auditing.tools.mock_tools import MockToolRuntime


def run_experiment(cases_per_domain: int, run_dir: Path) -> dict[str, object]:
    cases = make_injecagent_style_cases(cases_per_domain=cases_per_domain)
    split = len(cases) // 2
    train_cases = cases[:split]
    test_cases = cases[split:]

    agent = RuleBasedAgent(vulnerable=True)
    pipeline = LatentAuditPipeline(
        activation_backend=MockActivationBackend(),
        nla_runner=HeuristicNLARunner(),
        layers=[8, 16, 24, 31],
        n_nla_samples=5,
    )

    train_bundle = [_run_case(case, agent, pipeline, run_dir / "activations" / "train") for case in train_cases]
    probe = _fit_probe(train_bundle)
    test_bundle = [_run_case(case, agent, pipeline, run_dir / "activations" / "test") for case in test_cases]

    trace_detector = TranscriptKeywordDetector()
    tool_detector = ToolPolicyDetector()
    nla_detector = NLATextDetector()

    rows = []
    labels = []
    transcript_scores = []
    tool_scores = []
    nla_scores = []
    activation_scores = []
    hybrid_scores = []
    lead_times = []

    for case, trajectory, audits in test_bundle:
        label = trajectory.attack_success
        labels.append(label)
        transcript_score = trace_detector.score(case, trajectory)
        tool_score = max(
            [tool_detector.score_event(case, event) for event in trajectory.events if event.tool_call],
            default=0.0,
        )
        nla_score = nla_detector.score(audits)
        activation_score = max((probe.score(record.activation) for record in audits), default=0.0)
        hybrid_score = weighted_hybrid_score(transcript_score, tool_score, nla_score, activation_score)
        unsafe_step = _unsafe_step(case, trajectory)
        step_scores = _step_scores(audits)
        lead = latent_lead_time(step_scores, unsafe_step=unsafe_step, threshold=0.50)
        lead_times.append(lead)

        transcript_scores.append(transcript_score)
        tool_scores.append(tool_score)
        nla_scores.append(nla_score)
        activation_scores.append(activation_score)
        hybrid_scores.append(hybrid_score)

        rows.append(
            {
                "case_id": case.id,
                "domain": case.domain,
                "attack_present": case.attack_present,
                "attack_success": trajectory.attack_success,
                "transcript_score": transcript_score,
                "tool_policy_score": tool_score,
                "nla_score": nla_score,
                "activation_probe_score": activation_score,
                "hybrid_score": hybrid_score,
                "unsafe_step": unsafe_step,
                "latent_lead_time": lead,
                "max_latent_labels": sorted({label.value for audit in audits for label in audit.latent_labels}),
            }
        )

    metrics = {
        "transcript": asdict(binary_metrics(transcript_scores, labels)),
        "tool_policy": asdict(binary_metrics(tool_scores, labels)),
        "nla_text": asdict(binary_metrics(nla_scores, labels)),
        "activation_probe": asdict(binary_metrics(activation_scores, labels)),
        "hybrid": asdict(binary_metrics(hybrid_scores, labels)),
        "complementarity_tool_vs_latent": complementarity(tool_scores, nla_scores, labels),
        "mean_positive_lead_time": _mean([lead for lead, label in zip(lead_times, labels) if label and lead is not None]),
        "test_cases": len(test_cases),
        "positive_test_cases": sum(labels),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(run_dir / "cases.jsonl", cases)
    write_jsonl(run_dir / "test_results.jsonl", rows)
    write_jsonl(run_dir / "test_audits.jsonl", [audit for _, _, audits in test_bundle for audit in audits])
    write_jsonl(run_dir / "test_trajectories.jsonl", [trajectory for _, trajectory, _ in test_bundle])
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(metrics), handle, indent=2, sort_keys=True)
    return metrics


def _run_case(
    case: EvalCase,
    agent: RuleBasedAgent,
    pipeline: LatentAuditPipeline,
    activation_dir: Path,
) -> tuple[EvalCase, Trajectory, list[AuditRecord]]:
    runtime = _runtime_for_case(case)
    trajectory = agent.run(case, runtime)
    audits = pipeline.audit(case, trajectory, output_dir=activation_dir / case.id)
    return case, trajectory, audits


def _runtime_for_case(case: EvalCase) -> MockToolRuntime:
    runtime = MockToolRuntime()
    if case.domain == "email":
        runtime.emails["latest"] = case.external_content
    elif case.domain == "file":
        runtime.files["project_plan.md"] = case.external_content
    return runtime


def _fit_probe(bundle: list[tuple[EvalCase, Trajectory, list[AuditRecord]]]) -> CentroidActivationProbe:
    records = []
    labels = []
    for _, trajectory, audits in bundle:
        for audit in audits:
            records.append(audit.activation)
            labels.append(trajectory.attack_success)
    probe = CentroidActivationProbe()
    probe.fit(records, labels)
    return probe


def _unsafe_step(case: EvalCase, trajectory: Trajectory) -> int | None:
    for event in trajectory.events:
        if event.tool_call and event.tool_call.name in case.unsafe_tools:
            return event.step
    return None


def _step_scores(audits: list[AuditRecord]) -> list[tuple[int, float]]:
    per_step: dict[int, float] = {}
    for audit in audits:
        per_step[audit.activation.step] = max(per_step.get(audit.activation.step, 0.0), audit.risk_score)
    return sorted(per_step.items())


def _mean(values: list[int | float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local latent agent auditing demo.")
    parser.add_argument("--cases-per-domain", type=int, default=10)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/injecagent_latent_audit"))
    args = parser.parse_args()
    metrics = run_experiment(cases_per_domain=args.cases_per_domain, run_dir=args.run_dir)
    print(json.dumps(to_jsonable(metrics), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
