from __future__ import annotations

from pathlib import Path

from latent_agent_auditing.auditing.activation_capture import ActivationBackend
from latent_agent_auditing.auditing.decision_points import DEFAULT_CRITICAL_POINTS, is_security_critical
from latent_agent_auditing.auditing.nla_runner import NLARunner
from latent_agent_auditing.labels.labeler import NLAKeywordLabeler
from latent_agent_auditing.labels.taxonomy import labels_to_score
from latent_agent_auditing.models.schemas import AuditRecord, DecisionPoint, EvalCase, Trajectory


class LatentAuditPipeline:
    def __init__(
        self,
        activation_backend: ActivationBackend,
        nla_runner: NLARunner,
        labeler: NLAKeywordLabeler | None = None,
        layers: list[int] | None = None,
        n_nla_samples: int = 5,
        critical_points: set[DecisionPoint] | None = None,
    ) -> None:
        self.activation_backend = activation_backend
        self.nla_runner = nla_runner
        self.labeler = labeler or NLAKeywordLabeler()
        self.layers = layers or [8, 16, 24, 31]
        self.n_nla_samples = n_nla_samples
        self.critical_points = critical_points or DEFAULT_CRITICAL_POINTS

    def audit(self, case: EvalCase, trajectory: Trajectory, output_dir: Path | None = None) -> list[AuditRecord]:
        records: list[AuditRecord] = []
        for event in trajectory.events:
            if not is_security_critical(event, self.critical_points):
                continue
            activations = self.activation_backend.capture(case, event, self.layers, output_dir=output_dir)
            for activation in activations:
                explanations = self.nla_runner.explain(activation, n_samples=self.n_nla_samples)
                labels = self.labeler.label(explanations)
                records.append(
                    AuditRecord(
                        activation=activation,
                        nla_explanations=explanations,
                        latent_labels=labels,
                        risk_score=labels_to_score(labels),
                        metadata={
                            "proposed_tool": event.tool_call.name if event.tool_call else None,
                            "event_safe": event.safe,
                            "attack_success": trajectory.attack_success,
                        },
                    )
                )
        return records
