from __future__ import annotations

from latent_agent_auditing.labels.taxonomy import labels_to_score
from latent_agent_auditing.models.schemas import AuditRecord


class NLATextDetector:
    def score(self, audits: list[AuditRecord]) -> float:
        if not audits:
            return 0.0
        return max(labels_to_score(record.latent_labels) for record in audits)
