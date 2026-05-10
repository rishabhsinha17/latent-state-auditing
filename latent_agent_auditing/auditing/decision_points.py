from __future__ import annotations

from latent_agent_auditing.models.schemas import DecisionPoint, TraceEvent


DEFAULT_CRITICAL_POINTS: set[DecisionPoint] = {
    DecisionPoint.AFTER_UNTRUSTED_CONTENT,
    DecisionPoint.BEFORE_TOOL_RETRIEVAL,
    DecisionPoint.AFTER_TOOL_DOCS,
    DecisionPoint.BEFORE_TOOL_SELECTION,
    DecisionPoint.BEFORE_TOOL_ARGUMENTS,
    DecisionPoint.BEFORE_TOOL_EXECUTION,
    DecisionPoint.BEFORE_MEMORY_WRITE,
}


def is_security_critical(event: TraceEvent, critical_points: set[DecisionPoint] | None = None) -> bool:
    points = critical_points or DEFAULT_CRITICAL_POINTS
    return event.decision_point in points
