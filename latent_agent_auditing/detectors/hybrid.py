from __future__ import annotations


def weighted_hybrid_score(
    transcript_score: float,
    tool_policy_score: float,
    nla_score: float,
    activation_probe_score: float,
    weights: tuple[float, float, float, float] = (0.20, 0.25, 0.35, 0.20),
) -> float:
    score = (
        transcript_score * weights[0]
        + tool_policy_score * weights[1]
        + nla_score * weights[2]
        + activation_probe_score * weights[3]
    )
    return max(0.0, min(1.0, score))
