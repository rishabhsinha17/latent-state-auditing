from __future__ import annotations

import math

from latent_agent_auditing.models.schemas import ActivationRecord


class CentroidActivationProbe:
    """Tiny dependency-free raw activation probe baseline."""

    def __init__(self) -> None:
        self.positive_centroid: list[float] | None = None
        self.negative_centroid: list[float] | None = None

    def fit(self, records: list[ActivationRecord], labels: list[bool]) -> None:
        positives = [record.vector for record, label in zip(records, labels) if label]
        negatives = [record.vector for record, label in zip(records, labels) if not label]
        if not positives or not negatives:
            raise ValueError("need at least one positive and one negative activation")
        self.positive_centroid = _mean(positives)
        self.negative_centroid = _mean(negatives)

    def score(self, record: ActivationRecord) -> float:
        if self.positive_centroid is None or self.negative_centroid is None:
            raise RuntimeError("probe must be fit before scoring")
        pos = _distance(record.vector, self.positive_centroid)
        neg = _distance(record.vector, self.negative_centroid)
        if pos + neg == 0:
            return 0.5
        return neg / (pos + neg)

    def score_many(self, records: list[ActivationRecord]) -> list[float]:
        return [self.score(record) for record in records]


def _mean(vectors: list[list[float]]) -> list[float]:
    width = len(vectors[0])
    return [sum(vector[idx] for vector in vectors) / len(vectors) for idx in range(width)]


def _distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
