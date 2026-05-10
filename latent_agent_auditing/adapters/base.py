from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from latent_agent_auditing.models.schemas import EvalCase


class BenchmarkAdapter(ABC):
    benchmark_name: str

    @abstractmethod
    def from_record(self, record: dict[str, Any]) -> EvalCase:
        """Convert a benchmark-specific record to the common eval schema."""

    def from_records(self, records: list[dict[str, Any]]) -> list[EvalCase]:
        return [self.from_record(record) for record in records]
