from __future__ import annotations

from typing import Any

from latent_agent_auditing.adapters.base import BenchmarkAdapter
from latent_agent_auditing.models.schemas import AttackType, EvalCase


class AgentDojoAdapter(BenchmarkAdapter):
    """Adapter seam for AgentDojo task/security cases.

    This intentionally avoids importing AgentDojo so local CI remains light.
    Pass dictionaries exported from an AgentDojo run, or wrap its objects
    upstream and convert them here.
    """

    benchmark_name = "agentdojo"

    def from_record(self, record: dict[str, Any]) -> EvalCase:
        injection_task = record.get("injection_task") or {}
        user_task = record.get("user_task") or {}
        attacker_goal = str(record.get("attacker_goal", injection_task.get("goal", "")))
        attack_present = bool(attacker_goal or record.get("attack_present"))
        return EvalCase(
            id=str(record.get("id", record.get("case_id", ""))),
            benchmark=self.benchmark_name,
            domain=str(record.get("environment", record.get("domain", "unknown"))),
            user_instruction=str(record.get("user_instruction", user_task.get("instruction", ""))),
            external_content=str(record.get("external_content", record.get("injection_text", ""))),
            tools_available=list(record.get("tools_available", record.get("tools", []))),
            attack_present=attack_present,
            attack_type=AttackType.INDIRECT_PROMPT_INJECTION if attack_present else AttackType.NONE,
            attacker_goal=attacker_goal,
            unsafe_tools=list(record.get("unsafe_tools", injection_task.get("ground_truth_tools", []))),
            expected_safe_tools=list(record.get("expected_safe_tools", user_task.get("ground_truth_tools", []))),
            metadata={"source_record": record},
        )
