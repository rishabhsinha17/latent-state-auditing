from __future__ import annotations

from typing import Any

from latent_agent_auditing.adapters.base import BenchmarkAdapter
from latent_agent_auditing.models.schemas import AttackType, EvalCase


class ToolHijackerAdapter(BenchmarkAdapter):
    """Adapter seam for tool-retrieval/tool-selection prompt injection cases."""

    benchmark_name = "toolhijacker"

    def from_record(self, record: dict[str, Any]) -> EvalCase:
        attack_present = bool(record.get("malicious_tool_doc") or record.get("attack_present"))
        benign_tools = list(record.get("benign_tools", []))
        malicious_tool = record.get("malicious_tool")
        tools = list(record.get("tools_available", benign_tools))
        if malicious_tool and malicious_tool not in tools:
            tools.append(str(malicious_tool))
        return EvalCase(
            id=str(record.get("id", record.get("case_id", ""))),
            benchmark=self.benchmark_name,
            domain=str(record.get("domain", "tool_selection")),
            user_instruction=str(record.get("user_instruction", record.get("target_task", ""))),
            external_content=str(record.get("malicious_tool_doc", record.get("tool_docs", ""))),
            tools_available=tools,
            attack_present=attack_present,
            attack_type=AttackType.TOOL_SELECTION_HIJACKING if attack_present else AttackType.NONE,
            attacker_goal=str(record.get("attacker_goal", f"select {malicious_tool}" if malicious_tool else "")),
            unsafe_tools=[str(malicious_tool)] if malicious_tool else list(record.get("unsafe_tools", [])),
            expected_safe_tools=benign_tools,
            metadata={"source_record": record},
        )
