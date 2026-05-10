from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any

from latent_agent_auditing.labels.taxonomy import LatentRiskLabel


class DecisionPoint(str, Enum):
    AFTER_USER_INSTRUCTION = "after_user_instruction"
    AFTER_UNTRUSTED_CONTENT = "after_untrusted_content"
    BEFORE_TOOL_RETRIEVAL = "before_tool_retrieval"
    AFTER_TOOL_DOCS = "after_tool_docs"
    BEFORE_TOOL_SELECTION = "before_tool_selection"
    BEFORE_TOOL_ARGUMENTS = "before_tool_arguments"
    BEFORE_TOOL_EXECUTION = "before_tool_execution"
    AFTER_TOOL_RESULT = "after_tool_result"
    BEFORE_MEMORY_WRITE = "before_memory_write"
    BEFORE_FINAL_ANSWER = "before_final_answer"


class AttackType(str, Enum):
    NONE = "none"
    INDIRECT_PROMPT_INJECTION = "indirect_prompt_injection"
    TOOL_SELECTION_HIJACKING = "tool_selection_hijacking"
    DATA_STEALING = "data_stealing"
    DIRECT_HARM = "direct_harm"
    MEMORY_POISONING = "memory_poisoning"
    MONITOR_EVASION = "monitor_evasion"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    impact: str
    requires_confirmation: bool = False


@dataclass
class EvalCase:
    id: str
    benchmark: str
    domain: str
    user_instruction: str
    external_content: str
    tools_available: list[str]
    attack_present: bool
    attack_type: AttackType
    attacker_goal: str = ""
    unsafe_tools: list[str] = field(default_factory=list)
    expected_safe_tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceEvent:
    step: int
    decision_point: DecisionPoint
    message: str
    tool_call: ToolCall | None = None
    tool_result: str | None = None
    safe: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    case_id: str
    events: list[TraceEvent]
    attack_success: bool
    utility_success: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActivationRecord:
    id: str
    case_id: str
    step: int
    decision_point: DecisionPoint
    layer: int
    token_position: str
    vector: list[float]
    activation_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditRecord:
    activation: ActivationRecord
    nla_explanations: list[str]
    latent_labels: list[LatentRiskLabel]
    risk_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectorPrediction:
    case_id: str
    step: int
    decision_point: DecisionPoint
    score: float
    label: bool
    metadata: dict[str, Any] = field(default_factory=dict)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: to_jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value
