from pathlib import Path

from latent_agent_auditing.adapters.injecagent import make_injecagent_style_cases
from latent_agent_auditing.agent.rule_based import RuleBasedAgent
from latent_agent_auditing.auditing.activation_capture import MockActivationBackend, build_activation_prompt, resolve_activation_prompt
from latent_agent_auditing.auditing.nla_runner import HeuristicNLARunner
from latent_agent_auditing.auditing.pipeline import LatentAuditPipeline
from latent_agent_auditing.labels.taxonomy import LatentRiskLabel
from latent_agent_auditing.tools.mock_tools import MockToolRuntime


def test_attack_case_gets_latent_injection_label(tmp_path: Path) -> None:
    case = [case for case in make_injecagent_style_cases(cases_per_domain=2) if case.attack_present][0]
    runtime = MockToolRuntime(emails={"latest": case.external_content})
    trajectory = RuleBasedAgent(vulnerable=True).run(case, runtime)
    audits = LatentAuditPipeline(MockActivationBackend(), HeuristicNLARunner()).audit(
        case,
        trajectory,
        output_dir=tmp_path,
    )
    labels = {label for audit in audits for label in audit.latent_labels}
    assert LatentRiskLabel.EXTERNAL_INSTRUCTION_INTERNALIZED in labels
    assert trajectory.attack_success


def test_clean_case_stays_lower_risk(tmp_path: Path) -> None:
    case = [case for case in make_injecagent_style_cases(cases_per_domain=2) if not case.attack_present][0]
    runtime = MockToolRuntime(emails={"latest": case.external_content})
    trajectory = RuleBasedAgent(vulnerable=True).run(case, runtime)
    audits = LatentAuditPipeline(MockActivationBackend(), HeuristicNLARunner()).audit(
        case,
        trajectory,
        output_dir=tmp_path,
    )
    assert max(audit.risk_score for audit in audits) < 0.7
    assert not trajectory.attack_success


def test_activation_prompt_contains_security_context() -> None:
    case = [case for case in make_injecagent_style_cases(cases_per_domain=2) if case.attack_present][0]
    trajectory = RuleBasedAgent(vulnerable=True).run(case, MockToolRuntime(emails={"latest": case.external_content}))
    event = trajectory.events[3]
    prompt = build_activation_prompt(case, event)
    assert case.user_instruction in prompt
    assert case.external_content in prompt
    assert event.decision_point.value in prompt
    assert "Next proposed tool call" not in prompt
    assert "private payroll and invoice data" not in prompt


def test_activation_prompt_includes_tool_after_selection() -> None:
    case = [case for case in make_injecagent_style_cases(cases_per_domain=2) if case.attack_present][0]
    trajectory = RuleBasedAgent(vulnerable=True).run(case, MockToolRuntime(emails={"latest": case.external_content}))
    event = trajectory.events[4]
    prompt = build_activation_prompt(case, event)
    assert event.decision_point.value in prompt
    assert "send_email" in prompt


def test_exact_model_prefix_overrides_replay_prompt() -> None:
    case = [case for case in make_injecagent_style_cases(cases_per_domain=2) if case.attack_present][0]
    trajectory = RuleBasedAgent(vulnerable=True).run(case, MockToolRuntime(emails={"latest": case.external_content}))
    event = trajectory.events[3]
    event.metadata["model_prefix"] = "System: exact exported prefix\nAssistant:"

    prompt, prompt_source = resolve_activation_prompt(case, event)

    assert prompt == "System: exact exported prefix\nAssistant:"
    assert prompt_source == "exact_prefix"
