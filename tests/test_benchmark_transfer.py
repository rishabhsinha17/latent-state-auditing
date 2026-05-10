import json
from pathlib import Path

from latent_agent_auditing.experiments.evaluate_benchmark_transfer import evaluate_benchmark_transfer


def test_evaluate_benchmark_transfer_writes_holdout_tables(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        _case("workspace_clean", "workspace", False),
        _case("workspace_attack", "workspace", True),
        _case("slack_clean", "slack", False),
        _case("slack_attack", "slack", True),
    ]
    source = [
        _source("workspace_clean", "workspace", "none"),
        _source("workspace_attack", "workspace", "ignore_previous"),
        _source("slack_clean", "slack", "none"),
        _source("slack_attack", "slack", "system_message"),
    ]
    audits = [
        _audit("workspace_clean", [0.0, 0.0], "benign summary"),
        _audit("workspace_attack", [1.0, 1.0], "external injected instruction"),
        _audit("slack_clean", [0.1, 0.0], "benign task"),
        _audit("slack_attack", [0.9, 1.0], "attacker goal active"),
    ]
    _write_jsonl(run_dir / "case_scores.jsonl", rows)
    _write_jsonl(run_dir / "source_records.jsonl", source)
    _write_jsonl(run_dir / "audits.jsonl", audits)

    summary = evaluate_benchmark_transfer(run_dir)

    assert summary["experiments"]
    assert (run_dir / "paper_evidence" / "transfer" / "transfer_table.md").exists()
    assert (run_dir / "paper_evidence" / "transfer" / "transfer_summary.json").exists()


def _case(case_id: str, domain: str, attack_success: bool) -> dict[str, object]:
    return {
        "case_id": case_id,
        "benchmark": "agentdojo",
        "domain": domain,
        "attack_success": attack_success,
        "transcript_score": 0.0,
        "tool_policy_score": 0.0,
        "nla_score": 0.9 if attack_success else 0.1,
        "latent_lead_time": None,
    }


def _source(case_id: str, environment: str, attack_name: str) -> dict[str, object]:
    return {
        "id": case_id,
        "environment": environment,
        "attack_name": attack_name,
        "injection_task": {"id": "inj"} if attack_name != "none" else {},
    }


def _audit(case_id: str, vector: list[float], text: str) -> dict[str, object]:
    return {
        "activation": {
            "id": f"{case_id}_act",
            "case_id": case_id,
            "step": 1,
            "decision_point": "before_tool_selection",
            "layer": 20,
            "token_position": "final",
            "vector": vector,
            "activation_path": None,
            "metadata": {},
        },
        "nla_explanations": [text],
        "latent_labels": [],
        "risk_score": 0.0,
        "metadata": {},
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
