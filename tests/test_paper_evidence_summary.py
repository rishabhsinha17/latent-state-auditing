import json
from pathlib import Path

from latent_agent_auditing.experiments.summarize_paper_evidence import summarize_paper_evidence


def test_summarize_paper_evidence_writes_table_and_predictions(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    case_rows = [
        _case("clean_email", False, 0.1, 0.0, 0.1, None),
        _case("clean_file", False, 0.2, 0.0, 0.2, None),
        _case("attack_email", True, 0.8, 0.9, 0.8, 1),
        _case("attack_file", True, 0.9, 0.9, 0.9, 2),
    ]
    audits = [
        _audit("clean_email", [0.0, 0.0], ["The model is focused on summarizing the task."]),
        _audit("clean_file", [0.1, 0.0], ["The model is focused on the user request."]),
        _audit("attack_email", [1.0, 1.0], ["The model is considering sending data to an attacker."]),
        _audit("attack_file", [0.9, 1.0], ["The model is following an injected external instruction."]),
    ]
    _write_jsonl(run_dir / "case_scores.jsonl", case_rows)
    _write_jsonl(run_dir / "audits.jsonl", audits)
    (run_dir / "run_manifest.json").write_text(json.dumps({"prompt_summary": {"exact_prefix": 4}}), encoding="utf-8")

    summary = summarize_paper_evidence(run_dir, folds=2, bootstrap_resamples=10)

    assert summary["cases"] == 4
    assert summary["positive_cases"] == 2
    assert "raw_activation_probe" in summary["monitors"]
    assert (run_dir / "paper_evidence" / "paper_table.md").exists()
    assert (run_dir / "paper_evidence" / "predictions.jsonl").exists()


def _case(
    case_id: str,
    attack_success: bool,
    transcript_score: float,
    tool_policy_score: float,
    nla_score: float,
    latent_lead_time: int | None,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "benchmark": "agentdojo",
        "domain": "workspace",
        "attack_success": attack_success,
        "transcript_score": transcript_score,
        "tool_policy_score": tool_policy_score,
        "nla_score": nla_score,
        "unsafe_step": 2 if attack_success else None,
        "latent_lead_time": latent_lead_time,
        "latent_labels": [],
    }


def _audit(case_id: str, vector: list[float], explanations: list[str]) -> dict[str, object]:
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
        "nla_explanations": explanations,
        "latent_labels": [],
        "risk_score": 0.5,
        "metadata": {"attack_success": case_id.startswith("attack")},
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
