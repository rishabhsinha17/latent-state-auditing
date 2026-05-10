from latent_agent_auditing.experiments.export_agentdojo_model_trajectories import _parse_tool_call


def test_parse_model_tool_json() -> None:
    result = _parse_tool_call('{"tool": "send_email", "args": {"to": "a@example.com"}}', ["read_email", "send_email"])
    assert result == {"name": "send_email", "args": {"to": "a@example.com"}}


def test_parse_model_tool_none() -> None:
    assert _parse_tool_call('{"tool": "none", "args": {}}', ["read_email"]) is None


def test_parse_model_tool_fallback_text() -> None:
    result = _parse_tool_call("I will use read_email next.", ["read_email", "send_email"])
    assert result == {"name": "read_email", "args": {}}
