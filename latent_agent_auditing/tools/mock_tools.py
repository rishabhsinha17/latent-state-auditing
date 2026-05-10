from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from latent_agent_auditing.models.schemas import ToolSpec


DEFAULT_TOOL_SPECS: dict[str, ToolSpec] = {
    "search_email": ToolSpec("search_email", "Search email subjects and senders.", "read"),
    "read_email": ToolSpec("read_email", "Read a single email body.", "read"),
    "send_email": ToolSpec("send_email", "Send an email to a recipient.", "external_side_effect", True),
    "list_files": ToolSpec("list_files", "List files in a workspace.", "read"),
    "read_file": ToolSpec("read_file", "Read a file from the workspace.", "read"),
    "write_file": ToolSpec("write_file", "Write or overwrite a workspace file.", "write", True),
    "delete_file": ToolSpec("delete_file", "Delete a workspace file.", "destructive", True),
    "query_database": ToolSpec("query_database", "Run a read-only database query.", "read"),
    "export_table": ToolSpec("export_table", "Export a full database table.", "secret_or_bulk_data", True),
    "open_url": ToolSpec("open_url", "Open a URL and return text content.", "read"),
    "submit_form": ToolSpec("submit_form", "Submit a web form.", "external_side_effect", True),
    "run_shell": ToolSpec("run_shell", "Run a shell command in a sandbox.", "code_execution", True),
    "commit_code": ToolSpec("commit_code", "Commit code changes.", "external_side_effect", True),
}


@dataclass
class MockToolRuntime:
    """In-memory tools with no real external side effects."""

    emails: dict[str, str] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    database: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    web_pages: dict[str, str] = field(default_factory=dict)
    sent_emails: list[dict[str, Any]] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)
    shell_commands: list[str] = field(default_factory=list)

    def execute(self, tool_name: str, args: dict[str, Any]) -> str:
        if tool_name == "search_email":
            query = str(args.get("query", "")).lower()
            matches = [key for key, body in self.emails.items() if query in key.lower() or query in body.lower()]
            return "\n".join(matches)
        if tool_name == "read_email":
            return self.emails.get(str(args.get("id", "")), "")
        if tool_name == "send_email":
            self.sent_emails.append(dict(args))
            return "email_sent"
        if tool_name == "list_files":
            return "\n".join(sorted(self.files))
        if tool_name == "read_file":
            return self.files.get(str(args.get("path", "")), "")
        if tool_name == "write_file":
            self.files[str(args.get("path", ""))] = str(args.get("content", ""))
            return "file_written"
        if tool_name == "delete_file":
            path = str(args.get("path", ""))
            self.deleted_files.append(path)
            self.files.pop(path, None)
            return "file_deleted"
        if tool_name == "query_database":
            table = str(args.get("table", ""))
            return repr(self.database.get(table, []))
        if tool_name == "export_table":
            table = str(args.get("table", ""))
            return repr(self.database.get(table, []))
        if tool_name == "open_url":
            return self.web_pages.get(str(args.get("url", "")), "")
        if tool_name == "submit_form":
            return "form_submitted"
        if tool_name == "run_shell":
            command = str(args.get("command", ""))
            self.shell_commands.append(command)
            return "shell_command_recorded"
        if tool_name == "commit_code":
            return "commit_recorded"
        raise KeyError(f"unknown tool: {tool_name}")


def tool_impact(tool_name: str) -> str:
    spec = DEFAULT_TOOL_SPECS.get(tool_name)
    return spec.impact if spec else "unknown"
