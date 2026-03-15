"""Base ProviderProfile interface and common functionality."""

from __future__ import annotations

import datetime
from typing import Any

from attractor.agent.types import ExecutionEnvironment, ProviderProfile
from attractor.agent.registry import ToolRegistry, RegisteredTool
from attractor.llm.types import ToolDefinition
from attractor.agent.tools.common import (
    read_file_executor,
    write_file_executor,
    shell_executor,
    grep_executor,
    glob_executor,
)
from attractor.agent.subagents import register_subagent_tools


class BaseProfile(ProviderProfile):
    """Base implementation of ProviderProfile with shared logic."""

    def __init__(self, model: str):
        self.model = model
        self.tool_registry = ToolRegistry()
        self._register_common_tools()

    def _register_common_tools(self) -> None:
        self.tool_registry.register(RegisteredTool(
            definition=ToolDefinition(
                name="read_file",
                description="Read a file from the filesystem. Returns line-numbered content.",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "offset": {"type": "integer"},
                        "limit": {"type": "integer", "default": 2000},
                    },
                    "required": ["file_path"],
                }
            ),
            executor=read_file_executor
        ))
        
        self.tool_registry.register(RegisteredTool(
            definition=ToolDefinition(
                name="write_file",
                description="Write content to a file. Creates the file and parent directories if needed.",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["file_path", "content"],
                }
            ),
            executor=write_file_executor
        ))

        self.tool_registry.register(RegisteredTool(
            definition=ToolDefinition(
                name="shell",
                description="Execute a shell command. Returns stdout, stderr, and exit code.",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout_ms": {"type": "integer"},
                        "description": {"type": "string"},
                    },
                    "required": ["command"],
                }
            ),
            executor=shell_executor
        ))

        self.tool_registry.register(RegisteredTool(
            definition=ToolDefinition(
                name="grep",
                description="Search file contents using regex patterns.",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                        "case_insensitive": {"type": "boolean"},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["pattern"],
                }
            ),
            executor=grep_executor
        ))

        self.tool_registry.register(RegisteredTool(
            definition=ToolDefinition(
                name="glob",
                description="Find files matching a glob pattern.",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                    },
                    "required": ["pattern"],
                }
            ),
            executor=glob_executor
        ))
        
        register_subagent_tools(self.tool_registry)

    def build_system_prompt(self, environment: ExecutionEnvironment, project_docs: str = "") -> str:
        # Structured block with runtime information (Section 6.3)
        now = datetime.datetime.now().strftime("%Y-%m-%d")
        env_block = f"""<environment>
Working directory: {environment.working_directory()}
Platform: {environment.platform()}
OS version: {environment.os_version()}
Today's date: {now}
Model: {self.model}
</environment>"""

        # Tool descriptions (Section 6.1)
        # Note: In a real implementation, we might want more detailed tool descriptions.
        tool_block = "Available tools:\n" + "\n".join([
            f"- {t.name}: {t.description}" for t in self.tool_registry.definitions()
        ])

        return f"{self._get_base_instructions()}\n\n{env_block}\n\n{tool_block}\n\n{project_docs}"

    def _get_base_instructions(self) -> str:
        """Override in subclasses with provider-specific base instructions."""
        return "You are a coding agent."

    def tools(self) -> list[ToolDefinition]:
        return self.tool_registry.definitions()

    def provider_options(self) -> dict[str, Any] | None:
        return None

    @property
    def supports_reasoning(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_parallel_tool_calls(self) -> bool:
        return True

    @property
    def context_window_size(self) -> int:
        return 128000
