"""OpenAI provider profile (Section 3.4)."""

from __future__ import annotations

from typing import Any
from attractor.agent.profiles.base import BaseProfile
from attractor.agent.registry import RegisteredTool
from attractor.llm.types import ToolDefinition
from attractor.agent.patch import apply_v4a_patch
from attractor.agent.types import ExecutionEnvironment


def apply_patch_executor(args: dict[str, Any], env: ExecutionEnvironment) -> str:
    affected = apply_v4a_patch(args["patch"], env)
    return f"Applied patch to: {', '.join(affected)}"


class OpenAIProfile(BaseProfile):
    """OpenAI-aligned profile (codex-rs style)."""

    def __init__(self, model: str = "gpt-5.2-codex"):
        super().__init__(model)
        self.id = "openai"
        self._register_openai_tools()

    def _register_openai_tools(self) -> None:
        self.tool_registry.register(RegisteredTool(
            definition=ToolDefinition(
                name="apply_patch",
                description="Apply code changes using the v4a patch format. Supports creating, deleting, and modifying files.",
                parameters={
                    "type": "object",
                    "properties": {
                        "patch": {"type": "string"},
                    },
                    "required": ["patch"],
                }
            ),
            executor=apply_patch_executor
        ))

    def _get_base_instructions(self) -> str:
        return """You are a world-class coding assistant using the OpenAI codex models.
When modifying files, prefer using the `apply_patch` tool with the v4a format.
The v4a format uses:
*** Begin Patch
*** Update File: path/to/file
@@ context hint
- removed line
+ added line
*** End Patch"""

    def provider_options(self) -> dict[str, Any] | None:
        return {} # OpenAI specific options like reasoning_effort are handled by Session
