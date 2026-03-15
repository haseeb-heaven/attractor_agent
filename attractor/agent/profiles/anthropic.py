"""Anthropic provider profile (Section 3.5)."""

from __future__ import annotations

from typing import Any
from attractor.agent.profiles.base import BaseProfile
from attractor.agent.registry import RegisteredTool
from attractor.llm.types import ToolDefinition
from attractor.agent.types import ExecutionEnvironment


def edit_file_executor(args: dict[str, Any], env: ExecutionEnvironment) -> str:
    path = args["file_path"]
    old_string = args["old_string"]
    new_string = args["new_string"]
    replace_all = args.get("replace_all", False)
    
    content = env.read_file_raw(path)
    
    # Exact match check
    count = content.count(old_string)
    if count == 0:
        raise ValueError(f"Could not find exact match for 'old_string' in {path}")
    if count > 1 and not replace_all:
        raise ValueError(f"'old_string' matches multiple locations in {path}. Provide more context or set replace_all=True")
    
    new_content = content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)
    env.write_file(path, new_content)
    
    num_replacements = count if replace_all else 1
    return f"Successfully edited {path}. Made {num_replacements} replacements."


class AnthropicProfile(BaseProfile):
    """Anthropic-aligned profile (Claude Code style)."""

    def __init__(self, model: str = "claude-4.6-sonnet"):
        super().__init__(model)
        self.id = "anthropic"
        self._register_anthropic_tools()

    def _register_anthropic_tools(self) -> None:
        self.tool_registry.register(RegisteredTool(
            definition=ToolDefinition(
                name="edit_file",
                description="Replace an exact string occurrence in a file.",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                        "replace_all": {"type": "boolean", "default": False},
                    },
                    "required": ["file_path", "old_string", "new_string"],
                }
            ),
            executor=edit_file_executor
        ))

    def _get_base_instructions(self) -> str:
        return """You are Claude, a world-class coding assistant from Anthropic.
When modifying files, use the `edit_file` tool which searches for an exact string and replaces it.
Ensure the `old_string` is unique within the file."""
