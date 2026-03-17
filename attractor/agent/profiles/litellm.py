"""LiteLLM provider profile."""

from __future__ import annotations

from typing import Any

from attractor.agent.patch import apply_v4a_patch
from attractor.agent.profiles.base import BaseProfile
from attractor.agent.registry import RegisteredTool
from attractor.agent.types import ExecutionEnvironment
from attractor.llm.types import ToolDefinition


def apply_patch_executor(args: dict[str, Any], env: ExecutionEnvironment) -> str:
    affected = apply_v4a_patch(args["patch"], env)
    return f"Applied patch to: {', '.join(affected)}"


def edit_file_executor(args: dict[str, Any], env: ExecutionEnvironment) -> str:
    path = args["file_path"]
    old_string = args["old_string"]
    new_string = args["new_string"]
    if not old_string:
        raise ValueError("old_string must not be empty.")
    replace_all = args.get("replace_all", False)
    content = env.read_file_raw(path)
    count = content.count(old_string)
    if count == 0:
        raise ValueError(f"Could not find exact match for 'old_string' in {path}")
    if count > 1 and not replace_all:
        raise ValueError(
            f"'old_string' matches multiple locations in {path}. Provide more context or set replace_all=True"
        )
    new_content = content.replace(old_string, new_string) if replace_all else content.replace(
        old_string, new_string, 1
    )
    env.write_file(path, new_content)
    replacements = count if replace_all else 1
    return f"Successfully edited {path}. Made {replacements} replacements."


class LiteLLMProfile(BaseProfile):
    """Unified profile for the LiteLLM-backed runtime."""

    def __init__(self, model: str = "provider/model-name"):
        super().__init__(model)
        self.id = "litellm"
        self._register_litellm_tools()

    def _register_litellm_tools(self) -> None:
        self.tool_registry.register(
            RegisteredTool(
                definition=ToolDefinition(
                    name="apply_patch",
                    description=(
                        "Apply code changes using the v4a patch format. "
                        "Supports creating, deleting, and modifying files."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {"patch": {"type": "string"}},
                        "required": ["patch"],
                    },
                ),
                executor=apply_patch_executor,
            )
        )
        self.tool_registry.register(
            RegisteredTool(
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
                    },
                ),
                executor=edit_file_executor,
            )
        )

    def _get_base_instructions(self) -> str:
        return (
            "You are a coding assistant running through LiteLLM. "
            "Prefer apply_patch for structured code edits and use edit_file when exact replacement is safer."
        )

    def provider_options(self) -> dict[str, Any] | None:
        return {}
