from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from attractor.llm.types import ToolDefinition
from attractor.agent.types import ExecutionEnvironment


@dataclass
class RegisteredTool:
    """A tool registered with its definition and executor (Section 3.8)."""
    definition: ToolDefinition
    executor: Callable[[dict[str, Any], ExecutionEnvironment], str]

    def execute(self, arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
        """Execute the tool with the given arguments and environment."""
        # Note: Validation against JSON Schema should happen before this, 
        # but for now we'll just call the executor.
        return self.executor(arguments, env)


class ToolRegistry:
    """Registry of tools available to a provider profile."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        """Register or replace a tool."""
        self._tools[tool.definition.name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool by name."""
        if name in self._tools:
            del self._tools[name]

    def get(self, name: str) -> RegisteredTool | None:
        """Get a registered tool by name."""
        return self._tools.get(name)

    def definitions(self) -> list[ToolDefinition]:
        """Get all tool definitions."""
        return [t.definition for t in self._tools.values()]

    def names(self) -> list[str]:
        """Get names of all registered tools."""
        return list(self._tools.keys())
