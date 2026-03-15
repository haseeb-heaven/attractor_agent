"""Subagent management tools (Section 7)."""

from __future__ import annotations

import uuid
from typing import Any, Dict

from attractor.agent.types import ExecutionEnvironment
from attractor.agent.registry import RegisteredTool
from attractor.llm.types import ToolDefinition


# This would typically be managed by the Session or a SubAgentManager
_active_subagents: Dict[str, Any] = {}


def spawn_agent_executor(args: dict[str, Any], env: ExecutionEnvironment) -> str:
    """Executor for spawn_agent (Section 7.2)."""
    agent_id = str(uuid.uuid4())
    task = args["task"]
    
    # In a real implementation, we would need the llm_client and current profile
    # For now, this is a placeholder demonstrating the interface
    return f"Spawned subagent {agent_id} for task: {task}"


def send_input_executor(args: dict[str, Any], env: ExecutionEnvironment) -> str:
    agent_id = args["agent_id"]
    return f"Sent input to subagent {agent_id}"


def wait_executor(args: dict[str, Any], env: ExecutionEnvironment) -> str:
    agent_id = args["agent_id"]
    return f"Subagent {agent_id} completed successfully."


def close_agent_executor(args: dict[str, Any], env: ExecutionEnvironment) -> str:
    agent_id = args["agent_id"]
    return f"Closed subagent {agent_id}"


def register_subagent_tools(registry: Any) -> None:
    """Helper to register subagent tools to a registry."""
    registry.register(RegisteredTool(
        definition=ToolDefinition(
            name="spawn_agent",
            description="Spawn a subagent to handle a scoped task autonomously.",
            parameters={
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "working_dir": {"type": "string"},
                    "model": {"type": "string"},
                    "max_turns": {"type": "integer"},
                },
                "required": ["task"],
            }
        ),
        executor=spawn_agent_executor
    ))
    
    registry.register(RegisteredTool(
        definition=ToolDefinition(
            name="send_input",
            description="Send a message to a running subagent.",
            parameters={
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["agent_id", "message"],
            }
        ),
        executor=send_input_executor
    ))

    registry.register(RegisteredTool(
        definition=ToolDefinition(
            name="wait",
            description="Wait for a subagent to complete and return its result.",
            parameters={
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string"},
                },
                "required": ["agent_id"],
            }
        ),
        executor=wait_executor
    ))

    registry.register(RegisteredTool(
        definition=ToolDefinition(
            name="close_agent",
            description="Terminate a subagent.",
            parameters={
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string"},
                },
                "required": ["agent_id"],
            }
        ),
        executor=close_agent_executor
    ))
