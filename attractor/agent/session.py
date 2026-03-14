"""Coding Agent Session — core agentic loop."""

from __future__ import annotations

import enum
import json
import time
from dataclasses import dataclass, field
from typing import Any

from attractor.agent.config import SessionConfig
from attractor.agent.events import AgentEventEmitter, EventKind, SessionEvent
from attractor.agent.loop_detection import detect_loop
from attractor.agent.truncation import truncate_tool_output
from attractor.llm.types import (
    ContentKind,
    ContentPart,
    Message,
    Request,
    Response,
    Role,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


class SessionState(enum.Enum):
    """Session state machine."""
    IDLE = "idle"
    PROCESSING = "processing"
    AWAITING_INPUT = "awaiting_input"
    CLOSED = "closed"


@dataclass
class Turn:
    """A single turn in the session history."""
    role: str = "user"  # "user", "assistant", "tool", "steering"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class Session:
    """Core agentic loop session.

    Manages conversation history, tool execution, steering, and events.
    Uses Client.complete() directly (not the high-level generate())
    for maximum control over the loop.
    """

    def __init__(
        self,
        config: SessionConfig | None = None,
        llm_client: Any = None,
        tools: dict[str, Any] | None = None,
    ):
        self.config = config or SessionConfig()
        self._llm_client = llm_client
        self._tools: dict[str, Any] = tools or {}
        self._tool_definitions: list[ToolDefinition] = []
        self._state = SessionState.IDLE
        self._history: list[Message] = []
        self._turns: list[Turn] = []
        self._turn_count = 0
        self._tool_call_log: list[str] = []
        self._steering_queue: list[str] = []
        self._followup_queue: list[str] = []
        self._emitter = AgentEventEmitter()
        self._abort = False

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def history(self) -> list[Message]:
        return list(self._history)

    @property
    def emitter(self) -> AgentEventEmitter:
        return self._emitter

    def register_tool(
        self,
        name: str,
        handler: Any,
        definition: ToolDefinition | None = None,
    ) -> None:
        """Register a tool with its handler and optional definition."""
        self._tools[name] = handler
        if definition:
            self._tool_definitions.append(definition)

    def steer(self, message: str) -> None:
        """Queue a steering message to inject after the current tool round."""
        self._steering_queue.append(message)

    def follow_up(self, message: str) -> None:
        """Queue a follow-up message for after the current input completes."""
        self._followup_queue.append(message)

    def abort(self) -> None:
        """Signal the session to abort."""
        self._abort = True

    def process_input(self, user_input: str) -> str:
        """Process a user input through the agentic loop.

        The loop: LLM call → tool execution → repeat until natural completion.

        Returns:
            The final assistant text response.
        """
        if self._state == SessionState.CLOSED:
            raise RuntimeError("Session is closed")

        self._state = SessionState.PROCESSING
        self._emitter.emit_simple(EventKind.TURN_START, message=user_input)

        # Add user message to history
        self._history.append(Message.user(user_input))

        tool_rounds = 0
        final_text = ""

        while not self._abort:
            # Check turn limits
            if (self.config.max_turns > 0 and
                    self._turn_count >= self.config.max_turns):
                self._emitter.emit_simple(
                    EventKind.TURN_LIMIT,
                    message=f"Turn limit reached ({self.config.max_turns})",
                )
                self._state = SessionState.IDLE
                return "[Turn limit reached]"

            # Check tool round limit
            if (self.config.max_tool_rounds_per_input > 0 and
                    tool_rounds >= self.config.max_tool_rounds_per_input):
                self._state = SessionState.IDLE
                return final_text or "[Tool round limit reached]"

            # Make LLM call
            if self._llm_client is None:
                self._state = SessionState.IDLE
                return "[No LLM client configured]"

            self._emitter.emit_simple(EventKind.LLM_REQUEST)

            request = Request(
                model=self.config.model,
                messages=list(self._history),
                provider=self.config.provider or None,
                tools=self._tool_definitions if self._tool_definitions else None,
                reasoning_effort=self.config.reasoning_effort,
            )

            try:
                response: Response = self._llm_client.complete(request)
            except Exception as e:
                self._emitter.emit_simple(
                    EventKind.ERROR, message=f"LLM error: {e}"
                )
                self._state = SessionState.IDLE
                raise

            self._emitter.emit_simple(EventKind.LLM_RESPONSE)
            self._turn_count += 1

            # Add assistant response to history
            self._history.append(response.message)

            # Check for tool calls
            tool_calls = response.tool_calls
            if not tool_calls:
                # Natural completion — model responded with text only
                final_text = response.text
                break

            # Execute tool calls
            tool_results_messages: list[Message] = []
            for tc in tool_calls:
                self._emitter.emit_simple(
                    EventKind.TOOL_CALL_START,
                    tool_name=tc.name,
                    tool_call_id=tc.id,
                )
                self._tool_call_log.append(tc.name)

                result = self._execute_tool(tc)

                # Truncate output
                truncated = truncate_tool_output(
                    result.content, tc.name, self.config
                )
                result.content = truncated

                self._emitter.emit_simple(
                    EventKind.TOOL_CALL_END,
                    tool_name=tc.name,
                    is_error=result.is_error,
                )

                tool_results_messages.append(
                    Message.tool_result(tc.id, result.content, result.is_error)
                )

            # Add tool results to history
            self._history.extend(tool_results_messages)
            tool_rounds += 1

            # Loop detection
            if detect_loop(
                self._tool_call_log,
                window=self.config.loop_detection_window,
                threshold=self.config.loop_detection_threshold,
            ):
                self._emitter.emit_simple(
                    EventKind.LOOP_DETECTED,
                    message="Repeated tool call pattern detected",
                )
                # Inject steering message
                self._history.append(Message(
                    role=Role.USER,
                    content=[ContentPart.text_part(
                        "[SYSTEM WARNING: You appear to be in a loop, "
                        "repeating the same tool calls. Try a different approach.]"
                    )],
                ))

            # Inject steering messages
            while self._steering_queue:
                msg = self._steering_queue.pop(0)
                self._history.append(Message(
                    role=Role.USER,
                    content=[ContentPart.text_part(f"[STEERING]: {msg}")],
                ))
                self._emitter.emit_simple(EventKind.STEERING, message=msg)

            # Context window check
            self._check_context_usage()

        # Process follow-up queue
        if self._followup_queue and not self._abort:
            msg = self._followup_queue.pop(0)
            self._state = SessionState.IDLE
            return self.process_input(msg)

        self._state = SessionState.IDLE
        self._emitter.emit_simple(EventKind.TURN_END, message=final_text[:200])

        if self._abort:
            self._state = SessionState.CLOSED
            self._emitter.emit_simple(EventKind.SESSION_END, message="Aborted")

        return final_text

    def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        handler = self._tools.get(tool_call.name)
        if handler is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error: Unknown tool '{tool_call.name}'",
                is_error=True,
            )

        try:
            result = handler(**tool_call.arguments)
            content = result if isinstance(result, str) else json.dumps(result, default=str)
            return ToolResult(
                tool_call_id=tool_call.id,
                content=content,
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error executing {tool_call.name}: {e}",
                is_error=True,
            )

    def _check_context_usage(self) -> None:
        """Check approximate context window usage and warn if high."""
        total_chars = sum(
            len(msg.text) for msg in self._history
        )
        approx_tokens = total_chars // 4
        threshold = int(
            self.config.context_window_size * self.config.context_warning_threshold
        )
        if approx_tokens > threshold:
            pct = round(approx_tokens / self.config.context_window_size * 100)
            self._emitter.emit_simple(
                EventKind.WARNING,
                message=f"Context usage at ~{pct}% of context window",
            )

    def close(self) -> None:
        """Close the session."""
        self._state = SessionState.CLOSED
        self._emitter.emit_simple(EventKind.SESSION_END)
