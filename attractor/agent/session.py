from __future__ import annotations

import json
import uuid
from typing import Any

from attractor.agent.config import SessionConfig
from attractor.agent.types import (
    SessionState,
    UserTurn,
    AssistantTurn,
    ToolResultsTurn,
    SystemTurn,
    SteeringTurn,
    Turn,
    EventKind,
    SessionEvent,
    ExecutionEnvironment,
    ProviderProfile,
)
from attractor.agent.docs import discover_project_docs
from attractor.agent.events import AgentEventEmitter
from attractor.agent.loop_detection import detect_loop
from attractor.agent.truncation import truncate_tool_output
from attractor.llm.types import (
    Message,
    Request,
    Response,
    ToolCall,
    ToolResult,
)


class Session:
    """Core agentic loop session (Section 2.1)."""

    def __init__(
        self,
        profile: ProviderProfile,
        environment: ExecutionEnvironment,
        llm_client: Any,
        config: SessionConfig | None = None,
    ):
        self.id = str(uuid.uuid4())
        self.profile = profile
        self.env = environment
        self.llm_client = llm_client
        self.config = config or SessionConfig()
        
        self.history: list[Turn] = []
        self.state = SessionState.IDLE
        self.steering_queue: list[str] = []
        self.followup_queue: list[str] = []
        self.emitter = AgentEventEmitter()
        self.abort_signaled = False

    def emit(self, kind: EventKind, **data: Any) -> None:
        """Emit a session event (Section 2.9)."""
        self.emitter.emit(SessionEvent(
            kind=kind,
            session_id=self.id,
            data=data
        ))

    def steer(self, message: str) -> None:
        """Queue a steering message (Section 2.6)."""
        self.steering_queue.append(message)

    def follow_up(self, message: str) -> None:
        """Queue a follow-up message (Section 2.6)."""
        self.followup_queue.append(message)

    def abort(self) -> None:
        """Signal the session to abort."""
        self.abort_signaled = True

    def process_input(self, user_input: str) -> str:
        """Process user input through the agentic loop (Section 2.5)."""
        self.state = SessionState.PROCESSING
        self.history.append(UserTurn(content=user_input))
        self.emit(EventKind.USER_INPUT, content=user_input)

        # 1. Drain any pending steering messages
        self._drain_steering()

        round_count = 0
        final_text = ""

        while not self.abort_signaled:
            # 2. Check limits
            if (self.config.max_tool_rounds_per_input > 0 and 
                round_count >= self.config.max_tool_rounds_per_input):
                self.emit(EventKind.TURN_LIMIT, round=round_count)
                break

            total_turns = sum(1 for t in self.history if isinstance(t, (UserTurn, AssistantTurn)))
            if self.config.max_turns > 0 and total_turns >= self.config.max_turns:
                self.emit(EventKind.TURN_LIMIT, total_turns=total_turns)
                break

            # 3. Build LLM request (Section 6)
            project_docs = discover_project_docs(self.env, self.profile.id)
            system_prompt = self.profile.build_system_prompt(self.env, project_docs)
            
            messages = self._convert_history_to_messages()
            request = Request(
                model=self.profile.model,
                messages=[Message.system(system_prompt)] + messages,
                tools=self.profile.tools(),
                tool_choice="auto",
                reasoning_effort=self.config.reasoning_effort,
                provider=self.profile.id,
                provider_options=self.profile.provider_options(),
            )

            # 4. Call LLM
            self.emit(EventKind.ASSISTANT_TEXT_START)
            try:
                response: Response = self.llm_client.complete(request)
            except Exception as e:
                self.emit(EventKind.ERROR, message=str(e))
                self.state = SessionState.CLOSED
                raise

            # 5. Record assistant turn
            assistant_turn = AssistantTurn(
                content=response.text,
                tool_calls=response.tool_calls,
                reasoning=response.reasoning,
                usage=response.usage,
                response_id=response.id
            )
            self.history.append(assistant_turn)
            self.emit(EventKind.ASSISTANT_TEXT_END, text=response.text, reasoning=response.reasoning)

            # 6. If no tool calls, completion
            if not response.tool_calls:
                final_text = response.text
                break

            # 7. Execute tool calls
            round_count += 1
            results = self._execute_tool_calls(response.tool_calls)
            self.history.append(ToolResultsTurn(results=results))

            # 8. Drain steering messages injected during tool execution
            self._drain_steering()

            # 9. Loop detection (Section 2.10)
            if self.config.enable_loop_detection:
                signatures = self._extract_tool_call_signatures()
                if detect_loop(signatures, self.config.loop_detection_window):
                    warning = "Loop detected: tool calls follow a repeating pattern. Try a different approach."
                    self.history.append(SteeringTurn(content=warning))
                    self.emit(EventKind.LOOP_DETECTION, message=warning)

            # 10. Context window awareness (Section 5.5)
            self._check_context_usage()

        # Process follow-up messages
        if self.followup_queue and not self.abort_signaled:
            next_input = self.followup_queue.pop(0)
            return self.process_input(next_input)

        self.state = SessionState.IDLE
        self.emit(EventKind.SESSION_END)
        return final_text

    def _drain_steering(self) -> None:
        """Move queued steering messages to history."""
        while self.steering_queue:
            msg = self.steering_queue.pop(0)
            self.history.append(SteeringTurn(content=msg))
            self.emit(EventKind.STEERING_INJECTED, content=msg)

    def _execute_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute a batch of tool calls."""
        results = []
        # Parallel execution could be implemented here if self.profile.supports_parallel_tool_calls
        for tc in tool_calls:
            results.append(self._execute_single_tool(tc))
        return results

    def _execute_single_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute one tool call (Section 3.8)."""
        self.emit(EventKind.TOOL_CALL_START, tool_name=tool_call.name, call_id=tool_call.id)
        
        registered = self.profile.tool_registry.get(tool_call.name)
        if not registered:
            error_msg = f"Unknown tool: {tool_call.name}"
            self.emit(EventKind.TOOL_CALL_END, call_id=tool_call.id, error=error_msg)
            return ToolResult(tool_call_id=tool_call.id, content=error_msg, is_error=True)

        try:
            raw_output = registered.execute(tool_call.arguments, self.env)
            
            # Truncate output before sending to LLM (Section 5.3)
            truncated_output = truncate_tool_output(raw_output, tool_call.name, self.config)
            
            # Emit FULL untruncated output to events (Section 4.27)
            self.emit(EventKind.TOOL_CALL_END, call_id=tool_call.id, output=raw_output)
            
            return ToolResult(
                tool_call_id=tool_call.id,
                content=truncated_output,
                is_error=False
            )
        except Exception as e:
            error_msg = f"Tool error ({tool_call.name}): {str(e)}"
            self.emit(EventKind.TOOL_CALL_END, call_id=tool_call.id, error=error_msg)
            return ToolResult(tool_call_id=tool_call.id, content=error_msg, is_error=True)

    def _convert_history_to_messages(self) -> list[Message]:
        """Convert Turn objects to LLM Message objects."""
        messages: list[Message] = []
        for turn in self.history:
            if isinstance(turn, UserTurn):
                messages.append(Message.user(turn.content))
            elif isinstance(turn, AssistantTurn):
                # Need to reconstruct Assistant message with tool calls
                msg = Message.assistant(turn.content)
                # Re-add tool call parts to msg.content if needed
                # (Simple text content for now)
                messages.append(msg)
            elif isinstance(turn, ToolResultsTurn):
                for res in turn.results:
                    messages.append(Message.tool_result(res.tool_call_id, res.content, res.is_error))
            elif isinstance(turn, SteeringTurn):
                messages.append(Message.user(f"[STEERING]: {turn.content}"))
            elif isinstance(turn, SystemTurn):
                messages.append(Message.system(turn.content))
        return messages

    def _extract_tool_call_signatures(self) -> list[str]:
        """Extract name+args signatures for loop detection."""
        signatures = []
        for turn in self.history:
            if isinstance(turn, AssistantTurn):
                for tc in turn.tool_calls:
                    # Simple signature: name + sorted JSON args
                    args_str = json.dumps(tc.arguments, sort_keys=True)
                    signatures.append(f"{tc.name}:{args_str}")
        return signatures

    def _check_context_usage(self) -> None:
        """Heuristic context usage check (Section 5.5)."""
        total_chars = 0
        for turn in self.history:
            if isinstance(turn, (UserTurn, AssistantTurn, SystemTurn, SteeringTurn)):
                total_chars += len(turn.content)
            elif isinstance(turn, ToolResultsTurn):
                total_chars += sum(len(res.content) for res in turn.results)
        
        approx_tokens = total_chars // 4
        threshold = self.profile.context_window_size * self.config.context_warning_threshold
        if approx_tokens > threshold:
            pct = round(approx_tokens / self.profile.context_window_size * 100)
            self.emit(EventKind.WARNING, message=f"Context usage at ~{pct}% of context window")

    def close(self) -> None:
        """Close the session (Section 2.3)."""
        self.state = SessionState.CLOSED
        self.emit(EventKind.SESSION_END)
