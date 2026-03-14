"""Built-in handler implementations for all node types."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from attractor.pipeline.context import Context, Outcome, StageStatus
from attractor.pipeline.events import EventEmitter, PipelineEventKind
from attractor.pipeline.graph import Edge, Graph, Node
from attractor.pipeline.handlers.base import Handler
from attractor.pipeline.interviewer import (
    Answer,
    Interviewer,
    Option,
    Question,
)


class StartHandler(Handler):
    """No-op — marks the pipeline entry point."""

    def execute(self, node: Node, context: Context, graph: Graph,
                emitter: EventEmitter, **kwargs: Any) -> Outcome:
        return Outcome(status=StageStatus.SUCCESS)


class ExitHandler(Handler):
    """No-op — marks the pipeline exit point."""

    def execute(self, node: Node, context: Context, graph: Graph,
                emitter: EventEmitter, **kwargs: Any) -> Outcome:
        return Outcome(status=StageStatus.SUCCESS)


class ConditionalHandler(Handler):
    """Pass-through — routing is handled by the engine's edge selection."""

    def execute(self, node: Node, context: Context, graph: Graph,
                emitter: EventEmitter, **kwargs: Any) -> Outcome:
        return Outcome(status=StageStatus.SUCCESS)


class CodergenHandler(Handler):
    """LLM code generation handler.

    Uses a CodergenBackend if provided, otherwise runs in simulation mode.
    """

    def execute(self, node: Node, context: Context, graph: Graph,
                emitter: EventEmitter, **kwargs: Any) -> Outcome:
        backend = kwargs.get("codergen_backend")

        # Expand variables in prompt
        prompt = self._expand_variables(node.prompt, context)

        emitter.emit_simple(
            PipelineEventKind.LOG,
            node_id=node.id,
            message=f"Executing codergen: {node.label}",
            prompt_preview=prompt[:200],
        )

        if backend is None:
            # Simulation mode
            context.set(f"{node.id}.output", f"[SIMULATED] Output for: {prompt[:100]}")
            context.set(f"{node.id}.status", "simulated")
            return Outcome(
                status=StageStatus.SUCCESS,
                notes="Simulated (no backend)",
                context_updates={f"{node.id}.output": f"[SIMULATED] {prompt[:100]}"},
            )

        try:
            result = backend.generate(prompt, node=node, context=context)
            context.set(f"{node.id}.output", result)
            context.set(f"{node.id}.status", "completed")
            return Outcome(
                status=StageStatus.SUCCESS,
                context_updates={f"{node.id}.output": result},
            )
        except Exception as e:
            context.set(f"{node.id}.status", "failed")
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason=str(e),
            )

    def _expand_variables(self, prompt: str, context: Context) -> str:
        """Replace ${var} references in prompt with context values."""
        def replacer(m: re.Match) -> str:
            key = m.group(1)
            return context.get_string(key, f"${{{key}}}")
        return re.sub(r"\$\{([^}]+)\}", replacer, prompt)


class WaitForHumanHandler(Handler):
    """Present choices to a human and route based on their answer."""

    def execute(self, node: Node, context: Context, graph: Graph,
                emitter: EventEmitter, **kwargs: Any) -> Outcome:
        interviewer: Interviewer | None = kwargs.get("interviewer")

        if interviewer is None:
            # No interviewer — auto-approve (first option)
            edges = graph.outgoing_edges(node.id)
            if edges:
                return Outcome(
                    status=StageStatus.SUCCESS,
                    preferred_label=edges[0].label,
                    suggested_next_ids=[edges[0].to_node],
                )
            return Outcome(status=StageStatus.SUCCESS)

        # Build options from outgoing edges
        edges = graph.outgoing_edges(node.id)
        options: list[Option] = []
        for i, edge in enumerate(edges):
            label = edge.label or f"Option {i + 1}"
            # Parse accelerator key from label: [Y]es, [N]o
            key = ""
            accel_match = re.search(r"\[(\w)\]", label)
            if accel_match:
                key = accel_match.group(1).lower()
            options.append(Option(label=label, key=key or str(i + 1)))

        question = Question(
            id=f"q_{node.id}_{int(time.time())}",
            text=node.prompt or node.label or f"Review step: {node.id}",
            options=options,
            node_id=node.id,
        )

        emitter.emit_simple(PipelineEventKind.INTERVIEW_STARTED, node_id=node.id)

        answer = interviewer.ask(question)

        emitter.emit_simple(
            PipelineEventKind.INTERVIEW_COMPLETED,
            node_id=node.id,
            selected=answer.selected_label,
        )

        if answer.timed_out:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason="Human review timed out",
            )

        # Map answer to edge
        for edge in edges:
            edge_label = edge.label or ""
            if (edge_label == answer.selected_label or
                    edge_label.lower().startswith(answer.selected_label.lower())):
                return Outcome(
                    status=StageStatus.SUCCESS,
                    preferred_label=edge.label,
                    suggested_next_ids=[edge.to_node],
                )

        # Fallback — treat as success
        return Outcome(
            status=StageStatus.SUCCESS,
            preferred_label=answer.selected_label,
        )


class ParallelHandler(Handler):
    """Fan-out: execute branches concurrently."""

    def execute(self, node: Node, context: Context, graph: Graph,
                emitter: EventEmitter, **kwargs: Any) -> Outcome:
        out_edges = graph.outgoing_edges(node.id)
        if not out_edges:
            return Outcome(status=StageStatus.SUCCESS)

        # Get join policy
        allow_partial = node.allow_partial

        emitter.emit_simple(
            PipelineEventKind.PARALLEL_STARTED,
            node_id=node.id,
            branches=len(out_edges),
        )

        results: list[tuple[str, StageStatus]] = []

        # Execute branches (simple sequential for now; real parallelism
        # would use the engine recursively per-branch)
        for edge in out_edges:
            target_id = edge.to_node
            emitter.emit_simple(
                PipelineEventKind.PARALLEL_BRANCH_STARTED,
                node_id=node.id,
                branch=target_id,
            )
            # Mark branches as suggested; actual execution is engine's job
            results.append((target_id, StageStatus.SUCCESS))
            emitter.emit_simple(
                PipelineEventKind.PARALLEL_BRANCH_COMPLETED,
                node_id=node.id,
                branch=target_id,
            )

        emitter.emit_simple(
            PipelineEventKind.PARALLEL_COMPLETED,
            node_id=node.id,
        )

        return Outcome(
            status=StageStatus.SUCCESS,
            suggested_next_ids=[e.to_node for e in out_edges],
        )


class FanInHandler(Handler):
    """Consolidation point: wait for all parallel branches."""

    def execute(self, node: Node, context: Context, graph: Graph,
                emitter: EventEmitter, **kwargs: Any) -> Outcome:
        # Fan-in simply passes through — the engine handles branch collection
        return Outcome(status=StageStatus.SUCCESS)


class ToolHandler(Handler):
    """Execute a shell command from the tool_command attribute."""

    def execute(self, node: Node, context: Context, graph: Graph,
                emitter: EventEmitter, **kwargs: Any) -> Outcome:
        command = node.attrs.get("tool_command", "")
        if not command:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason="No tool_command attribute specified",
            )

        # Expand variables
        command = re.sub(
            r"\$\{([^}]+)\}",
            lambda m: context.get_string(m.group(1), f"${{{m.group(1)}}}"),
            command,
        )

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            context.set(f"{node.id}.stdout", result.stdout)
            context.set(f"{node.id}.stderr", result.stderr)
            context.set(f"{node.id}.returncode", result.returncode)

            if result.returncode == 0:
                return Outcome(
                    status=StageStatus.SUCCESS,
                    context_updates={
                        f"{node.id}.stdout": result.stdout,
                        f"{node.id}.returncode": 0,
                    },
                )
            else:
                return Outcome(
                    status=StageStatus.FAIL,
                    failure_reason=f"Command exited with code {result.returncode}: {result.stderr[:200]}",
                )
        except subprocess.TimeoutExpired:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason="Command timed out",
            )
        except Exception as e:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason=str(e),
            )


class ManagerLoopHandler(Handler):
    """Supervisor loop: observe -> steer -> wait pattern."""

    def execute(self, node: Node, context: Context, graph: Graph,
                emitter: EventEmitter, **kwargs: Any) -> Outcome:
        # Manager loop collects results from previous stages and
        # decides whether to continue or exit the loop.
        # In simulation mode, just pass through.
        return Outcome(status=StageStatus.SUCCESS)


def create_default_registry() -> "HandlerRegistry":
    """Create a HandlerRegistry with all built-in handlers registered."""
    from attractor.pipeline.handlers.base import HandlerRegistry

    registry = HandlerRegistry()
    registry.register("start", StartHandler())
    registry.register("exit", ExitHandler())
    registry.register("codergen", CodergenHandler())
    registry.register("wait.human", WaitForHumanHandler())
    registry.register("conditional", ConditionalHandler())
    registry.register("parallel", ParallelHandler())
    registry.register("parallel.fan_in", FanInHandler())
    registry.register("tool", ToolHandler())
    registry.register("stack.manager_loop", ManagerLoopHandler())

    # Default handler for unknown types — use codergen
    registry.set_default(CodergenHandler())

    return registry
