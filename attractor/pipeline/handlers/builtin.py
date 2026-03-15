"""Built-in handler implementations for all node types."""

from __future__ import annotations

import re
import subprocess
import time
from typing import Any

from attractor.pipeline.context import Context, Outcome, StageStatus
from attractor.pipeline.events import EventEmitter, PipelineEventKind
from attractor.pipeline.graph import Graph, Node
from attractor.pipeline.handlers.base import Handler, HandlerRegistry
from attractor.pipeline.handlers.scoring import SatisfactionScorerHandler
from attractor.pipeline.handlers.testing import TestExecutionHandler
from attractor.pipeline.handlers.twin import DigitalTwinHandler
from attractor.pipeline.interviewer import (
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

        # Expand variables
        prompt_text = node.prompt or node.label or f"Review step: {node.id}"
        prompt_text = re.sub(
            r"\$\{([^}]+)\}",
            lambda m: context.get_string(m.group(1), f"${{{m.group(1)}}}"),
            prompt_text,
        )

        question = Question(
            id=f"q_{node.id}_{int(time.time())}",
            text=prompt_text,
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
    """Fan-out: execute branches concurrently (spec §4.8)."""

    def execute(self, node: Node, context: Context, graph: Graph,
                emitter: EventEmitter, **kwargs: Any) -> Outcome:
        out_edges = graph.outgoing_edges(node.id)
        if not out_edges:
            return Outcome(status=StageStatus.SUCCESS)

        emitter.emit_simple(
            PipelineEventKind.PARALLEL_STARTED,
            node_id=node.id,
            branches=len(out_edges),
        )

        # The actual parallel execution is typically managed by a worker pool
        # or separate engine instances. Here we suggest all targets to the engine.
        # Spec §4.8: shape=component triggers concurrent branch execution.
        suggested = [e.to_node for e in out_edges]

        return Outcome(
            status=StageStatus.SUCCESS,
            suggested_next_ids=suggested,
            notes=f"Triggered {len(suggested)} parallel branches",
        )


class FanInHandler(Handler):
    """Consolidation point: wait for all parallel branches (spec §4.9)."""

    def execute(self, node: Node, context: Context, graph: Graph,
                emitter: EventEmitter, **kwargs: Any) -> Outcome:
        # Fan-in implementation: shape=tripleoctagon
        # It should verify that all incoming branches have completed.
        incoming = graph.incoming_edges(node.id)
        completed = context.get("completed_nodes", [])
        
        pending = [e.from_node for e in incoming if e.from_node not in completed]
        
        if pending:
            emitter.emit_simple(
                PipelineEventKind.LOG,
                node_id=node.id,
                message=f"Fan-in waiting for: {', '.join(pending)}",
            )
            # In a synchronous engine, this might return a special status to wait
            # or just continue if the engine handles the synchronization.
            return Outcome(
                status=StageStatus.PARTIAL_SUCCESS,
                notes=f"Waiting for {len(pending)} branches",
            )

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
    """Supervisor loop: observe -> steer -> wait pattern (spec §4.11)."""

    def execute(self, node: Node, context: Context, graph: Graph,
                emitter: EventEmitter, **kwargs: Any) -> Outcome:
        # Manager loop: shape=house.
        # Decisions based on context keys like 'loop_count', 'satisfaction_score'.
        loop_count = context.get(f"{node.id}.loop_count", 0) + 1
        context.set(f"{node.id}.loop_count", loop_count)
        
        max_loops = int(node.attrs.get("max_loops", "10"))
        
        emitter.emit_simple(
            PipelineEventKind.LOG,
            node_id=node.id,
            message=f"Manager loop '{node.id}' iteration {loop_count}/{max_loops}",
        )
        
        if loop_count >= max_loops:
            return Outcome(
                status=StageStatus.SUCCESS,
                preferred_label="exit",
                notes="Max loops reached",
            )

        return Outcome(status=StageStatus.SUCCESS)


class BugDiagnosisHandler(Handler):
    """Parses test failures to provide a diagnosis for the fixer."""

    def execute(self, node: Node, context: Context, graph: Graph,
                emitter: EventEmitter, **kwargs: Any) -> Outcome:
        # Search for test output
        test_stdout = context.get_string("RunTests.stdout", "") or context.get_string("test_runner.stdout", "")
        
        if not test_stdout:
            return Outcome(
                status=StageStatus.SUCCESS,
                context_updates={f"{node.id}.diagnosis": "No test output found to diagnose."}
            )

        # Basic "diagnosis" — extract everything after 'FAILURES' or 'FAIL'
        diagnosis = ""
        if "FAILURES" in test_stdout:
            diagnosis = test_stdout.split("FAILURES")[-1]
        elif "FAIL:" in test_stdout:
            diagnosis = test_stdout.split("FAIL:")[-1]
        else:
            # Maybe just a traceback?
            lines = test_stdout.splitlines()
            traceback = [line for line in lines if "Traceback" in line or "Error:" in line]
            diagnosis = "\n".join(traceback) or test_stdout[-500:]

        context.set(f"{node.id}.diagnosis", diagnosis)
        emitter.emit_simple(PipelineEventKind.LOG, node_id=node.id, message=f"Diagnosed bug: {diagnosis[:100]}...")
        
        return Outcome(
            status=StageStatus.SUCCESS,
            context_updates={f"{node.id}.diagnosis": diagnosis}
        )


class TargetedFixHandler(CodergenHandler):
    """LLM handler specifically tasked with fixing a diagnosed bug."""

    def execute(self, node: Node, context: Context, graph: Graph,
                emitter: EventEmitter, **kwargs: Any) -> Outcome:
        # Get diagnosis from context
        diagnosis = context.get_string("Diagnose.diagnosis", "") or context.get_string("bug_diagnosis.diagnosis", "")
        original_code = context.get_string("Generate.output", "")
        
        # Override prompt to include diagnosis
        node.prompt = (
            f"Original Code:\n{original_code}\n\n"
            f"Bug Diagnosis:\n{diagnosis}\n\n"
            f"Task: {node.prompt or 'Fix the bug identified above.'}"
        )
        
        # Use Codergen logic
        return super().execute(node, context, graph, emitter, **kwargs)


class ConvergenceHandler(Handler):
    """Decision point to exit SDLC loop based on quality metrics."""

    def execute(self, node: Node, context: Context, graph: Graph,
                emitter: EventEmitter, **kwargs: Any) -> Outcome:
        # Check test success and score
        test_rc = context.get("RunTests.returncode", 1)
        score = context.get("Score.satisfaction_score", 0)
        
        # User requirement: tests==0 (returncode 0) + score >= 95
        converged = (test_rc == 0 and score >= 95)
        
        emitter.emit_simple(
            PipelineEventKind.LOG,
            node_id=node.id,
            message=f"Convergence check: test_rc={test_rc}, score={score} -> converged={converged}"
        )
        
        if converged:
            return Outcome(
                status=StageStatus.SUCCESS,
                preferred_label="converged",
                suggested_next_ids=["Exit", "exit", "End", "end"]
            )
        else:
            return Outcome(
                status=StageStatus.SUCCESS,
                preferred_label="loop",
                suggested_next_ids=["Diagnose", "bug_diagnosis"]
            )


def create_default_registry() -> HandlerRegistry:
    """Create a HandlerRegistry with all built-in handlers registered."""

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
    
    # New Handlers
    registry.register("test_runner", TestExecutionHandler())
    registry.register("digital_twin", DigitalTwinHandler())
    registry.register("satisfaction_scorer", SatisfactionScorerHandler())
    registry.register("bug_diagnosis", BugDiagnosisHandler())
    registry.register("targeted_fix", TargetedFixHandler())
    registry.register("convergence", ConvergenceHandler())

    # Default handler for unknown types — use codergen
    registry.set_default(CodergenHandler())

    return registry
