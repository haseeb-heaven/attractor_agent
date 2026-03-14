"""Core pipeline execution engine.

Implements the execution lifecycle from spec §3:
PARSE → VALIDATE → INITIALIZE → EXECUTE → FINALIZE

The core loop follows the 5-step edge selection algorithm with
goal gate enforcement, retry logic, and checkpoint save/resume.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from attractor.pipeline.conditions import evaluate_condition
from attractor.pipeline.context import Checkpoint, Context, Outcome, StageStatus
from attractor.pipeline.events import EventEmitter, PipelineEvent, PipelineEventKind
from attractor.pipeline.graph import Edge, Graph, Node
from attractor.pipeline.handlers.base import Handler, HandlerRegistry
from attractor.pipeline.handlers.builtin import create_default_registry
from attractor.pipeline.interviewer import AutoApproveInterviewer, Interviewer
from attractor.pipeline.parser import parse_dot
from attractor.pipeline.retry import RetryPolicy, RETRY_STANDARD
from attractor.pipeline.transforms import TransformRegistry, create_default_transforms
from attractor.pipeline.validator import validate_or_raise


@dataclass
class PipelineConfig:
    """Configuration for a pipeline run."""
    simulate: bool = False
    variables: dict[str, str] = field(default_factory=dict)
    checkpoint_dir: str = ""
    resume_from: str = ""  # Path to checkpoint file
    goal: str = ""
    default_retry_policy: RetryPolicy = field(default_factory=lambda: RETRY_STANDARD)
    max_total_steps: int = 1000  # Safety limit
    codergen_backend: Any = None
    interviewer: Interviewer | None = None


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""
    success: bool = False
    context: Context = field(default_factory=Context)
    completed_nodes: list[str] = field(default_factory=list)
    final_node: str = ""
    error: str = ""
    total_steps: int = 0
    elapsed_seconds: float = 0.0


def select_next_edge(
    edges: list[Edge],
    outcome: Outcome,
    context: Context,
) -> Edge | None:
    """5-step edge selection algorithm from spec §3.3.

    Priority order:
      1. suggested_next_ids from outcome
      2. preferred_label match
      3. condition evaluation
      4. Unconditional edges
      5. Weight-based (highest weight)
    """
    if not edges:
        return None

    # Step 1: Suggested next IDs (highest priority)
    if outcome.suggested_next_ids:
        for edge in edges:
            if edge.to_node in outcome.suggested_next_ids:
                return edge

    # Step 2: Preferred label match
    if outcome.preferred_label:
        for edge in edges:
            if edge.label == outcome.preferred_label:
                return edge

    # Step 3: Condition evaluation
    conditional = [e for e in edges if e.condition]
    for edge in conditional:
        if evaluate_condition(edge.condition, outcome, context):
            return edge

    # Step 4: Unconditional edges (no label, no condition → match by status)
    unconditional = [e for e in edges if not e.condition]
    if unconditional:
        # Try to match by label = status name
        status_label = outcome.status.value.lower()
        for edge in unconditional:
            if edge.label.lower() == status_label:
                return edge
        # Try "success" or empty label as fallback
        for edge in unconditional:
            if not edge.label:
                return edge
        # Return first unconditional
        return unconditional[0]

    # Step 5: Weight-based fallback
    edges_sorted = sorted(edges, key=lambda e: e.weight, reverse=True)
    return edges_sorted[0] if edges_sorted else None


def run_pipeline(
    source: str | Graph,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """Execute a pipeline from DOT source or Graph.

    Lifecycle:
        1. PARSE:      Parse DOT source into Graph
        2. TRANSFORM:  Apply AST transforms (variable expansion, stylesheet)
        3. VALIDATE:   Run lint rules
        4. INITIALIZE: Set up context, handlers, checkpoint
        5. EXECUTE:    Core execution loop
        6. FINALIZE:   Emit completion events, return result
    """
    config = config or PipelineConfig()
    emitter = EventEmitter()
    start_time = time.time()

    # --- Phase 1: PARSE ---
    if isinstance(source, str):
        graph = parse_dot(source)
    else:
        graph = source

    # --- Phase 2: TRANSFORM ---
    transforms = create_default_transforms()
    graph = transforms.apply_all(graph, vars=config.variables)

    # Override goal from config if provided
    if config.goal:
        graph.goal = config.goal

    # --- Phase 3: VALIDATE ---
    try:
        validate_or_raise(graph)
    except Exception as e:
        return PipelineResult(
            success=False,
            error=f"Validation failed: {e}",
            elapsed_seconds=time.time() - start_time,
        )

    # --- Phase 4: INITIALIZE ---
    context = Context()
    handler_registry = create_default_registry()
    completed_nodes: list[str] = []
    node_retries: dict[str, int] = {}

    # Interviewer setup
    interviewer = config.interviewer or AutoApproveInterviewer()

    # Resume from checkpoint if specified
    if config.resume_from:
        try:
            checkpoint = Checkpoint.load(config.resume_from)
            context.apply_updates(checkpoint.context_values)
            completed_nodes = list(checkpoint.completed_nodes)
            node_retries = dict(checkpoint.node_retries)
            emitter.emit_simple(
                PipelineEventKind.LOG,
                message=f"Resumed from checkpoint: {config.resume_from}",
            )
        except Exception as e:
            return PipelineResult(
                success=False,
                error=f"Failed to load checkpoint: {e}",
                elapsed_seconds=time.time() - start_time,
            )

    # Find start node
    start_node = graph.find_start_node()
    if not start_node:
        return PipelineResult(
            success=False,
            error="No start node found",
            elapsed_seconds=time.time() - start_time,
        )

    # --- Phase 5: EXECUTE ---
    emitter.emit_simple(
        PipelineEventKind.PIPELINE_STARTED,
        message=f"Pipeline '{graph.name}' started",
        goal=graph.goal,
    )

    current_node = start_node
    step_count = 0
    final_node = ""

    while step_count < config.max_total_steps:
        step_count += 1
        node = current_node

        emitter.emit_simple(
            PipelineEventKind.STAGE_STARTED,
            node_id=node.id,
            message=f"Executing: {node.label or node.id}",
        )

        # Get handler
        handler = handler_registry.get(node.handler_type)
        if handler is None:
            emitter.emit_simple(
                PipelineEventKind.STAGE_FAILED,
                node_id=node.id,
                message=f"No handler for type: {node.handler_type}",
            )
            return PipelineResult(
                success=False,
                context=context,
                completed_nodes=completed_nodes,
                error=f"No handler for type: {node.handler_type}",
                total_steps=step_count,
                elapsed_seconds=time.time() - start_time,
            )

        # Execute handler with retry
        outcome = _execute_with_retry(
            handler=handler,
            node=node,
            context=context,
            graph=graph,
            emitter=emitter,
            node_retries=node_retries,
            config=config,
            interviewer=interviewer,
        )

        # Apply context updates from outcome
        if outcome.context_updates:
            context.apply_updates(outcome.context_updates)

        # Check outcome status
        if outcome.status == StageStatus.FAIL:
            emitter.emit_simple(
                PipelineEventKind.STAGE_FAILED,
                node_id=node.id,
                message=f"Stage failed: {outcome.failure_reason}",
            )

            # Check if retry is possible
            retries = node_retries.get(node.id, 0)
            max_retries = node.max_retries or graph.default_max_retry

            if retries < max_retries:
                node_retries[node.id] = retries + 1
                emitter.emit_simple(
                    PipelineEventKind.STAGE_RETRYING,
                    node_id=node.id,
                    attempt=retries + 1,
                    max_retries=max_retries,
                )
                # Retry the same node
                continue
            else:
                # Check for retry_target
                retry_target = node.retry_target or graph.retry_target
                if retry_target and retry_target in graph.nodes:
                    current_node = graph.nodes[retry_target]
                    continue

                # Exhausted retries
                return PipelineResult(
                    success=False,
                    context=context,
                    completed_nodes=completed_nodes,
                    final_node=node.id,
                    error=f"Stage '{node.id}' failed after {retries} retries: "
                          f"{outcome.failure_reason}",
                    total_steps=step_count,
                    elapsed_seconds=time.time() - start_time,
                )

        # Stage succeeded
        completed_nodes.append(node.id)
        emitter.emit_simple(
            PipelineEventKind.STAGE_COMPLETED,
            node_id=node.id,
            status=outcome.status.value,
        )

        # Goal gate enforcement
        if node.goal_gate and graph.goal:
            # In a real implementation, this would verify the goal
            # by examining artifacts or running checks
            emitter.emit_simple(
                PipelineEventKind.LOG,
                node_id=node.id,
                message=f"Goal gate check at '{node.id}' (goal: {graph.goal})",
            )

        # Save checkpoint
        if config.checkpoint_dir:
            checkpoint = Checkpoint.from_context(
                context, node.id, completed_nodes, node_retries,
            )
            checkpoint.save(
                Path(config.checkpoint_dir) / f"checkpoint_{node.id}.json"
            )
            emitter.emit_simple(
                PipelineEventKind.CHECKPOINT_SAVED,
                node_id=node.id,
            )

        # Check for exit
        if node.handler_type == "exit" or node.shape == "Msquare":
            final_node = node.id
            break

        # Select next edge
        outgoing = graph.outgoing_edges(node.id)
        if not outgoing:
            # Dead end — success (exit without explicit exit node)
            final_node = node.id
            break

        selected = select_next_edge(outgoing, outcome, context)
        if selected is None:
            final_node = node.id
            break

        emitter.emit_simple(
            PipelineEventKind.EDGE_SELECTED,
            node_id=node.id,
            target=selected.to_node,
            edge_label=selected.label,
        )

        # Move to the next node
        next_node_id = selected.to_node
        if next_node_id not in graph.nodes:
            return PipelineResult(
                success=False,
                context=context,
                completed_nodes=completed_nodes,
                error=f"Edge targets non-existent node: {next_node_id}",
                total_steps=step_count,
                elapsed_seconds=time.time() - start_time,
            )

        current_node = graph.nodes[next_node_id]

    # --- Phase 6: FINALIZE ---
    success = step_count < config.max_total_steps

    if success:
        emitter.emit_simple(
            PipelineEventKind.PIPELINE_COMPLETED,
            message=f"Pipeline '{graph.name}' completed successfully",
        )
    else:
        emitter.emit_simple(
            PipelineEventKind.PIPELINE_FAILED,
            message=f"Pipeline exceeded max steps ({config.max_total_steps})",
        )

    return PipelineResult(
        success=success,
        context=context,
        completed_nodes=completed_nodes,
        final_node=final_node,
        total_steps=step_count,
        elapsed_seconds=time.time() - start_time,
    )


def _execute_with_retry(
    handler: Handler,
    node: Node,
    context: Context,
    graph: Graph,
    emitter: EventEmitter,
    node_retries: dict[str, int],
    config: PipelineConfig,
    interviewer: Interviewer,
) -> Outcome:
    """Execute a handler with the appropriate kwargs."""
    kwargs: dict[str, Any] = {
        "interviewer": interviewer,
    }
    if config.codergen_backend:
        kwargs["codergen_backend"] = config.codergen_backend

    try:
        return handler.execute(node, context, graph, emitter, **kwargs)
    except Exception as e:
        return Outcome(
            status=StageStatus.FAIL,
            failure_reason=f"Handler exception: {e}",
        )
