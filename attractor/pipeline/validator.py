"""Validation and linting for DOT pipeline definitions.

Implements 13 built-in lint rules from spec §7.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Callable

from attractor.pipeline.graph import Graph, SHAPE_TO_HANDLER


class Severity(enum.Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Diagnostic:
    """A single validation finding."""
    rule: str = ""
    severity: Severity = Severity.ERROR
    message: str = ""
    node_id: str = ""
    edge: str = ""


# Type for lint rule functions
LintRule = Callable[[Graph], list[Diagnostic]]


def _check_start_node(graph: Graph) -> list[Diagnostic]:
    """Exactly one start node (Mdiamond) must exist."""
    starts = [n for n in graph.nodes.values() if n.shape == "Mdiamond"]
    if len(starts) == 0:
        return [Diagnostic(
            rule="start_node", severity=Severity.ERROR,
            message="No start node found (shape=Mdiamond is required)",
        )]
    if len(starts) > 1:
        return [Diagnostic(
            rule="start_node", severity=Severity.ERROR,
            message=f"Multiple start nodes found: {[s.id for s in starts]}. Exactly one is required.",
        )]
    return []


def _check_terminal_node(graph: Graph) -> list[Diagnostic]:
    """At least one terminal node (Msquare) must exist."""
    exits = [n for n in graph.nodes.values() if n.shape == "Msquare"]
    if not exits:
        return [Diagnostic(
            rule="terminal_node", severity=Severity.WARNING,
            message="No terminal node found (shape=Msquare is recommended)",
        )]
    return []


def _check_reachability(graph: Graph) -> list[Diagnostic]:
    """All nodes must be reachable from the start node."""
    reachable = graph.is_reachable_from_start()
    unreachable = set(graph.nodes.keys()) - reachable
    return [
        Diagnostic(
            rule="reachability", severity=Severity.WARNING,
            message=f"Node '{nid}' is not reachable from the start node",
            node_id=nid,
        )
        for nid in sorted(unreachable)
    ]


def _check_edge_target_exists(graph: Graph) -> list[Diagnostic]:
    """Every edge must reference existing nodes."""
    results: list[Diagnostic] = []
    for edge in graph.edges:
        if edge.from_node not in graph.nodes:
            results.append(Diagnostic(
                rule="edge_target_exists", severity=Severity.ERROR,
                message=f"Edge source '{edge.from_node}' does not exist as a node",
                edge=f"{edge.from_node} -> {edge.to_node}",
            ))
        if edge.to_node not in graph.nodes:
            results.append(Diagnostic(
                rule="edge_target_exists", severity=Severity.ERROR,
                message=f"Edge target '{edge.to_node}' does not exist as a node",
                edge=f"{edge.from_node} -> {edge.to_node}",
            ))
    return results


def _check_valid_shape(graph: Graph) -> list[Diagnostic]:
    """Node shapes must map to known handler types."""
    valid_shapes = set(SHAPE_TO_HANDLER.keys())
    return [
        Diagnostic(
            rule="valid_shape", severity=Severity.WARNING,
            message=f"Node '{n.id}' has unrecognized shape '{n.shape}'. "
                    f"Known shapes: {sorted(valid_shapes)}",
            node_id=n.id,
        )
        for n in graph.nodes.values()
        if n.shape not in valid_shapes and not n.type
    ]


def _check_conditional_edges(graph: Graph) -> list[Diagnostic]:
    """Conditional (diamond) nodes must have labelled outgoing edges."""
    results: list[Diagnostic] = []
    for n in graph.nodes.values():
        if n.handler_type == "conditional":
            out_edges = graph.outgoing_edges(n.id)
            unlabelled = [e for e in out_edges if not e.label and not e.condition]
            if len(unlabelled) > 1:
                results.append(Diagnostic(
                    rule="conditional_edges", severity=Severity.WARNING,
                    message=f"Conditional node '{n.id}' has {len(unlabelled)} unlabelled edges. "
                            "Edges should have labels or conditions for deterministic routing.",
                    node_id=n.id,
                ))
    return results


def _check_retry_target_exists(graph: Graph) -> list[Diagnostic]:
    """retry_target and fallback_retry_target must reference existing nodes."""
    results: list[Diagnostic] = []
    for n in graph.nodes.values():
        if n.retry_target and n.retry_target not in graph.nodes:
            results.append(Diagnostic(
                rule="retry_target_exists", severity=Severity.ERROR,
                message=f"Node '{n.id}' retry_target='{n.retry_target}' does not exist",
                node_id=n.id,
            ))
        if n.fallback_retry_target and n.fallback_retry_target not in graph.nodes:
            results.append(Diagnostic(
                rule="retry_target_exists", severity=Severity.ERROR,
                message=f"Node '{n.id}' fallback_retry_target='{n.fallback_retry_target}' does not exist",
                node_id=n.id,
            ))
    return results


def _check_start_has_outgoing(graph: Graph) -> list[Diagnostic]:
    """Start node must have at least one outgoing edge."""
    start = graph.find_start_node()
    if start and not graph.outgoing_edges(start.id):
        return [Diagnostic(
            rule="start_has_outgoing", severity=Severity.ERROR,
            message=f"Start node '{start.id}' has no outgoing edges",
            node_id=start.id,
        )]
    return []


def _check_exit_no_outgoing(graph: Graph) -> list[Diagnostic]:
    """Exit nodes should not have outgoing edges."""
    results: list[Diagnostic] = []
    for n in graph.nodes.values():
        if n.shape == "Msquare" and graph.outgoing_edges(n.id):
            results.append(Diagnostic(
                rule="exit_no_outgoing", severity=Severity.WARNING,
                message=f"Exit node '{n.id}' has outgoing edges (unexpected)",
                node_id=n.id,
            ))
    return results


def _check_goal_not_empty(graph: Graph) -> list[Diagnostic]:
    """Graph should have a goal attribute for goal gate enforcement."""
    if not graph.goal:
        has_goal_gates = any(n.goal_gate for n in graph.nodes.values())
        if has_goal_gates:
            return [Diagnostic(
                rule="goal_not_empty", severity=Severity.WARNING,
                message="Graph has goal_gate nodes but no graph-level 'goal' attribute",
            )]
    return []


def _check_duplicate_edges(graph: Graph) -> list[Diagnostic]:
    """Warn on duplicate edges between the same pair."""
    seen: set[tuple[str, str, str]] = set()
    results: list[Diagnostic] = []
    for e in graph.edges:
        key = (e.from_node, e.to_node, e.label)
        if key in seen:
            results.append(Diagnostic(
                rule="duplicate_edges", severity=Severity.WARNING,
                message=f"Duplicate edge: {e.from_node} -> {e.to_node} (label='{e.label}')",
                edge=f"{e.from_node} -> {e.to_node}",
            ))
        seen.add(key)
    return results


def _check_prompt_set(graph: Graph) -> list[Diagnostic]:
    """codergen nodes should have a prompt attribute."""
    results: list[Diagnostic] = []
    for n in graph.nodes.values():
        if n.handler_type == "codergen" and not n.prompt:
            results.append(Diagnostic(
                rule="prompt_set", severity=Severity.WARNING,
                message=f"Node '{n.id}' (codergen) has no 'prompt' attribute",
                node_id=n.id,
            ))
    return results


def _check_no_self_loops(graph: Graph) -> list[Diagnostic]:
    """Edges should not point to themselves (except with loop_restart)."""
    return [
        Diagnostic(
            rule="no_self_loops", severity=Severity.WARNING,
            message=f"Self-loop on node '{e.from_node}' without loop_restart=true",
            edge=f"{e.from_node} -> {e.to_node}",
        )
        for e in graph.edges
        if e.from_node == e.to_node and not e.loop_restart
    ]


# All built-in rules
BUILTIN_RULES: list[LintRule] = [
    _check_start_node,
    _check_terminal_node,
    _check_reachability,
    _check_edge_target_exists,
    _check_valid_shape,
    _check_conditional_edges,
    _check_retry_target_exists,
    _check_start_has_outgoing,
    _check_exit_no_outgoing,
    _check_goal_not_empty,
    _check_duplicate_edges,
    _check_prompt_set,
    _check_no_self_loops,
]


def validate(
    graph: Graph,
    *,
    extra_rules: list[LintRule] | None = None,
) -> list[Diagnostic]:
    """Run all lint rules against a graph. Returns a list of diagnostics."""
    all_rules = list(BUILTIN_RULES)
    if extra_rules:
        all_rules.extend(extra_rules)

    diagnostics: list[Diagnostic] = []
    for rule in all_rules:
        diagnostics.extend(rule(graph))
    return diagnostics


class ValidationError(Exception):
    """Raised when pipeline validation fails."""

    def __init__(self, diagnostics: list[Diagnostic]):
        self.diagnostics = diagnostics
        errors = [d for d in diagnostics if d.severity == Severity.ERROR]
        msg = f"Pipeline validation failed with {len(errors)} error(s):\n"
        msg += "\n".join(f"  [{d.rule}] {d.message}" for d in errors)
        super().__init__(msg)


def validate_or_raise(
    graph: Graph,
    *,
    extra_rules: list[LintRule] | None = None,
) -> list[Diagnostic]:
    """Validate and raise if any ERROR-severity diagnostics are found."""
    diagnostics = validate(graph, extra_rules=extra_rules)
    errors = [d for d in diagnostics if d.severity == Severity.ERROR]
    if errors:
        raise ValidationError(diagnostics)
    return diagnostics
