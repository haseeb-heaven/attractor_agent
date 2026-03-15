"""Graph data model — Node, Edge, Graph with all supported attributes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Shape-to-handler-type mapping (spec §2.8)
SHAPE_TO_HANDLER: dict[str, str] = {
    "Mdiamond": "start",
    "Msquare": "exit",
    "box": "codergen",
    "hexagon": "wait.human",
    "diamond": "conditional",
    "component": "parallel",
    "tripleoctagon": "parallel.fan_in",
    "parallelogram": "tool",
    "house": "stack.manager_loop",
}


def _parse_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes")
    return bool(val)


def _parse_int(val: Any, default: int = 0) -> int:
    if isinstance(val, int):
        return val
    try:
        return int(str(val))
    except (ValueError, TypeError):
        return default


@dataclass
class Node:
    """A node in the pipeline graph."""
    id: str = ""
    label: str = ""
    shape: str = "box"
    type: str = ""
    prompt: str = ""
    max_retries: int = 0
    goal_gate: bool = False
    retry_target: str = ""
    fallback_retry_target: str = ""
    fidelity: str = ""
    thread_id: str = ""
    node_class: str = ""  # 'class' attribute
    timeout: str = ""
    llm_model: str = ""
    llm_provider: str = ""
    reasoning_effort: str = "high"
    auto_status: bool = False
    allow_partial: bool = False
    attrs: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_attrs(node_id: str, attrs: dict[str, str]) -> Node:
        """Create a Node from a dictionary of parsed attributes."""
        return Node(
            id=node_id,
            label=attrs.get("label", node_id),
            shape=attrs.get("shape", "box"),
            type=attrs.get("type", ""),
            prompt=attrs.get("prompt", ""),
            max_retries=_parse_int(attrs.get("max_retries", "0")),
            goal_gate=_parse_bool(attrs.get("goal_gate", "false")),
            retry_target=attrs.get("retry_target", ""),
            fallback_retry_target=attrs.get("fallback_retry_target", ""),
            fidelity=attrs.get("fidelity", ""),
            thread_id=attrs.get("thread_id", ""),
            node_class=attrs.get("class", ""),
            timeout=attrs.get("timeout", ""),
            llm_model=attrs.get("llm_model", ""),
            llm_provider=attrs.get("llm_provider", ""),
            reasoning_effort=attrs.get("reasoning_effort", "high"),
            auto_status=_parse_bool(attrs.get("auto_status", "false")),
            allow_partial=_parse_bool(attrs.get("allow_partial", "false")),
            attrs=dict(attrs),
        )

    @property
    def handler_type(self) -> str:
        """Resolve handler: explicit type= > handler= attr > shape mapping > default."""
        if self.type:
            return self.type
        # Check handler="" attribute stored in attrs dict
        handler_attr = self.attrs.get("handler", "")
        if handler_attr:
            return handler_attr
        return SHAPE_TO_HANDLER.get(self.shape, "codergen")


@dataclass
class Edge:
    """A directed edge in the pipeline graph."""
    from_node: str = ""
    to_node: str = ""
    label: str = ""
    condition: str = ""
    weight: int = 0
    fidelity: str = ""
    thread_id: str = ""
    loop_restart: bool = False
    attrs: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_attrs(from_node: str, to_node: str, attrs: dict[str, str]) -> Edge:
        """Create an Edge from a dictionary of parsed attributes."""
        return Edge(
            from_node=from_node,
            to_node=to_node,
            label=attrs.get("label", ""),
            condition=attrs.get("condition", ""),
            weight=_parse_int(attrs.get("weight", "0")),
            fidelity=attrs.get("fidelity", ""),
            thread_id=attrs.get("thread_id", ""),
            loop_restart=_parse_bool(attrs.get("loop_restart", "false")),
            attrs=dict(attrs),
        )


@dataclass
class Graph:
    """A directed graph representing a pipeline workflow."""
    name: str = ""
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    # Graph-level attributes
    goal: str = ""
    label: str = ""
    model_stylesheet: str = ""
    default_max_retry: int = 50
    retry_target: str = ""
    fallback_retry_target: str = ""
    default_fidelity: str = ""
    attrs: dict[str, str] = field(default_factory=dict)

    def outgoing_edges(self, node_id: str) -> list[Edge]:
        """Get all edges leaving a node."""
        return [e for e in self.edges if e.from_node == node_id]

    def incoming_edges(self, node_id: str) -> list[Edge]:
        """Get all edges entering a node."""
        return [e for e in self.edges if e.to_node == node_id]

    def find_start_node(self) -> Node | None:
        """Find the entry point node (Mdiamond shape or id='start'/'Start')."""
        for n in self.nodes.values():
            if n.shape == "Mdiamond":
                return n
        for name in ("start", "Start"):
            if name in self.nodes:
                return self.nodes[name]
        return None

    def find_exit_node(self) -> Node | None:
        """Find the exit point node (Msquare shape or id='exit'/'end')."""
        for n in self.nodes.values():
            if n.shape == "Msquare":
                return n
        for name in ("exit", "end", "Exit", "End"):
            if name in self.nodes:
                return self.nodes[name]
        return None

    def is_reachable_from_start(self) -> set[str]:
        """BFS from start node, return set of reachable node IDs."""
        start = self.find_start_node()
        if not start:
            return set()
        visited: set[str] = set()
        queue = [start.id]
        while queue:
            nid = queue.pop(0)
            if nid in visited:
                continue
            visited.add(nid)
            for e in self.outgoing_edges(nid):
                if e.to_node not in visited:
                    queue.append(e.to_node)
        return visited
