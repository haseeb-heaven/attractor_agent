"""AST transforms — modify the graph after parsing but before validation."""

from __future__ import annotations

import abc
import re
from typing import Any

from attractor.pipeline.graph import Graph
from attractor.pipeline.stylesheet import parse_stylesheet, apply_stylesheet


class Transform(abc.ABC):
    """Interface for graph transforms."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @abc.abstractmethod
    def apply(self, graph: Graph, vars: dict[str, str] | None = None) -> Graph:
        """Apply the transform to a graph, returning the (possibly modified) graph."""
        ...


class VariableExpansionTransform(Transform):
    """Replace ${var} references in node attributes with provided values."""

    @property
    def name(self) -> str:
        return "variable_expansion"

    def apply(self, graph: Graph, vars: dict[str, str] | None = None) -> Graph:
        if not vars:
            return graph

        def expand(text: str) -> str:
            def replacer(m: re.Match) -> str:
                key = m.group(1)
                return vars.get(key, m.group(0))
            return re.sub(r"\$\{([^}]+)\}", replacer, text)

        for node in graph.nodes.values():
            node.label = expand(node.label)
            node.prompt = expand(node.prompt)
            for key, val in list(node.attrs.items()):
                node.attrs[key] = expand(val)

        for edge in graph.edges:
            edge.label = expand(edge.label)
            edge.condition = expand(edge.condition)

        if graph.goal:
            graph.goal = expand(graph.goal)

        return graph


class StylesheetApplicationTransform(Transform):
    """Apply a model stylesheet to the graph."""

    @property
    def name(self) -> str:
        return "stylesheet_application"

    def apply(self, graph: Graph, vars: dict[str, str] | None = None) -> Graph:
        if graph.model_stylesheet:
            stylesheet = parse_stylesheet(graph.model_stylesheet)
            apply_stylesheet(graph, stylesheet)
        return graph


class PreambleTransform(Transform):
    """Prepend a preamble to all codergen node prompts."""

    def __init__(self, preamble: str = ""):
        self._preamble = preamble

    @property
    def name(self) -> str:
        return "preamble"

    def apply(self, graph: Graph, vars: dict[str, str] | None = None) -> Graph:
        if not self._preamble:
            return graph
        for node in graph.nodes.values():
            if node.handler_type == "codergen" and node.prompt:
                node.prompt = self._preamble + "\n\n" + node.prompt
        return graph


class TransformRegistry:
    """Ordered collection of transforms to apply."""

    def __init__(self) -> None:
        self._transforms: list[Transform] = []

    def add(self, transform: Transform) -> None:
        self._transforms.append(transform)

    def apply_all(self, graph: Graph, vars: dict[str, str] | None = None) -> Graph:
        for transform in self._transforms:
            graph = transform.apply(graph, vars)
        return graph


def create_default_transforms() -> TransformRegistry:
    """Create the default transform pipeline."""
    registry = TransformRegistry()
    registry.add(VariableExpansionTransform())
    registry.add(StylesheetApplicationTransform())
    return registry
