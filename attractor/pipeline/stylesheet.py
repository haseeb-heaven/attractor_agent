"""Model stylesheet parser — CSS-like syntax for configuring LLM models.

Selectors: * (all), .class, #id
Properties: llm_model, llm_provider, reasoning_effort, etc.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from attractor.pipeline.graph import Graph, Node


@dataclass
class StyleRule:
    """A single stylesheet rule with selector and properties."""
    selector: str = ""
    properties: dict[str, str] = field(default_factory=dict)
    specificity: int = 0  # Higher = more specific

    @staticmethod
    def compute_specificity(selector: str) -> int:
        """Compute specificity: * = 0, .class = 1, #id = 2."""
        selector = selector.strip()
        if selector == "*":
            return 0
        if selector.startswith("#"):
            return 2
        if selector.startswith("."):
            return 1
        return 0


@dataclass
class Stylesheet:
    """A collection of style rules."""
    rules: list[StyleRule] = field(default_factory=list)


def parse_stylesheet(source: str) -> Stylesheet:
    """Parse a CSS-like model stylesheet.

    Grammar:
        selector { property: value; ... }

    Example:
        * { llm_model: "gpt-4o"; }
        .coding { reasoning_effort: "high"; }
        #review { llm_model: "claude-opus-4-6"; }
    """
    rules: list[StyleRule] = []

    # Match selector { ... } blocks
    pattern = re.compile(r"([^{]+)\{([^}]*)\}", re.DOTALL)
    for match in pattern.finditer(source):
        selector = match.group(1).strip()
        body = match.group(2).strip()

        properties: dict[str, str] = {}
        for prop_match in re.finditer(r"([\w_-]+)\s*:\s*([^;]+)\s*;?", body):
            key = prop_match.group(1).strip()
            value = prop_match.group(2).strip()
            # Remove quotes
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            properties[key] = value

        rules.append(StyleRule(
            selector=selector,
            properties=properties,
            specificity=StyleRule.compute_specificity(selector),
        ))

    # Sort by specificity (lower first, so higher specificity overrides)
    rules.sort(key=lambda r: r.specificity)

    return Stylesheet(rules=rules)


def matches_selector(node: Node, selector: str) -> bool:
    """Check if a node matches a CSS-like selector."""
    selector = selector.strip()

    if selector == "*":
        return True

    if selector.startswith("#"):
        target_id = selector[1:]
        return node.id == target_id

    if selector.startswith("."):
        target_class = selector[1:]
        if not node.node_class:
            return False
        classes = [c.strip() for c in node.node_class.split()]
        return target_class in classes

    # Bare name — match node ID
    return node.id == selector


def apply_stylesheet(graph: Graph, stylesheet: Stylesheet) -> None:
    """Apply a stylesheet to all nodes in a graph.

    Rules are applied in specificity order (lower first), so more
    specific rules override less specific ones.
    """
    for node in graph.nodes.values():
        for rule in stylesheet.rules:
            if matches_selector(node, rule.selector):
                for key, value in rule.properties.items():
                    # Map known properties to node fields
                    if key == "llm_model":
                        node.llm_model = value
                    elif key == "llm_provider":
                        node.llm_provider = value
                    elif key == "reasoning_effort":
                        node.reasoning_effort = value
                    elif key == "fidelity":
                        node.fidelity = value
                    elif key == "timeout":
                        node.timeout = value
                    # Store all properties in attrs for extensibility
                    node.attrs[key] = value
