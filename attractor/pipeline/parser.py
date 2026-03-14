"""DOT parser — parses the Attractor subset of Graphviz DOT syntax.

Implements the BNF grammar from spec §2.2: digraph, node/edge statements,
attribute blocks, chained edges, subgraphs, comments, defaults blocks.
"""

from __future__ import annotations

import re
from typing import Any

from attractor.pipeline.graph import Edge, Graph, Node

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# Token patterns
_PATTERNS = [
    ("COMMENT_LINE", r"//[^\n]*"),
    ("COMMENT_BLOCK", r"/\*[\s\S]*?\*/"),
    ("DIGRAPH", r"\bdigraph\b"),
    ("SUBGRAPH", r"\bsubgraph\b"),
    ("GRAPH_KW", r"\bgraph\b"),
    ("NODE_KW", r"\bnode\b"),
    ("EDGE_KW", r"\bedge\b"),
    ("TRUE", r"\btrue\b"),
    ("FALSE", r"\bfalse\b"),
    ("ARROW", r"->"),
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("LBRACKET", r"\["),
    ("RBRACKET", r"\]"),
    ("EQUALS", r"="),
    ("COMMA", r","),
    ("SEMI", r";"),
    ("STRING", r'"(?:[^"\\]|\\.)*"'),
    ("NUMBER", r"-?[0-9]+(?:\.[0-9]+)?"),
    ("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
    ("WS", r"\s+"),
]

_TOKEN_RE = re.compile("|".join(f"(?P<{name}>{pat})" for name, pat in _PATTERNS))


class _Token:
    __slots__ = ("type", "value", "pos")

    def __init__(self, type: str, value: str, pos: int):
        self.type = type
        self.value = value
        self.pos = pos


def _tokenize(source: str) -> list[_Token]:
    tokens: list[_Token] = []
    for m in _TOKEN_RE.finditer(source):
        kind = m.lastgroup
        if kind in ("WS", "COMMENT_LINE", "COMMENT_BLOCK"):
            continue
        assert kind is not None
        tokens.append(_Token(kind, m.group(), m.start()))
    return tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class _Parser:
    """Recursive descent parser for the Attractor DOT subset."""

    def __init__(self, tokens: list[_Token]):
        self.tokens = tokens
        self.pos = 0
        self.graph = Graph()
        self.node_defaults: dict[str, str] = {}
        self.edge_defaults: dict[str, str] = {}

    def peek(self) -> _Token | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def advance(self) -> _Token:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def expect(self, type: str) -> _Token:
        t = self.peek()
        if t is None or t.type != type:
            expected = type
            got = f"{t.type}({t.value!r})" if t else "EOF"
            raise SyntaxError(f"Expected {expected}, got {got}")
        return self.advance()

    def match(self, type: str) -> _Token | None:
        t = self.peek()
        if t and t.type == type:
            return self.advance()
        return None

    def parse(self) -> Graph:
        self.expect("DIGRAPH")
        name_tok = self.peek()
        if name_tok and name_tok.type == "IDENT":
            self.graph.name = self.advance().value
        elif name_tok and name_tok.type == "STRING":
            self.graph.name = self._unquote(self.advance().value)
        self.expect("LBRACE")
        self._parse_statements()
        self.expect("RBRACE")
        return self.graph

    def _parse_statements(self) -> None:
        while True:
            t = self.peek()
            if t is None or t.type == "RBRACE":
                break
            self._parse_statement()

    def _parse_statement(self) -> None:
        t = self.peek()
        if t is None:
            return

        # Skip semicolons
        if t.type == "SEMI":
            self.advance()
            return

        # graph [attrs] or graph attr = val
        if t.type == "GRAPH_KW":
            self.advance()
            if self.peek() and self.peek().type == "LBRACKET":  # type: ignore[union-attr]
                attrs = self._parse_attr_block()
                self._apply_graph_attrs(attrs)
            else:
                # graph attr = val
                pass
            self.match("SEMI")
            return

        # node [defaults]
        if t.type == "NODE_KW":
            self.advance()
            if self.peek() and self.peek().type == "LBRACKET":  # type: ignore[union-attr]
                self.node_defaults.update(self._parse_attr_block())
            self.match("SEMI")
            return

        # edge [defaults]
        if t.type == "EDGE_KW":
            self.advance()
            if self.peek() and self.peek().type == "LBRACKET":  # type: ignore[union-attr]
                self.edge_defaults.update(self._parse_attr_block())
            self.match("SEMI")
            return

        # subgraph
        if t.type == "SUBGRAPH":
            self._parse_subgraph()
            self.match("SEMI")
            return

        # ID = value (graph attribute declaration)
        if t.type == "IDENT":
            # Look ahead for = or -> or [
            next_t = self.tokens[self.pos + 1] if self.pos + 1 < len(self.tokens) else None
            if next_t and next_t.type == "EQUALS":
                # Check if the token after the value is not part of an attr block
                # This is a graph-level key=value declaration
                key = self.advance().value
                self.expect("EQUALS")
                val = self._parse_value()
                self._apply_graph_attrs({key: val})
                self.match("SEMI")
                return

            # Could be node or edge statement
            self._parse_node_or_edge()
            return

        if t.type == "STRING":
            self._parse_node_or_edge()
            return

        # Skip unknown tokens
        self.advance()

    def _parse_subgraph(self) -> None:
        """Parse a subgraph block — flatten nodes/edges into the main graph."""
        self.expect("SUBGRAPH")
        # Optional name
        subgraph_label = ""
        if self.peek() and self.peek().type in ("IDENT", "STRING"):  # type: ignore[union-attr]
            subgraph_label = self.advance().value

        self.expect("LBRACE")

        # Save defaults
        saved_node_defaults = dict(self.node_defaults)
        saved_edge_defaults = dict(self.edge_defaults)

        # Parse subgraph body
        self._parse_statements()

        # Restore defaults
        self.node_defaults = saved_node_defaults
        self.edge_defaults = saved_edge_defaults

        self.expect("RBRACE")

    def _parse_node_or_edge(self) -> None:
        """Parse a node statement or edge chain."""
        # Collect first identifier
        first_id = self._parse_id()

        # Check for edge chain
        node_chain = [first_id]
        while self.peek() and self.peek().type == "ARROW":  # type: ignore[union-attr]
            self.advance()  # consume ->
            node_chain.append(self._parse_id())

        # Parse optional attribute block
        attrs: dict[str, str] = {}
        if self.peek() and self.peek().type == "LBRACKET":  # type: ignore[union-attr]
            attrs = self._parse_attr_block()

        if len(node_chain) == 1:
            # Node statement
            self._add_node(node_chain[0], attrs)
        else:
            # Edge chain: A -> B -> C => edges A->B, B->C
            # Ensure all nodes exist
            for nid in node_chain:
                if nid not in self.graph.nodes:
                    self._add_node(nid, {})
            # Create edges
            for i in range(len(node_chain) - 1):
                edge_attrs = {**self.edge_defaults, **attrs}
                edge = Edge.from_attrs(node_chain[i], node_chain[i + 1], edge_attrs)
                self.graph.edges.append(edge)

        self.match("SEMI")

    def _parse_id(self) -> str:
        t = self.peek()
        if t and t.type == "IDENT":
            return self.advance().value
        if t and t.type == "STRING":
            return self._unquote(self.advance().value)
        raise SyntaxError(f"Expected identifier, got {t.type if t else 'EOF'}")

    def _parse_attr_block(self) -> dict[str, str]:
        """Parse [key=value, key=value, ...]."""
        self.expect("LBRACKET")
        attrs: dict[str, str] = {}
        while True:
            t = self.peek()
            if t is None or t.type == "RBRACKET":
                break
            if t.type in ("COMMA", "SEMI"):
                self.advance()
                continue
            # key = value
            key = self._parse_attr_key()
            self.expect("EQUALS")
            value = self._parse_value()
            attrs[key] = value
        self.expect("RBRACKET")
        return attrs

    def _parse_attr_key(self) -> str:
        """Parse a key (may be qualified: a.b.c)."""
        parts = [self._parse_id()]
        while self.peek() and self.peek().value == ".":  # type: ignore[union-attr]
            # Peek further to see if it's a qualified ID
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == "IDENT":
                self.advance()  # consume .
                parts.append(self._parse_id())
            else:
                break
        return ".".join(parts)

    def _parse_value(self) -> str:
        """Parse a value (string, number, boolean, or identifier)."""
        t = self.peek()
        if t is None:
            raise SyntaxError("Expected value, got EOF")
        if t.type == "STRING":
            return self._unquote(self.advance().value)
        if t.type == "NUMBER":
            return self.advance().value
        if t.type in ("TRUE", "FALSE"):
            return self.advance().value
        if t.type == "IDENT":
            return self.advance().value
        raise SyntaxError(f"Expected value, got {t.type}({t.value!r})")

    def _unquote(self, s: str) -> str:
        """Remove surrounding quotes and unescape."""
        if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
            s = s[1:-1]
        return s.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")

    def _add_node(self, node_id: str, explicit_attrs: dict[str, str]) -> None:
        """Add or update a node in the graph."""
        merged = {**self.node_defaults, **explicit_attrs}
        if node_id in self.graph.nodes:
            # Update existing node attrs
            existing = self.graph.nodes[node_id]
            existing.attrs.update(merged)
            # Re-apply key attrs
            if "label" in merged:
                existing.label = merged["label"]
            if "shape" in merged:
                existing.shape = merged["shape"]
            if "type" in merged:
                existing.type = merged["type"]
            if "prompt" in merged:
                existing.prompt = merged["prompt"]
            if "max_retries" in merged:
                existing.max_retries = int(merged["max_retries"])
            if "goal_gate" in merged:
                existing.goal_gate = merged["goal_gate"].lower() in ("true", "1")
            if "class" in merged:
                existing.node_class = merged["class"]
        else:
            node = Node.from_attrs(node_id, merged)
            self.graph.nodes[node_id] = node

    def _apply_graph_attrs(self, attrs: dict[str, str]) -> None:
        """Apply attributes to the graph."""
        self.graph.attrs.update(attrs)
        if "goal" in attrs:
            self.graph.goal = attrs["goal"]
        if "label" in attrs:
            self.graph.label = attrs["label"]
        if "model_stylesheet" in attrs:
            self.graph.model_stylesheet = attrs["model_stylesheet"]
        if "default_max_retry" in attrs:
            try:
                self.graph.default_max_retry = int(attrs["default_max_retry"])
            except ValueError:
                pass
        if "retry_target" in attrs:
            self.graph.retry_target = attrs["retry_target"]
        if "fallback_retry_target" in attrs:
            self.graph.fallback_retry_target = attrs["fallback_retry_target"]
        if "default_fidelity" in attrs:
            self.graph.default_fidelity = attrs["default_fidelity"]
        # rankdir and other visual attrs are stored but not semantically used
        for key in ("rankdir",):
            if key in attrs:
                self.graph.attrs[key] = attrs[key]


def parse_dot(source: str) -> Graph:
    """Parse a DOT source string into a Graph model.

    Args:
        source: DOT language source text (digraph only).

    Returns:
        Parsed Graph with nodes, edges, and attributes.

    Raises:
        SyntaxError: If the source is not valid DOT.
    """
    tokens = _tokenize(source)
    parser = _Parser(tokens)
    return parser.parse()
