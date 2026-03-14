"""Tests for condition expression evaluator."""

from attractor.pipeline.conditions import evaluate_condition, evaluate_clause, resolve_key
from attractor.pipeline.context import Context, Outcome, StageStatus


class TestConditions:

    def test_empty_condition(self):
        outcome = Outcome(status=StageStatus.SUCCESS)
        ctx = Context()
        assert evaluate_condition("", outcome, ctx) is True

    def test_outcome_equals(self):
        outcome = Outcome(status=StageStatus.SUCCESS)
        ctx = Context()
        assert evaluate_condition("outcome=success", outcome, ctx) is True
        assert evaluate_condition("outcome=fail", outcome, ctx) is False

    def test_outcome_not_equals(self):
        outcome = Outcome(status=StageStatus.FAIL)
        ctx = Context()
        assert evaluate_condition("outcome!=success", outcome, ctx) is True
        assert evaluate_condition("outcome!=fail", outcome, ctx) is False

    def test_preferred_label(self):
        outcome = Outcome(status=StageStatus.SUCCESS, preferred_label="approve")
        ctx = Context()
        assert evaluate_condition("preferred_label=approve", outcome, ctx) is True
        assert evaluate_condition("preferred_label=reject", outcome, ctx) is False

    def test_context_variable(self):
        outcome = Outcome(status=StageStatus.SUCCESS)
        ctx = Context()
        ctx.set("result", "pass")
        assert evaluate_condition("context.result=pass", outcome, ctx) is True
        assert evaluate_condition("context.result=fail", outcome, ctx) is False

    def test_conjunction(self):
        outcome = Outcome(status=StageStatus.SUCCESS)
        ctx = Context()
        ctx.set("quality", "high")
        assert evaluate_condition(
            "outcome=success && context.quality=high", outcome, ctx
        ) is True
        assert evaluate_condition(
            "outcome=success && context.quality=low", outcome, ctx
        ) is False

    def test_whitespace_tolerance(self):
        outcome = Outcome(status=StageStatus.SUCCESS)
        ctx = Context()
        assert evaluate_condition("  outcome = success  ", outcome, ctx) is True

    def test_context_missing_key(self):
        outcome = Outcome(status=StageStatus.SUCCESS)
        ctx = Context()
        assert evaluate_condition("context.missing=anything", outcome, ctx) is False
        assert evaluate_condition("context.missing!=anything", outcome, ctx) is True

    def test_resolve_key_context_prefix(self):
        outcome = Outcome(status=StageStatus.SUCCESS)
        ctx = Context()
        ctx.set("mykey", "myval")
        assert resolve_key("context.mykey", outcome, ctx) == "myval"

    def test_resolve_key_direct(self):
        outcome = Outcome(status=StageStatus.SUCCESS)
        ctx = Context()
        ctx.set("mykey", "myval")
        assert resolve_key("mykey", outcome, ctx) == "myval"


class TestEdgeSelection:
    """Tests for the edge selection algorithm."""

    def test_suggested_next_ids_priority(self):
        from attractor.pipeline.engine import select_next_edge
        from attractor.pipeline.graph import Edge

        edges = [
            Edge(from_node="A", to_node="B", label="path1"),
            Edge(from_node="A", to_node="C", label="path2"),
        ]
        outcome = Outcome(
            status=StageStatus.SUCCESS,
            suggested_next_ids=["C"],
        )
        ctx = Context()

        selected = select_next_edge(edges, outcome, ctx)
        assert selected is not None
        assert selected.to_node == "C"

    def test_preferred_label_priority(self):
        from attractor.pipeline.engine import select_next_edge
        from attractor.pipeline.graph import Edge

        edges = [
            Edge(from_node="A", to_node="B", label="approve"),
            Edge(from_node="A", to_node="C", label="reject"),
        ]
        outcome = Outcome(
            status=StageStatus.SUCCESS,
            preferred_label="reject",
        )
        ctx = Context()

        selected = select_next_edge(edges, outcome, ctx)
        assert selected is not None
        assert selected.to_node == "C"

    def test_condition_evaluation(self):
        from attractor.pipeline.engine import select_next_edge
        from attractor.pipeline.graph import Edge

        edges = [
            Edge(from_node="A", to_node="B", condition="outcome=success"),
            Edge(from_node="A", to_node="C", condition="outcome=fail"),
        ]
        outcome = Outcome(status=StageStatus.FAIL)
        ctx = Context()

        selected = select_next_edge(edges, outcome, ctx)
        assert selected is not None
        assert selected.to_node == "C"

    def test_unconditional_fallback(self):
        from attractor.pipeline.engine import select_next_edge
        from attractor.pipeline.graph import Edge

        edges = [
            Edge(from_node="A", to_node="B"),
        ]
        outcome = Outcome(status=StageStatus.SUCCESS)
        ctx = Context()

        selected = select_next_edge(edges, outcome, ctx)
        assert selected is not None
        assert selected.to_node == "B"

    def test_no_edges(self):
        from attractor.pipeline.engine import select_next_edge
        outcome = Outcome(status=StageStatus.SUCCESS)
        ctx = Context()
        assert select_next_edge([], outcome, ctx) is None
