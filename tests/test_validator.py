"""Tests for the validator."""

import pytest
from attractor.pipeline.parser import parse_dot
from attractor.pipeline.validator import Severity, validate, validate_or_raise, ValidationError


class TestValidator:

    def test_valid_pipeline(self):
        source = '''
        digraph test {
            Start [shape=Mdiamond];
            Code [shape=box, prompt="Generate code"];
            End [shape=Msquare];
            Start -> Code -> End;
        }
        '''
        graph = parse_dot(source)
        diagnostics = validate(graph)
        errors = [d for d in diagnostics if d.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_no_start_node(self):
        source = '''
        digraph test {
            A [shape=box];
            End [shape=Msquare];
            A -> End;
        }
        '''
        graph = parse_dot(source)
        diagnostics = validate(graph)
        errors = [d for d in diagnostics if d.rule == "start_node"]
        assert len(errors) == 1

    def test_unreachable_node(self):
        source = '''
        digraph test {
            Start [shape=Mdiamond];
            A [shape=box, prompt="x"];
            Orphan [shape=box, prompt="y"];
            End [shape=Msquare];
            Start -> A -> End;
        }
        '''
        graph = parse_dot(source)
        diagnostics = validate(graph)
        reachability_warnings = [d for d in diagnostics if d.rule == "reachability"]
        assert len(reachability_warnings) == 1
        assert "Orphan" in reachability_warnings[0].message

    def test_missing_edge_target(self):
        source = '''
        digraph test {
            Start [shape=Mdiamond];
            End [shape=Msquare];
            Start -> End;
        }
        '''
        graph = parse_dot(source)
        # Manually add a bad edge
        from attractor.pipeline.graph import Edge
        graph.edges.append(Edge(from_node="Start", to_node="NonExistent"))
        diagnostics = validate(graph)
        edge_errors = [d for d in diagnostics if d.rule == "edge_target_exists"]
        assert len(edge_errors) == 1

    def test_missing_prompt_warning(self):
        source = '''
        digraph test {
            Start [shape=Mdiamond];
            Code [shape=box, label="Generate"];
            End [shape=Msquare];
            Start -> Code -> End;
        }
        '''
        graph = parse_dot(source)
        diagnostics = validate(graph)
        prompt_warnings = [d for d in diagnostics if d.rule == "prompt_set"]
        assert len(prompt_warnings) == 1

    def test_validate_or_raise(self):
        source = '''
        digraph test {
            A [shape=box];
        }
        '''
        graph = parse_dot(source)
        with pytest.raises(ValidationError):
            validate_or_raise(graph)

    def test_start_no_outgoing(self):
        source = '''
        digraph test {
            Start [shape=Mdiamond];
        }
        '''
        graph = parse_dot(source)
        diagnostics = validate(graph)
        errors = [d for d in diagnostics if d.rule == "start_has_outgoing"]
        assert len(errors) == 1

    def test_no_self_loops(self):
        source = '''
        digraph test {
            Start [shape=Mdiamond];
            A [shape=box, prompt="x"];
            End [shape=Msquare];
            Start -> A;
            A -> A;
            A -> End;
        }
        '''
        graph = parse_dot(source)
        diagnostics = validate(graph)
        loop_warnings = [d for d in diagnostics if d.rule == "no_self_loops"]
        assert len(loop_warnings) == 1

    def test_self_loop_with_loop_restart_ok(self):
        source = '''
        digraph test {
            Start [shape=Mdiamond];
            A [shape=box, prompt="x"];
            End [shape=Msquare];
            Start -> A;
            A -> A [loop_restart=true];
            A -> End;
        }
        '''
        graph = parse_dot(source)
        diagnostics = validate(graph)
        loop_warnings = [d for d in diagnostics if d.rule == "no_self_loops"]
        assert len(loop_warnings) == 0
