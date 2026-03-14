"""Tests for the DOT parser."""

import pytest
from attractor.pipeline.parser import parse_dot
from attractor.pipeline.graph import Graph


class TestDotParser:
    """Tests for DOT parsing functionality."""

    def test_parse_simple_linear(self):
        source = '''
        digraph test {
            Start [shape=Mdiamond];
            A [shape=box, label="Step A"];
            End [shape=Msquare];
            Start -> A -> End;
        }
        '''
        graph = parse_dot(source)
        assert graph.name == "test"
        assert len(graph.nodes) == 3
        assert "Start" in graph.nodes
        assert "A" in graph.nodes
        assert "End" in graph.nodes
        assert len(graph.edges) == 2

    def test_parse_node_attributes(self):
        source = '''
        digraph test {
            Start [shape=Mdiamond];
            Code [shape=box, label="Generate", prompt="Write code",
                  max_retries="3", goal_gate=true];
            End [shape=Msquare];
            Start -> Code -> End;
        }
        '''
        graph = parse_dot(source)
        node = graph.nodes["Code"]
        assert node.label == "Generate"
        assert node.prompt == "Write code"
        assert node.max_retries == 3
        assert node.goal_gate is True
        assert node.handler_type == "codergen"

    def test_parse_edge_attributes(self):
        source = '''
        digraph test {
            Start [shape=Mdiamond];
            A [shape=box];
            B [shape=box];
            End [shape=Msquare];
            Start -> A;
            A -> B [label="success", condition="outcome=success"];
            A -> End [label="fail"];
            B -> End;
        }
        '''
        graph = parse_dot(source)
        assert len(graph.edges) == 4
        ab_edge = [e for e in graph.edges if e.from_node == "A" and e.to_node == "B"][0]
        assert ab_edge.label == "success"
        assert ab_edge.condition == "outcome=success"

    def test_parse_chained_edges(self):
        source = '''
        digraph chain {
            Start [shape=Mdiamond];
            A [shape=box];
            B [shape=box];
            C [shape=box];
            End [shape=Msquare];
            Start -> A -> B -> C -> End;
        }
        '''
        graph = parse_dot(source)
        assert len(graph.edges) == 4
        assert graph.edges[0].from_node == "Start"
        assert graph.edges[0].to_node == "A"
        assert graph.edges[3].from_node == "C"
        assert graph.edges[3].to_node == "End"

    def test_parse_graph_attributes(self):
        source = '''
        digraph my_pipeline {
            goal="Build a web app";
            rankdir=LR;
            Start [shape=Mdiamond];
            End [shape=Msquare];
            Start -> End;
        }
        '''
        graph = parse_dot(source)
        assert graph.name == "my_pipeline"
        assert graph.goal == "Build a web app"

    def test_parse_comments(self):
        source = '''
        // This is a comment
        digraph test {
            /* Block comment */
            Start [shape=Mdiamond]; // inline comment
            End [shape=Msquare];
            Start -> End;
        }
        '''
        graph = parse_dot(source)
        assert len(graph.nodes) == 2

    def test_parse_shape_handler_mapping(self):
        source = '''
        digraph test {
            S [shape=Mdiamond];
            A [shape=box];
            B [shape=hexagon];
            C [shape=diamond];
            D [shape=component];
            E [shape=Msquare];
            S -> A -> B -> C -> D -> E;
        }
        '''
        graph = parse_dot(source)
        assert graph.nodes["S"].handler_type == "start"
        assert graph.nodes["A"].handler_type == "codergen"
        assert graph.nodes["B"].handler_type == "wait.human"
        assert graph.nodes["C"].handler_type == "conditional"
        assert graph.nodes["D"].handler_type == "parallel"
        assert graph.nodes["E"].handler_type == "exit"

    def test_parse_node_defaults(self):
        source = '''
        digraph test {
            node [shape=box];
            Start [shape=Mdiamond];
            A;
            B;
            End [shape=Msquare];
            Start -> A -> B -> End;
        }
        '''
        graph = parse_dot(source)
        assert graph.nodes["A"].shape == "box"
        assert graph.nodes["B"].shape == "box"
        assert graph.nodes["Start"].shape == "Mdiamond"

    def test_parse_quoted_strings(self):
        source = '''
        digraph test {
            Start [shape=Mdiamond];
            "Node With Spaces" [shape=box, label="A Label"];
            End [shape=Msquare];
            Start -> "Node With Spaces" -> End;
        }
        '''
        graph = parse_dot(source)
        assert "Node With Spaces" in graph.nodes

    def test_parse_subgraph(self):
        source = '''
        digraph test {
            Start [shape=Mdiamond];
            End [shape=Msquare];
            subgraph cluster_sub {
                A [shape=box];
                B [shape=box];
                A -> B;
            }
            Start -> A;
            B -> End;
        }
        '''
        graph = parse_dot(source)
        assert "A" in graph.nodes
        assert "B" in graph.nodes
        assert len([e for e in graph.edges if e.from_node == "A" and e.to_node == "B"]) == 1

    def test_empty_pipeline_fails(self):
        with pytest.raises(SyntaxError):
            parse_dot("")

    def test_parse_explicit_type(self):
        source = '''
        digraph test {
            Start [shape=Mdiamond];
            Custom [type="wait.human", label="Review"];
            End [shape=Msquare];
            Start -> Custom -> End;
        }
        '''
        graph = parse_dot(source)
        assert graph.nodes["Custom"].handler_type == "wait.human"


class TestGraph:
    """Tests for Graph operations."""

    def test_find_start_node(self):
        source = '''
        digraph test {
            Begin [shape=Mdiamond];
            End [shape=Msquare];
            Begin -> End;
        }
        '''
        graph = parse_dot(source)
        start = graph.find_start_node()
        assert start is not None
        assert start.id == "Begin"

    def test_find_exit_node(self):
        source = '''
        digraph test {
            Start [shape=Mdiamond];
            Done [shape=Msquare];
            Start -> Done;
        }
        '''
        graph = parse_dot(source)
        exit_node = graph.find_exit_node()
        assert exit_node is not None
        assert exit_node.id == "Done"

    def test_reachability(self):
        source = '''
        digraph test {
            Start [shape=Mdiamond];
            A [shape=box];
            B [shape=box];
            Orphan [shape=box];
            End [shape=Msquare];
            Start -> A -> B -> End;
        }
        '''
        graph = parse_dot(source)
        reachable = graph.is_reachable_from_start()
        assert "Start" in reachable
        assert "A" in reachable
        assert "B" in reachable
        assert "End" in reachable
        assert "Orphan" not in reachable

    def test_outgoing_edges(self):
        source = '''
        digraph test {
            Start [shape=Mdiamond];
            A [shape=box];
            B [shape=box];
            End [shape=Msquare];
            Start -> A;
            Start -> B;
            A -> End;
            B -> End;
        }
        '''
        graph = parse_dot(source)
        out = graph.outgoing_edges("Start")
        assert len(out) == 2
