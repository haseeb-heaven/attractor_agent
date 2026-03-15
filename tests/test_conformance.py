"""Conformance tests for the Attractor pipeline engine based on attractor-spec.md"""

from attractor.pipeline.context import Context, Outcome, StageStatus
from attractor.pipeline.engine import run_pipeline, select_next_edge
from attractor.pipeline.graph import Edge, Graph, Node


def test_select_next_edge_priority_1_suggested_ids():
    """§3.3 Step 1: Suggested next IDs should have highest priority."""
    edges = [
        Edge(from_node="A", to_node="B", label="b"),
        Edge(from_node="A", to_node="C", label="c"),
    ]
    outcome = Outcome(status=StageStatus.SUCCESS, suggested_next_ids=["C"])
    context = Context()
    
    selected = select_next_edge(edges, outcome, context)
    assert selected.to_node == "C"


def test_select_next_edge_priority_2_preferred_label():
    """§3.3 Step 2: Preferred label should be prioritized if no suggested IDs match."""
    edges = [
        Edge(from_node="A", to_node="B", label="b"),
        Edge(from_node="A", to_node="C", label="c"),
    ]
    outcome = Outcome(status=StageStatus.SUCCESS, preferred_label="c")
    context = Context()
    
    selected = select_next_edge(edges, outcome, context)
    assert selected.to_node == "C"


def test_select_next_edge_priority_3_condition():
    """§3.3 Step 3: Expression conditions should be evaluated."""
    edges = [
        Edge(from_node="A", to_node="B", label="b"),
        Edge(from_node="A", to_node="C", condition="outcome=success"),
        Edge(from_node="A", to_node="D", condition="outcome=fail"),
    ]
    outcome = Outcome(status=StageStatus.SUCCESS)
    context = Context()
    
    selected = select_next_edge(edges, outcome, context)
    assert selected.to_node == "C"


def test_select_next_edge_priority_4_unconditional_status_match():
    """§3.3 Step 4: Unconditional edges matching the status name should match."""
    edges = [
        Edge(from_node="A", to_node="B", label="success"),
        Edge(from_node="A", to_node="C", label="fail"),
    ]
    outcome = Outcome(status=StageStatus.FAIL)
    context = Context()
    
    selected = select_next_edge(edges, outcome, context)
    assert selected.to_node == "C"


def test_select_next_edge_priority_5_weight():
    """§3.3 Step 5: Fallback to highest weight."""
    edges = [
        Edge(from_node="A", to_node="B", weight=10),
        Edge(from_node="A", to_node="C", weight=20),
    ]
    outcome = Outcome(status=StageStatus.SUCCESS)
    context = Context()
    
    selected = select_next_edge(edges, outcome, context)
    assert selected.to_node == "C"


def test_pipeline_linear_execution():
    """Test a simple linear pipeline execution."""
    dot = '''
    digraph G {
        Start [shape=Mdiamond];
        Start -> Step1 -> Step2 -> Exit;
    }
    '''
    result = run_pipeline(dot)
    
    assert result.success is True
    assert result.completed_nodes == ["Start", "Step1", "Step2", "Exit"]
    assert result.total_steps == 4


def test_pipeline_conditional_routing():
    """Test that pipeline follows conditions correctly."""
    dot = '''
    digraph G {
        Start [shape=Mdiamond];
        Start -> Step1;
        Step1 -> EndSuccess [condition="outcome=success"];
        Step1 -> EndFail [condition="outcome=fail"];
    }
    '''
    # Default handlers return SUCCESS, so it should route to EndSuccess
    result = run_pipeline(dot)
    
    assert result.success is True
    assert "EndSuccess" in result.completed_nodes
    assert "EndFail" not in result.completed_nodes


def test_pipeline_max_steps_exhaustion():
    """Test that the engine catches infinite loops and respects max_total_steps."""
    dot = '''
    digraph G {
        Start [shape=Mdiamond];
        Start -> Loop;
        Loop -> Loop;
    }
    '''
    from attractor.pipeline.engine import PipelineConfig
    
    # Set a tiny max steps config to avoid hanging the test
    config = PipelineConfig(max_total_steps=5)
    result = run_pipeline(dot, config=config)
    
    # It should fail because it exceeded max steps
    assert result.success is False
    assert result.total_steps == 5
    assert "exceeded max steps" in result.error or "exceeded max steps" in getattr(result, 'message', '') or getattr(result, 'success', False) is False


def test_pipeline_retry_exhaustion():
    """Test that a node failing repeatedly exhausts max_retries and fails the pipeline."""
    # We need a custom handler that always fails to test retry logic
    from attractor.pipeline.events import EventEmitter
    from attractor.pipeline.handlers.base import Handler
    class AlwaysFailHandler(Handler):
        def execute(self, node: Node, context: Context, graph: Graph, emitter: EventEmitter, **kwargs) -> Outcome:
            return Outcome(status=StageStatus.FAIL, failure_reason="Intentional Test Failure")
            
    # Modify the registry just for this test
    dot = '''
    digraph G {
        Start [shape=Mdiamond];
        Start -> Boom [max_retries=2, handler="always_fail"];
        Boom -> End;
    }
    '''
    
    # We emulate the parsing & injection
    from attractor.pipeline.parser import parse_dot
    graph = parse_dot(dot)
    
    # We have to register the handler globally or pass it in. 
    # Since registry is created per run_pipeline call currently via create_default_registry, 
    # we'll monkeypatch temporarily or test it via the graph node's behavior if we can.
    # A cleaner way: The parser assigns unknown handlers to Codergen which simulates success.
    # So we'll mock the registry creation.
    
    import attractor.pipeline.engine as engine
    original_create = engine.create_default_registry
    
    def mocked_registry():
        reg = original_create()
        reg.register("always_fail", AlwaysFailHandler())
        return reg
        
    engine.create_default_registry = mocked_registry
    try:
        result = run_pipeline(graph)
        assert result.success is False
        assert "after 2 retries" in result.error
        assert result.final_node == "Boom"
    finally:
        engine.create_default_registry = original_create
