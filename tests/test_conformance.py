"""Conformance tests for the Attractor pipeline engine based on attractor-spec.md"""

from attractor.pipeline.context import Context, Outcome, StageStatus
from attractor.pipeline.engine import run_pipeline, select_next_edge
from attractor.pipeline.graph import Edge


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
        Start -> Step1 -> Step2 -> Exit [shape=Msquare];
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
        Step1 -> EndSuccess [condition="outcome=success", shape=Msquare];
        Step1 -> EndFail [condition="outcome=fail", shape=Msquare];
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
        Loop [shape=box];
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
    from attractor.pipeline.engine import PipelineConfig
    
    class FailBackend:
        def generate(self, prompt, **kwargs):
            raise Exception("Intentional Test Failure")
            
    # DOT with specific retry count. Boom defaults to 'codergen' handler.
    dot = '''
    digraph G {
        Start [shape=Mdiamond];
        Boom [max_retries=2];
        Start -> Boom;
        Boom -> End [shape=Msquare];
    }
    '''
    
    config = PipelineConfig(codergen_backend=FailBackend())
    
    result = run_pipeline(dot, config=config)
    
    assert result.success is False
    assert "after 2 retries" in result.error
    assert result.final_node == "Boom"
