"""Tests for the pipeline execution engine."""

from attractor.pipeline.context import Context, Outcome, StageStatus, Checkpoint
from attractor.pipeline.engine import PipelineConfig, run_pipeline
from attractor.pipeline.interviewer import AutoApproveInterviewer, QueueInterviewer, Answer


class TestEngine:
    """Tests for pipeline execution."""

    def test_simple_linear_pipeline(self):
        source = '''
        digraph test {
            Start [shape=Mdiamond];
            Step1 [shape=box, prompt="Do step 1"];
            Step2 [shape=box, prompt="Do step 2"];
            End [shape=Msquare];
            Start -> Step1 -> Step2 -> End;
        }
        '''
        result = run_pipeline(source, PipelineConfig(simulate=True))
        assert result.success
        assert "Start" in result.completed_nodes
        assert "Step1" in result.completed_nodes
        assert "Step2" in result.completed_nodes
        assert "End" in result.completed_nodes

    def test_branching_pipeline(self):
        source = '''
        digraph test {
            Start [shape=Mdiamond];
            Code [shape=box, prompt="Generate code"];
            Review [shape=hexagon, prompt="Review"];
            End [shape=Msquare];
            Start -> Code;
            Code -> Review [label="success"];
            Review -> End [label="approve"];
            Review -> Code [label="reject"];
        }
        '''
        # Auto-approve will select first option "approve"
        result = run_pipeline(source, PipelineConfig(
            simulate=True,
            interviewer=AutoApproveInterviewer(),
        ))
        assert result.success

    def test_pipeline_with_variables(self):
        source = '''
        digraph test {
            Start [shape=Mdiamond];
            Code [shape=box, prompt="Generate ${language} code"];
            End [shape=Msquare];
            Start -> Code -> End;
        }
        '''
        result = run_pipeline(source, PipelineConfig(
            simulate=True,
            variables={"language": "Python"},
        ))
        assert result.success

    def test_pipeline_result_context(self):
        source = '''
        digraph test {
            Start [shape=Mdiamond];
            Step [shape=box, prompt="Do something"];
            End [shape=Msquare];
            Start -> Step -> End;
        }
        '''
        result = run_pipeline(source, PipelineConfig(simulate=True))
        assert result.success
        # Simulated output should be in context
        output = result.context.get("Step.output")
        assert output is not None
        assert "[SIMULATED]" in output

    def test_max_steps_safety(self):
        source = '''
        digraph test {
            Start [shape=Mdiamond];
            Loop [shape=box, prompt="Loop forever"];
            Start -> Loop;
            Loop -> Loop;
        }
        '''
        result = run_pipeline(source, PipelineConfig(
            simulate=True,
            max_total_steps=5,
        ))
        assert not result.success

    def test_pipeline_no_start_node(self):
        source = '''
        digraph test {
            A [shape=box];
        }
        '''
        result = run_pipeline(source)
        assert not result.success
        assert "start" in result.error.lower() or "Validation" in result.error


class TestContext:
    """Tests for Context operations."""

    def test_set_get(self):
        ctx = Context()
        ctx.set("key", "value")
        assert ctx.get("key") == "value"

    def test_get_default(self):
        ctx = Context()
        assert ctx.get("missing") is None
        assert ctx.get("missing", "default") == "default"

    def test_snapshot(self):
        ctx = Context()
        ctx.set("a", 1)
        ctx.set("b", 2)
        snap = ctx.snapshot()
        assert snap == {"a": 1, "b": 2}

    def test_clone(self):
        ctx = Context()
        ctx.set("key", "original")
        cloned = ctx.clone()
        cloned.set("key", "modified")
        assert ctx.get("key") == "original"
        assert cloned.get("key") == "modified"

    def test_apply_updates(self):
        ctx = Context()
        ctx.set("a", 1)
        ctx.apply_updates({"b": 2, "c": 3})
        assert ctx.get("b") == 2
        assert ctx.get("c") == 3

    def test_logs(self):
        ctx = Context()
        ctx.append_log("Entry 1")
        ctx.append_log("Entry 2")
        assert len(ctx.logs) == 2


class TestCheckpoint:
    """Tests for checkpoint save/load."""

    def test_round_trip(self, tmp_path):
        ctx = Context()
        ctx.set("data", "test_value")
        checkpoint = Checkpoint.from_context(
            ctx, "node_A", ["Start", "node_A"], {"node_A": 1},
        )
        path = tmp_path / "checkpoint.json"
        checkpoint.save(path)

        loaded = Checkpoint.load(path)
        assert loaded.current_node == "node_A"
        assert loaded.completed_nodes == ["Start", "node_A"]
        assert loaded.node_retries == {"node_A": 1}
        assert loaded.context_values["data"] == "test_value"
