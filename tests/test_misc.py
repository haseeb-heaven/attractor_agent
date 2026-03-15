"""Tests for truncation, loop detection, LLM types, retry, stylesheet, and interviewer."""

from attractor.agent.truncation import truncate_output, truncate_lines, truncate_tool_output
from attractor.agent.loop_detection import detect_loop
from attractor.llm.types import Message, ContentPart, ContentKind, Role, Response, ToolCall
from attractor.llm.retry import BackoffConfig, RetryConfig, is_retryable
from attractor.llm.errors import RateLimitError, AuthenticationError, SDKError
from attractor.pipeline.stylesheet import parse_stylesheet, apply_stylesheet, matches_selector
from attractor.pipeline.graph import Node
from attractor.pipeline.interviewer import (
    AutoApproveInterviewer, QueueInterviewer, RecordingInterviewer,
    Question, Answer, Option,
)


class TestTruncation:

    def test_no_truncation_needed(self):
        assert truncate_output("short", 1000) == "short"

    def test_character_truncation(self):
        text = "x" * 1000
        result = truncate_output(text, 100)
        assert len(result) < 1000
        assert "WARNING" in result
        assert "characters removed" in result

    def test_line_truncation(self):
        text = "\n".join(f"line {i}" for i in range(100))
        result = truncate_lines(text, 10)
        assert "omitted" in result

    def test_no_line_truncation_needed(self):
        text = "line1\nline2\nline3"
        assert truncate_lines(text, 100) == text

    def test_full_pipeline(self):
        from attractor.agent.config import SessionConfig
        text = "x" * 100_000
        result = truncate_tool_output(text, "read_file", SessionConfig())
        assert len(result) < 100_000


class TestLoopDetection:

    def test_no_loop(self):
        calls = ["read_file", "shell", "write_file"]
        assert detect_loop(calls) is False

    def test_single_repeat_loop(self):
        calls = ["read_file"] * 5
        assert detect_loop(calls, threshold=3) is True

    def test_pair_pattern_loop(self):
        calls = ["read_file", "write_file"] * 5
        assert detect_loop(calls, threshold=3) is True

    def test_threshold_not_met(self):
        calls = ["read_file", "read_file"]
        assert detect_loop(calls, threshold=3) is False


class TestLLMTypes:

    def test_message_text(self):
        msg = Message.user("Hello")
        assert msg.text == "Hello"
        assert msg.role == Role.USER

    def test_message_system(self):
        msg = Message.system("Be helpful")
        assert msg.role == Role.SYSTEM
        assert msg.text == "Be helpful"

    def test_message_tool_result(self):
        msg = Message.tool_result("call_1", "result text")
        assert msg.role == Role.TOOL
        assert msg.tool_call_id == "call_1"

    def test_response_tool_calls(self):
        resp = Response(
            message=Message(
                role=Role.ASSISTANT,
                content=[
                    ContentPart.text_part("Let me help"),
                    ContentPart.tool_call_part("call_1", "read_file", '{"path": "foo.py"}'),
                ],
            ),
        )
        assert resp.text == "Let me help"
        calls = resp.tool_calls
        assert len(calls) == 1
        assert calls[0].name == "read_file"
        assert calls[0].arguments == {"path": "foo.py"}


class TestRetry:

    def test_backoff_delay(self):
        config = BackoffConfig(initial_delay_ms=100, backoff_factor=2.0, jitter=False)
        assert config.delay_for_attempt(1) == 0.1
        assert config.delay_for_attempt(2) == 0.2
        assert config.delay_for_attempt(3) == 0.4

    def test_max_delay(self):
        config = BackoffConfig(initial_delay_ms=1000, backoff_factor=10.0,
                               max_delay_ms=5000, jitter=False)
        assert config.delay_for_attempt(10) == 5.0

    def test_retryable_errors(self):
        assert is_retryable(RateLimitError("rate limited")) is True
        assert is_retryable(AuthenticationError("bad key")) is False
        assert is_retryable(ConnectionError("network")) is True


class TestStylesheet:

    def test_parse_simple(self):
        source = '* { llm_model: "gpt-4o"; }'
        sheet = parse_stylesheet(source)
        assert len(sheet.rules) == 1
        assert sheet.rules[0].properties["llm_model"] == "gpt-4o"

    def test_parse_multiple_rules(self):
        source = '''
        * { llm_model: "gpt-4o"; }
        .coding { reasoning_effort: "high"; }
        #review { llm_model: "claude-opus-4-6"; }
        '''
        sheet = parse_stylesheet(source)
        assert len(sheet.rules) == 3

    def test_specificity_ordering(self):
        source = '''
        * { llm_model: "default"; }
        .coding { llm_model: "coding-model"; }
        #specific { llm_model: "specific-model"; }
        '''
        sheet = parse_stylesheet(source)
        # Rules sorted by specificity
        assert sheet.rules[0].specificity == 0  # *
        assert sheet.rules[1].specificity == 1  # .class
        assert sheet.rules[2].specificity == 2  # #id

    def test_apply_stylesheet(self):
        from attractor.pipeline.parser import parse_dot
        source = '''
        digraph test {
            Start [shape=Mdiamond];
            Code [shape=box, class="coding"];
            End [shape=Msquare];
            Start -> Code -> End;
        }
        '''
        graph = parse_dot(source)

        style = parse_stylesheet('* { llm_model: "gpt-4o"; } .coding { reasoning_effort: "high"; }')
        apply_stylesheet(graph, style)

        assert graph.nodes["Code"].llm_model == "gpt-4o"
        assert graph.nodes["Code"].reasoning_effort == "high"

    def test_selector_matching(self):
        node = Node(id="review", node_class="coding helper", shape="box")
        assert matches_selector(node, "*") is True
        assert matches_selector(node, "#review") is True
        assert matches_selector(node, "#other") is False
        assert matches_selector(node, ".coding") is True
        assert matches_selector(node, ".helper") is True
        assert matches_selector(node, ".unknown") is False


class TestInterviewer:

    def test_auto_approve(self):
        iv = AutoApproveInterviewer()
        q = Question(id="q1", text="Approve?", options=[
            Option(label="Approve"), Option(label="Reject"),
        ])
        answer = iv.ask(q)
        assert answer.selected_label == "Approve"

    def test_queue_interviewer(self):
        iv = QueueInterviewer([
            Answer(selected_label="Reject"),
            Answer(selected_label="Approve"),
        ])
        q = Question(id="q1", options=[Option(label="x")])
        assert iv.ask(q).selected_label == "Reject"
        assert iv.ask(q).selected_label == "Approve"

    def test_recording_interviewer(self):
        inner = AutoApproveInterviewer()
        iv = RecordingInterviewer(inner)
        q = Question(id="q1", text="Review?", options=[Option(label="OK")])
        iv.ask(q)
        assert len(iv.log) == 1
        assert iv.log[0][1].selected_label == "OK"
