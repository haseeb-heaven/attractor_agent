import json
from pathlib import Path

from attractor.pipeline.context import Context
from attractor.pipeline.events import EventEmitter
from attractor.pipeline.graph import Graph, Node
from attractor.pipeline.handlers.twin import DigitalTwinHandler

from attractor_agent.project import BuildRequest, build_dot, load_build_request


def test_build_dot_skips_review_for_autonomous_runs():
    spec = BuildRequest(
        request="Build a notes app",
        language="Python",
        require_human_review=False,
        auto_approve=True,
    )

    dot = build_dot(spec)

    assert "Human Review" not in dot
    assert "DeployTwin -> Done;" in dot
    assert "Review ->" not in dot


def test_load_build_request_from_json(tmp_path: Path):
    config_path = tmp_path / "build.json"
    config_path.write_text(
        json.dumps(
            {
                "request": "Build a calculator",
                "language": "TypeScript",
                "framework": "React",
                "include_tests": False,
                "require_human_review": True,
                "auto_approve": False,
                "retry_save_attempts": 5,
            }
        ),
        encoding="utf-8",
    )

    spec = load_build_request(config_path)

    assert spec.request == "Build a calculator"
    assert spec.language == "TypeScript"
    assert spec.framework == "React"
    assert spec.include_tests is False
    assert spec.require_human_review is True
    assert spec.auto_approve is False
    assert spec.retry_save_attempts == 5


def test_digital_twin_handler_writes_deployment_bundle(tmp_path: Path):
    handler = DigitalTwinHandler()
    node = Node(id="DeployTwin", type="digital_twin", label="DeployTwin")
    context = Context()
    context.set(
        "Generate.output",
        "```python\n# filename: main.py\nprint('hello')\n```",
    )
    graph = Graph(name="g", goal="Ship app")
    emitter = EventEmitter()

    class Config:
        checkpoint_dir = str(tmp_path)

    outcome = handler.execute(node, context, graph, emitter, config=Config())

    assert outcome.status.value == "success"
    assert (tmp_path / "twin_manifest.json").exists()
    assert (tmp_path / "deployment" / "current" / "main.py").exists()
