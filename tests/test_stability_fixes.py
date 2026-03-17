from __future__ import annotations

import types
from pathlib import Path

from fastapi.testclient import TestClient

import attractor_agent.api as api_mod
from attractor.agent.profiles.litellm import edit_file_executor
from attractor.pipeline.interviewer import Option, Question
from attractor_agent import __main__ as main_mod
from attractor_agent.extraction import extract_blocks_with_fallbacks
from attractor_agent.project import build_request_from_mapping
from attractor_agent.runtime import execute_build, save_output_files


class _DummyDB:
    def __init__(self):
        self.saved_runs: list[tuple[str, str, dict]] = []
        self.saved_questions: list[tuple[str, str]] = []

    def save_run(self, run_id, goal, config):
        self.saved_runs.append((run_id, goal, config))

    def save_question(self, run_id, question_id, node_id, text, options):
        self.saved_questions.append((run_id, question_id))

    def get_questions(self, run_id):
        return []

    def update_run(self, *args, **kwargs):
        return None

    def save_event(self, *args, **kwargs):
        return None

    def get_run(self, *args, **kwargs):
        return None

    def answer_question(self, *args, **kwargs):
        return None


class _Context:
    def __init__(self, data: dict[str, str]):
        self.data = dict(data)

    def get_string(self, key: str, default: str = "") -> str:
        return self.data.get(key, default)

    def set(self, key: str, value):
        self.data[key] = value


class _Result:
    def __init__(self, output: str):
        self.context = _Context({"Generate.output": output})


def test_create_run_returns_422_for_invalid_payload(monkeypatch):
    dummy_db = _DummyDB()
    monkeypatch.setattr(api_mod, "db", dummy_db)
    monkeypatch.setattr(api_mod, "_write_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(api_mod, "execute_pipeline_task", lambda *args, **kwargs: None)

    client = TestClient(api_mod.app)
    response = client.post("/api/v1/runs", json={"request": "", "prompt": ""})

    assert response.status_code == 422
    assert "request" in response.json()["detail"].lower()


def test_create_run_returns_running_for_valid_payload(monkeypatch):
    dummy_db = _DummyDB()
    monkeypatch.setattr(api_mod, "db", dummy_db)
    monkeypatch.setattr(api_mod, "_write_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(api_mod, "execute_pipeline_task", lambda *args, **kwargs: None)

    client = TestClient(api_mod.app)
    response = client.post("/api/v1/runs", json={"request": "Build a todo app"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "RUNNING"
    assert body["run_id"]
    assert len(dummy_db.saved_runs) == 1


def test_db_interviewer_polling_resumes_when_answer_available(monkeypatch):
    class _QuestionDB(_DummyDB):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def get_questions(self, run_id):
            self.calls += 1
            if self.calls == 1:
                return [{"question_id": "q1", "answer": None}]
            return [{"question_id": "q1", "answer": '{"selected_label":"[A]pprove","free_text":""}'}]

    monkeypatch.setattr(api_mod, "db", _QuestionDB())
    monkeypatch.setattr(api_mod.time, "sleep", lambda *args, **kwargs: None)

    interviewer = api_mod.DBInterviewer("run-1")
    answer = interviewer.ask(
        Question(
            id="q1",
            node_id="Review",
            text="Approve?",
            options=[Option(label="[A]pprove", key="a")],
        )
    )
    assert answer.selected_label == "[A]pprove"


def test_build_request_bool_alias_preserves_primary_false():
    spec = build_request_from_mapping(
        {
            "request": "Build app",
            "use_mock": False,
            "mock_llm": True,
            "require_human_review": False,
            "human_review": True,
        }
    )
    assert spec.use_mock is False
    assert spec.require_human_review is False


def test_markdown_extraction_requires_closing_fence_line():
    text = "```python\nprint('ok')\n```\nplain text"
    blocks = extract_blocks_with_fallbacks(text)
    assert len(blocks) == 1
    assert blocks[0].code == "print('ok')"


def test_save_output_files_supports_subfolders_and_no_synthetic_main(tmp_path: Path):
    result = _Result("```python file=src/app.py\nprint('ok')\n```")
    saved = save_output_files(result, tmp_path, "Python", "main.py")
    rel_paths = {path.relative_to(tmp_path).as_posix() for path in saved}
    assert "src/app.py" in rel_paths
    assert "main.py" not in rel_paths


def test_save_output_files_blocks_path_traversal(tmp_path: Path):
    result = _Result("```python file=../escape.py\nprint('x')\n```")
    saved = save_output_files(result, tmp_path, "Python", "main.py")
    assert saved == []
    assert not (tmp_path / "escape.py").exists()


def test_main_api_defaults_to_loopback_and_supports_host_override(monkeypatch):
    calls: list[tuple[str, int]] = []
    fake_uvicorn = types.SimpleNamespace(run=lambda app, host, port: calls.append((host, port)))
    fake_api = types.SimpleNamespace(app=object())
    monkeypatch.setitem(main_mod.sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setitem(main_mod.sys.modules, "attractor_agent.api", fake_api)

    monkeypatch.setattr(main_mod.sys, "argv", ["attractor_agent", "--api"])
    main_mod.main()
    monkeypatch.setattr(main_mod.sys, "argv", ["attractor_agent", "--api", "--host", "0.0.0.0", "--port", "9000"])
    main_mod.main()

    assert calls[0] == ("127.0.0.1", 8000)
    assert calls[1] == ("0.0.0.0", 9000)


def test_edit_file_executor_rejects_empty_old_string():
    env = types.SimpleNamespace(read_file_raw=lambda _: "hello", write_file=lambda *_: None)
    try:
        edit_file_executor(
            {"file_path": "a.txt", "old_string": "", "new_string": "x"},
            env,
        )
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty old_string")


def test_execute_build_raises_when_no_files_extracted(monkeypatch, tmp_path: Path):
    fake_result = types.SimpleNamespace(success=True, context=_Context({}), final_node="Generate")
    monkeypatch.setattr("attractor_agent.runtime.create_backend", lambda *_: (object(), lambda: None))
    monkeypatch.setattr("attractor_agent.runtime.run_pipeline", lambda *args, **kwargs: fake_result)
    monkeypatch.setattr("attractor_agent.runtime.save_output_files", lambda *args, **kwargs: [])
    monkeypatch.setattr("attractor_agent.runtime.build_dot", lambda *args, **kwargs: "digraph x {}")

    from attractor_agent.project import BuildRequest

    spec = BuildRequest(
        request="Build app",
        language="Python",
        retry_save_attempts=1,
        checkpoint_dir=str(tmp_path),
    )
    try:
        execute_build(spec)
    except RuntimeError as exc:
        assert "No files were extracted" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when extraction yields no files")
