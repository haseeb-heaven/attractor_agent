from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import gradio as gr

from attractor.pipeline.events import EventEmitter, PipelineEvent, PipelineEventKind
from attractor.pipeline.interviewer import Answer, Interviewer, Question

from attractor_agent.project import (
    BuildRequest,
    SUPPORTED_LANGUAGES,
    get_gradio_language,
    load_build_request,
)
from attractor_agent.runtime import execute_build, get_project_dir

logger = logging.getLogger("attractor_agent.gui")


class GradioInterviewer(Interviewer):
    def __init__(self) -> None:
        self.question: Question | None = None
        self.answer_event = threading.Event()
        self.selected_label: str | None = None

    def ask(self, question: Question) -> Answer:
        self.question = question
        self.answer_event.clear()
        self.answer_event.wait()
        selected_label = self.selected_label or ""
        self.question = None
        self.selected_label = None
        return Answer(question_id=question.id, selected_label=selected_label)

    def set_answer(self, label: str) -> None:
        self.selected_label = label
        self.answer_event.set()


active_interviewer: GradioInterviewer | None = None


def load_config_into_form(config_file: str | None):
    if not config_file:
        return ("", "Python", "", True, True, False, True, False, 3, "No config loaded.")

    spec = load_build_request(config_file)
    message = f"Loaded config from {Path(config_file).name}"
    return (
        spec.request,
        spec.language,
        spec.framework,
        spec.include_tests,
        spec.include_sdlc,
        spec.use_mock,
        spec.auto_approve,
        spec.require_human_review,
        spec.retry_save_attempts,
        message,
    )


def _resolve_spec(
    request: str,
    language: str,
    framework: str,
    include_tests: bool,
    include_sdlc: bool,
    use_mock: bool,
    auto_approve: bool,
    require_human_review: bool,
    retry_save_attempts: int,
    config_file: str | None,
) -> BuildRequest:
    if config_file:
        spec = load_build_request(config_file)
        spec.request = request or spec.request
        spec.language = language or spec.language
        spec.framework = framework
        spec.include_tests = include_tests
        spec.include_sdlc = include_sdlc
        spec.use_mock = use_mock
        spec.auto_approve = auto_approve
        spec.require_human_review = require_human_review
        spec.retry_save_attempts = retry_save_attempts
        return spec

    return BuildRequest(
        request=request.strip(),
        language=language,
        framework=framework.strip(),
        include_tests=include_tests,
        include_sdlc=include_sdlc,
        use_mock=use_mock,
        auto_approve=auto_approve,
        require_human_review=require_human_review,
        retry_save_attempts=max(1, int(retry_save_attempts)),
    )


def start_pipeline(
    request: str,
    language: str,
    framework: str,
    include_tests: bool,
    include_sdlc: bool,
    use_mock: bool,
    auto_approve: bool,
    require_human_review: bool,
    retry_save_attempts: int,
    config_file: str | None,
):
    global active_interviewer

    try:
        spec = _resolve_spec(
            request,
            language,
            framework,
            include_tests,
            include_sdlc,
            use_mock,
            auto_approve,
            require_human_review,
            retry_save_attempts,
            config_file,
        )
    except Exception as exc:
        yield str(exc), "", gr.update(visible=False), gr.update(visible=False)
        return

    if not spec.request:
        yield "Please enter a build request or upload a config file.", "", gr.update(visible=False), gr.update(visible=False)
        return

    logs = [f"Starting pipeline for: {spec.request}", f"Project dir: {get_project_dir(spec)}"]

    interviewer: Interviewer | None = None
    if spec.require_human_review and not spec.auto_approve:
        interviewer = GradioInterviewer()
        active_interviewer = interviewer

    emitter = EventEmitter()
    result_container = {}

    @emitter.on
    def on_event(event: PipelineEvent) -> None:
        message = event.message or ""
        node = event.node_id or "pipeline"
        if event.kind == PipelineEventKind.STAGE_STARTED:
            logs.append(f"Running: {node}")
        elif event.kind == PipelineEventKind.STAGE_COMPLETED:
            logs.append(f"Completed: {node}")
        elif event.kind == PipelineEventKind.STAGE_FAILED:
            logs.append(f"Failed: {node} - {message}")
        elif event.kind == PipelineEventKind.INTERVIEW_STARTED:
            logs.append(f"Review required: {node}")

    def run_it() -> None:
        try:
            result_container["artifacts"] = execute_build(spec, interviewer=interviewer, emitter=emitter)
        except Exception as exc:
            result_container["error"] = str(exc)

    thread = threading.Thread(target=run_it, daemon=True)
    thread.start()

    while True:
        if active_interviewer and active_interviewer.question:
            question = active_interviewer.question
            yield (
                "\n".join(logs + [f"Action required: {question.node_id}"]),
                question.text,
                gr.update(visible=True, interactive=True),
                gr.update(visible=True, interactive=True),
            )
            while active_interviewer and active_interviewer.question:
                time.sleep(0.2)
            yield "\n".join(logs), "", gr.update(visible=False), gr.update(visible=False)
        elif not thread.is_alive():
            break
        else:
            yield "\n".join(logs), "", gr.update(visible=False), gr.update(visible=False)
            time.sleep(0.5)

    if "error" in result_container:
        yield result_container["error"], "", gr.update(visible=False), gr.update(visible=False)
        return

    artifacts = result_container["artifacts"]
    if artifacts.result.success and artifacts.saved_files:
        preview_file = artifacts.saved_files[0]
        code = preview_file.read_text(encoding="utf-8")
        logs.append(f"Saved {len(artifacts.saved_files)} file(s) to {artifacts.project_dir}")
        yield "\n".join(logs), code, gr.update(visible=False), gr.update(visible=False)
        return

    error = artifacts.result.error or "No files were extracted from the pipeline output."
    yield "\n".join(logs + [error]), "", gr.update(visible=False), gr.update(visible=False)


def approve_action():
    global active_interviewer
    if active_interviewer:
        active_interviewer.set_answer("[A]pprove")
    return gr.update(visible=False), gr.update(visible=False)


def retry_action():
    global active_interviewer
    if active_interviewer:
        active_interviewer.set_answer("[R]etry")
    return gr.update(visible=False), gr.update(visible=False)


def run_gui(port: int = 8000) -> None:
    with gr.Blocks(title="Attractor Agent") as app:
        gr.Markdown("# Attractor Agent")
        gr.Markdown("Run the autonomous SDLC pipeline from form inputs or a JSON/TOML config file.")

        with gr.Row():
            request_input = gr.Textbox(label="What do you want to build?", lines=3)
            config_file = gr.File(label="Build config (.json or .toml)", file_count="single", type="filepath")

        with gr.Row():
            language_input = gr.Dropdown(choices=SUPPORTED_LANGUAGES, value="Python", label="Language")
            framework_input = gr.Textbox(label="Framework")
            retry_input = gr.Number(value=3, precision=0, label="Retry save attempts")

        with gr.Row():
            tests_input = gr.Checkbox(value=True, label="Include tests")
            sdlc_input = gr.Checkbox(value=True, label="Include SDLC review")
            mock_input = gr.Checkbox(value=False, label="Use local mock server")
            auto_approve_input = gr.Checkbox(value=True, label="Auto approve")
            human_review_input = gr.Checkbox(value=False, label="Require human review")

        load_button = gr.Button("Load Config")
        start_button = gr.Button("Build Project", variant="primary")
        status_box = gr.Textbox(label="Status", interactive=False)
        log_box = gr.Textbox(label="Execution logs", interactive=False, lines=14)
        code_box = gr.Code(label="Preview", interactive=False, language="python", lines=20)

        with gr.Row():
            approve_button = gr.Button("Approve", visible=False, variant="primary")
            retry_button = gr.Button("Retry", visible=False, variant="stop")

        load_button.click(
            load_config_into_form,
            inputs=[config_file],
            outputs=[
                request_input,
                language_input,
                framework_input,
                tests_input,
                sdlc_input,
                mock_input,
                auto_approve_input,
                human_review_input,
                retry_input,
                status_box,
            ],
        )

        start_button.click(
            start_pipeline,
            inputs=[
                request_input,
                language_input,
                framework_input,
                tests_input,
                sdlc_input,
                mock_input,
                auto_approve_input,
                human_review_input,
                retry_input,
                config_file,
            ],
            outputs=[log_box, code_box, approve_button, retry_button],
        )

        language_input.change(
            lambda lang: gr.update(language=get_gradio_language(lang)),
            inputs=[language_input],
            outputs=[code_box],
        )
        approve_button.click(approve_action, outputs=[approve_button, retry_button])
        retry_button.click(retry_action, outputs=[approve_button, retry_button])

    app.launch(server_name="127.0.0.1", server_port=port, theme=gr.themes.Soft())


if __name__ == "__main__":
    run_gui()
