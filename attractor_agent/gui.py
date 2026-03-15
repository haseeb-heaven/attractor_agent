import os
import re
import threading
import time
import subprocess
import socket
import atexit
import logging
from pathlib import Path

import gradio as gr
from attractor.pipeline.engine import PipelineConfig, run_pipeline
from attractor.pipeline.events import EventEmitter, PipelineEventKind, PipelineEvent
from attractor.pipeline.interviewer import Interviewer, Question, Answer
from attractor.pipeline.backend import LLMBackend
from attractor_agent.cli import build_dot

logger = logging.getLogger("attractor_agent.gui")


class GradioInterviewer(Interviewer):
    """Thread-safe interviewer for Gradio human review gates."""

    def __init__(self):
        self.question = None
        self.answer_event = threading.Event()
        self.selected_label = None

    def ask(self, question: Question) -> Answer:
        self.question = question
        self.answer_event.clear()
        self.answer_event.wait()
        ans = self.selected_label
        self.question = None
        self.selected_label = None
        return Answer(question_id=question.id, selected_label=ans)

    def set_answer(self, label: str):
        self.selected_label = label
        self.answer_event.set()


# ── Port Waiter ───────────────────────────────────────────────────────────────
def wait_for_port(host: str, port: int, timeout: float = 45.0) -> bool:
    """
    Poll until a TCP port accepts connections or timeout expires.
    """
    logger.info(f"Waiting for server at {host}:{port} (timeout={timeout}s)")
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                logger.info(f"✓ Server ready at {host}:{port}")
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)
    raise RuntimeError(
        f"Server at {host}:{port} did not respond within {timeout}s"
    )


def slugify(text: str) -> str:
    """Convert text to filesystem-safe slug."""
    slug = re.sub(r'[^a-zA-Z0-9_\-]', '_', text.lower())
    return re.sub(r'_+', '_', slug).strip('_')


def get_extension(language: str) -> str:
    """Map language name to file extension."""
    lang = language.lower()
    if "javascript" in lang or lang == "js":
        return ".js"
    elif "typescript" in lang or lang == "ts":
        return ".ts"
    elif "html" in lang:
        return ".html"
    elif "go" in lang:
        return ".go"
    elif "rust" in lang:
        return ".rs"
    elif "c++" in lang or "cpp" in lang:
        return ".cpp"
    elif "java" in lang and "script" not in lang:
        return ".java"
    return ".py"


def get_gradio_language(language: str) -> str:
    """Map language name to Gradio Code component language."""
    lang = language.lower()
    if "javascript" in lang:
        return "javascript"
    elif "typescript" in lang:
        return "typescript"
    elif "html" in lang:
        return "html"
    elif "go" in lang:
        return "go"
    elif "rust" in lang:
        return "rust"
    elif "c++" in lang or "cpp" in lang:
        return "cpp"
    elif "java" in lang and "script" not in lang:
        return "java"
    return "python"


# Global active interviewer for button callbacks
active_interviewer: GradioInterviewer | None = None


def start_pipeline(request: str, language: str, framework: str, include_tests: bool, include_sdlc: bool, use_mock: bool = False):
    """Main pipeline generator — streams logs + code to Gradio UI."""
    global active_interviewer

    # Validate input
    if not request.strip():
        yield "⚠️ Please enter what you want to build.", "", gr.update(visible=False), gr.update(visible=False)
        return

    # Setup project directory
    slug = slugify(request)[:30] or "project"
    project_dir = Path("projects") / slug
    os.makedirs(project_dir, exist_ok=True)
    app_file_name = f"main{get_extension(language)}"

    # Generate DOT pipeline
    dot_content = build_dot(request, language, framework, include_tests, include_sdlc)
    dot_file = project_dir / "pipeline.dot"
    dot_file.write_text(dot_content, encoding="utf-8")

    logs = [f"🚀 Pipeline started: {request}\n"]

    if use_mock:
        yield f"🚀 Pipeline started: {request}\nSaved: {dot_file}\n🚀 Starting Mock LLM server on port 5555...\n", "", gr.update(visible=False), gr.update(visible=False)
        
        # Start llmock as a subprocess using shell=True string for Windows compatibility
        mock_process = subprocess.Popen(
            "npx -y @copilotkit/llmock --port 5555",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Ensure it cleans up on exit
        atexit.register(lambda: mock_process.terminate())
        
        try:
            # Wait until the port is actually accepting connections
            wait_for_port("localhost", 5555, timeout=45)
            logs.append("✓ Mock LLM server is ready.")
        except RuntimeError as e:
            logs.append(f"❌ Mock server startup failed: {e}")
            yield "\n".join(logs), "", gr.update(visible=False), gr.update(visible=False)
            return
        
        from attractor.llm.client import Client
        from attractor.llm.adapters.openai import OpenAIAdapter
        mock_adapter = OpenAIAdapter(api_key="mock-key", base_url="http://localhost:5555/v1")
        client = Client(providers={"mock": mock_adapter}, default_provider="mock")
        backend = LLMBackend(client=client)
    else:
        backend = LLMBackend()
        yield f"🚀 Pipeline started: {request}\nSaved: {dot_file}\n", "", gr.update(visible=False), gr.update(visible=False)

    interviewer = GradioInterviewer()
    active_interviewer = interviewer

    config = PipelineConfig(
        simulate=False,
        codergen_backend=backend,
        interviewer=interviewer,
        checkpoint_dir=str(project_dir)
    )

    emitter = EventEmitter()

    @emitter.on
    def on_event(event: PipelineEvent):
        nonlocal logs
        msg = event.message or ""
        node = event.node_id
        if event.kind == PipelineEventKind.STAGE_STARTED:
            logs.append(f"⏳ Running: {node}")
        elif event.kind == PipelineEventKind.STAGE_COMPLETED:
            logs.append(f"✅ Done: {node}")
        elif event.kind == PipelineEventKind.STAGE_FAILED:
            logs.append(f"❌ Failed: {node} — {msg}")
        elif event.kind == PipelineEventKind.INTERVIEW_STARTED:
            logs.append(f"⚠️ Review required: {node}")

    # Run pipeline in background thread
    result_container = []

    def run_it():
        result_container.append(run_pipeline(dot_content, config=config, emitter=emitter))

    t = threading.Thread(target=run_it, daemon=True)
    t.start()

    # ✅ REFACTORED: Unified loop for streaming logs + handling human gates
    while True:
        # PRIORITY 1: Human review gate — show buttons prominently
        if active_interviewer and active_interviewer.question:
            q = active_interviewer.question
            
            # Extract clean code if this is a code review prompt
            code_text = q.text
            match = re.search(r"```[a-zA-Z0-9+\-#]*\s*(.*?)```", q.text, re.DOTALL)
            if match:
                code_text = match.group(1).strip()
            
            # Update UI: Show the question in logs and make buttons visible
            yield (
                "\n".join(logs) + f"\n\n👉 ACTION REQUIRED: {q.text[:200]}...",
                code_text,
                gr.update(visible=True, interactive=True),
                gr.update(visible=True, interactive=True)
            )

            # Wait for user input via button click
            while active_interviewer and active_interviewer.question:
                time.sleep(0.5)
            
            logs.append("✅ Human responded — continuing pipeline.")
            # Hide buttons immediately after response
            yield "\n".join(logs), code_text, gr.update(visible=False), gr.update(visible=False)

        # PRIORITY 2: Pipeline finished
        elif not t.is_alive():
            break

        # PRIORITY 3: Still running — stream latest logs
        else:
            # When running normally, ensure buttons stay hidden
            yield "\n".join(logs), "", gr.update(visible=False), gr.update(visible=False)
            time.sleep(1.0)

    # ── Post-Execution Visualization ──────────────────────────────────────────
    if not result_container:
        logs.append("❌ Pipeline crashed or yielded no results.")
        yield "\n".join(logs), "", gr.update(visible=False), gr.update(visible=False)
        return

    result = result_container[0]
    if result.success:
        # Final response extraction
        output = result.context.get_string("last_response", "Pipeline completed successfully.")
        code = output
        match = re.search(r"```[a-zA-Z0-9+\-#]*\s*(.*?)```", output, re.DOTALL)
        if match:
            code = match.group(1).strip()

        app_file = project_dir / app_file_name
        try:
            app_file.write_text(code, encoding="utf-8")
            logs.append(f"\n🎉 Deployment successful! Final code saved to: {app_file}")
        except Exception as e:
            logs.append(f"\n⚠️ Could not save file: {e}")

        yield "\n".join(logs), code, gr.update(visible=False), gr.update(visible=False)
    else:
        logs.append(f"\n❌ Pipeline failed: {result.error}")
        yield "\n".join(logs), "", gr.update(visible=False), gr.update(visible=False)


# ✅ FIX: Return gr.update to hide buttons after click
def approve_action():
    """Handle Approve — resume pipeline."""
    global active_interviewer
    if active_interviewer:
        active_interviewer.set_answer("[A]pprove")
    return gr.update(visible=False), gr.update(visible=False)


def retry_action():
    """Handle Retry — loop back to Generate."""
    global active_interviewer
    if active_interviewer:
        active_interviewer.set_answer("[R]etry")
    return gr.update(visible=False), gr.update(visible=False)


def run_gui():
    """Launch Gradio web UI."""
    custom_css = """
    .centered-layout {
        max-width: 1000px;
        margin: 0 auto !important;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .footer { text-align: center; margin-top: 20px; color: #666; }
    """

    with gr.Blocks(title="Attractor Agent", css=custom_css) as app:
        with gr.Column(elem_classes="centered-layout"):
            gr.Markdown("<h1 style='text-align: center;'>🚀 Attractor Agent</h1>")
            gr.Markdown("<p style='text-align: center;'>Configure your project and watch the SDLC pipeline run.</p>")

            with gr.Row():
                with gr.Column(scale=1):
                    request_input = gr.Textbox(
                        placeholder="e.g. Build a Flask books app with auth",
                        label="What do you want to build?",
                        lines=3
                    )
                    with gr.Row():
                        lang_input = gr.Dropdown(
                            choices=["Python", "JavaScript", "TypeScript", "HTML/CSS", "Go", "Rust", "C++", "Java"],
                            value="Python",
                            label="Programming Language",
                            scale=1
                        )
                        framework_input = gr.Textbox(
                            placeholder="e.g. Flask, React (optional)",
                            label="Framework",
                            scale=1
                        )
                    with gr.Row():
                        tests_input = gr.Checkbox(value=True, label="Include Unit Tests")
                        sdlc_input = gr.Checkbox(value=True, label="Include SDLC Review")
                        mock_input = gr.Checkbox(value=False, label="Use Mock LLM (localhost:5555)")

                    start_btn = gr.Button("🚀 Build Project", variant="primary", size="lg")

            with gr.Row():
                with gr.Column(scale=1):
                    log_box = gr.Textbox(label="Execution Logs", interactive=False, lines=6)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🛠️ Generated Code Review")
                    code_box = gr.Code(
                        label="Generated Code",
                        language=None,
                        interactive=False,
                        lines=20
                    )
                    with gr.Row():
                        btn_approve = gr.Button("✅ Accept & Save", variant="primary", visible=False, size="lg")
                        btn_retry = gr.Button("🔄 Reject & Retry", variant="stop", visible=False, size="lg")

            gr.Markdown("<div class='footer'>Powered by Attractor Pipeline Engine</div>")

            start_btn.click(
                start_pipeline,
                inputs=[request_input, lang_input, framework_input, tests_input, sdlc_input, mock_input],
                outputs=[log_box, code_box, btn_approve, btn_retry]
            )
            
            # Dynamic language highlighting
            def update_code_lang(lang):
                return gr.update(language=get_gradio_language(lang))
            
            lang_input.change(update_code_lang, inputs=[lang_input], outputs=[code_box])

        # ✅ FIX: outputs wired so buttons hide after click
        btn_approve.click(
            approve_action,
            inputs=[],
            outputs=[btn_approve, btn_retry]
        )
        btn_retry.click(
            retry_action,
            inputs=[],
            outputs=[btn_approve, btn_retry]
        )

    print("🚀 Launching on http://localhost:8000")
    app.launch(server_name="127.0.0.1", server_port=8000, theme=gr.themes.Soft())


if __name__ == "__main__":
    run_gui()
