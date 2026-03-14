import os
import re
import threading
import time
from pathlib import Path

from attractor.pipeline.engine import PipelineConfig, run_pipeline
from attractor.pipeline.events import EventEmitter, PipelineEventKind, PipelineEvent
from attractor.pipeline.interviewer import Interviewer, Question, Answer
from attractor.pipeline.backend import LLMBackend

class GradioInterviewer(Interviewer):
    def __init__(self):
        self.question = None
        self.answer_event = threading.Event()
        self.selected_label = None
        
    def ask(self, question: Question) -> Answer:
        self.question = question
        self.answer_event.clear()
        # Wait for Gradio UI to set the answer
        self.answer_event.wait()
        ans = self.selected_label
        self.question = None
        self.selected_label = None
        return Answer(question_id=question.id, selected_label=ans)

    def set_answer(self, label: str):
        self.selected_label = label
        self.answer_event.set()


def slugify(text: str) -> str:
    slug = re.sub(r'[^a-zA-Z0-9_\-]', '_', text.lower())
    return re.sub(r'_+', '_', slug).strip('_')

# Global state for Gradio
active_interviewer = None

def start_pipeline(request: str):
    import gradio as gr
    global active_interviewer
    
    if not request:
        yield "Please enter what you want to build.", "", gr.update(visible=False), gr.update(visible=False)
        return
        
    slug = slugify(request)[:30] or "project"
    project_dir = Path("projects") / slug
    os.makedirs(project_dir, exist_ok=True)
    
    dot_content = f"""
digraph generate_app {{
    rankdir=LR;
    goal="Generate an application based on user request: {request}";

    Start [shape=Mdiamond, label="Start"];
    
    Generate [shape=box, label="Generate Code", 
              type="codergen",
              prompt="User requested to build: {request}. Please provide the complete Python code for this application. Output only the code, wrapped in ```python ... ```."];
              
    Review [shape=hexagon, label="Human Review",
            type="wait.human",
            prompt="Review the generated code:\\n\\n${{Generate.output}}"];
          
    Done [shape=Msquare, label="Done"];

    Start -> Generate;
    Generate -> Review [label="success"];
    Generate -> Generate [label="retry", loop_restart=true];
    Review -> Done [label="[A]pprove"];
    Review -> Generate [label="[R]etry"];
}}
"""
    dot_file = project_dir / "pipeline.dot"
    dot_file.write_text(dot_content, encoding="utf-8")
    
    yield f"🚀 Started pipeline for: {request}\\nGenerating {dot_file}...\\n", "", gr.update(visible=False), gr.update(visible=False)
    
    backend = LLMBackend()
    interviewer = GradioInterviewer()
    active_interviewer = interviewer
    
    config = PipelineConfig(
        simulate=False,
        codergen_backend=backend,
        interviewer=interviewer,
        checkpoint_dir=str(project_dir)
    )
    
    emitter = EventEmitter()
    logs = [f"🚀 Started pipeline for: {request}\\n"]
    
    code_preview = ""
    
    @emitter.on
    def on_event(event: PipelineEvent):
        nonlocal logs
        msg = event.message or ""
        node = event.node_id
        
        if event.kind == PipelineEventKind.STAGE_STARTED:
            logs.append(f"⏳ Executing step: {node}")
        elif event.kind == PipelineEventKind.STAGE_COMPLETED:
            logs.append(f"✅ Completed: {node}")
        elif event.kind == PipelineEventKind.STAGE_FAILED:
            logs.append(f"❌ Failed: {node} - {msg}")
        elif event.kind == PipelineEventKind.INTERVIEW_STARTED:
            logs.append(f"⚠️ Human review required for: {node}")
            
    # Run pipeline in background thread
    result_container = []
    
    def run_it():
        result_container.append(run_pipeline(dot_content, config=config, emitter=emitter))
        
    t = threading.Thread(target=run_it)
    t.start()
    
    # Poll for events or completion
    while t.is_alive() or not getattr(t, 'completed', False):
        if not t.is_alive() and result_container:
            break
            
        if active_interviewer and active_interviewer.question:
            q = active_interviewer.question
            # Show the generated code
            code_text = q.text.replace("Review the generated code:\\n\\n", "")
            if "```" not in code_text:
                code_text = f"```python\\n{code_text}\\n```"
                
            yield "\\n".join(logs), code_text, gr.update(visible=True, interactive=True), gr.update(visible=True, interactive=True)
            # Sleep to wait until answer is provided
            while active_interviewer.question:
                 time.sleep(0.5)
            
            logs.append("Human responded.")
            yield "\\n".join(logs), code_text, gr.update(visible=False), gr.update(visible=False)
        else:
            yield "\\n".join(logs), "Thinking...", gr.update(visible=False), gr.update(visible=False)
            time.sleep(1.0)
            
    result = result_container[0]
    
    if result.success:
        import re as regex
        output = result.context.get_string("Generate.output", "")
        code = output
        
        match = regex.search(r"```(?:python)?\\n(.*?)\\n```", output, regex.DOTALL)
        if match:
            code = match.group(1).strip()
            
        app_file = project_dir / "app.py"
        app_file.write_text(code, encoding="utf-8")
        
        logs.append(f"\\n🎉 Project '{slug}' created at {app_file}")
        
        code_view = code
        if "```" not in code_view:
             code_view = f"```python\\n{code}\\n```"
             
        yield "\\n".join(logs), code_view, gr.update(visible=False), gr.update(visible=False)
    else:
        logs.append(f"\\n❌ Error: {result.error}")
        yield "\\n".join(logs), "Generation failed.", gr.update(visible=False), gr.update(visible=False)


def approve_action():
    global active_interviewer
    if active_interviewer:
        active_interviewer.set_answer("[A]pprove")

def retry_action():
    global active_interviewer
    if active_interviewer:
        active_interviewer.set_answer("[R]etry")

def run_gui():
    import gradio as gr
    with gr.Blocks(title="Attractor Agent GUI", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🚀 Attractor Agent")
        gr.Markdown("What do you want to build? Enter your idea, and I'll generate the pipeline, code, and show you live progress.")
        
        with gr.Row():
            with gr.Column(scale=2):
                request_input = gr.Textbox(placeholder="e.g. Build me a simple calculator script", label="Project Idea", lines=3)
                start_btn = gr.Button("Build It!", variant="primary")
                
                log_box = gr.Textbox(label="Live Progress Tracker", interactive=False, lines=15)
                
            with gr.Column(scale=3):
                gr.Markdown("### Generated Code (Human Review)")
                code_box = gr.Markdown("Code will appear here once generated.")
                with gr.Row():
                    btn_approve = gr.Button("Approve & Save", variant="primary", visible=False)
                    btn_retry = gr.Button("Reject & Retry", variant="stop", visible=False)
        
        start_btn.click(
            start_pipeline, 
            inputs=[request_input], 
            outputs=[log_box, code_box, btn_approve, btn_retry]
        )
        
        btn_approve.click(approve_action)
        btn_retry.click(retry_action)
        
    print("Launching Gradio GUI server on http://localhost:8000")
    app.launch(server_name="127.0.0.1", server_port=8000)

if __name__ == "__main__":
    run_gui()
