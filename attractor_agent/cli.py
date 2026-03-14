import os
import re
import sys
import threading
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown
from rich.status import Status

from attractor.pipeline.engine import PipelineConfig, run_pipeline
from attractor.pipeline.events import EventEmitter, PipelineEventKind, PipelineEvent
from attractor.pipeline.interviewer import Interviewer, Question, Answer
from attractor.pipeline.backend import LLMBackend

console = Console()

class RichInterviewer(Interviewer):
    def __init__(self, console_lock: threading.Lock, status: Status):
        self.console_lock = console_lock
        self.status = status

    def ask(self, question: Question) -> Answer:
        # Stop the spinner briefly
        self.status.stop()
        with self.console_lock:
            console.print("\n")
            console.print(Panel(Markdown(question.text), title=f"[bold yellow]Human Review[/bold yellow] ({question.node_id})", border_style="yellow"))
            options_text = " / ".join(f"[{o.key.upper()}] {o.label}" for o in question.options)
            console.print(f"[bold cyan]Options:[/bold cyan] {options_text}")
            
            valid_keys = [o.key.lower() for o in question.options]
            
            while True:
                choice = Prompt.ask("[bold green]Your choice[/bold green]").lower()
                for opt in question.options:
                    if choice == opt.key.lower() or choice == opt.label.lower() or (choice and opt.label.lower().startswith(choice)):
                        self.status.start()
                        return Answer(question_id=question.id, selected_label=opt.label)
                console.print(f"[red]Invalid choice. valid keys: {', '.join(valid_keys)}[/red]")

def slugify(text: str) -> str:
    slug = re.sub(r'[^a-zA-Z0-9_\-]', '_', text.lower())
    return re.sub(r'_+', '_', slug).strip('_')

def run_cli():
    console.print(Panel("[bold magenta]Attractor Agent[/bold magenta]\\nWelcome to the conversational AI builder.", expand=False))
    
    request = Prompt.ask("[bold cyan]What do you want to build?[/bold cyan]")
    if not request:
        console.print("[red]No request provided. Exiting.[/red]")
        sys.exit(1)
        
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
    console.print(f"[dim]Generated pipeline definition at {dot_file}[/dim]")
    
    backend = LLMBackend()
    console_lock = threading.Lock()
    status = Status("[bold green]Initializing pipeline...[/bold green]", console=console)
    interviewer = RichInterviewer(console_lock, status)
    
    config = PipelineConfig(
        simulate=False,
        codergen_backend=backend,
        interviewer=interviewer,
        checkpoint_dir=str(project_dir)
    )
    
    emitter = EventEmitter()
    
    @emitter.on
    def on_event(event: PipelineEvent):
        msg = event.message or ""
        node = event.node_id
        
        if event.kind == PipelineEventKind.STAGE_STARTED:
            with console_lock:
                status.update(f"[bold blue]Executing step:[/bold blue] {node}")
        elif event.kind == PipelineEventKind.STAGE_COMPLETED:
            with console_lock:
                console.print(f"  [green]✓[/green] Completed: {node}")
        elif event.kind == PipelineEventKind.STAGE_FAILED:
            with console_lock:
                console.print(f"  [red]✗[/red] Failed: {node} - {msg}")
                
    status.start()
    try:
        result = run_pipeline(dot_content, config=config, emitter=emitter)
    finally:
        status.stop()
        
    if result.success:
        import re as regex
        # Extract code from Generate block
        output = result.context.get_string("Generate.output", "")
        code = output
        
        # Try to parse python blocks
        match = regex.search(r"```(?:python)?\\n(.*?)\\n```", output, regex.DOTALL)
        if match:
            code = match.group(1).strip()
            
        app_file = project_dir / "app.py"
        app_file.write_text(code, encoding="utf-8")
        
        console.print(f"\\n[bold green]Project '{slug}' has been created at {app_file}[/bold green]")
        console.print(f"Elapsed time: {result.elapsed_seconds:.2f}s, total steps: {result.total_steps}")
    else:
        console.print(f"\\n[bold red]Generation encountered an error: {result.error}[/bold red]")

if __name__ == "__main__":
    run_cli()
