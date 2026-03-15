import os
import re
import sys
import threading
import logging
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown
from rich.status import Status
from rich.logging import RichHandler
from rich.tree import Tree

from attractor.pipeline.engine import PipelineConfig, run_pipeline
from attractor.pipeline.events import EventEmitter, PipelineEventKind, PipelineEvent
from attractor.pipeline.interviewer import Interviewer, Question, Answer
from attractor.pipeline.backend import LLMBackend

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        RichHandler(rich_tracebacks=True, show_path=False),
        logging.FileHandler("attractor_agent.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("attractor_agent.cli")

console = Console()


# ── Helpers ───────────────────────────────────────────────────────────────────
def slugify(text: str) -> str:
    """Convert text to filesystem-safe slug."""
    slug = re.sub(r'[^a-zA-Z0-9_\-]', '_', text.lower())
    return re.sub(r'_+', '_', slug).strip('_')


def get_extension(language: str) -> str:
    """Map language name to file extension."""
    lang = language.lower()
    mapping = {
        "javascript": ".js", "js": ".js",
        "typescript": ".ts", "ts": ".ts",
        "html": ".html", "html/css": ".html",
        "go": ".go", "rust": ".rs",
        "c++": ".cpp", "cpp": ".cpp",
        "java": ".java",
    }
    for key, ext in mapping.items():
        if key in lang:
            return ext
    return ".py"


def get_file_extension_from_tag(lang_tag: str, fallback_language: str) -> str:
    """Map code block language tag to file extension."""
    ext_map = {
        "javascript": ".js", "js": ".js",
        "typescript": ".ts", "ts": ".ts",
        "html": ".html", "css": ".css",
        "sql": ".sql", "python": ".py",
        "go": ".go", "rust": ".rs",
        "java": ".java", "cpp": ".cpp",
        "bash": ".sh", "shell": ".sh",
        "json": ".json", "yaml": ".yml",
        "markdown": ".md", "md": ".md",
    }
    return ext_map.get(lang_tag.lower().strip(), get_extension(fallback_language))


def get_smart_filename(lang_tag: str, index: int, language: str) -> str:
    """Generate meaningful filename based on language tag and index."""
    ext = get_file_extension_from_tag(lang_tag, language)
    name_map = {
        ".html": ["index.html", "login.html", "register.html", "dashboard.html"],
        ".css":  ["style.css", "auth.css", "dashboard.css"],
        ".js":   ["main.js", "app.js", "api.js", "utils.js"],
        ".ts":   ["main.ts", "app.ts", "api.ts", "utils.ts"],
        ".sql":  ["schema.sql"],
        ".py":   ["main.py", "app.py", "models.py", "utils.py"],
        ".sh":   ["setup.sh", "run.sh"],
        ".json": ["package.json", "config.json"],
    }
    names = name_map.get(ext, [])
    if index < len(names):
        return names[index]
    return f"file_{index + 1}{ext}"


def print_project_tree(project_dir: Path):
    """Print a Rich tree of generated project files."""
    tree = Tree(f"[bold green]📁 {project_dir}[/bold green]")
    for f in sorted(project_dir.iterdir()):
        if f.is_file():
            size = f.stat().st_size
            tree.add(f"[cyan]{f.name}[/cyan] [dim]({size} bytes)[/dim]")
    console.print(tree)


# ── DOT Builder ───────────────────────────────────────────────────────────────
def build_dot(request: str, language: str, framework: str,
              include_tests: bool, include_sdlc: bool) -> str:
    """Build spec-correct Attractor DOT pipeline with dynamic SDLC phases."""
    fw = f"using {framework}" if framework else ""
    logger.debug(f"Building DOT — lang={language} fw='{framework}' tests={include_tests} sdlc={include_sdlc}")

    dot = f'''digraph generate_app {{
    rankdir=LR;
    goal="Build: {request} in {language} {fw}";

    node [shape=box, timeout="1200s", max_retries=2]

    Start [shape=Mdiamond, label="Start"];
    Done  [shape=Msquare,  label="Done"];

    Plan [
        label="Plan Architecture",
        prompt="Plan architecture for: {request} in {language} {fw}.\\nOutput:\\n- Full folder and file structure with filenames\\n- Each module purpose\\n- Dependencies list\\n- Database schema if needed\\nNO CODE YET.",
        goal_gate=true
    ];

    Generate [
        label="Generate Code",
        prompt="Using the plan above, write COMPLETE {language} code for: {request} {fw}.\\nRules:\\n- Output EVERY file as a SEPARATE triple backtick block\\n- Label each block with language tag (```html, ```css, ```javascript, ```sql, ```python etc)\\n- Start each block with a comment: // filename: <filename>\\n- No explanations between blocks\\n- All files must be complete and runnable",
        goal_gate=true
    ];
'''

    if include_tests:
        dot += f'''
    Tests [
        label="Unit Tests",
        prompt="Write complete unit tests for the code above in {language}.\\nTest ALL functions and edge cases.\\nOutput ONLY test code in a single triple backtick block labeled with language tag.",
        goal_gate=true
    ];

    TestGate [shape=diamond, label="Tests Pass?"];
'''

    if include_sdlc:
        dot += '''
    SDLCCheck [
        label="SDLC Validation",
        prompt="Review ALL generated code files for:\\n- Error handling\\n- Input validation\\n- Security (SQL injection, XSS, auth)\\n- Logging\\n- Documentation\\nOutput: PASS if all good, or LIST of specific issues per file.",
        goal_gate=true
    ];
'''

    dot += '''
    Review [shape=hexagon, label="Human Review"];

    Start -> Plan -> Generate;
'''

    if include_tests:
        dot += '''    Generate -> Tests -> TestGate;
    TestGate -> Review [label="pass", condition="outcome=success"];
    TestGate -> Generate [label="retry", condition="outcome=fail"];
'''
    else:
        dot += '''    Generate -> Review;
'''

    if include_sdlc:
        dot += '''    Review -> SDLCCheck [label="[A]pprove"];
    SDLCCheck -> Done [condition="outcome=success"];
    SDLCCheck -> Generate [condition="outcome=fail", label="[F]ix"];
'''
    else:
        dot += '''    Review -> Done [label="[A]pprove"];
'''

    dot += '''    Review -> Generate [label="[R]etry"];
}
'''
    return dot


# ── File Saver ────────────────────────────────────────────────────────────────
def save_output_files(result, project_dir: Path, language: str, app_file_name: str):
    """
    Parse ALL code blocks from Generate output and save as separate files.
    Extracts filenames from inline comments: // filename: <name> or # filename: <name>
    """
    # ── Try all possible context keys ────────────────────────────────────────
    generate_output = (
        result.context.get_string("Generate.output", "")
        or result.context.get_string("generate_output", "")
        or result.context.get_string("last_response", "")
        or result.context.get_string("output", "")
    )
    tests_output = (
        result.context.get_string("Tests.output", "")
        or result.context.get_string("tests_output", "")
    )

    # ── DEBUG: dump all context keys to log ──────────────────────────────────
    logger.debug("=== ALL CONTEXT KEYS ===")
    try:
        for key in result.context.keys():
            val = str(result.context.get_string(key, ""))[:150]
            logger.debug(f"  KEY [{key}] = {val}")
    except Exception as e:
        logger.warning(f"Could not iterate context keys: {e}")
    logger.debug("========================")

    if not generate_output:
        logger.error("Generate output is EMPTY — check attractor_agent.log for context keys!")
        console.print("[bold red]❌ No generated code found. Check attractor_agent.log[/bold red]")
        return

    # ── Parse all code blocks with language tag ───────────────────────────────
    code_blocks = re.findall(
        r"```([a-zA-Z0-9+\-#]*)\s*(.*?)```",
        generate_output,
        re.DOTALL
    )

    logger.info(f"Found {len(code_blocks)} code block(s) in Generate output")

    # Track counts per extension for smart naming
    ext_counters: dict[str, int] = {}
    saved_files = []

    if code_blocks:
        for lang_tag, code in code_blocks:
            code = code.strip()
            if not code:
                continue

            ext = get_file_extension_from_tag(lang_tag, language)

            # ── Try to extract filename from inline comment ───────────────────
            filename = None
            for line in code.splitlines()[:3]:
                # Matches: // filename: index.html OR # filename: main.py
                m = re.match(r'^(?://|#)\s*filename:\s*(.+)', line.strip(), re.IGNORECASE)
                if m:
                    filename = m.group(1).strip()
                    # Remove the filename comment line from code
                    code = "\n".join(
                        l for l in code.splitlines()
                        if not re.match(r'^(?://|#)\s*filename:', l.strip(), re.IGNORECASE)
                    ).strip()
                    break

            # ── Fallback: smart name by type + counter ────────────────────────
            if not filename:
                count = ext_counters.get(ext, 0)
                filename = get_smart_filename(lang_tag, count, language)
                ext_counters[ext] = count + 1

            fpath = project_dir / filename
            fpath.write_text(code, encoding="utf-8")
            saved_files.append(fpath)
            logger.info(f"Saved: {fpath} ({len(code)} chars)")

    else:
        # ── Fallback: no code blocks — save raw output ────────────────────────
        logger.warning("No triple-backtick code blocks found — saving raw output")
        raw_file = project_dir / app_file_name
        raw_file.write_text(generate_output, encoding="utf-8")
        console.print(f"[yellow]⚠ No code blocks found — saved raw to {raw_file}[/yellow]")
        saved_files.append(raw_file)

    # ── Save tests ────────────────────────────────────────────────────────────
    if tests_output:
        test_blocks = re.findall(
            r"```[a-zA-Z0-9+\-#]*\s*(.*?)```",
            tests_output,
            re.DOTALL
        )
        test_code = test_blocks.strip() if test_blocks else tests_output.strip()
        test_file = project_dir / f"test_main{get_extension(language)}"
        test_file.write_text(test_code, encoding="utf-8")
        saved_files.append(test_file)
        logger.info(f"Tests saved: {test_file}")

    # ── Write README ──────────────────────────────────────────────────────────
    readme_lines = [
        f"# Generated Project\n",
        f"**Built by:** Attractor Agent\n",
        f"**Language:** {language}\n\n",
        "## Files\n",
    ]
    for f in saved_files:
        readme_lines.append(f"- `{f.name}`\n")
    readme_lines.append("\n## Run\n")
    if language.lower() == "python":
        readme_lines.append("```bash\npip install -r requirements.txt\npython main.py\n```\n")
    elif "javascript" in language.lower():
        readme_lines.append("```bash\nnpm install\nnode main.js\n```\n")

    readme_file = project_dir / "README.md"
    readme_file.write_text("".join(readme_lines), encoding="utf-8")
    logger.info(f"README saved: {readme_file}")

    return saved_files


# ── Rich Interviewer ──────────────────────────────────────────────────────────
class RichInterviewer(Interviewer):
    """Handles hexagon human-review gates in Rich TUI."""

    def __init__(self, console_lock: threading.Lock, status: Status):
        self.console_lock = console_lock
        self.status = status

    def ask(self, question: Question) -> Answer:
        logger.info(f"Human review gate: node={question.node_id}")
        self.status.stop()
        with self.console_lock:
            console.print("\n")
            console.print(Panel(
                Markdown(question.text),
                title=f"[bold yellow]👁 Human Review[/bold yellow] ({question.node_id})",
                border_style="yellow"
            ))
            options_text = " / ".join(f"[{o.key.upper()}] {o.label}" for o in question.options)
            console.print(f"[bold cyan]Options:[/bold cyan] {options_text}")
            valid_keys = [o.key.lower() for o in question.options]

            while True:
                choice = Prompt.ask("[bold green]Your choice[/bold green]").lower()
                for opt in question.options:
                    if (choice == opt.key.lower()
                            or choice == opt.label.lower()
                            or (choice and opt.label.lower().startswith(choice))):
                        logger.info(f"User chose: {opt.label}")
                        self.status.start()
                        return Answer(question_id=question.id, selected_label=opt.label)
                console.print(f"[red]Invalid. Valid keys: {', '.join(valid_keys)}[/red]")


# ── Main CLI ──────────────────────────────────────────────────────────────────
def run_cli():
    console.print(Panel(
        "Welcome to [bold magenta]Attractor Agent[/bold magenta] 🚀\n"
        "[dim]Full SDLC AI pipeline — Plan → Generate → Test → Review → Done[/dim]",
        expand=False,
        border_style="magenta"
    ))
    logger.info("Attractor Agent CLI started")

    # ── Collect inputs ────────────────────────────────────────────────────────
    request = Prompt.ask("[bold cyan]What do you want to build?[/bold cyan]")
    if not request.strip():
        console.print("[red]No request provided. Exiting.[/red]")
        logger.error("Empty request — exiting")
        sys.exit(1)

    language = Prompt.ask(
        "[bold cyan]Programming language?[/bold cyan]",
        choices=["Python", "JavaScript", "TypeScript", "HTML", "Go", "Rust", "C++", "Java"],
        default="Python"
    )
    framework = Prompt.ask(
        "[bold cyan]Framework? (Enter to skip)[/bold cyan]",
        default=""
    )
    include_tests = Prompt.ask(
        "[bold cyan]Include unit tests? (y/n)[/bold cyan]",
        default="y"
    ).lower() == "y"
    include_sdlc = Prompt.ask(
        "[bold cyan]Include SDLC review? (y/n)[/bold cyan]",
        default="y"
    ).lower() == "y"

    logger.info(
        f"Input — request='{request[:60]}' lang={language} fw='{framework}' "
        f"tests={include_tests} sdlc={include_sdlc}"
    )

    # ── Setup project dir ─────────────────────────────────────────────────────
    slug = slugify(request)[:30] or "project"
    project_dir = Path("projects") / slug
    os.makedirs(project_dir, exist_ok=True)
    app_file_name = f"main{get_extension(language)}"
    logger.info(f"Project dir: {project_dir}")
    console.print(f"[dim]📁 Project folder: {project_dir}[/dim]")

    # ── Generate DOT ──────────────────────────────────────────────────────────
    dot_content = build_dot(request, language, framework, include_tests, include_sdlc)
    dot_file = project_dir / "pipeline.dot"
    dot_file.write_text(dot_content, encoding="utf-8")
    console.print(f"[dim]🔧 Pipeline saved: {dot_file}[/dim]")
    logger.info(f"DOT saved: {dot_file}")

    # ── Pipeline engine setup ─────────────────────────────────────────────────
    backend = LLMBackend()
    console_lock = threading.Lock()
    status = Status("[bold green]Running pipeline...[/bold green]", console=console)
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
            logger.info(f"Stage started: {node}")
            with console_lock:
                status.update(f"[bold blue]⏳ Running:[/bold blue] {node}")
        elif event.kind == PipelineEventKind.STAGE_COMPLETED:
            logger.info(f"Stage completed: {node}")
            with console_lock:
                console.print(f"  [green]✓[/green] Completed: [bold]{node}[/bold]")
        elif event.kind == PipelineEventKind.STAGE_FAILED:
            logger.error(f"Stage failed: {node} — {msg}")
            with console_lock:
                console.print(f"  [red]✗[/red] Failed: [bold]{node}[/bold] — {msg}")

    # ── Run ───────────────────────────────────────────────────────────────────
    status.start()
    try:
        logger.info("Pipeline execution started")
        result = run_pipeline(dot_content, config=config, emitter=emitter)
    except Exception as e:
        logger.exception(f"Pipeline crashed: {e}")
        console.print(f"\n[bold red]❌ Pipeline crashed: {e}[/bold red]")
        sys.exit(1)
    finally:
        status.stop()

    # ── Handle result ─────────────────────────────────────────────────────────
    if result.success:
        logger.info(f"Pipeline success — elapsed={result.elapsed_seconds:.2f}s steps={result.total_steps}")
        console.print(f"\n[bold green]✅ Pipeline complete![/bold green] "
                      f"Elapsed: {result.elapsed_seconds:.2f}s | Steps: {result.total_steps}")

        saved = save_output_files(result, project_dir, language, app_file_name)

        # ── Print project tree ────────────────────────────────────────────────
        console.print(f"\n[bold green]📦 Project '{slug}' saved:[/bold green]")
        print_project_tree(project_dir)

    else:
        logger.error(f"Pipeline failed: {result.error}")
        console.print(f"\n[bold red]❌ Pipeline failed: {result.error}[/bold red]")


if __name__ == "__main__":
    run_cli()
