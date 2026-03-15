import os
import re
import sys
import socket
import threading
import logging
import subprocess
import time
import atexit
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


# ── Port Waiter ───────────────────────────────────────────────────────────────
def wait_for_port(host: str, port: int, timeout: float = 45.0) -> bool:
    """
    Poll until a TCP port accepts connections or timeout expires.
    Raises RuntimeError if server does not start in time.
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
        f"Server at {host}:{port} did not respond within {timeout}s — "
        "is Node.js installed? Run: node --version"
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
def slugify(text: str) -> str:
    """Convert text to filesystem-safe slug."""
    slug = re.sub(r'[^a-zA-Z0-9_\-]', '_', text.lower())
    return re.sub(r'_+', '_', slug).strip('_')


def get_extension(language: str) -> str:
    """Map language name to primary file extension."""
    lang = language.lower()
    mapping = {
        "javascript": ".js",
        "js":         ".js",
        "typescript": ".ts",
        "ts":         ".ts",
        "html":       ".html",
        "html/css":   ".html",
        "go":         ".go",
        "rust":       ".rs",
        "c++":        ".cpp",
        "cpp":        ".cpp",
        "java":       ".java",
    }
    for key, ext in mapping.items():
        if key in lang:
            return ext
    return ".py"


def get_file_extension_from_tag(lang_tag: str, fallback_language: str) -> str:
    """Map a code-fence language tag to a file extension."""
    ext_map = {
        "javascript": ".js",
        "js":         ".js",
        "typescript": ".ts",
        "ts":         ".ts",
        "html":       ".html",
        "css":        ".css",
        "sql":        ".sql",
        "python":     ".py",
        "go":         ".go",
        "rust":       ".rs",
        "java":       ".java",
        "cpp":        ".cpp",
        "c++":        ".cpp",
        "bash":       ".sh",
        "shell":      ".sh",
        "json":       ".json",
        "yaml":       ".yml",
        "yml":        ".yml",
        "markdown":   ".md",
        "md":         ".md",
        "xml":        ".xml",
        "toml":       ".toml",
    }
    return ext_map.get(lang_tag.lower().strip(), get_extension(fallback_language))


def get_smart_filename(lang_tag: str, index: int, language: str) -> str:
    """
    Generate a meaningful filename based on language tag and occurrence index.
    Falls back to file_N.<ext> when the name list is exhausted.
    """
    ext = get_file_extension_from_tag(lang_tag, language)
    name_map: dict[str, list[str]] = {
        ".html": ["index.html", "login.html", "register.html", "dashboard.html",
                  "profile.html", "404.html"],
        ".css":  ["style.css", "auth.css", "dashboard.css", "components.css"],
        ".js":   ["main.js", "app.js", "api.js", "utils.js", "db.js", "routes.js"],
        ".ts":   ["main.ts", "app.ts", "api.ts", "utils.ts", "db.ts"],
        ".sql":  ["schema.sql", "seed.sql", "migrations.sql"],
        ".py":   ["main.py", "app.py", "models.py", "utils.py", "routes.py",
                  "database.py", "config.py"],
        ".sh":   ["setup.sh", "run.sh", "build.sh"],
        ".json": ["package.json", "config.json", "tsconfig.json"],
        ".java": ["Main.java", "App.java", "Utils.java", "Database.java"],
        ".cpp":  ["main.cpp", "app.cpp", "utils.cpp"],
        ".go":   ["main.go", "app.go", "utils.go"],
    }
    names = name_map.get(ext, [])
    if index < len(names):
        return names[index]
    return f"file_{index + 1}{ext}"


def print_project_tree(project_dir: Path) -> None:
    """Print a Rich directory tree of all generated project files."""
    tree = Tree(f"[bold green]📁 {project_dir}[/bold green]")
    for f in sorted(project_dir.iterdir()):
        if f.is_file():
            size = f.stat().st_size
            icon = "📄"
            if f.suffix in (".html", ".css"):
                icon = "🌐"
            elif f.suffix in (".js", ".ts"):
                icon = "⚡"
            elif f.suffix == ".py":
                icon = "🐍"
            elif f.suffix == ".sql":
                icon = "🗄️"
            elif f.suffix == ".md":
                icon = "📝"
            elif f.suffix in (".sh",):
                icon = "🔧"
            tree.add(
                f"{icon} [cyan]{f.name}[/cyan] [dim]({size:,} bytes)[/dim]"
            )
    console.print(tree)


# ── DOT Builder ───────────────────────────────────────────────────────────────
def build_dot(request: str, language: str, framework: str,
              include_tests: bool, include_sdlc: bool) -> str:
    """
    Build a spec-correct Attractor DOT pipeline with:
      Plan → Generate → [Tests → TestGate] → Review → [SDLCCheck] → Done
    """
    fw = f"using {framework}" if framework else ""
    logger.debug(
        f"Building DOT — lang={language} fw='{framework}' "
        f"tests={include_tests} sdlc={include_sdlc}"
    )

    dot = f'''digraph generate_app {{
    rankdir=LR;
    goal="Build: {request} in {language} {fw}";

    node [shape=box, timeout="1200s", max_retries=2]

    Start [shape=Mdiamond, label="Start"];
    Done  [shape=Msquare,  label="Done"];

    Plan [
        label="Plan Architecture",
        prompt="Plan architecture for: {request} in {language} {fw}.\\nOutput:\\n- Full folder and file structure with exact filenames\\n- Each module/file purpose\\n- All dependencies list\\n- Database schema if needed\\nNO CODE YET.",
        goal_gate=true
    ];

    Generate [
        label="Generate Code",
        prompt="Using the plan above, write COMPLETE {language} code for: {request} {fw}.\\nStrict rules:\\n- Output EVERY file as a SEPARATE triple backtick code block\\n- Label EVERY block with its language tag e.g. ```html ```css ```javascript ```sql ```python\\n- First line inside EVERY block must be a comment: // filename: <exact-filename> (or # filename: for Python/SQL)\\n- ALL files must be complete, functional and runnable\\n- No explanations or text between blocks",
        goal_gate=true
    ];
'''

    if include_tests:
        dot += f'''
    Tests [
        label="Unit Tests",
        prompt="Write COMPLETE unit tests for ALL functions in the code above using {language}.\\n- Cover all edge cases\\n- Use the standard test framework for {language}\\n- Output ONLY test code in a single labeled triple backtick block\\n- First line: // filename: test_main{get_extension(language)}",
        goal_gate=true
    ];

    RunTests [handler="test_runner", label="Execute Tests", max_retries=2];
'''

    if include_sdlc:
        dot += '''
    SDLCCheck [
        label="SDLC Validation",
        prompt="Review ALL generated code files for:\\n1. Error handling and exceptions\\n2. Input validation and sanitization\\n3. Security (SQL injection, XSS, auth checks)\\n4. Logging and observability\\n5. Code documentation\\n6. Memory and performance issues\\nOutput: PASS if everything is good, or a numbered LIST of specific issues per file.",
        goal_gate=true
    ];
'''

    dot += '''
    Score [handler="satisfaction_scorer", label="Satisfaction Scorer"];
    DeployTwin [handler="digital_twin", label="Deploy to Digital Twin Universe"];
    Review [shape=hexagon, label="Human Review"];

    Start -> Plan -> Generate;
'''

    if include_tests:
        dot += '''    Generate -> Tests -> RunTests;
    RunTests -> Score [label="pass", condition="outcome=success"];
    RunTests -> Generate [label="retry_code", condition="outcome=fail"];
'''
    else:
        dot += '''    Generate -> Score;
'''

    if include_sdlc:
        dot += '''    Score -> SDLCCheck;
    SDLCCheck -> Review [condition="outcome=success"];
    SDLCCheck -> Generate [condition="outcome=fail", label="[F]ix"];
'''
    else:
        dot += '''    Score -> Review;
'''

    dot += '''    Review -> DeployTwin [label="[A]pprove"];
    DeployTwin -> Done;
    Review -> Generate [label="[R]etry"];
}
'''
    return dot


# ── File Saver ────────────────────────────────────────────────────────────────
def save_output_files(
    result,
    project_dir: Path,
    language: str,
    app_file_name: str
) -> list[Path]:
    """
    Parse ALL triple-backtick code blocks from Generate output.
    Saves each block as a separate named file.
    Extracts filename from inline comment: // filename: <name> OR # filename: <name>
    Falls back to smart naming by type + index counter.
    Also saves tests and auto-generates README.md.
    """
    # ── Try all possible context key names ───────────────────────────────────
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

    # ── DEBUG: dump every context key to log file ─────────────────────────────
    logger.debug("=== ALL CONTEXT KEYS ===")
    try:
        for key in result.context.keys():
            val = str(result.context.get_string(key, ""))[:200]
            logger.debug(f"  KEY [{key}] => {val!r}")
    except Exception as e:
        logger.warning(f"Could not iterate context keys: {e}")
    logger.debug("========================")

    if not generate_output:
        logger.error("Generate output is EMPTY — context key mismatch. Check attractor_agent.log.")
        console.print(
            "[bold red]❌ No generated code found in pipeline context.\n"
            "   Check attractor_agent.log for available KEY names.[/bold red]"
        )
        return []

    # ── Parse ALL code blocks ─────────────────────────────────────────────────
    code_blocks = re.findall(
        r"```([a-zA-Z0-9+\-#]*)\s*(.*?)```",
        generate_output,
        re.DOTALL
    )
    logger.info(f"Found {len(code_blocks)} code block(s) in Generate output")

    ext_counters: dict[str, int] = {}
    saved_files: list[Path] = []

    if code_blocks:
        for lang_tag, code in code_blocks:
            code = code.strip()
            if not code:
                continue

            ext = get_file_extension_from_tag(lang_tag, language)

            # ── Extract filename from first 3 lines of block ──────────────────
            filename: str | None = None
            lines = code.splitlines()
            for line in lines[:3]:
                m = re.match(
                    r'^(?://|#|/\*)\s*filename:\s*(.+?)(?:\s*\*/)?$',
                    line.strip(),
                    re.IGNORECASE
                )
                if m:
                    filename = m.group(1).strip()
                    # Strip the filename comment from saved code
                    code = "\n".join(
                        line_content for line_content in lines
                        if not re.match(
                            r'^(?://|#|/\*)\s*filename:',
                            line_content.strip(),
                            re.IGNORECASE
                        )
                    ).strip()
                    break

            # ── Fallback to smart naming ──────────────────────────────────────
            if not filename:
                count = ext_counters.get(ext, 0)
                filename = get_smart_filename(lang_tag, count, language)
                ext_counters[ext] = count + 1

            fpath = project_dir / filename
            try:
                os.makedirs(fpath.parent, exist_ok=True)
                fpath.write_text(code, encoding="utf-8")
                saved_files.append(fpath)
                logger.info(f"Saved: {fpath} ({len(code):,} chars)")
                console.print(
                    f"  [green]✓[/green] Saved [bold]{filename}[/bold] "
                    f"[dim]({len(code):,} chars)[/dim]"
                )
            except OSError as e:
                logger.error(f"Failed to save {fpath}: {e}")
                console.print(f"  [red]✗[/red] Could not save {filename}: {e}")

    else:
        # ── No code blocks found — save raw output as fallback ────────────────
        logger.warning("No triple-backtick code blocks found — saving raw output")
        raw_file = project_dir / app_file_name
        raw_file.write_text(generate_output, encoding="utf-8")
        console.print(
            f"[yellow]⚠ No code blocks found — saved raw output to "
            f"[bold]{raw_file.name}[/bold][/yellow]"
        )
        saved_files.append(raw_file)

    # ── Save test file ────────────────────────────────────────────────────────
    if tests_output:
        test_blocks = re.findall(
            r"```[a-zA-Z0-9+\-#]*\s*(.*?)```",
            tests_output,
            re.DOTALL
        )
        # FIX: test_blocks is a list — use  not .strip() directly
        test_code = test_blocks[0].strip() if test_blocks else tests_output.strip()
        test_ext = get_extension(language)
        test_file = project_dir / f"test_main{test_ext}"
        try:
            test_file.write_text(test_code, encoding="utf-8")
            saved_files.append(test_file)
            logger.info(f"Tests saved: {test_file} ({len(test_code):,} chars)")
            console.print(
                f"  [green]✓[/green] Tests saved [bold]{test_file.name}[/bold] "
                f"[dim]({len(test_code):,} chars)[/dim]"
            )
        except OSError as e:
            logger.error(f"Failed to save tests: {e}")

    # ── Auto-generate README.md ───────────────────────────────────────────────
    run_instructions = {
        "python":     "```bash\npip install -r requirements.txt\npython main.py\n```",
        "javascript": "```bash\nnpm install\nnode main.js\n```",
        "typescript": "```bash\nnpm install\nnpx ts-node main.ts\n```",
        "java":       "```bash\njavac Main.java\njava Main\n```",
        "c++":        "```bash\ng++ -std=c++17 -o app main.cpp\n./app\n```",
        "go":         "```bash\ngo run main.go\n```",
        "rust":       "```bash\ncargo run\n```",
        "html":       "Open `index.html` in your browser.",
    }
    run_cmd = run_instructions.get(language.lower(), "See language documentation.")

    readme_lines = [
        "# Generated Project\n\n",
        "**Built by:** Attractor Agent 🚀\n",
        f"**Language:** {language}\n\n",
        "## Project Files\n\n",
    ]
    for f in saved_files:
        readme_lines.append(f"- `{f.name}`\n")

    readme_lines.append(f"\n## How to Run\n\n{run_cmd}\n")

    readme_file = project_dir / "README.md"
    try:
        readme_file.write_text("".join(readme_lines), encoding="utf-8")
        logger.info(f"README saved: {readme_file}")
        console.print("  [green]✓[/green] README saved [bold]README.md[/bold]")
    except OSError as e:
        logger.error(f"Failed to save README: {e}")

    return saved_files


# ── Rich Interviewer ──────────────────────────────────────────────────────────
class RichInterviewer(Interviewer):
    """Handles hexagon human-review gate prompts in the Rich TUI."""

    def __init__(self, console_lock: threading.Lock, status: Status):
        self.console_lock = console_lock
        self.status = status

    def ask(self, question: Question) -> Answer:
        logger.info(f"Human review gate triggered: node={question.node_id}")
        self.status.stop()
        with self.console_lock:
            console.print("\n")
            console.print(Panel(
                Markdown(question.text),
                title=f"[bold yellow]👁 Human Review[/bold yellow] "
                      f"[dim]({question.node_id})[/dim]",
                border_style="yellow",
                padding=(1, 2)
            ))
            options_text = "  ".join(
                f"[[bold]{option.key.upper()}[/bold]] {option.label}"
                for option in question.options
            )
            console.print(f"\n[bold cyan]Options:[/bold cyan]  {options_text}\n")
            valid_keys = [option.key.lower() for option in question.options]

            while True:
                choice = Prompt.ask("[bold green]Your choice[/bold green]").strip().lower()
                for opt in question.options:
                    if (
                        choice == opt.key.lower()
                        or choice == opt.label.lower()
                        or (choice and opt.label.lower().startswith(choice))
                    ):
                        logger.info(f"User selected: '{opt.label}'")
                        self.status.start()
                        return Answer(
                            question_id=question.id,
                            selected_label=opt.label
                        )
                console.print(
                    f"[red]❌ Invalid choice.[/red] "
                    f"Valid keys: [bold]{', '.join(valid_keys)}[/bold]"
                )


# ── Main CLI ──────────────────────────────────────────────────────────────────
def run_cli() -> None:
    console.print(Panel(
        "Welcome to [bold magenta]Attractor Agent[/bold magenta] 🚀\n"
        "[dim]Full SDLC AI pipeline — Plan → Generate → Test → Review → Done[/dim]",
        expand=False,
        border_style="magenta",
        padding=(1, 4)
    ))
    logger.info("=" * 60)
    logger.info("Attractor Agent CLI started")
    logger.info("=" * 60)

    # ── Collect user inputs ───────────────────────────────────────────────────
    request = Prompt.ask("[bold cyan]What do you want to build?[/bold cyan]").strip()
    if not request:
        console.print("[red]No request provided. Exiting.[/red]")
        logger.error("Empty build request — exiting")
        sys.exit(1)

    language = Prompt.ask(
        "[bold cyan]Programming language?[/bold cyan]",
        choices=["Python", "JavaScript", "TypeScript", "HTML",
                 "Go", "Rust", "C++", "Java"],
        default="Python"
    )
    framework = Prompt.ask(
        "[bold cyan]Framework? (Enter to skip)[/bold cyan]",
        default=""
    ).strip()
    include_tests = Prompt.ask(
        "[bold cyan]Include unit tests? (y/n)[/bold cyan]",
        default="y"
    ).strip().lower() == "y"
    include_sdlc = Prompt.ask(
        "[bold cyan]Include SDLC review? (y/n)[/bold cyan]",
        default="y"
    ).strip().lower() == "y"
    use_mock = Prompt.ask(
        "[bold cyan]Use Mock LLM for testing? (localhost:5555) (y/n)[/bold cyan]",
        default="n"
    ).strip().lower() == "y"

    logger.info(
        f"Config — request='{request[:80]}' lang={language} "
        f"fw='{framework}' tests={include_tests} "
        f"sdlc={include_sdlc} mock={use_mock}"
    )

    # ── Project directory setup ───────────────────────────────────────────────
    slug = slugify(request)[:30] or "project"
    project_dir = Path("projects") / slug
    os.makedirs(project_dir, exist_ok=True)
    app_file_name = f"main{get_extension(language)}"
    logger.info(f"Project dir: {project_dir} | Main file: {app_file_name}")
    console.print(f"[dim]📁 Project folder: [bold]{project_dir}[/bold][/dim]")

    # ── Build and save DOT ────────────────────────────────────────────────────
    dot_content = build_dot(request, language, framework, include_tests, include_sdlc)
    dot_file = project_dir / "pipeline.dot"
    dot_file.write_text(dot_content, encoding="utf-8")
    console.print(f"[dim]🔧 Pipeline DOT saved: {dot_file}[/dim]")
    logger.info(f"DOT saved: {dot_file}")

    # ── Backend setup: Mock or Real ───────────────────────────────────────────
    if use_mock:
        console.print(
            "\n[bold yellow]🚀 Starting Mock LLM server on port 5555...[/bold yellow]"
        )
        logger.info("Launching mock LLM: npx -y @copilotkit/llmock --port 5555")

        # Use shell=True with a string command — fixes Windows npx.cmd issue
        mock_process = subprocess.Popen(
            "npx -y @copilotkit/llmock --port 5555",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Ensure mock server is killed when CLI exits
        atexit.register(lambda: mock_process.terminate())

        try:
            # Wait until the port is actually accepting connections
            wait_for_port("localhost", 5555, timeout=45)
            console.print("[dim]✓ Mock LLM server is ready.[/dim]\n")
        except RuntimeError as e:
            logger.error(f"Mock server startup failed: {e}")
            console.print(f"[bold red]❌ {e}[/bold red]")
            sys.exit(1)

        from attractor.llm.client import Client
        from attractor.llm.adapters.openai import OpenAIAdapter

        mock_adapter = OpenAIAdapter(
            api_key="mock-key",
            base_url="http://localhost:5555/v1"
        )
        client = Client(providers={"mock": mock_adapter}, default_provider="mock")
        backend = LLMBackend(client=client)
        logger.info("Mock LLM backend ready at http://localhost:5555/v1")

    else:
        backend = LLMBackend()
        logger.info("Using default LLMBackend (OpenRouter / configured provider)")

    # ── Pipeline setup ────────────────────────────────────────────────────────
    console_lock = threading.Lock()
    status = Status(
        "[bold green]Running pipeline...[/bold green]",
        console=console
    )
    interviewer = RichInterviewer(console_lock, status)

    config = PipelineConfig(
        simulate=False,
        codergen_backend=backend,
        interviewer=interviewer,
        checkpoint_dir=str(project_dir)
    )

    emitter = EventEmitter()

    # ── Event handler ─────────────────────────────────────────────────────────
    @emitter.on
    def on_event(event: PipelineEvent) -> None:
        msg = event.message or ""
        node = event.node_id
        if event.kind == PipelineEventKind.STAGE_STARTED:
            logger.info(f"[STAGE STARTED] {node}")
            with console_lock:
                status.update(f"[bold blue]⏳ Running:[/bold blue] {node}")
        elif event.kind == PipelineEventKind.STAGE_COMPLETED:
            logger.info(f"[STAGE COMPLETED] {node}")
            with console_lock:
                console.print(f"  [green]✓[/green] Completed: [bold]{node}[/bold]")
        elif event.kind == PipelineEventKind.STAGE_FAILED:
            logger.error(f"[STAGE FAILED] {node} — {msg}")
            with console_lock:
                console.print(
                    f"  [red]✗[/red] Failed: [bold]{node}[/bold] "
                    f"[dim]— {msg}[/dim]"
                )

    # ── Execute pipeline ──────────────────────────────────────────────────────
    console.print(
        f"\n[bold]Starting pipeline for:[/bold] [magenta]{request}[/magenta]\n"
    )
    status.start()
    try:
        logger.info("Pipeline execution started")
        result = run_pipeline(dot_content, config=config, emitter=emitter)
    except Exception as e:
        logger.exception(f"Pipeline crashed with unhandled exception: {e}")
        console.print(f"\n[bold red]❌ Pipeline crashed: {e}[/bold red]")
        sys.exit(1)
    finally:
        status.stop()

    # ── Handle result ─────────────────────────────────────────────────────────
    if result.success:
        logger.info(
            f"Pipeline SUCCESS — "
            f"elapsed={result.elapsed_seconds:.2f}s "
            f"steps={result.total_steps}"
        )
        console.print(
            f"\n[bold green]✅ Pipeline complete![/bold green]  "
            f"Elapsed: [bold]{result.elapsed_seconds:.2f}s[/bold]  "
            f"Steps: [bold]{result.total_steps}[/bold]\n"
        )

        console.print("[bold]Saving generated files...[/bold]")
        saved = save_output_files(result, project_dir, language, app_file_name)

        if saved:
            console.print(
                f"\n[bold green]📦 Project '[cyan]{slug}[/cyan]' "
                f"saved to [cyan]{project_dir}[/cyan][/bold green]\n"
            )
            print_project_tree(project_dir)
        else:
            console.print(
                "[bold yellow]⚠ Pipeline succeeded but no files were saved. "
                "Check attractor_agent.log.[/bold yellow]"
            )

    else:
        logger.error(f"Pipeline FAILED: {result.error}")
        console.print(f"\n[bold red]❌ Pipeline failed:[/bold red] {result.error}")
        console.print(
            "[dim]→ Check [bold]attractor_agent.log[/bold] for full details[/dim]"
        )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_cli()
