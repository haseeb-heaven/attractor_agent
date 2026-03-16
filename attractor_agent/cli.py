from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status
from rich.tree import Tree

from attractor.pipeline.events import EventEmitter, PipelineEvent, PipelineEventKind
from attractor.pipeline.interviewer import Answer, Interviewer, Question

from attractor_agent.extraction import ExtractedBlock, extract_blocks_with_fallbacks
from attractor_agent.project import (
    BuildRequest,
    SUPPORTED_LANGUAGES,
    build_dot as _project_build_dot,
    get_extension,
    get_gradio_language,
    load_build_request,
    slugify,
)
from attractor_agent.runtime import (
    _default_filename_for_language,
    execute_build,
    get_project_dir,
    get_smart_filename,
    save_output_files,
    wait_for_port,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        RichHandler(rich_tracebacks=True, show_path=False),
        logging.FileHandler("attractor_agent.log", encoding="utf-8"),
    ],
)

logger = logging.getLogger("attractor_agent.cli")
console = Console()


def build_dot(
    request: str | BuildRequest,
    language: str | None = None,
    framework: str = "",
    include_tests: bool = True,
    include_sdlc: bool = True,
    attempt: int = 0,
) -> str:
    if isinstance(request, BuildRequest):
        spec = request
    else:
        spec = BuildRequest(
            request=request,
            language=language or "Python",
            framework=framework,
            include_tests=include_tests,
            include_sdlc=include_sdlc,
        )
    return _project_build_dot(spec, attempt=attempt)


class RichInterviewer(Interviewer):
    """Handles review prompts in the Rich CLI."""

    def __init__(self, console_lock: threading.Lock, status: Status):
        self.console_lock = console_lock
        self.status = status

    def ask(self, question: Question) -> Answer:
        self.status.stop()
        with self.console_lock:
            console.print()
            console.print(
                Panel(
                    Markdown(question.text),
                    title=f"[bold yellow]Human Review[/bold yellow] [dim]({question.node_id})[/dim]",
                    border_style="yellow",
                    padding=(1, 2),
                )
            )
            options_text = "  ".join(
                f"[[bold]{option.key.upper()}[/bold]] {option.label}"
                for option in question.options
            )
            console.print(f"\n[bold cyan]Options:[/bold cyan] {options_text}\n")
            valid_keys = [option.key.lower() for option in question.options]

            while True:
                choice = Prompt.ask("[bold green]Your choice[/bold green]").strip().lower()
                for option in question.options:
                    if (
                        choice == option.key.lower()
                        or choice == option.label.lower()
                        or (choice and option.label.lower().startswith(choice))
                    ):
                        self.status.start()
                        return Answer(question_id=question.id, selected_label=option.label)
                console.print(
                    f"[red]Invalid choice.[/red] Valid keys: [bold]{', '.join(valid_keys)}[/bold]"
                )


def print_project_tree(project_dir: Path) -> None:
    tree = Tree(f"[bold green]{project_dir}[/bold green]")
    for file_path in sorted(project_dir.rglob("*")):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(project_dir).as_posix()
        size = file_path.stat().st_size
        tree.add(f"[cyan]{relative}[/cyan] [dim]({size:,} bytes)[/dim]")
    console.print(tree)


def _build_spec_from_args(args: argparse.Namespace) -> BuildRequest:
    if args.config:
        spec = load_build_request(args.config)
    elif args.request:
        spec = BuildRequest(
            request=args.request,
            language=args.language or "Python",
            framework=args.framework or "",
            include_tests=True if args.include_tests is None else args.include_tests,
            include_sdlc=True if args.include_sdlc is None else args.include_sdlc,
            use_mock=args.use_mock,
            auto_approve=args.auto_approve,
            require_human_review=args.require_human_review,
            retry_save_attempts=args.retry_save_attempts or 3,
            project_name=args.project_name or "",
            checkpoint_dir=args.project_dir or "",
        )
    else:
        raise ValueError("A request or --config file is required.")

    if args.language is not None:
        spec.language = args.language
    if args.framework is not None:
        spec.framework = args.framework
    if args.project_name:
        spec.project_name = args.project_name
    if args.project_dir:
        spec.checkpoint_dir = args.project_dir
    if args.use_mock is True:
        spec.use_mock = True
    if args.require_human_review is True:
        spec.require_human_review = True
    if args.auto_approve is True:
        spec.auto_approve = True
    if args.no_auto_approve:
        spec.auto_approve = False
    if args.retry_save_attempts is not None:
        spec.retry_save_attempts = args.retry_save_attempts
    return spec


def _interactive_spec() -> BuildRequest:
    request = Prompt.ask("[bold cyan]What do you want to build?[/bold cyan]").strip()
    if not request:
        raise ValueError("No request provided.")

    language = Prompt.ask(
        "[bold cyan]Programming language?[/bold cyan]",
        choices=SUPPORTED_LANGUAGES,
        default="Python",
    )
    framework = Prompt.ask("[bold cyan]Framework? (Enter to skip)[/bold cyan]", default="").strip()
    include_tests = Prompt.ask("[bold cyan]Include unit tests? (y/n)[/bold cyan]", default="y").lower() == "y"
    include_sdlc = Prompt.ask("[bold cyan]Include SDLC review? (y/n)[/bold cyan]", default="y").lower() == "y"
    use_mock = Prompt.ask("[bold cyan]Use mock LLM? (y/n)[/bold cyan]", default="n").lower() == "y"
    require_human_review = (
        Prompt.ask("[bold cyan]Require human review before deploy? (y/n)[/bold cyan]", default="n").lower() == "y"
    )
    auto_approve = not require_human_review

    return BuildRequest(
        request=request,
        language=language,
        framework=framework,
        include_tests=include_tests,
        include_sdlc=include_sdlc,
        use_mock=use_mock,
        auto_approve=auto_approve,
        require_human_review=require_human_review,
    )


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Attractor Agent CLI")
    parser.add_argument("--request", help="What to build")
    parser.add_argument("--config", help="Path to a JSON or TOML build config")
    parser.add_argument("--language", default=None, help="Target language")
    parser.add_argument("--framework", default=None, help="Framework name")
    parser.add_argument("--include-tests", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--include-sdlc", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-mock", action="store_true", help="Use the local mock LLM server")
    parser.add_argument("--auto-approve", action="store_true", help="Auto-approve review steps")
    parser.add_argument("--no-auto-approve", action="store_true", help="Disable auto-approve")
    parser.add_argument("--require-human-review", action="store_true", help="Insert a blocking review node")
    parser.add_argument("--retry-save-attempts", type=int, default=None, help="Max extraction retry attempts")
    parser.add_argument("--project-name", help="Optional project slug override")
    parser.add_argument("--project-dir", help="Optional project output directory")
    parser.add_argument("--interactive", action="store_true", help="Force interactive prompts")
    return parser


def run_cli(argv: list[str] | None = None) -> int:
    parser = _create_parser()
    args = parser.parse_args(argv)

    console.print(
        Panel(
            "Attractor Agent\n[dim]Autonomous SDLC pipeline with optional human review[/dim]",
            expand=False,
            border_style="magenta",
            padding=(1, 4),
        )
    )

    try:
        if args.interactive or not (args.request or args.config):
            spec = _interactive_spec()
        else:
            spec = _build_spec_from_args(args)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    console.print(f"[dim]Project folder: [bold]{get_project_dir(spec)}[/bold][/dim]")

    console_lock = threading.Lock()
    status = Status("[bold green]Running pipeline...[/bold green]", console=console)
    interviewer: Interviewer | None = None
    if spec.require_human_review and not spec.auto_approve:
        interviewer = RichInterviewer(console_lock, status)

    emitter = EventEmitter()

    @emitter.on
    def on_event(event: PipelineEvent) -> None:
        node = event.node_id
        message = event.message or ""
        with console_lock:
            if event.kind == PipelineEventKind.STAGE_STARTED:
                status.update(f"[bold blue]Running:[/bold blue] {node}")
            elif event.kind == PipelineEventKind.STAGE_COMPLETED:
                console.print(f"  [green]✓[/green] Completed: [bold]{node}[/bold]")
            elif event.kind == PipelineEventKind.STAGE_FAILED:
                console.print(f"  [red]✗[/red] Failed: [bold]{node}[/bold] [dim]{message}[/dim]")

    status.start()
    try:
        artifacts = execute_build(spec, interviewer=interviewer, emitter=emitter)
    except Exception as exc:
        status.stop()
        logger.exception("Build execution failed: %s", exc)
        console.print(f"[bold red]Pipeline crashed:[/bold red] {exc}")
        return 1
    finally:
        status.stop()

    result = artifacts.result
    if not result.success:
        console.print(f"[bold red]Pipeline failed:[/bold red] {result.error}")
        return 1

    if not artifacts.saved_files:
        console.print(
            f"[bold yellow]Pipeline completed but no files were extracted after {artifacts.attempts_used} attempt(s).[/bold yellow]"
        )
        return 1

    console.print(
        f"[bold green]Pipeline complete.[/bold green] Saved {len(artifacts.saved_files)} files after {artifacts.attempts_used} attempt(s)."
    )
    print_project_tree(artifacts.project_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
