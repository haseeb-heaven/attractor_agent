"""CLI entry point for Attractor.

Commands:
    attractor run <file.dot>       Execute a pipeline
    attractor validate <file.dot>  Validate a pipeline
    attractor serve                Start HTTP server mode
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from attractor.pipeline.engine import PipelineConfig, run_pipeline
from attractor.pipeline.events import PipelineEvent
from attractor.pipeline.interviewer import AutoApproveInterviewer, ConsoleInterviewer
from attractor.pipeline.parser import parse_dot
from attractor.pipeline.validator import Severity, validate
from attractor.pipeline.backend import LLMBackend
import subprocess
import logging

logger = logging.getLogger(__name__)

# Kill any existing llmock on port 5555 before starting
def kill_port(port: int):
    """Kill any process occupying the given port."""
    try:
        result = subprocess.run(
            f'netstat -ano | findstr :{port}',
            shell=True, capture_output=True, text=True
        )
        for line in result.stdout.strip().splitlines():
            parts = line.strip().split()
            if parts:
                pid = parts[-1]
                subprocess.run(f'taskkill /PID {pid} /F', shell=True, capture_output=True)
                logger.info(f"Killed process {pid} on port {port}")
    except Exception as e:
        logger.warning(f"Could not kill port {port}: {e}")

# Call before launching mock
kill_port(5555)


def _log_event(event: PipelineEvent) -> None:
    """Default event logger — prints to stderr."""
    kind = event.kind.value
    node = f" [{event.node_id}]" if event.node_id else ""
    msg = event.message or ""
    data_str = ""
    if event.data:
        data_str = " " + json.dumps(event.data, default=str)[:200]
    print(f"  [{kind}]{node} {msg}{data_str}", file=sys.stderr)


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a DOT pipeline file."""
    path = Path(args.file)
    if not path.exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        return 1

    source = path.read_text(encoding="utf-8")

    try:
        graph = parse_dot(source)
    except SyntaxError as e:
        print(f"Parse error: {e}", file=sys.stderr)
        return 1

    diagnostics = validate(graph)

    if not diagnostics:
        print(f"✓ Pipeline '{graph.name}' is valid ({len(graph.nodes)} nodes, {len(graph.edges)} edges)")
        return 0

    errors = [d for d in diagnostics if d.severity == Severity.ERROR]
    warnings = [d for d in diagnostics if d.severity == Severity.WARNING]

    for d in diagnostics:
        icon = "✗" if d.severity == Severity.ERROR else "⚠"
        node_info = f" (node: {d.node_id})" if d.node_id else ""
        edge_info = f" (edge: {d.edge})" if d.edge else ""
        print(f"  {icon} [{d.rule}] {d.message}{node_info}{edge_info}")

    if errors:
        print(f"\n✗ {len(errors)} error(s), {len(warnings)} warning(s)")
        return 1
    else:
        print(f"\n✓ Valid with {len(warnings)} warning(s)")
        return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run a DOT pipeline."""
    path = Path(args.file)
    if not path.exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        return 1

    source = path.read_text(encoding="utf-8")

    # Parse variables from command line
    variables: dict[str, str] = {}
    for v in (args.var or []):
        if "=" in v:
            key, val = v.split("=", 1)
            variables[key] = val

    # Set up interviewer (default to interactive if not auto-approving)
    if args.auto_approve:
        interviewer = AutoApproveInterviewer()
    else:
        # Default to interactive for a better out-of-the-box experience
        interviewer = ConsoleInterviewer()

    # Set up LLM backend
    backend = None
    if not args.simulate:
        backend = LLMBackend()

    config = PipelineConfig(
        simulate=args.simulate,
        variables=variables,
        checkpoint_dir=args.checkpoint_dir or "",
        resume_from=args.resume or "",
        goal=args.goal or "",
        interviewer=interviewer,
        codergen_backend=backend,
    )

    print(f"Running pipeline: {path.name}", file=sys.stderr)

    result = run_pipeline(source, config)

    print(f"\n{'='*60}")
    if result.success:
        print("✓ Pipeline completed successfully")
    else:
        print(f"✗ Pipeline failed: {result.error}")

    print(f"  Steps: {result.total_steps}")
    print(f"  Time:  {result.elapsed_seconds:.2f}s")
    print(f"  Nodes: {', '.join(result.completed_nodes)}")

    if result.final_node:
        print(f"  Final: {result.final_node}")

    return 0 if result.success else 1


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="attractor",
        description="Attractor: DOT-based AI pipeline orchestration",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # validate
    val_parser = subparsers.add_parser("validate", help="Validate a DOT pipeline")
    val_parser.add_argument("file", help="DOT file to validate")

    # run
    run_parser = subparsers.add_parser("run", help="Run a DOT pipeline")
    run_parser.add_argument("file", help="DOT file to run")
    run_parser.add_argument("--simulate", action="store_true",
                            help="Run in simulation mode (no LLM calls)")
    run_parser.add_argument("--var", action="append",
                            help="Set variable: --var key=value")
    run_parser.add_argument("--goal", help="Override pipeline goal")
    run_parser.add_argument("--checkpoint-dir", help="Directory for checkpoints")
    run_parser.add_argument("--resume", help="Resume from checkpoint file")
    run_parser.add_argument("--auto-approve", action="store_true",
                            help="Auto-approve all human review steps")
    run_parser.add_argument("--interactive", action="store_true",
                            help="Interactive mode for human review steps")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start HTTP server")
    serve_parser.add_argument("--port", type=int, default=8080)
    serve_parser.add_argument("--host", default="127.0.0.1")

    args = parser.parse_args()

    if args.command == "validate":
        sys.exit(cmd_validate(args))
    elif args.command == "run":
        sys.exit(cmd_run(args))
    elif args.command == "serve":
        print("HTTP server mode is not yet implemented.", file=sys.stderr)
        sys.exit(1)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
