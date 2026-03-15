from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    cli_flags = {
        "--request",
        "--config",
        "--language",
        "--framework",
        "--include-tests",
        "--no-include-tests",
        "--include-sdlc",
        "--no-include-sdlc",
        "--use-mock",
        "--auto-approve",
        "--no-auto-approve",
        "--require-human-review",
        "--retry-save-attempts",
        "--project-name",
        "--project-dir",
        "--interactive",
    }

    argv = sys.argv[1:]
    if any(arg in cli_flags for arg in argv):
        from attractor_agent.cli import main as cli_main

        raise SystemExit(cli_main(argv))

    parser = argparse.ArgumentParser(description="Attractor Agent")
    parser.add_argument("dot_file", nargs="?", help="Optional DOT file to execute directly")
    parser.add_argument("--gui", action="store_true", help="Launch the Gradio GUI")
    parser.add_argument("--api", action="store_true", help="Launch the REST API")
    parser.add_argument("--port", type=int, default=None, help="Port for GUI or API")
    args, remaining = parser.parse_known_args()

    if args.api:
        try:
            import uvicorn

            from attractor_agent.api import app

            uvicorn.run(app, host="0.0.0.0", port=args.port or 8000)
            return
        except ImportError as exc:
            print(f"Failed to start API: {exc}")
            raise SystemExit(1)

    if args.gui:
        try:
            from attractor_agent.gui import run_gui

            run_gui(port=args.port or 8000)
            return
        except ImportError as exc:
            print(f"Failed to start GUI: {exc}")
            raise SystemExit(1)

    if args.dot_file and not remaining:
        from attractor.pipeline.engine import run_pipeline

        dot_path = Path(args.dot_file)
        if not dot_path.exists():
            print(f"Error: File not found: {args.dot_file}")
            raise SystemExit(1)
        result = run_pipeline(dot_path.read_text(encoding="utf-8"))
        if not result.success:
            print(f"Pipeline failed: {result.error}")
            raise SystemExit(1)
        print("Pipeline completed successfully.")
        return

    from attractor_agent.cli import main as cli_main

    raise SystemExit(cli_main(([args.dot_file] if args.dot_file else []) + remaining))


if __name__ == "__main__":
    main()
