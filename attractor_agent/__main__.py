import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Attractor Agent - Conversational AI Builder")
    parser.add_argument("dot_file", nargs="?", help="DOT file to execute")
    parser.add_argument("--gui", action="store_true", help="Launch the Web GUI")
    parser.add_argument("--api", action="store_true", help="Launch the REST API")
    parser.add_argument("--port", type=int, help="Port for GUI or API", default=None)
    
    args = parser.parse_args()
    
    if args.api:
        try:
            import uvicorn
            from attractor_agent.api import app
            port = args.port or 8000
            print(f"Starting REST API on port {port}...")
            uvicorn.run(app, host="0.0.0.0", port=port)
        except ImportError as e:
            print(f"Failed to load API: {e}")
            print("Please ensure you have fastapi and uvicorn installed.")
            sys.exit(1)
    elif args.gui:
        try:
            from attractor_agent.gui import run_gui
            run_gui()
        except ImportError as e:
            print(f"Failed to load GUI: {e}")
            print("Please ensure you have gradio installed: pip install gradio")
            sys.exit(1)
    elif args.dot_file:
        from attractor.pipeline.engine import run_pipeline
        from pathlib import Path
        dot_path = Path(args.dot_file)
        if not dot_path.exists():
            print(f"Error: File not found: {args.dot_file}")
            sys.exit(1)
        
        print(f"Running pipeline from {args.dot_file}...")
        result = run_pipeline(dot_path.read_text())
        if result.success:
            print("Pipeline completed successfully.")
            print(f"Nodes: {result.completed_nodes}")
        else:
            print(f"Pipeline failed: {result.error}")
            sys.exit(1)
    else:
        from attractor_agent.cli import run_cli
        run_cli()

if __name__ == "__main__":
    main()
