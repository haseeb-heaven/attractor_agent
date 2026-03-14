import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Attractor Agent - Conversational AI Builder")
    parser.add_argument("--gui", action="store_true", help="Launch the Web GUI")
    
    args = parser.parse_args()
    
    if args.gui:
        try:
            from attractor_agent.gui import run_gui
            run_gui()
        except ImportError as e:
            print(f"Failed to load GUI: {e}")
            print("Please ensure you have gradio installed: pip install gradio")
            sys.exit(1)
    else:
        from attractor_agent.cli import run_cli
        run_cli()

if __name__ == "__main__":
    main()
