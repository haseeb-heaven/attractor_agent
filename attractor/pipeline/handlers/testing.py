"""Handler for real test execution."""

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from attractor.pipeline.context import Context, Outcome, StageStatus
from attractor.pipeline.events import EventEmitter, PipelineEventKind
from attractor.pipeline.graph import Graph, Node
from attractor.pipeline.handlers.base import Handler


def _extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """Extract (language, code) from markdown text."""
    blocks = re.findall(r"```([a-zA-Z0-9+\-#]*)\s*(.*?)```", text, re.DOTALL)
    # Strip whitespace from code
    return [(lang, code.strip()) for lang, code in blocks if code.strip()]


def _extract_filename(code: str) -> str | None:
    """Extract filename from first few lines of code comment."""
    lines = code.splitlines()
    for line in lines[:3]:
        m = re.match(
            r'^(?://|#|/\*)\s*filename:\s*(.+?)(?:\s*\*/)?$',
            line.strip(),
            re.IGNORECASE
        )
        if m:
            return m.group(1).strip()
    return None


def _safe_relative_path(filename: str) -> Path | None:
    raw = Path(filename)
    if raw.is_absolute():
        return None
    if raw.drive or (raw.parts and raw.parts[0].endswith(":")):
        return None
    normalized = Path(*[part for part in raw.parts if part not in ("", ".")])
    if any(part == ".." for part in normalized.parts):
        return None
    return normalized


class TestExecutionHandler(Handler):
    """Executes generated unit tests against generated code in a sandbox."""

    def execute(self, node: Node, context: Context, graph: Graph,
                emitter: EventEmitter, **kwargs: Any) -> Outcome:
        
        # 1. Retrieve the latest code and test output from Context
        # Using a fallback hierarchy
        generate_output = (
            context.get_string("Generate.output", "")
            or context.get_string("generate_output", "")
            or context.get_string("last_response", "")
        )
        
        tests_output = context.get_string("Tests.output", "")

        if not generate_output or not tests_output:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason="Missing Generate.output or Tests.output in context"
            )

        # 2. Extract code blocks
        code_blocks = _extract_code_blocks(generate_output)
        test_blocks = _extract_code_blocks(tests_output)

        if not code_blocks:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason="No code blocks found in generated output"
            )
        
        if not test_blocks:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason="No test blocks found in Tests output"
            )

        # Determine language broadly from the first test block
        test_lang = test_blocks[0][0].lower()

        # 3. Create a temporary sandbox directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Write application code files
            for lang, code in code_blocks:
                filename = _extract_filename(code)
                if not filename:
                    # Very simple fallback for unsaved logic
                    ext = ".py" if "py" in lang else ".js" if "js" in lang else ".txt"
                    filename = f"main{ext}"
                try:
                    safe_rel = _safe_relative_path(filename)
                    if safe_rel is None:
                        emitter.emit_simple(
                            PipelineEventKind.LOG,
                            message=f"Skipping unsafe code filename in sandbox: {filename}",
                        )
                        continue
                    file_path = temp_path / safe_rel
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(code, encoding="utf-8")
                except OSError as e:
                    emitter.emit_simple(PipelineEventKind.LOG, message=f"Failed to write code file in sandbox: {e}")

            # Write test code files
            for lang, code in test_blocks:
                filename = _extract_filename(code)
                if not filename:
                    ext = ".py" if "py" in lang else ".js" if "js" in lang else ".txt"
                    filename = f"test_main{ext}"
                try:
                    safe_rel = _safe_relative_path(filename)
                    if safe_rel is None:
                        emitter.emit_simple(
                            PipelineEventKind.LOG,
                            message=f"Skipping unsafe test filename in sandbox: {filename}",
                        )
                        continue
                    file_path = temp_path / safe_rel
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(code, encoding="utf-8")
                except OSError as e:
                    emitter.emit_simple(PipelineEventKind.LOG, message=f"Failed to write test file in sandbox: {e}")

            # 4. Determine test command based on language
            cmd = ""
            if "py" in test_lang:
                # Assuming pytest is available in the environment
                cmd = "python -m pytest"
            elif "js" in test_lang or "javascript" in test_lang:
                cmd = "npm test" # Usually expects package.json, might need fallback
                if not (temp_path / "package.json").exists():
                    cmd = "node --test" # Node.js 18+ built-in test runner
            elif "ts" in test_lang or "typescript" in test_lang:
                cmd = "npx jest"
            elif "go" in test_lang:
                cmd = "go test ./..."
            elif "rust" in test_lang or "rs" in test_lang:
                cmd = "cargo test"
            elif "java" in test_lang:
                # Basic java compile and run assuming JUnit isn't fully stubbed easily
                # Realistically requires maven/gradle, fallback to just compile check
                cmd = "javac *.java"
            else:
                return Outcome(
                    status=StageStatus.FAIL,
                    failure_reason=f"Unsupported testing language: {test_lang}"
                )

            emitter.emit_simple(
                PipelineEventKind.LOG,
                node_id=node.id,
                message=f"Running tests with command: {cmd} in {temp_dir}"
            )

            # 5. Execute tests
            try:
                result = subprocess.run(
                    cmd,
                    cwd=temp_dir,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=60, # 1 minute max for tests
                )
                
                # Expose test output back to the context
                updates = {
                    f"{node.id}.stdout": result.stdout,
                    f"{node.id}.stderr": result.stderr,
                    f"{node.id}.returncode": result.returncode,
                }

                if result.returncode == 0:
                    return Outcome(
                        status=StageStatus.SUCCESS,
                        context_updates=updates,
                        notes="Tests passed successfully"
                    )
                else:
                    error_msg = result.stderr.strip() or result.stdout.strip()
                    return Outcome(
                        status=StageStatus.FAIL,
                        failure_reason=f"Tests failed (code {result.returncode}):\n{error_msg[-500:]}",
                        context_updates=updates
                    )

            except subprocess.TimeoutExpired:
                return Outcome(
                    status=StageStatus.FAIL,
                    failure_reason="Test execution timed out after 60 seconds",
                )
            except Exception as e:
                return Outcome(
                    status=StageStatus.FAIL,
                    failure_reason=f"Test execution error: {e}",
                )
