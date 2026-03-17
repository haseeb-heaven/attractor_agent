"""Handler for compile checks on compiled languages."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from attractor.pipeline.context import Context, Outcome, StageStatus
from attractor.pipeline.events import EventEmitter, PipelineEventKind
from attractor.pipeline.graph import Graph, Node
from attractor.pipeline.handlers.base import Handler
from attractor_agent.extraction import extract_blocks_with_fallbacks


COMPILED_EXTS = {".go", ".rs", ".java", ".cpp", ".cc", ".cxx"}


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


class CompileExecutionHandler(Handler):
    """Compile generated code in a temp sandbox for compiled languages."""

    def execute(self, node: Node, context: Context, graph: Graph,
                emitter: EventEmitter, **kwargs: Any) -> Outcome:
        generate_output = (
            context.get_string("Generate.output", "")
            or context.get_string("generate_output", "")
            or context.get_string("last_response", "")
        )

        if not generate_output:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason="Missing Generate.output in context",
            )

        blocks = extract_blocks_with_fallbacks(generate_output)
        if not blocks:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason="No code blocks found for compile check",
            )

        filenames: list[Path] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for idx, block in enumerate(blocks):
                filename = (
                    block.attribute_filename
                    or block.filename_comment
                    or block.header_filename
                    or f"file_{idx + 1}.txt"
                )
                safe_rel = _safe_relative_path(filename)
                if safe_rel is None:
                    emitter.emit_simple(
                        PipelineEventKind.LOG,
                        message=f"Skipping unsafe filename in compile sandbox: {filename}",
                    )
                    continue
                file_path = temp_path / safe_rel
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(block.code, encoding="utf-8")
                filenames.append(file_path)

            compiled_files = [p for p in filenames if p.suffix.lower() in COMPILED_EXTS]
            if not compiled_files:
                return Outcome(status=StageStatus.SUCCESS, notes="No compiled files found.")

            cmd = None
            if any(p.suffix.lower() == ".go" for p in compiled_files):
                cmd = "go build ./..."
            elif any(p.suffix.lower() == ".rs" for p in compiled_files):
                if (temp_path / "Cargo.toml").exists():
                    cmd = "cargo build"
                else:
                    target = compiled_files[0].name
                    cmd = f"rustc {target} -o app"
            elif any(p.suffix.lower() == ".java" for p in compiled_files):
                cmd = "javac *.java"
            elif any(p.suffix.lower() in {".cpp", ".cc", ".cxx"} for p in compiled_files):
                sources = " ".join(p.name for p in compiled_files if p.suffix.lower() in {".cpp", ".cc", ".cxx"})
                cmd = f"g++ -std=c++17 -o app {sources}"

            if not cmd:
                return Outcome(status=StageStatus.SUCCESS, notes="No compile command matched.")

            emitter.emit_simple(
                PipelineEventKind.LOG,
                node_id=node.id,
                message=f"Compile check command: {cmd} in {temp_dir}",
            )

            try:
                result = subprocess.run(
                    cmd,
                    cwd=temp_dir,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                updates = {
                    f"{node.id}.stdout": result.stdout,
                    f"{node.id}.stderr": result.stderr,
                    f"{node.id}.returncode": result.returncode,
                }

                if result.returncode == 0:
                    return Outcome(
                        status=StageStatus.SUCCESS,
                        context_updates=updates,
                        notes="Compilation succeeded",
                    )

                error_msg = result.stderr.strip() or result.stdout.strip()
                return Outcome(
                    status=StageStatus.FAIL,
                    failure_reason=f"Compilation failed (code {result.returncode}):\n{error_msg[-500:]}",
                    context_updates=updates,
                )
            except subprocess.TimeoutExpired:
                return Outcome(
                    status=StageStatus.FAIL,
                    failure_reason="Compilation timed out after 60 seconds",
                )
            except Exception as exc:
                return Outcome(
                    status=StageStatus.FAIL,
                    failure_reason=f"Compilation error: {exc}",
                )
