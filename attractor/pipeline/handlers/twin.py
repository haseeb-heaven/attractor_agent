"""Deployment artifact handler."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from attractor.pipeline.context import Context, Outcome, StageStatus
from attractor.pipeline.events import EventEmitter, PipelineEventKind
from attractor.pipeline.graph import Graph, Node
from attractor.pipeline.handlers.base import Handler
from attractor_agent.extraction import extract_blocks_with_fallbacks, infer_language_from_filename


def _launch_command(language: str) -> str:
    commands = {
        "python": "python main.py",
        "javascript": "node main.js",
        "typescript": "npx ts-node main.ts",
        "go": "go run main.go",
        "rust": "cargo run",
        "java": "javac Main.java && java Main",
        "cpp": "g++ -std=c++17 -o app main.cpp && ./app",
        "html": "open index.html",
    }
    return commands.get(language, "See generated README.md")


class DigitalTwinHandler(Handler):
    """Create deployment artifacts from generated output."""

    def execute(
        self,
        node: Node,
        context: Context,
        graph: Graph,
        emitter: EventEmitter,
        **kwargs: Any,
    ) -> Outcome:
        config = kwargs.get("config")
        project_dir = Path(config.checkpoint_dir) if config and config.checkpoint_dir else None
        if not project_dir:
            return Outcome(status=StageStatus.FAIL, failure_reason="Missing checkpoint_dir for deployment artifacts")

        generate_output = (
            context.get_string("Generate.output", "")
            or context.get_string("generate_output", "")
            or context.get_string("last_response", "")
        )
        if not generate_output:
            return Outcome(status=StageStatus.FAIL, failure_reason="No generated output available for deployment")

        extracted_blocks = extract_blocks_with_fallbacks(generate_output)
        if not extracted_blocks:
            return Outcome(status=StageStatus.FAIL, failure_reason="Generated output did not contain deployable files")

        deployment_dir = project_dir / "deployment" / "current"
        deployment_dir.mkdir(parents=True, exist_ok=True)

        files: list[dict[str, str]] = []
        dominant_language = "text"
        for index, block in enumerate(extracted_blocks):
            filename = (
                block.attribute_filename
                or block.filename_comment
                or block.header_filename
                or f"artifact_{index + 1}.txt"
            )
            target = deployment_dir / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(block.code, encoding="utf-8")
            detected_language = block.language or infer_language_from_filename(filename) or "text"
            if index == 0:
                dominant_language = detected_language
            files.append({"filename": filename, "language": detected_language})

        manifest = {
            "project_goal": graph.goal,
            "deployment_id": f"deploy_{int(time.time())}",
            "status": "READY",
            "deployed_at": datetime.now().isoformat(),
            "environment": "local-artifacts",
            "launch_command": _launch_command(dominant_language),
            "files": files,
            "nodes_completed": context.keys(),
        }
        manifest_path = project_dir / "twin_manifest.json"
        (project_dir / "deployment" / "README.md").write_text(
            "# Deployment Artifacts\n\n"
            f"- Environment: local-artifacts\n"
            f"- Launch command: `{manifest['launch_command']}`\n"
            f"- Bundle path: `{deployment_dir}`\n",
            encoding="utf-8",
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        emitter.emit_simple(
            PipelineEventKind.LOG,
            node_id=node.id,
            message=f"Deployment artifacts created at {deployment_dir}",
        )
        return Outcome(
            status=StageStatus.SUCCESS,
            notes=f"Deployment artifacts created with manifest {manifest_path.name}",
            context_updates={
                f"{node.id}.manifest_path": str(manifest_path),
                f"{node.id}.deployment_dir": str(deployment_dir),
            },
        )
