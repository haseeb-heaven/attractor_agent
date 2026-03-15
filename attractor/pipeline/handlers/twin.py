"""Handler for Digital Twin Universe deployment."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from attractor.pipeline.context import Context, Outcome, StageStatus
from attractor.pipeline.events import EventEmitter, PipelineEventKind
from attractor.pipeline.graph import Graph, Node
from attractor.pipeline.handlers.base import Handler


class DigitalTwinHandler(Handler):
    """Mocks deployment of the generated application to the Digital Twin Universe."""

    def execute(self, node: Node, context: Context, graph: Graph,
                emitter: EventEmitter, **kwargs: Any) -> Outcome:
        
        project_dir = kwargs.get("config").checkpoint_dir if "config" in kwargs else ""
        if not project_dir:
            return Outcome(status=StageStatus.FAIL, failure_reason="Missing checkpoint_dir for Digital Twin metadata")

        twin_data = {
            "project_goal": graph.goal,
            "twin_id": f"twin_{int(time.time())}",
            "status": "LIVE",
            "deployed_at": datetime.now().isoformat(),
            "nodes_completed": list(context.inner.keys()),
            "environment": "Simulation Sandbox"
        }

        manifest_path = Path(project_dir) / "twin_manifest.json"
        
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(twin_data, f, indent=2)
            
            emitter.emit_simple(
                PipelineEventKind.LOG,
                node_id=node.id,
                message=f"Digital Twin Universe connection established. Metadata saved to {manifest_path}"
            )
            
            return Outcome(
                status=StageStatus.SUCCESS,
                notes=f"Deployed to Twin Universe with ID {twin_data['twin_id']}"
            )
            
        except OSError as e:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason=f"Failed to generate Twin manifest: {e}"
            )
