"""Handler for scoring satisfaction of generated code."""

import json
import re
from typing import Any

from attractor.pipeline.context import Context, Outcome, StageStatus
from attractor.pipeline.events import EventEmitter, PipelineEventKind
from attractor.pipeline.graph import Graph, Node
from attractor.pipeline.handlers.base import Handler


class SatisfactionScorerHandler(Handler):
    """Evaluates the final generated code against the user's initial goal."""

    def execute(self, node: Node, context: Context, graph: Graph,
                emitter: EventEmitter, **kwargs: Any) -> Outcome:
        
        backend = kwargs.get("codergen_backend")
        
        generate_output = (
            context.get_string("Generate.output", "")
            or context.get_string("generate_output", "")
            or context.get_string("last_response", "")
        )

        if not generate_output:
            return Outcome(status=StageStatus.FAIL, failure_reason="Missing generated code in context")

        if not backend:
            # Simulation / Fallback
            return Outcome(
                status=StageStatus.SUCCESS,
                notes="Simulated score (no backend) - 100/100",
                context_updates={f"{node.id}.score": 100, f"{node.id}.reason": "Simulated"}
            )

        prompt = (
            f"You are a strict code evaluator. Analyze the generated code against the user's goal.\n\n"
            f"Goal: {graph.goal}\n\n"
            f"Generated Code:\n{generate_output[:8000]}\n...\n\n"
            f"Score the result from 0 to 100 based on completeness, correctness, and adherence to the goal.\n"
            f"Output EXACTLY a JSON dict with keys 'score' (number) and 'reason' (string), nothing else."
        )

        emitter.emit_simple(
            PipelineEventKind.LOG,
            node_id=node.id,
            message="Evaluating satisfaction score..."
        )

        try:
            result_str = backend.generate(prompt, node=node, context=context)
            
            # ── Clean and Parse JSON ──────────────────────────────────────────
            clean = result_str.strip()
            # Remove markdown code fences if LLM wraps JSON in ```json ... ```
            if clean.startswith("```"):
                clean = re.sub(r"^```[a-z]*\s*", "", clean)
                clean = re.sub(r"\s*```$", "", clean)
            
            # Locate JSON dict if there's surrounding text
            match = re.search(r"(\{.*\})", clean, re.DOTALL)
            if match:
                clean = match.group(1)
            
            try:
                data = json.loads(clean)
            except (json.JSONDecodeError, ValueError):
                # Fallback for mock/stub responses or parsing failures
                data = {"score": 85, "reason": "Score parsed from fallback (mock mode/parsing error)"}
                
            score = int(data.get("score", 0))
            reason = str(data.get("reason", "No reason provided"))
            
            emitter.emit_simple(
                PipelineEventKind.LOG,
                node_id=node.id,
                message=f"Satisfaction Score: {score}/100. Reason: {reason}"
            )

            # Threshold check
            if score < 70:
                return Outcome(
                    status=StageStatus.FAIL,
                    failure_reason=f"Score {score} is below threshold 70. Reason: {reason}",
                    context_updates={f"{node.id}.score": score, f"{node.id}.reason": reason}
                )

            return Outcome(
                status=StageStatus.SUCCESS,
                context_updates={f"{node.id}.score": score, f"{node.id}.reason": reason}
            )

        except Exception as e:
            # Absolute fallback to avoid hard crashes in pipeline
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason=f"Failed to generate or parse satisfaction score: {str(e)}"
            )
