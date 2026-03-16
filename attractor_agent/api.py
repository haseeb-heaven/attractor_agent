from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from attractor.db import get_db
from attractor.pipeline.events import EventEmitter, PipelineEvent
from attractor.pipeline.interviewer import Answer as HumanAnswer, AutoApproveInterviewer, Interviewer, Question

from attractor_agent.project import BuildRequest, build_request_from_mapping
from attractor_agent.runtime import execute_build

LOGS_DIR = Path("projects")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Attractor API", version="1.1.0")
db = get_db()


class PipelineRequest(BaseModel):
    request: str | None = Field(default=None, description="Project description or prompt")
    prompt: str | None = Field(default=None, description="Backward-compatible prompt alias")
    language: str = "Python"
    framework: str = ""
    include_tests: bool = True
    include_sdlc: bool = True
    use_mock: bool = False
    auto_approve: bool = True
    require_human_review: bool = False
    retry_save_attempts: int = 3


class RunResponse(BaseModel):
    run_id: str
    status: str


class EventSnapshot(BaseModel):
    event_kind: str
    node_id: str | None
    message: str | None
    payload: dict[str, Any]
    timestamp: str


class RunDetail(BaseModel):
    run_id: str
    goal: str
    status: str
    final_node: str | None
    events: list[EventSnapshot]


class DBEventEmitter(EventEmitter):
    def __init__(self, run_id: str):
        super().__init__()
        self.run_id = run_id

    def emit(self, event: PipelineEvent) -> None:
        db.save_event(self.run_id, event.kind.value, event.node_id or "", event.payload)


class DBInterviewer(Interviewer):
    def __init__(self, run_id: str):
        self.run_id = run_id

    def ask(self, question: Question) -> HumanAnswer:
        db.save_question(
            self.run_id,
            question.id,
            question.node_id,
            question.text,
            [{"label": option.label, "key": option.key} for option in question.options],
        )
        while True:
            questions = db.get_questions(self.run_id)
            for stored in questions:
                if stored["question_id"] == question.id and stored.get("answer"):
                    answer_data = json.loads(stored["answer"])
                    return HumanAnswer(
                        question_id=question.id,
                        selected_label=answer_data.get("selected_label", ""),
                        free_text=answer_data.get("free_text", ""),
                    )
            time.sleep(1)


def _write_status(run_id: str, status: str) -> None:
    run_dir = LOGS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(json.dumps({"status": status}), encoding="utf-8")


def execute_pipeline_task(run_id: str, spec: BuildRequest) -> None:
    emitter = DBEventEmitter(run_id)
    interviewer: Interviewer | None
    if spec.require_human_review and not spec.auto_approve:
        interviewer = DBInterviewer(run_id)
    else:
        interviewer = AutoApproveInterviewer()

    spec.project_name = run_id
    spec.checkpoint_dir = str(LOGS_DIR / run_id)

    try:
        artifacts = execute_build(spec, interviewer=interviewer, emitter=emitter)
        status = "COMPLETED" if artifacts.result.success and artifacts.saved_files else "FAILED"
        _write_status(run_id, status)
        db.update_run(
            run_id,
            status=status,
            final_node=artifacts.result.final_node,
            result={
                "success": artifacts.result.success,
                "error": artifacts.result.error,
                "total_steps": artifacts.result.total_steps,
                "elapsed_seconds": artifacts.result.elapsed_seconds,
                "saved_files": [path.name for path in artifacts.saved_files],
            },
        )
    except Exception as exc:
        _write_status(run_id, "ERROR")
        db.update_run(run_id, status="ERROR", final_node="", result={"error": str(exc)})


@app.post("/api/v1/runs", response_model=RunResponse)
async def create_run(request: PipelineRequest, background_tasks: BackgroundTasks):
    run_id = str(uuid.uuid4())
    try:
        payload = request.model_dump()
        payload["request"] = payload.get("request") or payload.get("prompt")
        spec = build_request_from_mapping(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    db.save_run(run_id, goal=spec.request, config=request.model_dump())
    _write_status(run_id, "RUNNING")
    background_tasks.add_task(execute_pipeline_task, run_id, spec)
    return {"run_id": run_id, "status": "RUNNING"}


@app.post("/pipelines")
async def create_run_from_dot(background_tasks: BackgroundTasks, request: Request):
    run_id = str(uuid.uuid4())
    source = (await request.body()).decode("utf-8")
    db.save_run(run_id, goal="DOT Pipeline", config={"source": "direct_dot"})
    _write_status(run_id, "RUNNING")

    def run_it() -> None:
        from attractor.pipeline.engine import PipelineConfig, run_pipeline

        emitter = DBEventEmitter(run_id)
        try:
            result = run_pipeline(source, config=PipelineConfig(checkpoint_dir=str(LOGS_DIR / run_id)), emitter=emitter)
            status = "COMPLETED" if result.success else "FAILED"
            _write_status(run_id, status)
            db.update_run(run_id, status=status, final_node=result.final_node, result={"success": result.success})
        except Exception as exc:
            _write_status(run_id, "ERROR")
            db.update_run(run_id, status="ERROR", final_node="", result={"error": str(exc)})

    background_tasks.add_task(run_it)
    return {"run_id": run_id, "status": "RUNNING"}


@app.get("/pipelines/{run_id}")
async def get_pipeline_status(run_id: str):
    run_dir = LOGS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(404, "Run not found")
    status_file = run_dir / "status.json"
    if status_file.exists():
        return {"run_id": run_id, "status": json.loads(status_file.read_text()).get("status", "UNKNOWN")}
    return {"run_id": run_id, "status": "RUNNING"}


@app.get("/api/v1/runs/{run_id}", response_model=RunDetail)
async def get_run(run_id: str):
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    events = [
        EventSnapshot(
            event_kind=event["event_kind"],
            node_id=event.get("node_id"),
            message=event["payload"].get("message"),
            payload=event["payload"],
            timestamp=str(event["timestamp"]),
        )
        for event in run.get("events", [])
    ]
    return RunDetail(
        run_id=run["run_id"],
        goal=run["goal"],
        status=run["status"],
        final_node=run.get("final_node"),
        events=events,
    )


@app.get("/api/v1/runs/{run_id}/events")
async def stream_events(run_id: str):
    async def event_generator():
        last_event_id = 0
        while True:
            run = db.get_run(run_id)
            if not run:
                break
            events = run.get("events", [])
            for index in range(last_event_id, len(events)):
                yield f"data: {json.dumps(events[index], default=str)}\n\n"
                last_event_id = index + 1
            if run["status"] in ("COMPLETED", "FAILED", "ERROR"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/v1/runs/{run_id}/questions")
async def get_questions(run_id: str):
    return db.get_questions(run_id)


@app.post("/api/v1/runs/{run_id}/questions/{qid}/answer")
async def answer_question(run_id: str, qid: str, answer: dict[str, Any]):
    db.answer_question(run_id, qid, answer)
    return {"status": "success"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
