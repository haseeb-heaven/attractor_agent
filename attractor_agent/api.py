"""FastAPI REST service for Attractor Pipeline Engine."""

import uuid
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field

from attractor.pipeline.engine import run_pipeline, PipelineConfig
from attractor.db import get_db
from attractor.pipeline.events import EventEmitter, PipelineEvent
from attractor.pipeline.interviewer import Interviewer, Question, Answer as HumanAnswer
from fastapi.responses import StreamingResponse
import asyncio
import json
import time
from pathlib import Path

LOGS_DIR = Path("projects")
if not LOGS_DIR.exists():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Attractor API", version="1.0.0")

# --- Models ---

class PipelineRequest(BaseModel):
    prompt: str = Field(..., description="Project description or prompt")
    language: str = Field("Python", description="Target programming language")
    include_tests: bool = True
    include_sdlc: bool = True
    variables: Dict[str, str] = {}
    mock_llm: bool = False

class RunResponse(BaseModel):
    run_id: str
    status: str

class EventSnapshot(BaseModel):
    event_kind: str
    node_id: Optional[str]
    message: Optional[str]
    payload: Dict[str, Any]
    timestamp: str

class RunDetail(BaseModel):
    run_id: str
    goal: str
    status: str
    final_node: Optional[str]
    events: List[EventSnapshot]


@app.get("/pipelines/{run_id}")
async def get_pipeline_status(run_id: str):
    run_dir = LOGS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(404, "Run not found")
    
    status_file = run_dir / "status.json"
    if status_file.exists():
        status = json.loads(status_file.read_text())
        return {"run_id": run_id, "status": status.get("status", "UNKNOWN")}
    return {"run_id": run_id, "status": "RUNNING"}



# --- Database Integration ---

db = get_db()

class DBEventEmitter(EventEmitter):
    """Custom emitter that stream events to the database."""
    def __init__(self, run_id: str):
        super().__init__()
        self.run_id = run_id

    def emit(self, event: PipelineEvent) -> None:
        db.save_event(
            self.run_id,
            event.kind.value,
            event.node_id or "",
            event.payload
        )

class DBInterviewer(Interviewer):
    """Interviewer that saves questions to DB and polls for answers."""
    def __init__(self, run_id: str):
        self.run_id = run_id

    def ask(self, question: Question) -> HumanAnswer:
        # Save question to DB
        db.save_question(
            self.run_id,
            question.id,
            question.node_id,
            question.text,
            [{"label": o.label, "key": o.key} for o in question.options]
        )
        
        # Poll for answer
        while True:
            run = db.get_run(self.run_id)
            for q in run.get("questions", []):
                if q["question_id"] == question.id and q["answer"]:
                    ans_data = json.loads(q["answer"])
                    return HumanAnswer(
                        question_id=question.id,
                        selected_label=ans_data.get("selected_label", ""),
                        free_text=ans_data.get("free_text", "")
                    )
            time.sleep(1) # Wait for human

# --- Background Task ---

def execute_pipeline_task(run_id: str, prompt: str, language: str, include_tests: bool, include_sdlc: bool, mock_llm: bool):
    from attractor_agent.cli import build_dot
    
    # Generate DOT
    dot = build_dot(prompt, language, "", include_tests, include_sdlc)
    
    # Setup Emitter
    emitter = DBEventEmitter(run_id)
    
    # Setup Config
    config = PipelineConfig(
        checkpoint_dir=f"projects/{run_id}",
        goal=prompt,
        interviewer=DBInterviewer(run_id)
    )
    
    # Run
    try:
        result = run_pipeline(dot, config=config, emitter=emitter)
        status = "COMPLETED" if result.success else "FAILED"
        
        # Write status file for /pipelines/{id}
        run_dir = LOGS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "status.json").write_text(json.dumps({"status": status}))

        db.update_run(
            run_id,
            status=status,
            final_node=result.final_node,
            result={
                "success": result.success,
                "error": result.error,
                "total_steps": result.total_steps,
                "elapsed_seconds": result.elapsed_seconds,
            }
        )
    except Exception as e:
        db.update_run(run_id, status="ERROR", final_node="", result={"error": str(e)})

# --- Endpoints ---

@app.post("/api/v1/runs", response_model=RunResponse)
async def create_run(request: PipelineRequest, background_tasks: BackgroundTasks):
    run_id = str(uuid.uuid4())
    
    # Initialize run in DB
    db.save_run(
        run_id,
        goal=request.prompt,
        config=request.dict()
    )
    
    # Create directory immediately so /pipelines/{id} works (monitoring)
    run_dir = LOGS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Start background task
    background_tasks.add_task(
        execute_pipeline_task,
        run_id,
        request.prompt,
        request.language,
        request.include_tests,
        request.include_sdlc,
        request.mock_llm
    )
    
    return {"run_id": run_id, "status": "RUNNING"}

@app.post("/pipelines")
async def create_run_from_dot(background_tasks: BackgroundTasks, request: Request):
    """Create a run from a DOT file directly (Section 9.5)."""
    run_id = str(uuid.uuid4())
    source_bytes = await request.body()
    source = source_bytes.decode("utf-8")
    
    # Setup Emitter
    emitter = DBEventEmitter(run_id)
    
    # Save initial run record
    db.save_run(run_id, goal="DOT Pipeline", config={"source": "direct_dot"})
    
    # Create directory immediately so /pipelines/{id} works (monitoring)
    run_dir = LOGS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Background execute
    def run_it():
        try:
            # Setup Config with checkpoint_dir
            config = PipelineConfig(checkpoint_dir=str(LOGS_DIR / run_id))
            
            result = run_pipeline(source, config=config, emitter=emitter)
            status = "COMPLETED" if result.success else "FAILED"
            
            # Write status file for /pipelines/{id}
            run_dir = LOGS_DIR / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "status.json").write_text(json.dumps({"status": status}))

            db.update_run(run_id, status=status, final_node=result.final_node, result={"success": result.success})
        except Exception as e:
            db.update_run(run_id, status="ERROR", final_node="", result={"error": str(e)})
            
    background_tasks.add_task(run_it)
    return {"run_id": run_id, "status": "RUNNING"}

@app.get("/api/v1/runs/{run_id}", response_model=RunDetail)
async def get_run(run_id: str):
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    # Convert DB model to response model
    events = [
        EventSnapshot(
            event_kind=e["event_kind"],
            node_id=e.get("node_id"),
            message=e["payload"].get("message"),
            payload=e["payload"],
            timestamp=str(e["timestamp"])
        ) for e in run.get("events", [])
    ]
    
    return RunDetail(
        run_id=run["run_id"],
        goal=run["goal"],
        status=run["status"],
        final_node=run.get("final_node"),
        events=events
    )

@app.get("/api/v1/runs/{run_id}/events")
async def stream_events(run_id: str):
    """SSE stream of pipeline events (spec §9.5)."""
    async def event_generator():
        last_event_id = 0
        while True:
            run = db.get_run(run_id)
            if not run:
                break
                
            events = run.get("events", [])
            for i in range(last_event_id, len(events)):
                event = events[i]
                yield f"data: {json.dumps(event, default=str)}\n\n"
                last_event_id = i + 1
            
            if run["status"] in ("COMPLETED", "FAILED", "ERROR"):
                break
            await asyncio.sleep(0.5)
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/v1/runs/{run_id}/questions")
async def get_questions(run_id: str):
    """Get unanswered questions (spec §9.5)."""
    return db.get_questions(run_id)

@app.post("/api/v1/runs/{run_id}/questions/{qid}/answer")
async def answer_question(run_id: str, qid: str, answer: Dict[str, Any]):
    """Answer a question (spec §9.5)."""
    db.answer_question(run_id, qid, answer)
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
