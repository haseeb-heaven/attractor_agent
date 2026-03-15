"""FastAPI REST service for Attractor Pipeline Engine."""

import uuid
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from attractor.pipeline.engine import run_pipeline, PipelineConfig
from attractor.db import get_db
from attractor.pipeline.events import EventEmitter, PipelineEvent

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
        goal=prompt
    )
    
    # Run
    try:
        result = run_pipeline(dot, config=config, emitter=emitter)
        status = "COMPLETED" if result.success else "FAILED"
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

@app.get("/api/v1/runs", response_model=List[RunResponse])
async def list_runs(limit: int = 10, offset: int = 0):
    runs = db.list_runs(limit=limit, offset=offset)
    return [{"run_id": r["run_id"], "status": r["status"]} for r in runs]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
