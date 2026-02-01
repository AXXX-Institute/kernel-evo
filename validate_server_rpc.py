from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional
import uuid
import asyncio

app = FastAPI()

# In-memory storage for simplicity as per requirement
# In a real-world scenario, this should be a persistent database or Redis.
jobs: Dict[str, Dict[str, Any]] = {}

class ValidationRequest(BaseModel):
    cfg: Dict[str, Any]
    payload: Any

class ValidationResponse(BaseModel):
    job_id: str

class ResultResponse(BaseModel):
    status: str
    result: Optional[Dict[str, float]] = None
    error_msg: Optional[str] = None
    error_type: Optional[str] = None

# This will be injected by validate_server.py
worker_function = None

@app.post("/schedule_validate", response_model=ValidationResponse)
async def schedule_validate(request: ValidationRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "result": None,
        "error_msg": None,
        "error_type": None,
    }
    
    if worker_function:
        background_tasks.add_task(worker_function, job_id, request.cfg, request.payload)
    else:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error_msg"] = "Worker function not initialized"
        
    return ValidationResponse(job_id=job_id)

@app.get("/fetch_validate_results", response_model=ResultResponse)
async def fetch_validate_results(job_id: str, wait_seconds: int = 5):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Simple async waiting for result
    for _ in range(wait_seconds):
        if jobs[job_id]["status"] in ["completed", "failed"]:
            break
        await asyncio.sleep(1)
        
    return ResultResponse(
        status=jobs[job_id]["status"],
        result=jobs[job_id]["result"],
        error_msg=jobs[job_id].get("error_msg"),
        error_type=jobs[job_id].get("error_type"),
    )

