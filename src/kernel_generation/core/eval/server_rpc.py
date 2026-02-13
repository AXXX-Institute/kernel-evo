from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional
import uuid
import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import sys

app = FastAPI()

# In-memory storage for simplicity as per requirement
# In a real-world scenario, this should be a persistent database or Redis.
jobs: Dict[str, Dict[str, Any]] = {}

# ProcessPoolExecutor for running validation in isolated subprocess
# Using spawn context to avoid CUDA context issues
mp_ctx = mp.get_context("spawn")
# Python 3.11+: max_tasks_per_child makes the worker process restart periodically
if sys.version_info >= (3, 11):
    executor = ProcessPoolExecutor(max_workers=1, mp_context=mp_ctx, max_tasks_per_child=1)
else:
    executor = ProcessPoolExecutor(max_workers=1, mp_context=mp_ctx)


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


# Wrapper function that can be pickled and run in subprocess
def _run_worker_in_subprocess(job_id: str, cfg: Dict[str, Any], payload: Any):
    """Run validation core in subprocess and return result dict."""
    from kernel_generation.core.eval.server import run_validation_core

    return run_validation_core(job_id, cfg, payload)


async def _update_job_result(job_id: str, future):
    """Update job result from subprocess future"""
    try:
        # Wait for the future result in a thread pool (future.result() is blocking)
        result = await asyncio.get_event_loop().run_in_executor(None, future.result)
        jobs[job_id].update(result)
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error_msg"] = str(e)
        jobs[job_id]["error_type"] = type(e).__name__


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup ProcessPoolExecutor on shutdown"""
    executor.shutdown(wait=True)


@app.post("/schedule_validate", response_model=ValidationResponse)
async def schedule_validate(request: ValidationRequest):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "result": None,
        "error_msg": None,
        "error_type": None,
    }

    # Submit task to ProcessPoolExecutor
    future = executor.submit(_run_worker_in_subprocess, job_id, request.cfg, request.payload)

    # Update job status to running
    jobs[job_id]["status"] = "running"

    # Schedule async task to update job result when subprocess completes
    asyncio.create_task(_update_job_result(job_id, future))

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
