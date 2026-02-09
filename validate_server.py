import argparse
import uvicorn
import sys
from pathlib import Path
from typing import Any, Dict
from loguru import logger

# Add current directory to path so we can import validate
sys.path.append(str(Path(__file__).parent.parent))

from kernel_generation.validate import run_local_validation, _extract_custom_model_src, _find_problem_dir, _add_kernelbench_to_sys_path
from kernel_generation.validate_server_rpc import app, jobs
import kernel_generation.validate_server_rpc as rpc

def _run_validation_core(job_id: str, cfg: Dict[str, Any], payload: Any) -> Dict[str, Any]:
    """Core validation logic that returns result dict. Can be run in subprocess."""
    try:
        problem_dir = _find_problem_dir()
        _add_kernelbench_to_sys_path(problem_dir)
        
        custom_model_src = _extract_custom_model_src(payload)
        
        ref_arch_src = cfg.get("ref_arch_src") or cfg.get("original_model_src")
        if not ref_arch_src:
            raise ValueError("No reference model source code provided")
        
        logger.info(f"Run validation for job {job_id}")

        # Run the local validation logic we decoupled
        result = run_local_validation(
            problem_dir=problem_dir,
            cfg=cfg,
            payload=payload,
            custom_model_src=custom_model_src,
            ref_arch_src=ref_arch_src
        )
        
        logger.info(f"Validation result for job {job_id}: {result}")
        
        return {
            "status": "completed",
            "result": result,
            "error_msg": None,
            "error_type": None,
        }
    except Exception as e:
        import traceback
        logger.error(f"Error for job {job_id}: {traceback.format_exc()}")
        return {
            "status": "failed",
            "result": None,
            "error_msg": str(e),
            "error_type": type(e).__name__,
        }

# explicitly made sync to run in thread to avoid segfaults
def worker_task(job_id: str, cfg: Dict[str, Any], payload: Any):
    """Worker task that updates jobs dict directly (for backward compatibility)"""
    jobs[job_id]["status"] = "running"
    result_dict = _run_validation_core(job_id, cfg, payload)
    jobs[job_id].update(result_dict)

def main():
    parser = argparse.ArgumentParser(description="Kernel Generation Validation Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()

    # Inject the worker function into the RPC module
    rpc.worker_function = worker_task

    print(f"Starting validation server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()

