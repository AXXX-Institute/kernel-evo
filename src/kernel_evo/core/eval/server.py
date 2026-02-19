from typing import Any, Dict
from loguru import logger

from kernel_evo.resources.paths import get_problem_dir
from kernel_evo.resources.validate import (
    run_local_validation,
    _extract_custom_model_src,
)


def run_validation_core(job_id: str, cfg: Dict[str, Any], payload: Any) -> Dict[str, Any]:
    """Core validation logic that returns result dict. Can be run in subprocess."""
    try:
        problem_dir = get_problem_dir()

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
            ref_arch_src=ref_arch_src,
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
