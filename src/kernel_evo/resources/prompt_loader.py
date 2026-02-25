"""Load prompt templates from resources/prompts.

Default prompts live at package prompts dir. Per-backend overrides live under
prompts/backends/<backend>/ (e.g. backends/cuda_inline/mutation/). On each
experiment start we copy default into the experiment dir, then overlay backend
overrides so backend-specific stages (e.g. mutation for cuda_inline) override.
"""

import shutil
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# Top-level stage dirs in the default prompts dir (copied as-is; exclude backends subdir)
DEFAULT_STAGE_NAMES: tuple[str, ...] = ("mutation", "repair", "lineage", "insights")

# Subdir under prompts that holds per-backend override trees (not copied as a stage)
BACKENDS_SUBDIR = "backends"

# Mapping: backend name -> path relative to prompts dir for override content, or None to use default only.
# Override dir can contain only some stages (e.g. mutation/); those overwrite the copied default.
BACKEND_PROMPT_OVERRIDE_DIRS: dict[str, str | None] = {
    "triton": None,  # use default only
    "cuda_inline": "backends/cuda_inline",  # e.g. mutation/ overrides default mutation
}


def get_prompts_dir() -> Path:
    """Return the package's default prompts directory (works when installed via uv/pip)."""
    return _PROMPTS_DIR


def load_prompt(agent_name: str, prompt_type: str, prompts_dir: Path | None = None) -> str:
    """Load a prompt template from <prompts_dir>/<agent_name>/<prompt_type>.txt.
    If prompts_dir is None, uses KERNEL_EVO_PROMPTS_DIR env (when set) or package default."""
    if prompts_dir is not None:
        root = Path(prompts_dir).resolve()
    else:
        import os
        env_dir = os.environ.get("KERNEL_EVO_PROMPTS_DIR")
        root = Path(env_dir).resolve() if env_dir else _PROMPTS_DIR.resolve()
    path = root / agent_name / f"{prompt_type}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def prepare_prompts_for_experiment(experiment_dir: Path, backend: str) -> Path:
    """Copy default prompts into experiment_dir/prompts, then overlay backend overrides.
    Returns the path to use as prompts dir for this run (experiment_dir/prompts).
    If experiment_dir is the package resources dir, uses experiment_dir/prompts_run to avoid copying onto source."""
    default_dir = _PROMPTS_DIR.resolve()
    exp = Path(experiment_dir).resolve()
    dest = exp / "prompts"
    # Avoid copying onto ourselves when experiment dir is the resources dir (problem_dir = get_resources_dir())
    if dest.resolve() == default_dir:
        dest = exp / "prompts_run"
    dest.mkdir(parents=True, exist_ok=True)

    # 1) Copy default stage dirs
    for stage in DEFAULT_STAGE_NAMES:
        src = default_dir / stage
        if src.exists() and src.is_dir():
            shutil.copytree(src, dest / stage, dirs_exist_ok=True)

    # 2) Overlay backend overrides (overwrite same stage names)
    override_rel = BACKEND_PROMPT_OVERRIDE_DIRS.get(backend.lower())
    if override_rel:
        override_root = default_dir / override_rel
        if override_root.exists() and override_root.is_dir():
            for entry in override_root.iterdir():
                if entry.is_dir():
                    shutil.copytree(entry, dest / entry.name, dirs_exist_ok=True)

    return dest
