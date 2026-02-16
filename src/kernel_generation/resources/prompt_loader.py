"""Load prompt templates from resources/prompts."""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def get_prompts_dir() -> Path:
    """Return the package's prompts directory (works when installed via uv/pip)."""
    return _PROMPTS_DIR


def load_prompt(agent_name: str, prompt_type: str) -> str:
    """Load a prompt template from resources/prompts/<agent_name>/<prompt_type>.txt."""
    path = _PROMPTS_DIR / agent_name / f"{prompt_type}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8").strip()
