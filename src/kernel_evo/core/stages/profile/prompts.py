from __future__ import annotations

from pathlib import Path

from kernel_evo.resources.prompt_loader import load_prompt


class ProfileExtractPrompts:
    @staticmethod
    def system(prompts_dir: str | Path | None = None) -> str:
        return load_prompt("profile_extract", "system", Path(prompts_dir) if prompts_dir else None)

    @staticmethod
    def user(prompts_dir: str | Path | None = None) -> str:
        return load_prompt("profile_extract", "user", Path(prompts_dir) if prompts_dir else None)
