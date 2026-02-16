from kernel_generation.resources.prompt_loader import load_prompt


class RepairPrompts:
    """Repair agent prompt templates."""

    @staticmethod
    def system() -> str:
        """System prompt for repair."""
        return load_prompt("repair", "system")

    @staticmethod
    def user() -> str:
        """User prompt template for repair."""
        return load_prompt("repair", "user")
