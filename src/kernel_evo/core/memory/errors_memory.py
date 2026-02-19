"""Build error-fix memory from validation logs (invalid -> valid program pairs)."""

import os
import difflib

try:
    from evo_memory_agent.shared_memory.memory import AmemGamMemory
except ImportError:
    # TODO: Should be refactored on gigaevo side
    print("Memory would be shipped later as a separate package")
    
from kernel_evo.core.memory.extract_fixes_from_logs import parse_fixes


def create_errors_memory(
    source_dirs: list[str],
    memory_dir: str,
    *,
    rebuild_interval: int = 1000,
) -> int:
    """Scan source_dirs for validate logs, extract fix pairs, save to memory. Returns count of saved docs."""
    memory = AmemGamMemory(checkpoint_path=memory_dir, rebuild_interval=rebuild_interval)
    total_saved = 0
    for source_dir in source_dirs:
        if not os.path.isdir(source_dir):
            continue
        for exp_dir in os.listdir(source_dir):
            exp_path = os.path.join(source_dir, exp_dir)
            if not os.path.isdir(exp_path):
                continue
            traces = os.listdir(exp_path)
            experiments_traces = {exp_dir: traces}
            fixes = parse_fixes(source_dir, experiments_traces)
            for fix in fixes:
                diff = difflib.unified_diff(fix.source_code.splitlines(), fix.fixed_code.splitlines())
                doc = f"There was an error in the code: {fix.error}\nThe diff what fixed the error is: {diff}"
                memory.save(doc)
                total_saved += 1
    memory.rebuild()
    return total_saved
