"""
Build memory of top-N best (correct, fastest) programs per problem from validation logs.
Problem identity: custom tasks by problem_path; kernelbench by (level, problem_id).
"""

import os
from collections import defaultdict

try:
    from evo_memory_agent.shared_memory.memory import AmemGamMemory
except ImportError:
    # TODO: Should be refactored on gigaevo side
    print("Memory would be shipped later as a separate package")

from kernel_evo.core.memory.extract_fixes_from_logs import parse_log_file


def problem_key_from_context(context: dict) -> tuple | None:
    """
    Unique key for grouping runs across source-dirs.
    - custom: ("custom", problem_path)
    - kernelbench: ("kernelbench", level, problem_id)
    Returns None if context does not define a known problem.
    """
    kind = context.get("problem_kind")
    if kind == "custom":
        path = context.get("problem_path")
        if path:
            return ("custom", path)
    elif kind == "kernelbench":
        level = context.get("level")
        pid = context.get("problem_id")
        if level is not None and pid is not None:
            return ("kernelbench", int(level), int(pid))
    return None


def collect_best_per_problem(source_dirs: list[str], top_n: int = 4):
    """
    Scan source_dirs for validate log dirs; for each log with correct+compiled and
    valid runtime, group by problem key and keep top N by runtime (ascending).
    Returns: dict[problem_key, list[(runtime_ms, ref_runtime_ms, program_code)]],
    each list sorted by runtime ascending, at most top_n entries.
    """
    # problem_key -> list of (runtime, ref_runtime, code); we keep the top N by runtime (min = best)
    by_key = defaultdict(list)

    for source_dir in source_dirs:
        if not os.path.isdir(source_dir):
            continue
        for exp_dir in os.listdir(source_dir):
            exp_path = os.path.join(source_dir, exp_dir)
            if not os.path.isdir(exp_path):
                continue
            for trace in os.listdir(exp_path):
                log_path = os.path.join(exp_path, trace)
                if not os.path.isfile(log_path) or not trace.endswith(".log"):
                    continue
                try:
                    parsed = parse_log_file(log_path, include_context=True)
                except OSError:
                    continue
                context = parsed.get("context") or {}
                key = problem_key_from_context(context)
                if key is None:
                    continue
                results = parsed.get("kernelbench_results") or {}
                code = (parsed.get("program_code") or "").strip()
                if not code:
                    continue
                if not results.get("compiled") or not results.get("correctness"):
                    continue
                runtime = results.get("runtime")
                if runtime is None or (isinstance(runtime, (int, float)) and runtime <= 0):
                    continue
                ref_runtime = results.get("ref_runtime")
                if ref_runtime is None or (isinstance(ref_runtime, (int, float)) and ref_runtime <= 0):
                    ref_runtime = float(runtime)  # no baseline: speedup 1.0
                else:
                    ref_runtime = float(ref_runtime)
                runtime_ms = float(runtime)
                by_key[key].append((runtime_ms, ref_runtime, code))

    # For each key keep top N by runtime (ascending), dedupe by code
    best = {}
    for key, candidates in by_key.items():
        # Sort by runtime ascending; dedupe by code (keep first = fastest)
        seen_codes = set()
        unique = []
        for r, ref_r, c in sorted(candidates, key=lambda x: x[0]):
            if c in seen_codes:
                continue
            seen_codes.add(c)
            unique.append((r, ref_r, c))
            if len(unique) >= top_n:
                break
        best[key] = unique

    return best


def key_display(key: tuple) -> str:
    if key[0] == "custom":
        return f"custom path={key[1]}"
    return f"kernelbench level={key[1]} problem_id={key[2]}"


def create_best_programs_memory(
    source_dirs: list[str],
    memory_dir: str,
    *,
    top_n: int = 4,
    rebuild_interval: int = 1000,
) -> int:
    """Build best-programs memory from validation logs. Returns count of saved docs."""
    best = collect_best_per_problem(source_dirs, top_n=top_n)
    memory = AmemGamMemory(checkpoint_path=memory_dir, rebuild_interval=rebuild_interval)
    total_saved = 0
    for key, entries in best.items():
        if not entries:
            continue
        problem_description = key_display(key)
        for runtime_ms, ref_runtime_ms, code in entries:
            speedup = ref_runtime_ms / runtime_ms if runtime_ms > 0 else 0.0
            doc = (
                f"Got speedup {speedup} from baseline for problem ({problem_description}) with following code\n\n{code}"
            )
            memory.save(doc)
            total_saved += 1
    memory.rebuild()
    return total_saved
