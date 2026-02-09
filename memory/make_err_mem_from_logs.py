import os
import argparse
import difflib
from evo_memory_agent.shared_memory.memory import AmemGamMemory
from extract_fixes_from_logs import parse_fixes

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dirs", type=str, nargs='+', required=True)
    parser.add_argument("--err-memory-dir", type=str, required=True)
    args = parser.parse_args()
    
    memory = AmemGamMemory(checkpoint_path=args.err_memory_dir, rebuild_interval=1000)
    
    for source_dir in args.source_dirs:
        dirs_for_exp = os.listdir(source_dir)
        for exp_dir in dirs_for_exp:
            exp_dir_path = os.path.join(source_dir, exp_dir)
            traces_for_exp = os.listdir(exp_dir_path)
            experiments_traces = {exp_dir: traces_for_exp}
            list_of_fixes = parse_fixes(
                source_dir,
                experiments_traces
            )
            print(f"Found {len(list_of_fixes)} fixes for {exp_dir_path}")
            for fix in list_of_fixes:
                diff = difflib.unified_diff(fix.source_code.splitlines(), fix.fixed_code.splitlines())
                formated = f"There was an error in the code: {fix.error}\nThe diff what fixed the error is: {diff}"
                memory.save(formated)

    memory.rebuild()
