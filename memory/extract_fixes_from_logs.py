from dataclasses import dataclass
import os
import json
import ast 
import difflib

@dataclass
class Fix:
    id: int
    source_code: str
    fixed_code: str
    error: str

def _parse_context_from_lines(lines):
    """Extract run_config JSON from section '=== context (run_config.json / context.py) ==='."""
    try:
        start_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith("=== context ") and line.strip().endswith("==="):
                start_idx = i + 1
                break
        if start_idx is None:
            return {}
        json_lines = []
        brace_count = 0
        for i in range(start_idx, len(lines)):
            line = lines[i]
            if line.strip().startswith("===") and line.strip().endswith("==="):
                break
            json_lines.append(line)
            brace_count += line.count("{") - line.count("}")
            if brace_count == 0 and line.strip().endswith("}"):
                break
        json_str = "".join(json_lines).strip()
        if not json_str:
            return {}
        return json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        return {}


def _parse_kernelbench_results_from_lines(lines):
    """Extract JSON from section starting with '=== kernelbench result ==='"""
    try:
        # Find the section start
        start_idx = None
        for i, line in enumerate(lines):
            if line.strip() == "=== kernelbench result ===":
                start_idx = i + 1
                break
        
        if start_idx is None:
            return {}
        
        # Collect JSON lines until we find the closing brace or next section
        json_lines = []
        brace_count = 0
        for i in range(start_idx, len(lines)):
            line = lines[i]
            
            # Stop if we hit another section marker
            if line.strip().startswith("===") and line.strip().endswith("==="):
                break
            
            json_lines.append(line)
            
            # Count braces to find the end of JSON
            brace_count += line.count('{') - line.count('}')
            
            # If we've closed all braces, we're done
            if brace_count == 0 and line.strip().endswith('}'):
                break
        
        json_str = ''.join(json_lines).strip()
        if not json_str:
            return {}
        
        return json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        return {}

def _parse_exception_section_from_lines(lines):
    """Extract JSON from exception section, starting after 'Exception: Runtime error:'.
    If prefix not found, parse the whole section text until next section marker."""
    try:
        # Find the exception section start
        section_start_idx = None
        for i, line in enumerate(lines):
            if line.strip() == "=== exception ===":
                section_start_idx = i + 1
                break
        
        if section_start_idx is None:
            return {}
        
        # Collect all lines until the next section marker (or end of file)
        section_lines = []
        for i in range(section_start_idx, len(lines)):
            line = lines[i]
            # Stop if we hit another section marker
            if line.strip().startswith("===") and line.strip().endswith("==="):
                break
            section_lines.append(line)
        
        section_text = ''.join(section_lines).strip()
        if not section_text:
            return {}
        
        # Try to find "Exception: Runtime error:" prefix
        prefix = "Exception: Runtime error:"
        if prefix in section_text:
            # Extract the part after "Exception: Runtime error:"
            json_start = section_text.find(prefix)
            if json_start != -1:
                json_str = section_text[json_start + len(prefix):].strip()
                # Try to parse as Python dict literal (using ast.literal_eval)
                try:
                    return ast.literal_eval(json_str)
                except (ValueError, SyntaxError):
                    # If that fails, try JSON parsing
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        return {}
        else:
            # No prefix found, try to parse the whole section text as JSON/dict
            try:
                return ast.literal_eval(section_text)
            except (ValueError, SyntaxError):
                # If that fails, try JSON parsing
                try:
                    return json.loads(section_text)
                except json.JSONDecodeError:
                    # If it's not JSON/dict, store the raw text as runtime_error
                    # This handles cases where the exception section is just a traceback
                    return {"runtime_error": section_text}
        
        return {}
    except (ValueError, SyntaxError):
        return {}

def _parse_program_code_from_lines(lines):
    """Extract code from section starting with '=== program code ==='"""
    try:
        # Find the section start
        start_idx = None
        for i, line in enumerate(lines):
            if line.strip() == "=== program code ===":
                start_idx = i + 1
                break
        
        if start_idx is None:
            return ""
        
        # Collect all lines until the next section (or end of file)
        code_lines = []
        for i in range(start_idx, len(lines)):
            line = lines[i]
            # Stop if we hit another section marker
            if line.strip().startswith("===") and line.strip().endswith("==="):
                break
            code_lines.append(line)
        
        return ''.join(code_lines).rstrip()
    except Exception:
        return ""

def parse_log_file(file_path: str, include_context: bool = False):
    """Read log file once and parse sections. Returns dict with keys:
    'kernelbench_results', 'exception_json', 'program_code', and optionally 'context'.
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        out = {
            'kernelbench_results': _parse_kernelbench_results_from_lines(lines),
            'exception_json': _parse_exception_section_from_lines(lines),
            'program_code': _parse_program_code_from_lines(lines)
        }
        if include_context:
            out['context'] = _parse_context_from_lines(lines)
        return out
    except FileNotFoundError:
        out = {
            'kernelbench_results': {},
            'exception_json': {},
            'program_code': ""
        }
        if include_context:
            out['context'] = {}
        return out

def extract_kernelbench_results_section_json(file_path: str):
    """Extract JSON from section starting with '=== kernelbench result ==='"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        return _parse_kernelbench_results_from_lines(lines)
    except FileNotFoundError:
        return {}

def extract_exception_section_json(file_path: str):
    """Extract JSON from exception section, starting after 'Exception: Runtime error:'"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        return _parse_exception_section_from_lines(lines)
    except FileNotFoundError:
        return {}

def extract_program_code(file_path: str):
    """Extract code from section starting with '=== program code ==='"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        return _parse_program_code_from_lines(lines)
    except FileNotFoundError:
        return ""

def take_unshortened_version(results_json: dict, exception_json: dict):
    kb_logged = results_json.get("metadata", {}).get("runtime_error_traceback", None)
    if kb_logged is None:
        kb_logged = results_json.get("metadata", {}).get("correctness_issue", None)
    exp_logged = exception_json.get("runtime_error", None)
    
    # print(kb_logged, exp_logged)
    
    if kb_logged is not None and not kb_logged.endswith("..."):
        return kb_logged
    elif exp_logged is not None and not exp_logged.endswith("..."):
        return exp_logged
    else:
        raise ValueError("No unshortened version found")


def parse_fixes(directory: str, experiments_traces: dict, limit_documents: int = None):
    
    last_valid_prog = {}
    last_invalid_prog = {}
    documents_processed = 0
    
    for experiment, traces in experiments_traces.items():
        for trace in traces:
            # Check limit BEFORE processing
            if limit_documents is not None and documents_processed >= limit_documents:
                break
            
            # print(experiment, trace)
            file_path = os.path.join(directory, experiment, trace)
            run_key = trace.split("_")[0]
            last_modified_ts = os.path.getmtime(file_path)

            # Read file once and parse all sections
            parsed = parse_log_file(file_path)
            results_json = parsed['kernelbench_results']
            
            # Skip files without kernelbench results section
            if not results_json:
                continue
            
            documents_processed += 1
            
            if results_json.get("correctness") and results_json.get("compiled"):
                if run_key not in last_valid_prog or last_modified_ts > last_valid_prog[run_key][1]:
                    program_code = parsed['program_code']
                    last_valid_prog[run_key] = (program_code, last_modified_ts)  
            else:
                if run_key not in last_invalid_prog or last_modified_ts > last_invalid_prog[run_key][1]:
                    program_code = parsed['program_code']
                    exception_json = parsed['exception_json']
                    error = take_unshortened_version(results_json, exception_json)
                    last_invalid_prog[run_key] = (program_code, last_modified_ts, error)
        
        # Break outer loop if limit reached
        if limit_documents is not None and documents_processed >= limit_documents:
            break

    fix_pairs = []
    for run_key in last_invalid_prog:
        if run_key not in last_valid_prog:
            # raise ValueError(f"Run key {run_key} not found in last valid program")
            continue
        fix = Fix(
            id=run_key,
            source_code=last_valid_prog[run_key][0],
            fixed_code=last_invalid_prog[run_key][0],
            error=last_invalid_prog[run_key][2]
        )
        fix_pairs.append(fix)

    return fix_pairs

if __name__ == "__main__":
    directory = "/home/sivtsov/kernel_generation/outputs_bestruns/validate_logs"
    # directory = "/home/sivtsov/kernel_generation/outputs/validate_logs"
    # directory = "/home/sivtsov/kernel_generation/outputs2/validate_logs"
    experiments = os.listdir(directory)
    experiments_traces = {}
    for experiment in experiments:
        experiment_path = os.path.join(directory, experiment)
        if os.path.isdir(experiment_path):
            experiments_traces[experiment] = os.listdir(experiment_path)
    fixes = parse_fixes(directory, experiments_traces, limit_documents=1000)
    
    for fix in fixes:
        print("1--------------------------------")
        print(fix.id)
        # print(fix.source_code)
        # print(fix.fixed_code)
        # print(fix.error)
        split_str = "File \""
        cut_last = 1
        # print(split_str.join(fix.error.rsplit(split_str, cut_last)[-cut_last:]))
        
        diff = difflib.unified_diff(fix.source_code.splitlines(), fix.fixed_code.splitlines())
        print("\n".join(diff))
        
        print("2--------------------------------")
    
    # print(fixes)