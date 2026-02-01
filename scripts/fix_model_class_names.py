#!/usr/bin/env python3
"""
Post-process script to fix Model class names in JSON response files.

The generated code must define `class ModelNew(torch.nn.Module)`, but LLMs sometimes
forget and use `class Model` instead. This script fixes that by replacing all
`class Model` with `class ModelNew` in the response code field.

Note: We only modify the response code, not the request (which contains the reference
model that should remain as `class Model`).
"""

import json
import re
from pathlib import Path
from typing import Tuple


def fix_model_class_name(code: str) -> Tuple[str, bool]:
    """
    Replace `class Model` with `class ModelNew` in the generated code.
    
    Returns:
        Tuple of (fixed_code, was_modified)
    """
    # Pattern to match class Model with optional inheritance
    # We need to be careful not to match:
    # - class ModelNew (already correct)
    # - class Model(nn.Module) in reference code (shouldn't be in response)
    # - class Model in comments/docstrings
    
    # First, check if ModelNew already exists (code is already correct)
    if re.search(r'class\s+ModelNew\b', code):
        return code, False
    
    # Check if Model exists without New
    if not re.search(r'class\s+Model\b', code):
        return code, False
    
    # Replace class Model with class ModelNew
    # Use word boundary to avoid matching ModelNew or other variations
    fixed_code = re.sub(
        r'class\s+Model\b',
        'class ModelNew',
        code
    )
    
    was_modified = fixed_code != code
    return fixed_code, was_modified


def process_json_file(json_path: Path) -> Tuple[int, int]:
    """
    Process a single JSON file, fixing Model class names in response code.
    
    Returns:
        Tuple of (files_modified, files_skipped)
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  Error reading {json_path.name}: {e}")
        return 0, 0
    
    # Check if response has code field
    if 'response' not in data or 'json' not in data['response']:
        return 0, 0
    
    response_json = data['response']['json']
    if 'code' not in response_json:
        return 0, 0
    
    original_code = response_json['code']
    fixed_code, was_modified = fix_model_class_name(original_code)
    
    if was_modified:
        response_json['code'] = fixed_code
        
        # Write back the modified JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ Fixed: {json_path.name}")
        return 1, 0
    else:
        return 0, 1


def main():
    llm5_1_dir = Path("/home/sivtsov/kernel_generation/llm5_1")
    
    if not llm5_1_dir.exists():
        print(f"Error: Directory not found: {llm5_1_dir}")
        return
    
    # Find all JSON files in the directory
    json_files = sorted(llm5_1_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {llm5_1_dir}")
        return
    
    print(f"Processing {len(json_files)} JSON files in {llm5_1_dir}...\n")
    
    total_modified = 0
    total_skipped = 0
    
    for json_path in json_files:
        modified, skipped = process_json_file(json_file)
        total_modified += modified
        total_skipped += skipped
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Files modified: {total_modified}")
    print(f"  Files skipped (already correct): {total_skipped}")
    print(f"  Total files processed: {len(json_files)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()












