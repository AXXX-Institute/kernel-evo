from __future__ import annotations

import argparse
import json
from pathlib import Path

from kernel_evo.core.profile.torch_runner import run_torch_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Run torch.profiler for a candidate program.")
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--candidate-file", required=True)
    parser.add_argument("--reference-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-only", action="store_true")
    args = parser.parse_args()

    run_config = json.loads(Path(args.run_config).read_text(encoding="utf-8"))
    candidate_code = Path(args.candidate_file).read_text(encoding="utf-8")
    reference_code = Path(args.reference_file).read_text(encoding="utf-8")
    summary = run_torch_profile(
        run_config=run_config,
        ref_arch_src=reference_code,
        custom_model_src=candidate_code,
        out_dir=Path(args.out_dir).expanduser().resolve(),
        target_only=bool(args.target_only),
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
