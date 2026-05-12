from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path

from kernel_evo.core.profile.contracts import DEFAULT_NCU_PATH, run_profile_subprocess

_STABLE_SECTION_EXTRA_ARGS = "--section LaunchStats --section Occupancy"
_DEFAULT_NCU_FALLBACKS: tuple[str, ...] = ("/usr/local/cuda/bin/ncu",)


def _target_run_work_dir_for_ncu_child() -> Path:
    """Writable directory for ``profile_target --out-dir`` while wrapped by ``ncu``.

    ``sudo ncu ... python`` typically traces the app as the invoking user (``SUDO_UID``),
    not root. Artifact trees like ``.../ncu/`` are often ``root:root`` after a prior
    ``sudo`` run, so the traced process cannot create ``target_run/`` there and exits
    before any CUDA kernels — Nsight then reports only ``No kernels were profiled``.
    """
    work = Path(tempfile.mkdtemp(prefix="kernel_evo_ncu_target_run_"))
    sudo_uid = os.environ.get("SUDO_UID")
    if sudo_uid is not None:
        try:
            shutil.chown(work, int(sudo_uid), int(os.environ.get("SUDO_GID", sudo_uid)))
        except Exception:
            pass
    return work


def _resolve_executable(configured: str, *, fallbacks: tuple[str, ...] = ()) -> str | None:
    candidate = str(configured or "").strip()
    if candidate:
        if "/" in candidate:
            path = Path(candidate).expanduser().resolve()
            if path.exists():
                return str(path)
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    for fallback in fallbacks:
        path = Path(fallback).expanduser().resolve()
        if path.exists():
            return str(path)

    return None


def _cache_file(run_config: dict[str, object], out_dir: Path) -> Path:
    profile_artifacts_dir = str(run_config.get("profile_artifacts_dir", "") or "").strip()
    if profile_artifacts_dir:
        return Path(profile_artifacts_dir).expanduser().resolve() / "ncu_host_preflight.json"
    experiment_dir = str(run_config.get("experiment_dir", "") or "").strip()
    if experiment_dir:
        return Path(experiment_dir).expanduser().resolve() / "ncu_host_preflight.json"
    return out_dir.parent / "ncu_host_preflight.json"


def _device_index_from_run_config(run_config: dict[str, object]) -> str:
    device = str(run_config.get("device", "cuda:0") or "cuda:0").strip()
    if device.startswith("cuda:"):
        _, _, suffix = device.partition(":")
        if suffix.strip():
            return suffix.strip()
    return "0"


def _resolve_ncu_options(
    *,
    run_config: dict[str, object],
    set_override: str | None = None,
    kernel_name_override: str | None = None,
    extra_args_override: str | None = None,
) -> tuple[str, str, str, str]:
    devices = _device_index_from_run_config(run_config)
    section_set = (
        set_override
        if set_override is not None
        else str(run_config.get("profile_ncu_set", "full") or "full")
    ).strip()
    kernel_name = (
        kernel_name_override
        if kernel_name_override is not None
        else str(run_config.get("profile_ncu_kernel_name", "") or "")
    ).strip()
    extra_args = (
        extra_args_override
        if extra_args_override is not None
        else str(run_config.get("profile_ncu_extra_args", "") or "")
    ).strip()

    return devices, section_set, kernel_name, extra_args


def _effective_target_device(
    run_config: dict[str, object],
    *,
    target_device_override: str | None = None,
) -> str:
    explicit = str(target_device_override or "").strip()
    if explicit:
        return explicit

    return str(run_config.get("device", "cuda:0") or "cuda:0").strip() or "cuda:0"


def _build_ncu_option_args(
    *,
    run_config: dict[str, object],
    set_override: str | None = None,
    kernel_name_override: str | None = None,
    extra_args_override: str | None = None,
) -> list[str]:
    args: list[str] = []
    devices, section_set, kernel_name, extra_args = _resolve_ncu_options(
        run_config=run_config,
        set_override=set_override,
        kernel_name_override=kernel_name_override,
        extra_args_override=extra_args_override,
    )

    if devices:
        args.extend(["--devices", devices])
    if section_set:
        args.extend(["--set", section_set])
    if kernel_name:
        args.extend(["--kernel-name", kernel_name])
    if extra_args:
        args.extend(shlex.split(extra_args))
    return args


def _should_retry_with_stable_sections(
    *,
    section_set: str,
    extra_args: str,
    no_kernels_profiled: bool,
    report_exists: bool,
) -> bool:
    if report_exists or not no_kernels_profiled:
        return False
    if section_set.lower() not in {"full", "speedoflight"}:
        return False
    try:
        extra_tokens = shlex.split(extra_args)
    except ValueError:
        extra_tokens = extra_args.split()
    return "--section" not in extra_tokens


def _run_preflight(
    resolved_ncu: str,
    *,
    run_config: dict[str, object],
) -> dict[str, object]:
    nvcc_path = _resolve_executable("nvcc", fallbacks=("/usr/local/cuda/bin/nvcc",))
    if nvcc_path is None:
        return {"available": False, "reason": "nvcc executable not found for ncu preflight"}

    devices, _, _, _ = _resolve_ncu_options(run_config=run_config)

    with tempfile.TemporaryDirectory(prefix="kernel_evo_ncu_") as tmpdir:
        tmp = Path(tmpdir)
        source_file = tmp / "vecadd.cu"
        binary_file = tmp / "vecadd"
        report_base = tmp / "report"
        source_file.write_text(
            textwrap.dedent(
                """
                #include <cstdio>
                #include <cuda_runtime.h>

                __global__ void vecadd(const float* a, const float* b, float* c, int n) {
                  int i = blockIdx.x * blockDim.x + threadIdx.x;
                  if (i < n) c[i] = a[i] + b[i];
                }

                int main() {
                  const int n = 1 << 20;
                  const size_t bytes = n * sizeof(float);
                  float *a, *b, *c;
                  cudaMallocManaged(&a, bytes);
                  cudaMallocManaged(&b, bytes);
                  cudaMallocManaged(&c, bytes);
                  for (int i = 0; i < n; ++i) { a[i] = 1.0f; b[i] = 2.0f; }
                  for (int iter = 0; iter < 32; ++iter) {
                    vecadd<<<(n + 255) / 256, 256>>>(a, b, c, n);
                  }
                  cudaDeviceSynchronize();
                  std::printf("%f\\n", c[0]);
                  cudaFree(a); cudaFree(b); cudaFree(c);
                  return 0;
                }
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        compile_proc = run_profile_subprocess(
            [nvcc_path, "-O2", "-o", str(binary_file), str(source_file)],
            text=True,
            capture_output=True,
        )
        if compile_proc.returncode != 0:
            return {
                "available": False,
                "reason": "ncu preflight cuda compile failed",
                "compile_returncode": compile_proc.returncode,
                "compile_stdout_excerpt": compile_proc.stdout[-12000:],
                "compile_stderr_excerpt": compile_proc.stderr[-12000:],
            }

        probe_cmd = [
            resolved_ncu,
            "--section",
            "LaunchStats",
            "--target-processes",
            "all",
            "--force-overwrite",
            "--launch-count",
            "1",
            "--devices",
            devices,
            "--export",
            str(report_base),
            str(binary_file),
        ]
        probe_proc = run_profile_subprocess(probe_cmd, text=True, capture_output=True)
        report_file = report_base.with_suffix(".ncu-rep")
        no_kernels_profiled = "No kernels were profiled" in probe_proc.stdout
        return {
            "available": probe_proc.returncode == 0 and report_file.exists() and not no_kernels_profiled,
            "reason": (
                "ncu preflight succeeded"
                if probe_proc.returncode == 0 and report_file.exists() and not no_kernels_profiled
                else "ncu preflight captured no kernels on a native CUDA probe"
            ),
            "returncode": probe_proc.returncode,
            "command": probe_cmd,
            "devices": devices,
            "report_file": str(report_file),
            "report_exists": report_file.exists(),
            "stdout_excerpt": probe_proc.stdout[-12000:],
            "stderr_excerpt": probe_proc.stderr[-12000:],
        }


def _load_or_run_preflight(
    *,
    run_config: dict[str, object],
    resolved_ncu: str,
    out_dir: Path,
) -> dict[str, object]:
    cache_file = _cache_file(run_config, out_dir)
    expected_devices, _, _, _ = _resolve_ncu_options(run_config=run_config)
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if (
                isinstance(cached, dict)
                and "available" in cached
                and str(cached.get("devices", "") or "").strip() == expected_devices
            ):
                return cached
        except Exception:
            pass

    preflight = _run_preflight(resolved_ncu, run_config=run_config)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(preflight, ensure_ascii=False, indent=2), encoding="utf-8")
    return preflight


def _run_ncu_attempt(
    *,
    resolved_ncu: str,
    report_base: Path,
    run_config_path: Path,
    candidate_file: Path,
    reference_file: Path,
    target_run_work: Path,
    run_config: dict[str, object],
    set_override: str | None,
    kernel_name_override: str | None,
    extra_args_override: str | None,
    label: str,
) -> dict[str, object]:
    target_cmd = [
        resolved_ncu,
        "--target-processes",
        "all",
        "--force-overwrite",
        *_build_ncu_option_args(
            run_config=run_config,
            set_override=set_override,
            kernel_name_override=kernel_name_override,
            extra_args_override=extra_args_override,
        ),
        "--export",
        str(report_base),
        sys.executable,
        "-m",
        "kernel_evo.tools.profile_target",
        "--run-config",
        str(run_config_path),
        "--candidate-file",
        str(candidate_file),
        "--reference-file",
        str(reference_file),
        "--out-dir",
        str(target_run_work),
    ]
    proc = run_profile_subprocess(target_cmd, text=True, capture_output=True)
    report_file = report_base.with_suffix(".ncu-rep")
    devices, section_set, _, extra_args = _resolve_ncu_options(
        run_config=run_config,
        set_override=set_override,
        kernel_name_override=kernel_name_override,
        extra_args_override=extra_args_override,
    )
    no_kernels_profiled = "No kernels were profiled" in proc.stdout
    return {
        "label": label,
        "returncode": proc.returncode,
        "command": target_cmd,
        "report_file": str(report_file),
        "report_exists": report_file.exists(),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "stdout_excerpt": proc.stdout[-12000:],
        "stderr_excerpt": proc.stderr[-12000:],
        "no_kernels_profiled": no_kernels_profiled,
        "devices": devices,
        "section_set": section_set,
        "extra_args": extra_args,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Nsight Compute CLI for a candidate program.")
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--candidate-file", required=True)
    parser.add_argument("--reference-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ncu-set", dest="ncu_set", default=None)
    parser.add_argument("--ncu-kernel-name", default=None)
    parser.add_argument("--ncu-extra-args", default=None)
    parser.add_argument("--target-device", default=None)
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_config = json.loads(Path(args.run_config).read_text(encoding="utf-8"))
    ncu_path = str(run_config.get("profile_ncu_path", DEFAULT_NCU_PATH))
    resolved_ncu = _resolve_executable(
        ncu_path,
        fallbacks=_DEFAULT_NCU_FALLBACKS,
    )
    if resolved_ncu is None:
        summary = {"status": "skipped", "reason": f"ncu executable not found: {ncu_path}"}
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False))
        return

    preflight: dict[str, object] | None = None
    if not args.skip_preflight:
        preflight = _load_or_run_preflight(
            run_config=run_config,
            resolved_ncu=resolved_ncu,
            out_dir=out_dir,
        )

    target_run_config_path = Path(args.run_config).expanduser().resolve()
    temp_run_config_path: Path | None = None
    original_target_device = str(run_config.get("device", "cuda:0") or "cuda:0").strip() or "cuda:0"
    effective_target_device = _effective_target_device(
        run_config,
        target_device_override=args.target_device,
    )
    effective_devices, requested_set, _, requested_extra_args = _resolve_ncu_options(
        run_config=run_config,
        set_override=args.ncu_set,
        kernel_name_override=args.ncu_kernel_name,
        extra_args_override=args.ncu_extra_args,
    )
    if effective_target_device != original_target_device:
        run_config = dict(run_config)
        run_config["device"] = effective_target_device
        temp_dir = Path(tempfile.mkdtemp(prefix="kernel_evo_ncu_run_config_"))
        temp_run_config_path = temp_dir / "run_config.json"
        temp_run_config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
        target_run_config_path = temp_run_config_path

    report_base = out_dir / "report"
    target_run_work = _target_run_work_dir_for_ncu_child()
    attempts: list[dict[str, object]] = []
    try:
        attempts.append(
            _run_ncu_attempt(
                resolved_ncu=resolved_ncu,
                report_base=report_base,
                run_config_path=target_run_config_path,
                candidate_file=Path(args.candidate_file).expanduser().resolve(),
                reference_file=Path(args.reference_file).expanduser().resolve(),
                target_run_work=target_run_work,
                run_config=run_config,
                set_override=args.ncu_set,
                kernel_name_override=args.ncu_kernel_name,
                extra_args_override=args.ncu_extra_args,
                label="requested",
            )
        )
        first_attempt = attempts[0]
        if _should_retry_with_stable_sections(
            section_set=requested_set,
            extra_args=requested_extra_args,
            no_kernels_profiled=bool(first_attempt["no_kernels_profiled"]),
            report_exists=bool(first_attempt["report_exists"]),
        ):
            attempts.append(
                _run_ncu_attempt(
                    resolved_ncu=resolved_ncu,
                    report_base=report_base,
                    run_config_path=target_run_config_path,
                    candidate_file=Path(args.candidate_file).expanduser().resolve(),
                    reference_file=Path(args.reference_file).expanduser().resolve(),
                    target_run_work=target_run_work,
                    run_config=run_config,
                    set_override="",
                    kernel_name_override=args.ncu_kernel_name,
                    extra_args_override=_STABLE_SECTION_EXTRA_ARGS,
                    label="fallback_sections",
                )
            )
    finally:
        dest_run = out_dir / "target_run"
        dest_run.mkdir(parents=True, exist_ok=True)
        summary_tmp = target_run_work / "summary.json"
        if summary_tmp.exists():
            shutil.copy2(summary_tmp, dest_run / "summary.json")
        shutil.rmtree(target_run_work, ignore_errors=True)
        if temp_run_config_path is not None:
            shutil.rmtree(temp_run_config_path.parent, ignore_errors=True)
    report_file = report_base.with_suffix(".ncu-rep")
    final_attempt = next(
        (
            attempt
            for attempt in reversed(attempts)
            if bool(attempt["report_exists"]) and not bool(attempt["no_kernels_profiled"])
        ),
        attempts[-1],
    )

    summary: dict[str, object] = {
        "status": "completed" if int(final_attempt["returncode"]) == 0 else "failed",
        "returncode": final_attempt["returncode"],
        "command": final_attempt["command"],
        "report_file": str(report_file),
        "report_exists": report_file.exists(),
        "stdout_excerpt": final_attempt["stdout_excerpt"],
        "stderr_excerpt": final_attempt["stderr_excerpt"],
        "host_preflight": preflight,
        "effective_ncu_devices": effective_devices,
        "effective_target_device": effective_target_device,
        "attempts": [
            {
                key: value
                for key, value in attempt.items()
                if key not in {"stdout", "stderr", "no_kernels_profiled"}
            }
            for attempt in attempts
        ],
    }
    warnings: list[str] = []
    if preflight and not bool(preflight.get("available", False)):
        warnings.append("ncu host preflight failed, but target profiling was still attempted")
    if effective_target_device != original_target_device:
        warnings.append(
            f"overrode target device from {original_target_device} to {effective_target_device} "
            f"to match ncu devices {effective_devices}"
        )

    no_kernels_profiled = bool(final_attempt["no_kernels_profiled"])
    if no_kernels_profiled:
        summary["status"] = "skipped"
        summary["reason"] = "ncu collected no kernels"
        if len(attempts) > 1:
            summary["reason"] = "ncu collected no kernels across requested and fallback section attempts"

    if report_file.exists():
        import_cmd = [resolved_ncu, "--import", str(report_file), "--page", "raw", "--csv"]
        imported = run_profile_subprocess(import_cmd, text=True, capture_output=True)
        (out_dir / "report_raw.csv").write_text(imported.stdout, encoding="utf-8")
        if imported.returncode == 0:
            lines = [line for line in imported.stdout.splitlines() if line.strip()]
            summary["raw_csv_file"] = str(out_dir / "report_raw.csv")
            summary["raw_csv_preview"] = lines[:80]
        else:
            summary["import_failed"] = True
            summary["import_returncode"] = imported.returncode
            summary["import_stderr_excerpt"] = imported.stderr[-12000:]
    elif int(final_attempt["returncode"]) == 0 and not no_kernels_profiled:
        summary["status"] = "failed"
        summary["reason"] = "ncu finished without producing a report"

    if warnings:
        summary["warnings"] = warnings

    combined_stdout = "\n".join(
        [f"== ATTEMPT {attempt['label']} ==\n{attempt['stdout']}".rstrip() for attempt in attempts]
    ).strip()
    combined_stderr = "\n".join(
        [f"== ATTEMPT {attempt['label']} ==\n{attempt['stderr']}".rstrip() for attempt in attempts]
    ).strip()
    (out_dir / "stdout.txt").write_text(combined_stdout + ("\n" if combined_stdout else ""), encoding="utf-8")
    (out_dir / "stderr.txt").write_text(combined_stderr + ("\n" if combined_stderr else ""), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
