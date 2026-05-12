from __future__ import annotations

from kernel_evo.core.stages.profile.summary_compaction import summarize_ncu_for_llm


def test_summarize_ncu_for_llm_compacts_raw_csv_into_kernel_overview() -> None:
    summary = {
        "status": "completed",
        "returncode": 0,
        "report_exists": True,
        "report_file": "/tmp/report.ncu-rep",
        "effective_ncu_devices": "7",
        "effective_target_device": "cuda:7",
        "warnings": ["ncu host preflight failed, but target profiling was still attempted"],
        "host_preflight": {
            "available": False,
            "reason": "ncu preflight captured no kernels on a native CUDA probe",
            "returncode": 0,
            "devices": "7",
            "stdout_excerpt": "No kernels were profiled.",
            "stderr_excerpt": "",
        },
        "attempts": [
            {
                "label": "requested",
                "returncode": 0,
                "report_exists": True,
                "devices": "7",
                "section_set": "speedOfLight",
                "extra_args": "",
                "stdout_excerpt": "large success stdout that should not be copied for successful attempts",
            }
        ],
        "raw_csv_preview": [
            (
                '"ID","Kernel Name","Block Size","Grid Size",'
                '"launch__occupancy_cluster_gpu_pct","launch__occupancy_cluster_pct",'
                '"launch__registers_per_thread","launch__shared_mem_per_block_allocated",'
                '"sm__maximum_warps_per_active_cycle_pct","profiler__replayer_passes"'
            ),
            '"","","block","grid","%","%","register/thread","Kbyte/block","%","pass"',
            (
                '"0","void cudnn::nchwToNhwcKernel<__half>(...)","(256, 1, 1)",'
                '"(1, 2, 64)","0.0","0.0","32","3.2","100.0","1"'
            ),
            (
                '"1","fused_epilogue_kernel_half_vec(__half *, const __half *, float)",'
                '"(256, 1, 1)","(262144, 1, 1)","100.0","100.0","32","1.152",'
                '"100.0","1"'
            ),
            (
                '"2","fused_epilogue_kernel_half_vec(__half *, const __half *, float)",'
                '"(256, 1, 1)","(262144, 1, 1)","100.0","100.0","36","1.280",'
                '"96.0","1"'
            ),
            '"3","sm90_xmma_dgrad_implicit_gemm","(384, 1, 1)","(30, 4, 1)","12.0","18.75","168","233.472","18.75","4"',
        ],
    }

    compact = summarize_ncu_for_llm(summary)

    assert "raw_csv_preview" not in compact
    assert compact["status"] == "completed"
    assert compact["host_preflight"]["reason"] == "ncu preflight captured no kernels on a native CUDA probe"
    assert compact["attempts"] == [
        {
            "label": "requested",
            "returncode": 0,
            "report_exists": True,
            "devices": "7",
            "section_set": "speedOfLight",
            "extra_args": "",
        }
    ]

    kernel_overview = compact["kernel_overview"]
    assert kernel_overview["preview_row_count"] == 4
    assert kernel_overview["unique_kernel_count"] == 3
    assert kernel_overview["layout_transform_occurrences"] == 1

    fused_kernel = kernel_overview["kernels"][0]
    assert fused_kernel["kernel_name"] == "fused_epilogue_kernel_half_vec(__half *, const __half *, float)"
    assert fused_kernel["occurrences"] == 2
    assert fused_kernel["sample_ids"] == ["1", "2"]
    assert fused_kernel["registers_per_thread"] == {"min": 32.0, "max": 36.0}
    assert fused_kernel["shared_mem_per_block_kbyte"] == {"min": 1.152, "max": 1.28}


def test_summarize_ncu_for_llm_keeps_failure_diagnostics_without_raw_csv() -> None:
    summary = {
        "status": "skipped",
        "reason": "ncu collected no kernels",
        "returncode": 0,
        "report_exists": False,
        "stdout_excerpt": "x" * 1400,
        "stderr_excerpt": "",
        "attempts": [
            {
                "label": "requested",
                "returncode": 0,
                "report_exists": False,
                "devices": "2",
                "section_set": "speedOfLight",
                "stdout_excerpt": "No kernels were profiled.\n" * 80,
                "stderr_excerpt": "",
            }
        ],
    }

    compact = summarize_ncu_for_llm(summary)

    assert compact["status"] == "skipped"
    assert compact["reason"] == "ncu collected no kernels"
    assert "kernel_overview" not in compact
    assert compact["stdout_excerpt"].endswith("...[truncated]")
    assert compact["attempts"][0]["stdout_excerpt"].endswith("...[truncated]")
