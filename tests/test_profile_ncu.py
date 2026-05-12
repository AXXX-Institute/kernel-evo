from __future__ import annotations

from pathlib import Path

from kernel_evo.tools import profile_ncu


def test_effective_target_device_uses_run_config_device_by_default() -> None:
    run_config = {
        "device": "cuda:7",
    }

    assert profile_ncu._effective_target_device(run_config) == "cuda:7"


def test_resolve_ncu_options_uses_device_index_by_default() -> None:
    run_config = {
        "device": "cuda:7",
    }

    devices, section_set, kernel_name, extra_args = profile_ncu._resolve_ncu_options(run_config=run_config)

    assert devices == "7"
    assert section_set == "full"
    assert kernel_name == ""
    assert extra_args == ""


def test_effective_target_device_ignores_removed_profile_ncu_devices_field() -> None:
    run_config = {
        "device": "cuda:7",
        "profile_ncu_devices": "2",
    }

    assert profile_ncu._effective_target_device(run_config) == "cuda:7"


def test_retry_with_stable_sections_for_speed_of_light_without_explicit_sections() -> None:
    should_retry = profile_ncu._should_retry_with_stable_sections(
        section_set="speedOfLight",
        extra_args="",
        no_kernels_profiled=True,
        report_exists=False,
    )

    assert should_retry is True


def test_no_retry_when_sections_are_already_explicit() -> None:
    should_retry = profile_ncu._should_retry_with_stable_sections(
        section_set="",
        extra_args="--section LaunchStats --section Occupancy",
        no_kernels_profiled=True,
        report_exists=False,
    )

    assert should_retry is False


def test_preflight_cache_is_invalidated_when_device_changes(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "artifact" / "ncu"
    out_dir.mkdir(parents=True)
    cache_file = tmp_path / "artifact" / "ncu_host_preflight.json"
    cache_file.write_text('{"available": true, "devices": "7"}', encoding="utf-8")

    run_config = {
        "profile_artifacts_dir": str(tmp_path / "artifact"),
        "device": "cuda:2",
    }

    calls: list[str] = []

    def fake_run_preflight(
        resolved_ncu: str,
        *,
        run_config: dict[str, object],
    ) -> dict[str, object]:
        devices, _, _, _ = profile_ncu._resolve_ncu_options(run_config=run_config)
        calls.append(devices)
        return {"available": False, "devices": "2", "reason": "rerun"}

    monkeypatch.setattr(profile_ncu, "_run_preflight", fake_run_preflight)

    result = profile_ncu._load_or_run_preflight(
        run_config=run_config,
        resolved_ncu="/fake/ncu",
        out_dir=out_dir,
    )

    assert result["devices"] == "2"
    assert calls == ["2"]
