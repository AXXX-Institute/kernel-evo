from __future__ import annotations

import argparse

import pytest

pytest.importorskip("torch")

from kernel_evo.commands import compare as compare_cmd  # noqa: E402
from kernel_evo.commands import evolve as evolve_cmd  # noqa: E402


def test_evolve_parser_accepts_runtime_precision() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    evolve_cmd.setup_parser(subparsers)
    args = parser.parse_args(
        ["evolve", "--experiment-name", "x", "--model-name", "m", "--precision", "fp8", "--runtime-precision", "bf16"]
    )
    assert args.runtime_precision == "bf16"


def test_compare_parser_accepts_runtime_precision() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    compare_cmd.setup_parser(subparsers)
    args = parser.parse_args(
        ["compare", "--program-a", "a.py", "--program-b", "b.py", "--precision", "fp8", "--runtime-precision", "fp32"]
    )
    assert args.runtime_precision == "fp32"
