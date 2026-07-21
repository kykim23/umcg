#!/usr/bin/env python
"""Compare exported weights or complete DDP checkpoints for validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file


class Comparison:
    def __init__(self, *, rtol: float, atol: float) -> None:
        self.rtol = rtol
        self.atol = atol
        self.tensor_count = 0
        self.maximum_absolute_error = 0.0
        self.maximum_absolute_error_location: str | None = None
        self.failure_count = 0
        self.first_failure: str | None = None
        self.floating_element_count = 0
        self.sum_absolute_error = 0.0
        self.sum_squared_error = 0.0
        self.reference_sum_squared = 0.0

    def _record_failure(self, message: str) -> None:
        self.failure_count += 1
        if self.first_failure is None:
            self.first_failure = message

    def _record_absolute_error(self, error: float, location: str) -> None:
        if error > self.maximum_absolute_error:
            self.maximum_absolute_error = error
            self.maximum_absolute_error_location = location

    def _record_numpy_statistics(self, left: np.ndarray, right: np.ndarray) -> None:
        difference = np.abs(left - right).astype(np.float64, copy=False)
        reference = np.abs(right).astype(np.float64, copy=False)
        self.floating_element_count += int(difference.size)
        self.sum_absolute_error += float(difference.sum())
        self.sum_squared_error += float(np.square(difference).sum())
        self.reference_sum_squared += float(np.square(reference).sum())

    def _record_torch_statistics(self, left: torch.Tensor, right: torch.Tensor) -> None:
        difference = (left - right).abs().float()
        reference = right.abs().float()
        self.floating_element_count += difference.numel()
        self.sum_absolute_error += float(difference.sum().double().cpu())
        self.sum_squared_error += float((difference * difference).sum().double().cpu())
        self.reference_sum_squared += float((reference * reference).sum().double().cpu())

    def compare(self, left: Any, right: Any, location: str = "root") -> None:
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
                self._record_failure(f"array type differs at {location}")
                return
            if left.shape != right.shape or left.dtype != right.dtype:
                self._record_failure(f"array metadata differs at {location}")
                return
            if np.issubdtype(left.dtype, np.floating):
                if left.size:
                    self._record_absolute_error(
                        float(np.max(np.abs(left - right))), location
                    )
                    self._record_numpy_statistics(left, right)
                try:
                    np.testing.assert_allclose(left, right, rtol=self.rtol, atol=self.atol)
                except AssertionError as error:
                    self._record_failure(f"{location}: {error}")
            elif not np.array_equal(left, right):
                self._record_failure(f"array value differs at {location}")
            self.tensor_count += 1
            return
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
                self._record_failure(f"tensor type differs at {location}")
                return
            if left.shape != right.shape or left.dtype != right.dtype:
                self._record_failure(f"tensor metadata differs at {location}")
                return
            if left.is_floating_point() or left.is_complex():
                if left.numel():
                    error = float((left - right).abs().max().detach().float().cpu())
                    self._record_absolute_error(error, location)
                    self._record_torch_statistics(left, right)
                try:
                    torch.testing.assert_close(left, right, rtol=self.rtol, atol=self.atol)
                except AssertionError as error:
                    self._record_failure(f"{location}: {error}")
            elif not torch.equal(left, right):
                self._record_failure(f"tensor value differs at {location}")
            self.tensor_count += 1
            return
        if isinstance(left, dict) or isinstance(right, dict):
            if not isinstance(left, dict) or not isinstance(right, dict):
                self._record_failure(f"mapping type differs at {location}")
                return
            if set(left) != set(right):
                self._record_failure(f"mapping keys differ at {location}")
            for key in sorted(set(left) & set(right), key=str):
                self.compare(left[key], right[key], f"{location}.{key}")
            return
        if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            if type(left) is not type(right) or len(left) != len(right):
                self._record_failure(f"sequence differs at {location}")
                return
            for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
                self.compare(left_item, right_item, f"{location}[{index}]")
            return
        if isinstance(left, float) or isinstance(right, float):
            if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
                self._record_failure(f"scalar type differs at {location}")
                return
            error = abs(float(left) - float(right))
            self._record_absolute_error(error, location)
            if not torch.isclose(
                torch.tensor(float(left)),
                torch.tensor(float(right)),
                rtol=self.rtol,
                atol=self.atol,
            ):
                self._record_failure(
                    f"floating scalar differs at {location}: {left} != {right}"
                )
            return
        if left != right:
            self._record_failure(f"value differs at {location}: {left!r} != {right!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--mode", required=True, choices=("weights", "ddp-checkpoints"))
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--maximum-relative-l2-error", type=float)
    return parser


def _load_complete_ddp_checkpoint(path: Path) -> dict[str, Any]:
    if not (path / "COMPLETE").is_file():
        raise ValueError(f"checkpoint is incomplete: {path}")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("distributed_backend") != "ddp":
        raise ValueError(f"checkpoint is not DDP: {path}")
    rank_paths = sorted(path.glob("rank-*.pt"))
    if not rank_paths:
        raise ValueError(f"checkpoint has no rank state: {path}")
    return {
        "optimizer_update": manifest["optimizer_update"],
        "replicated_state": torch.load(
            path / "replicated-state.pt", map_location="cpu", weights_only=False
        ),
        "rank_states": [
            torch.load(rank_path, map_location="cpu", weights_only=False)
            for rank_path in rank_paths
        ],
    }


def main() -> None:
    arguments = build_parser().parse_args()
    output = Path(arguments.output).resolve()
    if output.exists():
        raise FileExistsError(f"comparison output already exists: {output}")
    left_path = Path(arguments.left).resolve()
    right_path = Path(arguments.right).resolve()
    if arguments.mode == "weights":
        left = load_file(str(left_path), device="cpu")
        right = load_file(str(right_path), device="cpu")
    else:
        left = _load_complete_ddp_checkpoint(left_path)
        right = _load_complete_ddp_checkpoint(right_path)
    comparison = Comparison(rtol=arguments.rtol, atol=arguments.atol)
    comparison.compare(left, right)
    relative_l2_error = (
        (comparison.sum_squared_error / comparison.reference_sum_squared) ** 0.5
        if comparison.reference_sum_squared
        else 0.0
    )
    if (
        arguments.maximum_relative_l2_error is not None
        and relative_l2_error > arguments.maximum_relative_l2_error
    ):
        comparison._record_failure(
            f"relative L2 error {relative_l2_error} exceeds "
            f"{arguments.maximum_relative_l2_error}"
        )
    report = {
        "schema_version": 2,
        "mode": arguments.mode,
        "left": str(left_path),
        "right": str(right_path),
        "rtol": arguments.rtol,
        "atol": arguments.atol,
        "tensor_count": comparison.tensor_count,
        "floating_element_count": comparison.floating_element_count,
        "maximum_absolute_error": comparison.maximum_absolute_error,
        "maximum_absolute_error_location": comparison.maximum_absolute_error_location,
        "mean_absolute_error": (
            comparison.sum_absolute_error / comparison.floating_element_count
            if comparison.floating_element_count
            else 0.0
        ),
        "root_mean_square_error": (
            (comparison.sum_squared_error / comparison.floating_element_count) ** 0.5
            if comparison.floating_element_count
            else 0.0
        ),
        "relative_l2_error": relative_l2_error,
        "maximum_relative_l2_error": arguments.maximum_relative_l2_error,
        "failure_count": comparison.failure_count,
        "first_failure": comparison.first_failure,
        "passed": comparison.failure_count == 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if comparison.failure_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
