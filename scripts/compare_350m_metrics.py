#!/usr/bin/env python
"""Validate Q=1 full/RR and native-resume metric equivalence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

EXACT_FIELDS = (
    "optimizer_update",
    "valid_tokens",
    "total_valid_tokens",
    "full_context_equivalent_tokens",
    "total_full_context_equivalent_tokens",
    "selected_level_indices",
    "selected_level_counts",
    "selected_level_frequencies",
    "selected_level_tail_frequencies",
    "global_target_counts",
    "tail_probabilities",
    "learning_rate",
    "distributed_backend",
    "zero_stage",
    "attention_backend",
    "precision",
    "fp8",
    "source_tree_sha256",
)


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"metric row {line_number} is not an object: {path}")
        rows.append(value)
    return rows


def _maximum_numeric_difference(
    left: list[dict[str, Any]], right: list[dict[str, Any]], field: str
) -> float:
    differences = []
    for left_row, right_row in zip(left, right, strict=False):
        left_value = left_row.get(field)
        right_value = right_row.get(field)
        if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            differences.append(abs(float(left_value) - float(right_value)))
    return max(differences, default=0.0)


def compare_metrics(
    full: list[dict[str, Any]],
    rr: list[dict[str, Any]],
    resumed: list[dict[str, Any]],
    *,
    validation_loss_atol: float,
) -> dict[str, Any]:
    errors: list[str] = []

    def fail(message: str) -> None:
        if len(errors) < 100:
            errors.append(message)

    expected_full_updates = list(range(1, 21))
    expected_resumed_updates = list(range(11, 21))
    for label, rows, expected in (
        ("full", full, expected_full_updates),
        ("rr", rr, expected_full_updates),
        ("resumed", resumed, expected_resumed_updates),
    ):
        updates = [row.get("optimizer_update") for row in rows]
        if updates != expected:
            fail(f"{label} optimizer updates differ: {updates!r} != {expected!r}")
        for row_index, row in enumerate(rows):
            update = row.get("optimizer_update", row_index + 1)
            for field in ("signed_estimator_scalar", "gradient_norm", "learning_rate"):
                value = row.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    fail(f"{label} update {update} has non-finite {field}: {value!r}")
            tail_probabilities = row.get("tail_probabilities")
            if not isinstance(tail_probabilities, list) or not tail_probabilities:
                fail(f"{label} update {update} has invalid tail probabilities")
                continue
            if any(float(value) != 1.0 for value in tail_probabilities):
                fail(f"{label} update {update} is not a Q=1 baseline")
            selected = row.get("selected_level_indices")
            maximum_level = len(tail_probabilities) - 1
            if not isinstance(selected, list) or any(
                value != maximum_level for value in selected
            ):
                fail(f"{label} update {update} did not select only maximum context")
            counts = row.get("selected_level_counts")
            if isinstance(selected, list) and isinstance(counts, list):
                expected_count = int(update) * len(selected)
                if sum(int(value) for value in counts) != expected_count:
                    fail(
                        f"{label} update {update} selected count total differs from "
                        f"{expected_count}"
                    )

    def compare_exact_rows(
        label: str, left: list[dict[str, Any]], right: list[dict[str, Any]]
    ) -> None:
        if len(left) != len(right):
            fail(f"{label} row count differs: {len(left)} != {len(right)}")
        for row_index, (left_row, right_row) in enumerate(zip(left, right, strict=False)):
            update = left_row.get("optimizer_update", row_index)
            for field in EXACT_FIELDS:
                if left_row.get(field) != right_row.get(field):
                    fail(f"{label} update {update} differs at {field}")

    rr_resume_reference = rr[10:] if len(rr) >= 10 else []
    compare_exact_rows("full-vs-rr", full, rr)
    compare_exact_rows("rr-vs-resumed", rr_resume_reference, resumed)

    maximum_full_rr_validation_loss_error = 0.0
    maximum_rr_resume_validation_loss_error = 0.0

    def compare_validation_losses(
        label: str, left: list[dict[str, Any]], right: list[dict[str, Any]]
    ) -> float:
        maximum_error = 0.0
        for left_row, right_row in zip(left, right, strict=False):
            update = left_row.get("optimizer_update")
            left_has_loss = "full_validation_loss" in left_row
            right_has_loss = "full_validation_loss" in right_row
            if left_has_loss != right_has_loss:
                fail(f"{label} update {update} validation-loss presence differs")
                continue
            if not left_has_loss:
                continue
            left_loss = float(left_row["full_validation_loss"])
            right_loss = float(right_row["full_validation_loss"])
            if not math.isfinite(left_loss) or not math.isfinite(right_loss):
                fail(f"{label} update {update} has non-finite validation loss")
                continue
            error = abs(left_loss - right_loss)
            maximum_error = max(maximum_error, error)
            if error > validation_loss_atol:
                fail(
                    f"{label} update {update} validation loss error {error} exceeds "
                    f"{validation_loss_atol}"
                )
            for row_name, row, loss in (
                ("left", left_row, left_loss),
                ("right", right_row, right_loss),
            ):
                perplexity = row.get("full_validation_perplexity")
                if not isinstance(perplexity, (int, float)) or not math.isclose(
                    float(perplexity), math.exp(loss), rel_tol=1e-12, abs_tol=1e-12
                ):
                    fail(f"{label} update {update} {row_name} perplexity is inconsistent")
        return maximum_error

    maximum_full_rr_validation_loss_error = compare_validation_losses(
        "full-vs-rr", full, rr
    )
    maximum_rr_resume_validation_loss_error = compare_validation_losses(
        "rr-vs-resumed", rr_resume_reference, resumed
    )

    return {
        "schema_version": 1,
        "full_row_count": len(full),
        "rr_row_count": len(rr),
        "resumed_row_count": len(resumed),
        "validation_loss_atol": validation_loss_atol,
        "maximum_full_rr_validation_loss_absolute_error": (
            maximum_full_rr_validation_loss_error
        ),
        "maximum_rr_resume_validation_loss_absolute_error": (
            maximum_rr_resume_validation_loss_error
        ),
        "maximum_full_rr_estimator_scalar_absolute_error": _maximum_numeric_difference(
            full, rr, "signed_estimator_scalar"
        ),
        "maximum_rr_resume_estimator_scalar_absolute_error": _maximum_numeric_difference(
            rr_resume_reference, resumed, "signed_estimator_scalar"
        ),
        "maximum_full_rr_gradient_norm_absolute_error": _maximum_numeric_difference(
            full, rr, "gradient_norm"
        ),
        "maximum_rr_resume_gradient_norm_absolute_error": _maximum_numeric_difference(
            rr_resume_reference, resumed, "gradient_norm"
        ),
        "failure_count": len(errors),
        "first_failure": errors[0] if errors else None,
        "passed": not errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--full", required=True)
    parser.add_argument("--rr", required=True)
    parser.add_argument("--resumed", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-loss-atol", type=float, default=1e-3)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    output = Path(arguments.output).resolve()
    if output.exists():
        raise FileExistsError(f"comparison output already exists: {output}")
    paths = {
        "full": Path(arguments.full).resolve(),
        "rr": Path(arguments.rr).resolve(),
        "resumed": Path(arguments.resumed).resolve(),
    }
    report = compare_metrics(
        _read_metrics(paths["full"]),
        _read_metrics(paths["rr"]),
        _read_metrics(paths["resumed"]),
        validation_loss_atol=arguments.validation_loss_atol,
    )
    report["paths"] = {key: str(value) for key, value in paths.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
