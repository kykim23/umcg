import copy
import importlib.util
import math
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


state_comparison = _load_script("compare_validation_states")
metric_comparison = _load_script("compare_350m_metrics")


def test_state_comparison_records_failure_report_statistics_without_stopping_early():
    comparison = state_comparison.Comparison(rtol=0.0, atol=0.1)
    comparison.compare(
        {
            "first": torch.tensor([0.0, 1.0]),
            "second": torch.tensor([2.0]),
        },
        {
            "first": torch.tensor([0.05, 1.3]),
            "second": torch.tensor([2.2]),
        },
    )

    assert comparison.tensor_count == 2
    assert comparison.failure_count == 2
    assert comparison.first_failure is not None
    assert comparison.maximum_absolute_error == pytest.approx(0.3)
    assert comparison.maximum_absolute_error_location == "root.first"


def _metric_row(update: int) -> dict:
    validation_loss = 8.0 - update / 100.0
    row = {
        "optimizer_update": update,
        "valid_tokens": 100 + update,
        "total_valid_tokens": sum(100 + value for value in range(1, update + 1)),
        "full_context_equivalent_tokens": 100 + update,
        "total_full_context_equivalent_tokens": sum(
            100 + value for value in range(1, update + 1)
        ),
        "selected_level_indices": [3, 3, 3, 3],
        "selected_level_counts": [0, 0, 0, update * 4],
        "selected_level_frequencies": [0.0, 0.0, 0.0, 1.0],
        "selected_level_tail_frequencies": [1.0, 1.0, 1.0, 1.0],
        "global_target_counts": [25, 50, 75, 100 + update],
        "tail_probabilities": [1.0, 1.0, 1.0, 1.0],
        "learning_rate": 0.001 / update,
        "distributed_backend": "ddp",
        "zero_stage": None,
        "attention_backend": "pytorch_sdpa_cudnn",
        "precision": "bfloat16",
        "fp8": None,
        "source_tree_sha256": "a" * 64,
        "signed_estimator_scalar": 9.0 / update,
        "gradient_norm": 3.0 / update,
    }
    if update % 10 == 0:
        row["full_validation_loss"] = validation_loss
        row["full_validation_perplexity"] = math.exp(validation_loss)
    return row


def test_350m_metric_comparison_enforces_exact_trajectory_and_loss_tolerance():
    full = [_metric_row(update) for update in range(1, 21)]
    rr = copy.deepcopy(full)
    for row in rr:
        row["signed_estimator_scalar"] += 1e-4
        row["gradient_norm"] += 2e-4
        if "full_validation_loss" in row:
            row["full_validation_loss"] += 5e-4
            row["full_validation_perplexity"] = math.exp(row["full_validation_loss"])
    resumed = copy.deepcopy(rr[10:])

    report = metric_comparison.compare_metrics(
        full, rr, resumed, validation_loss_atol=1e-3
    )

    assert report["passed"] is True
    assert report["maximum_full_rr_validation_loss_absolute_error"] == pytest.approx(5e-4)

    resumed[0]["global_target_counts"][-1] += 1
    failed = metric_comparison.compare_metrics(
        full, rr, resumed, validation_loss_atol=1e-3
    )
    assert failed["passed"] is False
    assert "global_target_counts" in failed["first_failure"]
