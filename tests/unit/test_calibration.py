import json

import pytest
import torch

from umcg.calibration.exact import (
    audit_schedule,
    candidate_schedules,
    correction_grams,
    gradient_level_and_correction_grams,
    maximum_level_probabilities,
    outcome_coefficients,
    score_schedule,
)
from umcg.cli.calibrate import (
    CalibrationParentCache,
    SplitMeasurements,
    _calibration_split_name,
    _collect_calibration_samples,
    _pack_split,
    _resolve_parent_batch_counts,
    _split_statistics,
    _validate_arguments,
    _write_outputs,
    build_parser,
)


class FakeParentStream:
    def __init__(self, samples):
        self.samples = iter(samples)

    def next_sample(self):
        return next(self.samples)


def parent_sample(document_hash: str, chunk_index: int):
    input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    attention_mask = torch.ones(4, dtype=torch.bool)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "causal_target_mask": attention_mask[:-1] & attention_mask[1:],
        "position_ids": torch.arange(4),
        "document_hash": document_hash,
        "chunk_index": chunk_index,
        "token_start": chunk_index * 4,
        "token_end": (chunk_index + 1) * 4,
        "url": "",
        "timestamp": "",
    }


def required_calibration_arguments():
    return [
        "--model_backend",
        "reference",
        "--model_config",
        "model.json",
        "--precision",
        "bfloat16",
        "--context_preset",
        "4096",
        "--c4_source",
        "local_raw",
        "--c4_local_path",
        "/data/c4",
        "--output",
        "estimator.json",
    ]


def level_statistics(level_vectors: torch.Tensor):
    batch_grams = torch.einsum("bkd,bld->bkl", level_vectors, level_vectors)
    mean_vectors = level_vectors.mean(dim=0)
    mean_gram = mean_vectors @ mean_vectors.T
    return batch_grams, mean_gram


def test_calibration_parser_uses_exact_protocol_defaults_and_aliases():
    parser = build_parser()
    arguments = parser.parse_args(required_calibration_arguments())
    assert _resolve_parent_batch_counts(arguments) == {
        "measurement": 64,
        "selection": 32,
        "audit": 32,
    }
    assert arguments.logical_parent_batch_size == 128
    assert arguments.max_parent_batch_size_per_gpu == 64
    assert arguments.memory_limit_fraction == 0.85
    assert arguments.timing_repeats == 1
    assert arguments.activation_checkpointing is True
    _validate_arguments(arguments, world_size=2)

    legacy = parser.parse_args(
        [
            *required_calibration_arguments(),
            "--parent_batches",
            "4",
            "--batch_size",
            "32",
        ]
    )
    assert _resolve_parent_batch_counts(legacy)["measurement"] == 4
    assert legacy.max_parent_batch_size_per_gpu == 32
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                *required_calibration_arguments(),
                "--measurement_parent_batches",
                "8",
                "--parent_batches",
                "4",
            ]
        )
    invalid_repeats = parser.parse_args(
        [*required_calibration_arguments(), "--timing_repeats", "2"]
    )
    with pytest.raises(ValueError, match="timing_repeats=1"):
        _validate_arguments(invalid_repeats, world_size=2)
    with pytest.raises(SystemExit):
        parser.parse_args(
            [*required_calibration_arguments(), "--sketch_dimension", "256"]
        )


def test_calibration_help_explains_logical_and_physical_batches():
    help_text = " ".join(build_parser().format_help().split())
    assert "model, precision, and attention:" in help_text
    assert "C4 data and immutable parent cache:" in help_text
    assert "logical batches, physical chunks, and split sizes:" in help_text
    assert "Global parent samples represented by one gradient observation" in help_text
    assert "Maximum exact-path allocated VRAM fraction" in help_text


def test_document_hash_split_is_deterministic_and_disjoint():
    seed = 777
    samples = [
        parent_sample(f"document-{document_index}", chunk_index)
        for document_index in range(500)
        for chunk_index in range(2)
    ]
    splits, metadata = _collect_calibration_samples(
        FakeParentStream(samples),
        logical_batch_size=2,
        batch_counts={"measurement": 2, "selection": 1, "audit": 1},
        seed=seed,
    )
    hash_sets = {
        name: {sample["document_hash"] for sample in split_samples}
        for name, split_samples in splits.items()
    }
    assert {name: len(values) for name, values in splits.items()} == {
        "measurement": 4,
        "selection": 2,
        "audit": 2,
    }
    assert not (hash_sets["measurement"] & hash_sets["selection"])
    assert not (hash_sets["measurement"] & hash_sets["audit"])
    assert not (hash_sets["selection"] & hash_sets["audit"])
    assert set(metadata["overlap_document_counts"].values()) == {0}
    for name, hashes in hash_sets.items():
        assert {_calibration_split_name(value, seed) for value in hashes} == {name}


def test_parent_cache_preserves_one_logical_batch_across_two_ranks():
    samples = [parent_sample(f"document-{index}", 0) for index in range(8)]
    cache = CalibrationParentCache(
        manifest={},
        splits={"measurement": _pack_split(samples)},
    )
    rank_zero = cache.rank_batch(
        "measurement", 1, logical_batch_size=4, rank=0, world_size=2
    )
    rank_one = cache.rank_batch(
        "measurement", 1, logical_batch_size=4, rank=1, world_size=2
    )
    assert rank_zero.input_ids.shape == (2, 4)
    assert rank_one.input_ids.shape == (2, 4)
    assert rank_zero.document_hashes == ["document-4", "document-5"]
    assert rank_one.document_hashes == ["document-6", "document-7"]


def test_fixed_monotone_grid_has_exactly_35_schedules():
    candidates = candidate_schedules(4)
    assert len(candidates) == 35
    assert len(set(candidates)) == 35
    assert (1.0, 1.0, 1.0, 1.0) in candidates
    assert (1.0, 0.5, 0.25, 0.125) in candidates
    assert all(
        left >= right
        for candidate in candidates
        for left, right in zip(candidate, candidate[1:], strict=False)
    )
    torch.testing.assert_close(
        maximum_level_probabilities((1.0, 0.5, 0.25, 0.125)),
        torch.tensor([0.5, 0.25, 0.125, 0.125], dtype=torch.float64),
        rtol=0,
        atol=0,
    )


def test_full_coordinate_level_and_correction_grams_match_flat_vectors():
    levels = [
        (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([5.0, 6.0])),
        (torch.tensor([[2.0, 4.0], [6.0, 8.0]]), torch.tensor([7.0, 9.0])),
        (torch.tensor([[1.0, 3.0], [8.0, 5.0]]), torch.tensor([11.0, 10.0])),
    ]
    flattened = torch.stack(
        [torch.cat([parameter.reshape(-1) for parameter in level]).double() for level in levels]
    )
    corrections = torch.cat((flattened[:1], flattened[1:] - flattened[:-1]))
    level_gram, correction_gram = gradient_level_and_correction_grams(
        levels, chunk_size=2
    )
    torch.testing.assert_close(level_gram, flattened @ flattened.T, rtol=0, atol=0)
    torch.testing.assert_close(
        correction_gram, corrections @ corrections.T, rtol=0, atol=0
    )
    torch.testing.assert_close(
        correction_gram, correction_grams(level_gram), rtol=0, atol=0
    )


def test_analytic_schedule_score_matches_explicit_four_outcomes():
    level_vectors = torch.tensor(
        [
            [[1.0, 2.0], [2.0, 4.0], [3.0, 7.0], [5.0, 8.0]],
            [[2.0, -1.0], [3.0, 0.0], [4.0, 2.0], [7.0, 3.0]],
        ],
        dtype=torch.float64,
    )
    schedule = (1.0, 0.5, 0.25, 0.125)
    batch_grams, mean_gram = level_statistics(level_vectors)
    costs = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
    evaluation = score_schedule(schedule, batch_grams, mean_gram, costs)

    corrections = torch.cat(
        (level_vectors[:, :1], level_vectors[:, 1:] - level_vectors[:, :-1]), dim=1
    )
    coefficients = outcome_coefficients(schedule)
    probabilities = maximum_level_probabilities(schedule)
    outcomes = torch.einsum("mk,bkd->bmd", coefficients, corrections)
    expected_second_moment = torch.einsum(
        "m,bm->b", probabilities, outcomes.square().sum(dim=2)
    )
    expected_variance = expected_second_moment.mean() - mean_gram[-1, -1]
    torch.testing.assert_close(
        evaluation.per_batch_second_moment, expected_second_moment, rtol=1e-14, atol=1e-14
    )
    assert evaluation.summary["gradient_variance"] == pytest.approx(expected_variance)
    assert evaluation.summary["expected_cuda_cost_ms"] == pytest.approx(2.5)
    assert evaluation.summary["expected_correction_coefficients"] == pytest.approx(
        [1.0, 1.0, 1.0, 1.0]
    )


def test_independent_exact_audit_passes_for_cheaper_low_noise_schedule():
    generator = torch.Generator().manual_seed(11)
    parent_count = 32
    base = torch.randn((parent_count, 8), generator=generator, dtype=torch.float64) * 10
    increments = torch.randn(
        (parent_count, 3, 8), generator=generator, dtype=torch.float64
    ) * 0.001
    levels = [base]
    for level_index in range(3):
        levels.append(levels[-1] + increments[:, level_index])
    level_vectors = torch.stack(levels, dim=1)
    batch_grams, mean_gram = level_statistics(level_vectors)
    full_vectors = level_vectors[:, -1]

    audit, passed = audit_schedule(
        (1.0, 0.5, 0.25, 0.125),
        batch_grams,
        mean_gram,
        full_vectors @ full_vectors.T,
        torch.tensor([1.0, 2.0, 4.0, 8.0]),
        bootstrap_samples=2_000,
        bootstrap_seed=2718,
    )
    assert passed
    assert audit["unbiasedness_passed"]
    assert audit["efficiency_passed"]
    assert audit["relative_l2_error"] <= 1e-12
    assert audit["cosine_similarity"] == pytest.approx(1.0)
    assert audit["efficiency_ratio_95_percent_confidence_interval"][1] < 1.0


def test_split_statistics_compare_direct_and_derived_correction_grams():
    level_vectors = torch.tensor(
        [
            [[1.0, 2.0], [2.0, 3.0], [4.0, 3.0]],
            [[2.0, 1.0], [3.0, 2.0], [3.0, 5.0]],
        ],
        dtype=torch.float64,
    )
    batch_level, mean_level = level_statistics(level_vectors)
    corrections = torch.cat(
        (level_vectors[:, :1], level_vectors[:, 1:] - level_vectors[:, :-1]), dim=1
    )
    batch_correction = torch.einsum("bkd,bld->bkl", corrections, corrections)
    mean_corrections = corrections.mean(dim=0)
    measurements = SplitMeasurements(
        batch_level_grams=batch_level,
        batch_correction_grams=batch_correction,
        mean_level_gram=mean_level,
        mean_correction_gram=mean_corrections @ mean_corrections.T,
        level_times_ms=torch.tensor([[1.0, 2.0, 5.0], [3.0, 4.0, 7.0]]),
        level_peak_memory_bytes=torch.tensor([[10, 20, 30], [11, 19, 31]]),
    )
    statistics = _split_statistics(measurements)
    assert statistics["correction_second_moment_v_k"] == pytest.approx(
        torch.diagonal(batch_correction.mean(dim=0)).tolist()
    )
    assert statistics["mean_level_cuda_time_ms_c_k"] == pytest.approx([2.0, 3.0, 6.0])
    assert statistics["incremental_cuda_time_ms"] == pytest.approx([2.0, 1.0, 3.0])
    assert statistics["maximum_level_peak_memory_bytes"] == [11, 20, 31]
    assert (
        statistics["per_batch_level_to_direct_correction_gram_relative_residual"]
        <= 1e-15
    )
    assert (
        statistics["mean_gradient_level_to_direct_correction_gram_relative_residual"]
        <= 1e-15
    )


def test_failed_audit_writes_only_a_diagnostic_report(tmp_path):
    output = tmp_path / "estimator.json"
    estimator = {"schema_version": 1}
    report = {"schema_version": 3, "independent_audit": {"passed": False}}
    with pytest.raises(RuntimeError, match="audit failed"):
        _write_outputs(
            output,
            estimator=estimator,
            report=report,
            audit_passed=False,
            diagnostic_only=False,
        )
    assert not output.exists()
    report_path = output.with_suffix(".json.report.json")
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_passed_audit_writes_report_then_estimator(tmp_path):
    output = tmp_path / "estimator.json"
    estimator = {"schema_version": 1}
    report = {"schema_version": 3, "independent_audit": {"passed": True}}
    report_path = _write_outputs(
        output,
        estimator=estimator,
        report=report,
        audit_passed=True,
        diagnostic_only=False,
    )
    assert json.loads(output.read_text(encoding="utf-8")) == estimator
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
