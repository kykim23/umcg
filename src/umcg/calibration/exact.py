"""Projection-free gradient statistics and analytic Russian-roulette scoring."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ScheduleEvaluation:
    summary: dict[str, object]
    per_batch_second_moment: torch.Tensor


def candidate_schedules(level_count: int) -> list[tuple[float, ...]]:
    """Return the monotone tail schedules on the fixed five-value grid."""
    if level_count <= 0:
        raise ValueError("level_count must be positive")
    grid = (1.0, 0.75, 0.5, 0.25, 0.125)
    schedules = []
    for tail in itertools.product(grid, repeat=level_count - 1):
        schedule = (1.0, *tail)
        if all(left >= right for left, right in zip(schedule, schedule[1:], strict=False)):
            schedules.append(schedule)
    return schedules


def maximum_level_probabilities(schedule: tuple[float, ...]) -> torch.Tensor:
    if not schedule:
        raise ValueError("tail schedule must be non-empty")
    values = [
        schedule[index] - schedule[index + 1] for index in range(len(schedule) - 1)
    ] + [schedule[-1]]
    probabilities = torch.tensor(values, dtype=torch.float64)
    if (probabilities < 0).any() or not torch.isclose(
        probabilities.sum(), torch.ones((), dtype=torch.float64), rtol=0, atol=1e-12
    ):
        raise ValueError("tail schedule does not define maximum-level probabilities")
    return probabilities


def difference_operator(level_count: int) -> torch.Tensor:
    if level_count <= 0:
        raise ValueError("level_count must be positive")
    result = torch.zeros((level_count, level_count), dtype=torch.float64)
    result[0, 0] = 1.0
    for index in range(1, level_count):
        result[index, index - 1] = -1.0
        result[index, index] = 1.0
    return result


def correction_grams(level_grams: torch.Tensor) -> torch.Tensor:
    if level_grams.ndim not in {2, 3} or level_grams.shape[-1] != level_grams.shape[-2]:
        raise ValueError("level Gram input must have shape [K,K] or [B,K,K]")
    transform = difference_operator(level_grams.shape[-1]).to(level_grams)
    return transform @ level_grams @ transform.transpose(-1, -2)


def cosine_matrix(gram: torch.Tensor) -> torch.Tensor:
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("cosine input must be a square Gram matrix")
    norms = torch.sqrt(torch.clamp(torch.diagonal(gram), min=0.0))
    denominator = norms[:, None] * norms[None, :]
    cosine = torch.where(denominator > 0, gram / denominator, torch.zeros_like(gram))
    zero = norms == 0
    cosine[zero[:, None] & zero[None, :]] = 1.0
    return torch.clamp(cosine, min=-1.0, max=1.0)


def outcome_coefficients(schedule: tuple[float, ...]) -> torch.Tensor:
    coefficients = torch.zeros((len(schedule), len(schedule)), dtype=torch.float64)
    coefficients[:, 0] = 1.0
    for maximum_index in range(len(schedule)):
        for correction_index in range(1, maximum_index + 1):
            coefficients[maximum_index, correction_index] = 1.0 / schedule[correction_index]
    return coefficients


def score_schedule(
    schedule: tuple[float, ...],
    batch_level_grams: torch.Tensor,
    mean_level_gram: torch.Tensor,
    level_costs_ms: torch.Tensor,
) -> ScheduleEvaluation:
    if batch_level_grams.ndim != 3:
        raise ValueError("batch level Grams must have shape [B,K,K]")
    level_count = batch_level_grams.shape[1]
    if batch_level_grams.shape[2] != level_count or len(schedule) != level_count:
        raise ValueError("schedule and level Gram dimensions differ")
    if mean_level_gram.shape != (level_count, level_count):
        raise ValueError("mean level Gram has the wrong shape")
    if level_costs_ms.shape != (level_count,):
        raise ValueError("level costs have the wrong shape")

    correction = correction_grams(batch_level_grams.double())
    coefficients = outcome_coefficients(schedule)
    probabilities = maximum_level_probabilities(schedule)
    outcome_norms_squared = torch.einsum(
        "mi,bij,mj->bm", coefficients, correction, coefficients
    )
    per_batch_second_moment = outcome_norms_squared @ probabilities
    mean_gradient_norm_squared = torch.clamp(mean_level_gram[-1, -1].double(), min=0.0)
    gradient_variance = torch.clamp(
        per_batch_second_moment.mean() - mean_gradient_norm_squared, min=0.0
    )
    full_norms_squared = torch.clamp(batch_level_grams[:, -1, -1].double(), min=0.0)
    estimator_noise = torch.clamp(
        per_batch_second_moment - full_norms_squared, min=0.0
    ).mean()
    expected_cost = torch.dot(probabilities, level_costs_ms.double())
    expected_correction_coefficients = probabilities @ coefficients
    summary: dict[str, object] = {
        "tail_probabilities": list(schedule),
        "maximum_level_probabilities": probabilities.tolist(),
        "expected_correction_coefficients": expected_correction_coefficients.tolist(),
        "gradient_variance": float(gradient_variance),
        "estimator_noise_second_moment": float(estimator_noise),
        "expected_cuda_cost_ms": float(expected_cost),
        "objective": float(gradient_variance * expected_cost),
        "schedule_evaluation": "analytic_four_outcome_enumeration",
    }
    return ScheduleEvaluation(summary, per_batch_second_moment)


def pareto_candidates(scored: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for candidate in scored:
        dominated = any(
            float(other["gradient_variance"]) <= float(candidate["gradient_variance"])
            and float(other["expected_cuda_cost_ms"])
            <= float(candidate["expected_cuda_cost_ms"])
            and (
                float(other["gradient_variance"]) < float(candidate["gradient_variance"])
                or float(other["expected_cuda_cost_ms"])
                < float(candidate["expected_cuda_cost_ms"])
            )
            for other in scored
        )
        if not dominated:
            result.append(candidate)
    return sorted(result, key=lambda item: float(item["expected_cuda_cost_ms"]))


def bootstrap_efficiency_interval(
    candidate: ScheduleEvaluation,
    baseline: ScheduleEvaluation,
    full_gradient_cross_gram: torch.Tensor,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    batch_count = candidate.per_batch_second_moment.numel()
    if baseline.per_batch_second_moment.numel() != batch_count:
        raise ValueError("candidate and baseline batch counts differ")
    if full_gradient_cross_gram.shape != (batch_count, batch_count):
        raise ValueError("full-gradient cross Gram has the wrong shape")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randint(batch_count, (samples, batch_count), generator=generator)
    weights = torch.nn.functional.one_hot(indices, num_classes=batch_count).sum(dim=1).double()
    weights /= batch_count
    mean_norms_squared = torch.einsum(
        "bi,ij,bj->b", weights, full_gradient_cross_gram.double(), weights
    )

    def objectives(evaluation: ScheduleEvaluation) -> torch.Tensor:
        second = weights @ evaluation.per_batch_second_moment.double()
        variance = torch.clamp(second - mean_norms_squared, min=0.0)
        return variance * float(evaluation.summary["expected_cuda_cost_ms"])

    candidate_objectives = objectives(candidate)
    baseline_objectives = objectives(baseline)
    ratios = torch.where(
        baseline_objectives > 0,
        candidate_objectives / baseline_objectives,
        torch.full_like(candidate_objectives, float("inf")),
    )
    return float(torch.quantile(ratios, 0.025)), float(torch.quantile(ratios, 0.975))


def audit_schedule(
    schedule: tuple[float, ...],
    audit_level_grams: torch.Tensor,
    audit_mean_level_gram: torch.Tensor,
    full_gradient_cross_gram: torch.Tensor,
    level_costs_ms: torch.Tensor,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, object], bool]:
    candidate = score_schedule(
        schedule, audit_level_grams, audit_mean_level_gram, level_costs_ms
    )
    q_one = score_schedule(
        tuple(1.0 for _ in schedule),
        audit_level_grams,
        audit_mean_level_gram,
        level_costs_ms,
    )

    probabilities = maximum_level_probabilities(schedule)
    coefficients = outcome_coefficients(schedule)
    expected_coefficients = probabilities @ coefficients
    coefficient_error = expected_coefficients - torch.ones_like(expected_coefficients)
    mean_correction_gram = correction_grams(audit_mean_level_gram.double())
    error_squared = torch.clamp(
        coefficient_error @ mean_correction_gram @ coefficient_error, min=0.0
    )
    full_coefficients = torch.ones_like(expected_coefficients)
    full_norm_squared = torch.clamp(
        full_coefficients @ mean_correction_gram @ full_coefficients, min=0.0
    )
    expected_norm_squared = torch.clamp(
        expected_coefficients @ mean_correction_gram @ expected_coefficients, min=0.0
    )
    expected_full_dot = expected_coefficients @ mean_correction_gram @ full_coefficients
    error_l2 = float(torch.sqrt(error_squared))
    full_l2 = float(torch.sqrt(full_norm_squared))
    relative_l2_error = error_l2 / full_l2 if full_l2 > 0 else error_l2
    denominator = math.sqrt(float(expected_norm_squared) * float(full_norm_squared))
    cosine_similarity = float(expected_full_dot) / denominator if denominator > 0 else 1.0

    baseline_objective = float(q_one.summary["objective"])
    efficiency_ratio = (
        float(candidate.summary["objective"]) / baseline_objective
        if baseline_objective > 0
        else float("inf")
    )
    confidence_interval = bootstrap_efficiency_interval(
        candidate,
        q_one,
        full_gradient_cross_gram,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    unbiasedness_passed = relative_l2_error <= 1e-6 and cosine_similarity >= 0.99999999
    efficiency_passed = math.isfinite(confidence_interval[1]) and confidence_interval[1] < 1.0
    passed = unbiasedness_passed and efficiency_passed
    report: dict[str, object] = {
        "passed": passed,
        "unbiasedness_passed": unbiasedness_passed,
        "efficiency_passed": efficiency_passed,
        "relative_l2_error": relative_l2_error,
        "cosine_similarity": cosine_similarity,
        "error_l2": error_l2,
        "efficiency_objective_ratio_vs_q_one": (
            efficiency_ratio if math.isfinite(efficiency_ratio) else None
        ),
        "efficiency_ratio_95_percent_confidence_interval": [
            value if math.isfinite(value) else None for value in confidence_interval
        ],
        "bootstrap_parent_batch_resamples": bootstrap_samples,
        "locked_schedule": candidate.summary,
        "q_one_baseline": q_one.summary,
    }
    return report, passed


def gradient_level_and_correction_grams(
    level_gradients: list[tuple[torch.Tensor | None, ...]],
    *,
    chunk_size: int = 250_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute level and direct-correction Grams in exact gradient coordinates."""
    if not level_gradients:
        raise ValueError("at least one level gradient is required")
    parameter_count = len(level_gradients[0])
    if any(len(gradients) != parameter_count for gradients in level_gradients):
        raise ValueError("all level gradients must have the same parameter count")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    level_result = torch.zeros(
        (len(level_gradients), len(level_gradients)), dtype=torch.float64
    )
    correction_result = torch.zeros_like(level_result)
    for parameter_index in range(parameter_count):
        tensors = [gradients[parameter_index] for gradients in level_gradients]
        reference = next((tensor for tensor in tensors if tensor is not None), None)
        if reference is None:
            continue
        if any(tensor is not None and tensor.shape != reference.shape for tensor in tensors):
            raise ValueError("gradient shapes differ between context levels")
        flattened = [
            (
                torch.zeros_like(reference).reshape(-1)
                if tensor is None
                else tensor.detach().reshape(-1)
            )
            for tensor in tensors
        ]
        for start in range(0, reference.numel(), chunk_size):
            end = min(start + chunk_size, reference.numel())
            block = torch.stack([tensor[start:end].double() for tensor in flattened])
            correction_block = torch.empty_like(block)
            correction_block[0] = block[0]
            correction_block[1:] = block[1:] - block[:-1]
            level_result += (block @ block.transpose(0, 1)).cpu()
            correction_result += (
                correction_block @ correction_block.transpose(0, 1)
            ).cpu()
    return level_result, correction_result


def gradient_gram(
    level_gradients: list[tuple[torch.Tensor | None, ...]],
    *,
    chunk_size: int = 250_000,
) -> torch.Tensor:
    """Compute an exact-coordinate level Gram without flattening the model gradient."""
    return gradient_level_and_correction_grams(
        level_gradients, chunk_size=chunk_size
    )[0]


def gradient_correction_gram(
    level_gradients: list[tuple[torch.Tensor | None, ...]],
    *,
    chunk_size: int = 250_000,
) -> torch.Tensor:
    """Compute a correction Gram from explicit gradient differences."""
    return gradient_level_and_correction_grams(
        level_gradients, chunk_size=chunk_size
    )[1]


def vector_cross_gram(
    vectors: list[torch.Tensor],
    *,
    device: torch.device | str = "cpu",
    chunk_size: int = 250_000,
) -> torch.Tensor:
    if not vectors:
        raise ValueError("at least one vector is required")
    length = vectors[0].numel()
    if any(vector.ndim != 1 or vector.numel() != length for vector in vectors):
        raise ValueError("cross-Gram vectors must be equal-length one-dimensional tensors")
    target = torch.device(device)
    result = torch.zeros((len(vectors), len(vectors)), dtype=torch.float64)
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        block = torch.stack([vector[start:end] for vector in vectors]).to(
            device=target, dtype=torch.float64
        )
        result += (block @ block.transpose(0, 1)).cpu()
    return result
