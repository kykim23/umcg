"""Global token-average full and Russian-roulette update objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as distributed

from umcg.data.collate import ParentBatch
from umcg.estimators.levels import LevelSpec
from umcg.estimators.russian_roulette import LevelSampler


@dataclass(frozen=True)
class GlobalUpdatePlan:
    sampled_level_indices: tuple[int, ...]
    global_target_counts: torch.Tensor


def local_target_counts(
    batches: list[ParentBatch], levels: LevelSpec, device: torch.device
) -> torch.Tensor:
    counts = torch.zeros(levels.num_levels, dtype=torch.long, device=device)
    for batch in batches:
        batch.validate()
        for index, length in enumerate(levels.lengths):
            counts[index] += batch.causal_target_mask[:, : length - 1].sum().to(device)
    return counts


def build_global_update_plan(
    batches: list[ParentBatch],
    *,
    levels: LevelSpec,
    gradient_estimator: str,
    level_sampler: LevelSampler,
    rank: int,
    device: torch.device,
) -> GlobalUpdatePlan:
    counts = local_target_counts(batches, levels, device)
    if distributed.is_available() and distributed.is_initialized():
        distributed.all_reduce(counts, op=distributed.ReduceOp.SUM)
    if (counts <= 0).any():
        empty_levels = [
            levels.lengths[index] for index, count in enumerate(counts.tolist()) if count <= 0
        ]
        raise RuntimeError(f"global target count is zero at context levels {empty_levels}")
    if gradient_estimator == "full":
        sampled = [levels.num_levels - 1] * len(batches)
    elif gradient_estimator == "russian_roulette":
        tensor = torch.zeros(len(batches), dtype=torch.long, device=device)
        if rank == 0:
            tensor.copy_(
                torch.tensor(
                    [level_sampler.sample() for _ in batches],
                    dtype=torch.long,
                    device=device,
                )
            )
        if distributed.is_available() and distributed.is_initialized():
            distributed.broadcast(tensor, src=0)
        sampled = [int(item) for item in tensor.tolist()]
        if rank != 0:
            for item in sampled:
                level_sampler.observe_broadcast(item)
    else:
        raise ValueError("gradient_estimator must be full or russian_roulette")
    return GlobalUpdatePlan(tuple(sampled), counts)


def global_token_coefficients(
    local_target_mask: torch.Tensor,
    *,
    levels: LevelSpec,
    global_target_counts: torch.Tensor,
    sampled_level_index: int,
    gradient_estimator: str,
    gradient_scale: float,
) -> torch.Tensor:
    if local_target_mask.ndim != 2 or local_target_mask.dtype != torch.bool:
        raise ValueError("local_target_mask must have shape [batch, maximum_context-1]")
    if global_target_counts.shape != (levels.num_levels,):
        raise ValueError("global_target_counts has the wrong shape")
    if not 0 <= sampled_level_index < levels.num_levels:
        raise ValueError("sampled_level_index is outside the configured levels")
    if gradient_estimator not in {"full", "russian_roulette"}:
        raise ValueError("gradient_estimator must be full or russian_roulette")
    active_width = levels.lengths[sampled_level_index] - 1
    coefficients = torch.zeros(
        (local_target_mask.shape[0], active_width),
        dtype=torch.float32,
        device=local_target_mask.device,
    )

    def normalized_mask(level_index: int) -> torch.Tensor:
        width = levels.lengths[level_index] - 1
        result = torch.zeros_like(coefficients)
        denominator = global_target_counts[level_index].to(
            device=coefficients.device, dtype=torch.float32
        )
        if denominator.item() <= 0:
            raise RuntimeError("global target denominator must be positive")
        result[:, :width] = local_target_mask[:, :width].to(torch.float32) / denominator
        return result

    coefficients += normalized_mask(0)
    for level_index in range(1, sampled_level_index + 1):
        correction = normalized_mask(level_index) - normalized_mask(level_index - 1)
        probability = (
            1.0 if gradient_estimator == "full" else levels.tail_probabilities[level_index]
        )
        coefficients += correction / probability
    coefficients *= float(gradient_scale)
    if not torch.isfinite(coefficients).all():
        raise FloatingPointError("non-finite global UMCG coefficients")
    return coefficients


def estimator_scalar(token_losses: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    if token_losses.shape != coefficients.shape:
        raise ValueError("token_losses and coefficients must have identical shapes")
    scalar = (token_losses.float() * coefficients).sum()
    if not torch.isfinite(scalar):
        raise FloatingPointError("non-finite distributed estimator scalar")
    return scalar
