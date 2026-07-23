import json
import math

import pytest
import torch
from torch import nn

from umcg.data.collate import ParentBatch
from umcg.estimators.global_objective import (
    build_global_update_plan,
    estimator_scalar,
    global_token_coefficients,
)
from umcg.estimators.levels import LevelSpec
from umcg.estimators.russian_roulette import LevelSampler


def make_batch(mask_rows):
    attention_mask = torch.tensor(mask_rows, dtype=torch.bool)
    batch_size, sequence_length = attention_mask.shape
    batch = ParentBatch(
        input_ids=torch.arange(sequence_length).repeat(batch_size, 1),
        attention_mask=attention_mask,
        causal_target_mask=attention_mask[:, :-1] & attention_mask[:, 1:],
        position_ids=torch.arange(sequence_length).repeat(batch_size, 1),
        document_hashes=[f"document-{index}" for index in range(batch_size)],
        chunk_indices=[0] * batch_size,
        token_starts=[0] * batch_size,
        token_ends=[int(row.sum()) for row in attention_mask],
        urls=[""] * batch_size,
        timestamps=[""] * batch_size,
    )
    batch.validate()
    return batch


class TinyCausalGradientModel(nn.Module):
    """Small prefix-invariant model with a real parameter-gradient vector."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(11, 4)
        self.readout = nn.Linear(4, 1)
        self.forward_lengths: list[int] = []

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.forward_lengths.append(input_ids.shape[1])
        predictions = self.readout(self.embedding(input_ids[:, :-1])).squeeze(-1)
        targets = input_ids[:, 1:].float() / 10.0
        return (predictions - targets).square() * 0.001


def flattened_gradient(scalar: torch.Tensor, model: nn.Module) -> torch.Tensor:
    gradients = torch.autograd.grad(scalar, tuple(model.parameters()))
    return torch.cat([gradient.reshape(-1) for gradient in gradients])


def segmented_4096_tokens() -> torch.Tensor:
    input_ids = torch.empty((1, 4096), dtype=torch.long)
    input_ids[:, :512] = 1
    input_ids[:, 512:1024] = 3
    input_ids[:, 1024:2048] = 6
    input_ids[:, 2048:] = 9
    input_ids[:, 1::31] = (input_ids[:, 1::31] + 1) % 11
    return input_ids


def test_q_one_gradient_is_the_full_global_token_average_across_microbatches():
    levels = LevelSpec((2, 4), (1.0, 1.0))
    batches = [
        make_batch([[True, True, True, True]]),
        make_batch([[True, True, False, False], [True, True, True, False]]),
    ]
    sampler = LevelSampler(levels, seed=7)
    plan = build_global_update_plan(
        batches,
        levels=levels,
        gradient_estimator="russian_roulette",
        level_sampler=sampler,
        rank=0,
        device=torch.device("cpu"),
    )
    parameter = torch.tensor(2.0, requires_grad=True)
    features = [
        torch.tensor([[1.0, 2.0, 3.0]]),
        torch.tensor([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
    ]
    accumulated = torch.zeros(())
    for batch, values in zip(batches, features, strict=True):
        coefficients = global_token_coefficients(
            batch.causal_target_mask,
            levels=levels,
            global_target_counts=plan.global_target_counts,
            sampled_level_index=1,
            gradient_estimator="russian_roulette",
            gradient_scale=1.0,
        )
        accumulated = accumulated + estimator_scalar(parameter * values, coefficients)
    accumulated.backward()

    valid_features = torch.cat(
        [values[batch.causal_target_mask] for batch, values in zip(batches, features, strict=True)]
    )
    expected_gradient = valid_features.mean()
    assert torch.allclose(parameter.grad, expected_gradient)


def test_accumulation_split_and_concatenated_batch_produce_the_same_scalar():
    levels = LevelSpec((2, 4), (1.0, 1.0))
    first = make_batch([[True, True, True, True]])
    second = make_batch([[True, True, False, False], [True, True, True, False]])
    batches = [first, second]
    plan = build_global_update_plan(
        batches,
        levels=levels,
        gradient_estimator="full",
        level_sampler=LevelSampler(levels, 1),
        rank=0,
        device=torch.device("cpu"),
    )
    losses = [torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])]
    split_scalar = sum(
        estimator_scalar(
            local_losses,
            global_token_coefficients(
                batch.causal_target_mask,
                levels=levels,
                global_target_counts=plan.global_target_counts,
                sampled_level_index=1,
                gradient_estimator="full",
                gradient_scale=1.0,
            ),
        )
        for batch, local_losses in zip(batches, losses, strict=True)
    )
    combined = make_batch(
        [[True, True, True, True], [True, True, False, False], [True, True, True, False]]
    )
    combined_losses = torch.cat(losses, dim=0)
    combined_scalar = estimator_scalar(
        combined_losses,
        global_token_coefficients(
            combined.causal_target_mask,
            levels=levels,
            global_target_counts=plan.global_target_counts,
            sampled_level_index=1,
            gradient_estimator="full",
            gradient_scale=1.0,
        ),
    )
    assert torch.allclose(split_scalar, combined_scalar)


def test_local_zero_targets_are_valid_when_global_denominators_are_positive():
    levels = LevelSpec((2, 4), (1.0, 1.0))
    local_mask = torch.zeros((1, 3), dtype=torch.bool)
    coefficients = global_token_coefficients(
        local_mask,
        levels=levels,
        global_target_counts=torch.tensor([3, 7]),
        sampled_level_index=1,
        gradient_estimator="russian_roulette",
        gradient_scale=2.0,
    )
    assert torch.equal(coefficients, torch.zeros_like(coefficients))


def test_full_gradient_ignores_russian_roulette_tail_probabilities():
    levels = LevelSpec((2, 4), (1.0, 0.125))
    target_mask = torch.ones((1, 3), dtype=torch.bool)
    token_losses = torch.tensor([[1.0, 2.0, 9.0]])
    coefficients = global_token_coefficients(
        target_mask,
        levels=levels,
        global_target_counts=torch.tensor([1, 3]),
        sampled_level_index=1,
        gradient_estimator="full",
        gradient_scale=1.0,
    )
    scalar = estimator_scalar(token_losses, coefficients)
    assert scalar == pytest.approx(float(token_losses.mean()))


def test_global_zero_targets_stop_the_whole_update():
    levels = LevelSpec((2, 4), (1.0, 1.0))
    empty = make_batch([[False, False, False, False]])
    with pytest.raises(RuntimeError, match="global target count is zero"):
        build_global_update_plan(
            [empty],
            levels=levels,
            gradient_estimator="full",
            level_sampler=LevelSampler(levels, 1),
            rank=0,
            device=torch.device("cpu"),
        )


def test_russian_roulette_scalar_is_monte_carlo_unbiased():
    levels = LevelSpec((2, 4, 6), (1.0, 0.5, 0.25))
    target_mask = torch.ones((1, 5), dtype=torch.bool)
    counts = torch.tensor([1, 3, 5])
    token_losses = torch.tensor([[0.5, 1.0, 2.0, 4.0, 8.0]])
    exact = token_losses.mean()
    values = []
    sampler = LevelSampler(levels, seed=1729)
    cached = {
        index: float(
            estimator_scalar(
                token_losses[:, : levels.lengths[index] - 1],
                global_token_coefficients(
                    target_mask,
                    levels=levels,
                    global_target_counts=counts,
                    sampled_level_index=index,
                    gradient_estimator="russian_roulette",
                    gradient_scale=1.0,
                ),
            )
        )
        for index in range(levels.num_levels)
    }
    for _ in range(30_000):
        values.append(cached[sampler.sample()])
    assert sum(values) / len(values) == pytest.approx(float(exact), abs=0.05)


def test_tail_probabilities_induce_the_expected_maximum_level_probabilities():
    levels = LevelSpec((512, 1024, 2048, 4096), (1.0, 0.5, 0.25, 0.125))
    sampler = LevelSampler(levels, seed=1729)
    torch.testing.assert_close(
        sampler.maximum_level_probabilities,
        torch.tensor([0.5, 0.25, 0.125, 0.125], dtype=torch.float64),
        rtol=0,
        atol=0,
    )


def test_selected_context_gradient_accumulates_every_lower_correction_in_one_forward():
    torch.manual_seed(11)
    model = TinyCausalGradientModel()
    input_ids = segmented_4096_tokens()
    levels = LevelSpec((512, 1024, 2048, 4096), (1.0, 0.5, 0.25, 0.125))
    target_mask = torch.ones((1, 4095), dtype=torch.bool)
    counts = torch.tensor([511, 1023, 2047, 4095])

    full_level_gradients = []
    for length in levels.lengths:
        losses = model(input_ids[:, :length])
        full_level_gradients.append(flattened_gradient(losses.mean(), model))

    selected_gradients = []
    for level_index, length in enumerate(levels.lengths):
        model.forward_lengths.clear()
        losses = model(input_ids[:, :length])
        coefficients = global_token_coefficients(
            target_mask,
            levels=levels,
            global_target_counts=counts,
            sampled_level_index=level_index,
            gradient_estimator="russian_roulette",
            gradient_scale=1.0,
        )
        selected_gradients.append(flattened_gradient(estimator_scalar(losses, coefficients), model))
        assert model.forward_lengths == [length]

    g_512, g_1024, g_2048, g_4096 = full_level_gradients
    expected_2048 = g_512 + 2 * (g_1024 - g_512) + 4 * (g_2048 - g_1024)
    expected_4096 = expected_2048 + 8 * (g_4096 - g_2048)
    torch.testing.assert_close(selected_gradients[2], expected_2048, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(selected_gradients[3], expected_4096, rtol=1e-5, atol=1e-6)


def test_4096_monte_carlo_mean_parameter_gradient_converges_to_full_gradient():
    torch.manual_seed(11)
    model = TinyCausalGradientModel()
    input_ids = segmented_4096_tokens()
    levels = LevelSpec((512, 1024, 2048, 4096), (1.0, 0.5, 0.25, 0.125))
    target_mask = torch.ones((1, 4095), dtype=torch.bool)
    counts = torch.tensor([511, 1023, 2047, 4095])

    full_gradient = flattened_gradient(model(input_ids).mean(), model)
    outcome_gradients = []
    for level_index, length in enumerate(levels.lengths):
        losses = model(input_ids[:, :length])
        coefficients = global_token_coefficients(
            target_mask,
            levels=levels,
            global_target_counts=counts,
            sampled_level_index=level_index,
            gradient_estimator="russian_roulette",
            gradient_scale=1.0,
        )
        outcome_gradients.append(flattened_gradient(estimator_scalar(losses, coefficients), model))
    outcomes = torch.stack(outcome_gradients).double()
    probabilities = LevelSampler(levels, seed=1729).maximum_level_probabilities

    exact_expectation = (probabilities.unsqueeze(1) * outcomes).sum(dim=0)
    torch.testing.assert_close(exact_expectation, full_gradient.double(), rtol=1e-5, atol=1e-6)

    sample_count = 65_536
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1729)
    sampled_indices = torch.multinomial(
        probabilities,
        num_samples=sample_count,
        replacement=True,
        generator=generator,
    )
    sampled_gradients = outcomes[sampled_indices]
    mean_gradient = sampled_gradients.mean(dim=0)
    error_l2 = torch.linalg.vector_norm(mean_gradient - full_gradient.double())
    full_l2 = torch.linalg.vector_norm(full_gradient.double())
    relative_l2_error = error_l2 / full_l2
    cosine_similarity = torch.nn.functional.cosine_similarity(
        mean_gradient, full_gradient.double(), dim=0
    )
    standard_error_l2 = torch.sqrt(
        sampled_gradients.var(dim=0, unbiased=True).sum() / sample_count
    )
    error_over_standard_error = error_l2 / standard_error_l2
    metrics = {
        "samples": sample_count,
        "gradient_dimension": full_gradient.numel(),
        "relative_l2_error": float(relative_l2_error),
        "cosine_similarity": float(cosine_similarity),
        "standard_error_l2": float(standard_error_l2),
        "relative_standard_error_l2": float(standard_error_l2 / full_l2),
        "error_over_standard_error": float(error_over_standard_error),
    }
    print("UMCG_4096_GRADIENT_METRICS=" + json.dumps(metrics, sort_keys=True))

    assert math.isfinite(metrics["relative_l2_error"])
    assert relative_l2_error <= 0.02
    assert cosine_similarity >= 0.999
    assert error_l2 <= 4 * standard_error_l2
