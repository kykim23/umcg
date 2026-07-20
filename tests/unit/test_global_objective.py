import pytest
import torch

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
