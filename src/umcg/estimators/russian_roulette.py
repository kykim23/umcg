"""Replayable categorical sampler induced by survival probabilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from umcg.estimators.levels import LevelSpec


class LevelSampler:
    def __init__(self, levels: LevelSpec, seed: int) -> None:
        levels.validate()
        self.levels = levels
        self.seed = int(seed)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(self.seed)
        self.counts = torch.zeros(levels.num_levels, dtype=torch.long)
        self.sample_count = 0

    @property
    def maximum_level_probabilities(self) -> torch.Tensor:
        tail = self.levels.tail_probabilities
        values = [tail[index] - tail[index + 1] for index in range(len(tail) - 1)]
        values.append(tail[-1])
        result = torch.tensor(values, dtype=torch.float64)
        if (result < 0).any() or not torch.isclose(result.sum(), result.new_ones(())):
            raise ValueError("tail probabilities do not define a categorical distribution")
        return result

    def sample(self) -> int:
        sampled = int(
            torch.multinomial(
                self.maximum_level_probabilities,
                num_samples=1,
                replacement=True,
                generator=self.generator,
            ).item()
        )
        self.counts[sampled] += 1
        self.sample_count += 1
        return sampled

    def observe_broadcast(self, sampled_level_index: int) -> None:
        if not 0 <= sampled_level_index < self.levels.num_levels:
            raise ValueError("broadcast level index is outside the configured levels")
        self.counts[sampled_level_index] += 1
        self.sample_count += 1

    def empirical_frequencies(self) -> tuple[float, ...]:
        if self.sample_count == 0:
            return tuple(0.0 for _ in range(self.levels.num_levels))
        return tuple(float(value) / self.sample_count for value in self.counts.tolist())

    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "lengths": self.levels.lengths,
            "tail_probabilities": self.levels.tail_probabilities,
            "generator_state": self.generator.get_state().clone(),
            "counts": self.counts.clone(),
            "sample_count": self.sample_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if tuple(state["lengths"]) != self.levels.lengths:
            raise ValueError("LevelSampler state has different lengths")
        if tuple(state["tail_probabilities"]) != self.levels.tail_probabilities:
            raise ValueError("LevelSampler state has different tail probabilities")
        counts = torch.as_tensor(state["counts"], dtype=torch.long)
        if counts.shape != self.counts.shape:
            raise ValueError("LevelSampler state has invalid counts shape")
        self.seed = int(state["seed"])
        self.generator.set_state(torch.as_tensor(state["generator_state"], dtype=torch.uint8))
        self.counts.copy_(counts)
        self.sample_count = int(state["sample_count"])
        if self.sample_count != int(self.counts.sum().item()):
            raise ValueError("LevelSampler state sample_count does not match counts")
