"""Validated context hierarchy specification."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LevelSpec:
    lengths: tuple[int, ...]
    tail_probabilities: tuple[float, ...]

    def validate(self, *, max_position_embeddings: int | None = None) -> None:
        if len(self.lengths) < 2:
            raise ValueError("LevelSpec requires at least two lengths")
        if len(self.tail_probabilities) != len(self.lengths):
            raise ValueError("tail_probabilities length must equal lengths")
        if any(
            not isinstance(length, int) or isinstance(length, bool) or length <= 0
            for length in self.lengths
        ):
            raise ValueError("level lengths must be positive integers")
        if any(left >= right for left, right in zip(self.lengths, self.lengths[1:], strict=False)):
            raise ValueError("level lengths must be strictly increasing")
        if self.tail_probabilities[0] != 1.0:
            raise ValueError("first tail probability must be 1")
        if any(
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0 < probability <= 1
            for probability in self.tail_probabilities
        ):
            raise ValueError("tail probabilities must be finite and in (0, 1]")
        if any(
            left < right
            for left, right in zip(
                self.tail_probabilities, self.tail_probabilities[1:], strict=False
            )
        ):
            raise ValueError("tail probabilities must be non-increasing")
        if max_position_embeddings is not None and self.lengths[-1] > max_position_embeddings:
            raise ValueError(
                "longest level exceeds model capacity: "
                f"{self.lengths[-1]} > {max_position_embeddings}"
            )

    @property
    def num_levels(self) -> int:
        return len(self.lengths)

    @classmethod
    def from_lists(
        cls,
        lengths: list[int],
        tail_probabilities: list[float],
        *,
        max_position_embeddings: int | None = None,
    ) -> LevelSpec:
        spec = cls(tuple(lengths), tuple(float(value) for value in tail_probabilities))
        spec.validate(max_position_embeddings=max_position_embeddings)
        return spec
