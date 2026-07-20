"""Serializable global counters at optimizer-update boundaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class TrainingState:
    optimizer_update: int = 0
    valid_tokens: int = 0
    full_context_equivalent_tokens: int = 0
    selected_level_counts: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> TrainingState:
        return cls(
            optimizer_update=int(value["optimizer_update"]),
            valid_tokens=int(value["valid_tokens"]),
            full_context_equivalent_tokens=int(value["full_context_equivalent_tokens"]),
            selected_level_counts=[int(item) for item in value["selected_level_counts"]],
        )
