"""Batched single-document parent contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ParentBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    causal_target_mask: torch.Tensor
    position_ids: torch.Tensor
    document_hashes: list[str]
    chunk_indices: list[int]
    token_starts: list[int]
    token_ends: list[int]
    urls: list[str]
    timestamps: list[str]

    def validate(self) -> None:
        if self.input_ids.ndim != 2 or self.input_ids.dtype != torch.long:
            raise ValueError("input_ids must have shape [batch, sequence] and dtype torch.long")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        if self.attention_mask.dtype != torch.bool:
            raise ValueError("attention_mask must use torch.bool")
        expected_target_shape = (self.input_ids.shape[0], self.input_ids.shape[1] - 1)
        if self.causal_target_mask.shape != expected_target_shape:
            raise ValueError("causal_target_mask must have shape [batch, sequence - 1]")
        if self.causal_target_mask.dtype != torch.bool:
            raise ValueError("causal_target_mask must use torch.bool")
        if self.position_ids.shape != self.input_ids.shape or self.position_ids.dtype != torch.long:
            raise ValueError("position_ids must match input_ids and use torch.long")
        if (self.position_ids < 0).any():
            raise ValueError("position_ids must be non-negative")
        if not torch.equal(
            self.causal_target_mask,
            self.attention_mask[:, :-1] & self.attention_mask[:, 1:],
        ):
            raise ValueError("causal_target_mask does not match the attention mask")
        batch_size = self.input_ids.shape[0]
        metadata = (
            self.document_hashes,
            self.chunk_indices,
            self.token_starts,
            self.token_ends,
            self.urls,
            self.timestamps,
        )
        if any(len(values) != batch_size for values in metadata):
            raise ValueError("every metadata field must have one value per sample")

    def to(self, device: torch.device | str) -> ParentBatch:
        self.validate()
        return ParentBatch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            causal_target_mask=self.causal_target_mask.to(device, non_blocking=True),
            position_ids=self.position_ids.to(device, non_blocking=True),
            document_hashes=list(self.document_hashes),
            chunk_indices=list(self.chunk_indices),
            token_starts=list(self.token_starts),
            token_ends=list(self.token_ends),
            urls=list(self.urls),
            timestamps=list(self.timestamps),
        )

    def prefix(self, active_length: int) -> ParentBatch:
        if not 2 <= active_length <= self.input_ids.shape[1]:
            raise ValueError("active_length is outside the parent sequence")
        result = ParentBatch(
            input_ids=self.input_ids[:, :active_length],
            attention_mask=self.attention_mask[:, :active_length],
            causal_target_mask=self.causal_target_mask[:, : active_length - 1],
            position_ids=self.position_ids[:, :active_length],
            document_hashes=list(self.document_hashes),
            chunk_indices=list(self.chunk_indices),
            token_starts=list(self.token_starts),
            token_ends=list(self.token_ends),
            urls=list(self.urls),
            timestamps=list(self.timestamps),
        )
        result.validate()
        return result


def collate_parent_samples(samples: list[dict[str, Any]]) -> ParentBatch:
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    batch = ParentBatch(
        input_ids=torch.stack([sample["input_ids"] for sample in samples]),
        attention_mask=torch.stack([sample["attention_mask"] for sample in samples]),
        causal_target_mask=torch.stack([sample["causal_target_mask"] for sample in samples]),
        position_ids=torch.stack([sample["position_ids"] for sample in samples]),
        document_hashes=[str(sample["document_hash"]) for sample in samples],
        chunk_indices=[int(sample["chunk_index"]) for sample in samples],
        token_starts=[int(sample["token_start"]) for sample in samples],
        token_ends=[int(sample["token_end"]) for sample in samples],
        urls=[str(sample.get("url", "")) for sample in samples],
        timestamps=[str(sample.get("timestamp", "")) for sample in samples],
    )
    batch.validate()
    return batch
