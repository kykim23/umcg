"""Deterministic single-document data used only by smoke tests."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from umcg.data.parent_dataset import ParentSample


class SyntheticParentDataset(Dataset):
    def __init__(
        self,
        *,
        num_parents: int,
        parent_length: int,
        vocab_size: int,
        seed: int,
        pad_token_id: int = 0,
    ) -> None:
        if num_parents <= 0 or parent_length < 2 or vocab_size < 4:
            raise ValueError("invalid synthetic dataset dimensions")
        self.num_parents = num_parents
        self.parent_length = parent_length
        self.vocab_size = vocab_size
        self.seed = seed
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return self.num_parents

    def __getitem__(self, index: int) -> ParentSample:
        if not 0 <= index < self.num_parents:
            raise IndexError(index)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + index * 1_000_003)
        values = torch.randint(0, self.vocab_size - 1, (self.parent_length,), generator=generator)
        input_ids = values + (values >= self.pad_token_id).to(values.dtype)
        attention_mask = torch.ones(self.parent_length, dtype=torch.bool)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "causal_target_mask": attention_mask[:-1] & attention_mask[1:],
            "position_ids": torch.arange(self.parent_length, dtype=torch.long),
            "document_hash": f"synthetic-{index:08d}",
            "chunk_index": 0,
            "token_start": 0,
            "token_end": self.parent_length,
            "url": "",
            "timestamp": "",
        }
