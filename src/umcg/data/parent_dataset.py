"""Single-document parent sample type."""

from __future__ import annotations

from typing import TypedDict

import torch


class ParentSample(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    causal_target_mask: torch.Tensor
    position_ids: torch.Tensor
    document_hash: str
    chunk_index: int
    token_start: int
    token_end: int
    url: str
    timestamp: str
